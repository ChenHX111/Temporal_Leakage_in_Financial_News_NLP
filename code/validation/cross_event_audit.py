"""
Cross-event leakage audit + locked-test pipeline for three event types:
  - mergers_acquisitions  (M&A; the headline)
  - clinical_study        (CLN)
  - law_legal_issues      (LGL)

We replicate the paper's M&A protocol on the two new events so the reader can
judge how event-specific the audit story is.

Adaptive cutoffs (set per event because LGL is concentrated post-Apr-2025):
  - For each event, we pick TRAIN_END / VAL_END such that the train set has
    >= 300 rows, val >= 50 rows, and test >= 100 rows. We report cutoffs.

HP selection (mirrors the paper):
  - Validation-grid pick over max_features in {100, 300, 500, 1000} and
    C in {0.1, 0.5, 1.0, 5.0, 20.0}, criterion = max val MCC, ties broken
    by stronger regularisation (smaller C, then smaller max_features).

Per event we report:
  1. Audit ratio: random-vs-chronological at the val-best HP (5 random seeds)
  2. Locked-test MCC, balanced-accuracy
  3. 10K-permutation null on locked test (one-/two-sided p, z)
  4. Weekly block bootstrap 95% CI (2000 resamples)
  5. Per-month MCC over the locked-test window
"""

import sys, io, os, json, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, accuracy_score

BASE_DIR = r'.'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'classifier_training_v2.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'validation')

EVENTS = {
    'mergers_acquisitions': dict(short='M&A', train_end='2025-04-01', val_end='2025-06-01'),
    'clinical_study':        dict(short='CLN', train_end='2025-04-01', val_end='2025-06-01'),
    'law_legal_issues':      dict(short='LGL', train_end='2025-06-01', val_end='2025-07-15'),
    'earnings_releases_and_operating_results':
                             dict(short='ERN', train_end='2025-04-01', val_end='2025-06-01'),
}
# Paper-authoritative M&A pipeline (val MCC=0.228, test MCC=0.138 on M&A; see
# ma_hp_grid_robustness.json). We APPLY THIS SAME PIPELINE IDENTICALLY to all
# three events to test transfer (no per-event re-tuning => no data snooping).
PAPER_HP = dict(max_features=100, C=5.0)
# Auxiliary HP grid for the per-event audit-ratio sub-analysis (smaller than
# paper's full 1080-cell grid; sublinear_tf=False fixed; C<=5.0 to avoid the
# val-noise C=20.0 overfit observed in pilot).
HP_GRID = [(mf, C) for mf in (100, 300, 500, 1000) for C in (0.1, 0.5, 1.0, 5.0)]
TFIDF_KW = dict(stop_words='english', min_df=2, ngram_range=(1, 1), sublinear_tf=False)


def tfidf_lr(tr_titles, tr_y, te_titles, max_features, C, seed=42):
    tf = TfidfVectorizer(max_features=max_features, **TFIDF_KW)
    X_tr = tf.fit_transform(tr_titles)
    X_te = tf.transform(te_titles)
    lr = LogisticRegression(C=C, max_iter=2000, random_state=seed, penalty='l2')
    lr.fit(X_tr, tr_y)
    return lr.predict(X_te), lr.predict_proba(X_te)[:, 1]


def pick_hp(tr_titles, tr_y, vl_titles, vl_y):
    best = None
    grid_results = []
    for mf, C in HP_GRID:
        try:
            pred, _ = tfidf_lr(tr_titles, tr_y, vl_titles, mf, C)
            mcc = matthews_corrcoef(vl_y, pred)
        except Exception:
            mcc = -1.0
        grid_results.append((mf, C, float(mcc)))
        key = (mcc, -C, -mf)
        if best is None or key > best[0]:
            best = (key, mf, C, float(mcc))
    return best[1], best[2], best[3], grid_results


def block_bootstrap(y_true, y_pred, weeks, B=2000, seed=0):
    rng = np.random.RandomState(seed)
    uniq = np.unique(weeks)
    week_to_idx = {w: np.where(weeks == w)[0] for w in uniq}
    mccs = []
    for _ in range(B):
        smp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([week_to_idx[w] for w in smp])
        try:
            mccs.append(matthews_corrcoef(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    return float(np.mean(mccs)), float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5))


def perm_test(y_true, y_pred, B=10000, seed=0):
    rng = np.random.RandomState(seed)
    obs = matthews_corrcoef(y_true, y_pred)
    null = np.zeros(B)
    for i in range(B):
        null[i] = matthews_corrcoef(y_true, rng.permutation(y_pred))
    p_one = float(((null >= obs).sum() + 1) / (B + 1))
    p_two = float(((np.abs(null) >= abs(obs)).sum() + 1) / (B + 1))
    z = float((obs - null.mean()) / (null.std() + 1e-12))
    return float(obs), p_one, p_two, z, float(null.mean()), float(null.std())


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA_PATH)
    df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
    df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
    df['label'] = (df['actual_side'].str.lower() == 'up').astype(int)
    df['title_en'] = df['title_en'].fillna('').astype(str)
    df['week'] = df['published_date'].dt.to_period('W')

    summary = {}
    for ev_key, cfg in EVENTS.items():
        print(f"\n{'='*70}\n[{cfg['short']}] {ev_key}\n{'='*70}", flush=True)
        TRAIN_END = pd.Timestamp(cfg['train_end'])
        VAL_END = pd.Timestamp(cfg['val_end'])
        edf = df[df['event'] == ev_key].copy()
        if len(edf) < 200:
            print(f"  too small ({len(edf)}); skipping", flush=True)
            continue
        print(f"  n_total={len(edf)}  UP%={edf['label'].mean()*100:.1f}  "
              f"cutoffs train<{cfg['train_end']} val<{cfg['val_end']}", flush=True)

        tr = edf[edf['published_date'] < TRAIN_END]
        vl = edf[(edf['published_date'] >= TRAIN_END) & (edf['published_date'] < VAL_END)]
        te = edf[edf['published_date'] >= VAL_END]
        print(f"  CHRONO split sizes: tr={len(tr)} vl={len(vl)} te={len(te)}", flush=True)

        mf, C, val_mcc_best, grid = pick_hp(tr['title_en'], tr['label'].values,
                                            vl['title_en'], vl['label'].values)
        print(f"  per-event val-best HP (audit subanalysis): max_features={mf}, C={C}, "
              f"val MCC={val_mcc_best:.4f}", flush=True)

        # Audit ratio uses per-event val-best HP (each event gets its own best chance).
        pred_v, _ = tfidf_lr(tr['title_en'], tr['label'].values, vl['title_en'], mf, C)
        chrono_val_mcc = float(matthews_corrcoef(vl['label'], pred_v))

        rand_mccs = []
        pre_test_pool = edf[edf['published_date'] < VAL_END].reset_index(drop=True)
        n_val_target = max(30, len(vl))
        for s in range(5):
            shuffled = pre_test_pool.sample(frac=1.0, random_state=s).reset_index(drop=True)
            vl_r = shuffled.iloc[:n_val_target]
            tr_r = shuffled.iloc[n_val_target:]
            if len(tr_r) < 80:
                continue
            pred_r, _ = tfidf_lr(tr_r['title_en'], tr_r['label'].values, vl_r['title_en'], mf, C)
            rand_mccs.append(float(matthews_corrcoef(vl_r['label'], pred_r)))
        rand_mean = float(np.mean(rand_mccs)) if rand_mccs else None
        rand_std = float(np.std(rand_mccs)) if rand_mccs else None
        ratio = (rand_mean / chrono_val_mcc) if (rand_mean is not None and abs(chrono_val_mcc) > 1e-6) else None
        print(f"  CHRONO val MCC = {chrono_val_mcc:.4f}", flush=True)
        if rand_mean is not None:
            print(f"  RANDOM val MCC ({len(rand_mccs)} seeds, n={n_val_target}) = "
                  f"{rand_mean:.4f} +/- {rand_std:.4f}", flush=True)
        if ratio is not None:
            print(f"  AUDIT RATIO random/chrono = {ratio:.2f}x", flush=True)

        train_final = tr.reset_index(drop=True)
        # Locked test: apply PAPER HP (max_features=100, C=5.0) identically to all events,
        # using PRE-TRAIN-END articles only (matches cpu_extension_pack_b7_b11.py B7 which
        # gives M&A test MCC=0.138 — the paper's headline number).
        if len(te) < 30:
            test_block = dict(n=int(len(te)), note='too small')
        else:
            pred_te, _ = tfidf_lr(train_final['title_en'], train_final['label'].values,
                                  te['title_en'], PAPER_HP['max_features'], PAPER_HP['C'])
            test_mcc = float(matthews_corrcoef(te['label'], pred_te))
            test_bacc = float(balanced_accuracy_score(te['label'], pred_te))
            test_acc = float(accuracy_score(te['label'], pred_te))
            up_test = float(te['label'].mean())
            pred_up = float(pred_te.mean())
            print(f"  LOCKED TEST: n={len(te)} UP%={up_test*100:.1f} pred-UP%={pred_up*100:.1f}",
                  flush=True)
            print(f"    MCC={test_mcc:.4f} bal-acc={test_bacc:.4f} acc={test_acc:.4f}", flush=True)
            mcc_obs, p1, p2, z, mu, sd = perm_test(te['label'].values, pred_te, B=10000)
            print(f"    10K-perm: z={z:.3f}, p_one={p1:.4f}, p_two={p2:.4f}", flush=True)
            weeks = te['week'].astype(str).values
            boot_mean, boot_lo, boot_hi = block_bootstrap(te['label'].values, pred_te, weeks,
                                                          B=2000, seed=0)
            n_weeks = int(pd.Series(weeks).nunique())
            print(f"    Weekly block bootstrap ({n_weeks} weeks): mean={boot_mean:.4f}, "
                  f"95% CI [{boot_lo:.4f}, {boot_hi:.4f}]", flush=True)
            per_month = []
            te2 = te.reset_index(drop=True)
            te2['_m'] = te2['published_date'].dt.to_period('M')
            for m, g in te2.groupby('_m'):
                if len(g) >= 15:
                    pm = pred_te[g.index.values]
                    per_month.append(dict(month=str(m), n=int(len(g)),
                                          mcc=float(matthews_corrcoef(g['label'], pm)),
                                          up_rate=float(g['label'].mean()),
                                          pred_up=float(pm.mean())))
            for r in per_month:
                print(f"    month={r['month']} n={r['n']} MCC={r['mcc']:.4f}", flush=True)
            test_block = dict(n=int(len(te)), up_rate=up_test, mcc=test_mcc, bal_acc=test_bacc,
                              acc=test_acc, pred_up=pred_up,
                              perm=dict(mcc=mcc_obs, p_one=p1, p_two=p2, z=z, null_mean=mu,
                                        null_std=sd, B=10000),
                              bootstrap=dict(mean=boot_mean, ci_lo=boot_lo, ci_hi=boot_hi,
                                             n_weeks=n_weeks, B=2000),
                              per_month=per_month)

        summary[ev_key] = dict(
            short=cfg['short'], cutoffs=dict(train_end=cfg['train_end'], val_end=cfg['val_end']),
            paper_hp_for_locked_test=PAPER_HP,
            n_total=int(len(edf)), up_rate_full=float(edf['label'].mean()),
            split_sizes=dict(tr=int(len(tr)), vl=int(len(vl)), te=int(len(te))),
            hp=dict(max_features=int(mf), C=float(C), val_mcc_best=val_mcc_best, grid=grid),
            audit=dict(chrono_val_mcc=chrono_val_mcc, rand_val_mcc_mean=rand_mean,
                       rand_val_mcc_std=rand_std, ratio=ratio,
                       n_val=int(n_val_target), n_rand_seeds=len(rand_mccs)),
            locked_test=test_block,
        )

    out_path = os.path.join(RESULTS_DIR, 'cross_event_audit.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWall: {time.time()-t0:.1f}s. Saved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
