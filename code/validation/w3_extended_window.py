"""
W3 closure: extended M&A locked-test window (Mar-Aug 2025 ≈ n=1500)
addresses reviewer concern that the paper's headline rests on a single
3-month near-temporal test (Jun-Aug 2025, n=786).

Protocol (analogous to App R B8 train+val merge, but with shifted cutoff):
- Train: M&A articles with published_date < 2025-03-01
- Test:  M&A articles with published_date >= 2025-03-01 (Mar-Aug 2025)
- HP fixed at paper-authoritative (max_features=100, C=5.0, sublinear_tf=False,
  min_df=2, ngram_range=(1,1)).
- Tests: 10K-permutation p, weekly block-bootstrap 95% CI, per-month MCC.

Output: results/validation/w3_extended_window.json
"""
import os, sys, io, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

BASE = r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package"
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(BASE, "results", "validation", "w3_extended_window.json")

EXTENDED_TEST_START = pd.Timestamp('2025-03-01')
PAPER_TEST_START = pd.Timestamp('2025-06-01')

MA_HP = dict(max_features=100, C=5.0, sublinear_tf=False, min_df=2,
             ngram_range=(1, 1))


def safe_mcc(y, yp):
    if len(np.unique(y)) < 2 or len(np.unique(yp)) < 2:
        return 0.0
    return float(matthews_corrcoef(y, yp))


def load_ma():
    df = pd.read_parquet(DATA)
    df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
    df = df[df['actual_side'].astype(str).str.lower().isin(['up', 'down'])].copy()
    df['y'] = (df['actual_side'].astype(str).str.lower() == 'up').astype(int)
    df['title_en'] = df['title_en'].fillna('').astype(str)
    ma = df[df['event'] == 'mergers_acquisitions'].copy()
    ma = ma.sort_values('published_date').reset_index(drop=True)
    return ma


def perm_test(yhat, ytrue, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    obs = safe_mcc(ytrue, yhat)
    null = np.array([safe_mcc(rng.permutation(ytrue), yhat) for _ in range(n)])
    p_one = float((null >= obs).mean())
    p_two = float((np.abs(null) >= abs(obs)).mean())
    z = (obs - null.mean()) / (null.std() + 1e-12)
    return {"mcc": obs, "z": float(z), "p_one": p_one, "p_two": p_two,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "n_perm": n}


def weekly_block_bootstrap(yhat, ytrue, dates, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    weeks = pd.to_datetime(dates).to_period('W-MON')
    week_to_idx = {}
    for i, w in enumerate(weeks):
        week_to_idx.setdefault(w, []).append(i)
    week_keys = list(week_to_idx.keys())
    W = len(week_keys)
    boot = []
    for _ in range(B):
        sampled_weeks = rng.choice(W, size=W, replace=True)
        idx = []
        for sw in sampled_weeks:
            idx.extend(week_to_idx[week_keys[sw]])
        if len(idx) == 0:
            continue
        boot.append(safe_mcc(ytrue[idx], yhat[idx]))
    boot = np.array(boot)
    return {"mean": float(boot.mean()), "lo": float(np.quantile(boot, 0.025)),
            "hi": float(np.quantile(boot, 0.975)), "n_blocks": int(W),
            "n_resamples": int(B)}


def evaluate(ma, train_end, test_start, label):
    tr = ma[ma['published_date'] < train_end]
    te = ma[ma['published_date'] >= test_start].copy()
    print(f"\n[{label}] train_end<{train_end.date()}  test_start>={test_start.date()}")
    print(f"  n_train = {len(tr):>5}  (UP={int(tr['y'].sum())}, DOWN={int((1-tr['y']).sum())})")
    print(f"  n_test  = {len(te):>5}  (UP={int(te['y'].sum())}, DOWN={int((1-te['y']).sum())})")
    print(f"  test period: {te['published_date'].min()} .. {te['published_date'].max()}")
    tf = TfidfVectorizer(max_features=MA_HP['max_features'], stop_words='english',
                         min_df=MA_HP['min_df'], sublinear_tf=MA_HP['sublinear_tf'],
                         ngram_range=MA_HP['ngram_range'])
    Xtr = tf.fit_transform(tr['title_en'])
    Xte = tf.transform(te['title_en'])
    clf = LogisticRegression(max_iter=2000, C=MA_HP['C'], random_state=42)
    clf.fit(Xtr, tr['y'].values)
    yhat = clf.predict(Xte)
    ytrue = te['y'].values
    mcc = safe_mcc(ytrue, yhat)
    bacc = float(balanced_accuracy_score(ytrue, yhat))
    print(f"  MCC = {mcc:+.4f}   BalAcc = {bacc:.4f}   pred_up = {float(yhat.mean()):.4f}")

    perm = perm_test(yhat, ytrue, n=10000, seed=42)
    print(f"  10K-perm: z={perm['z']:.2f}  p_one={perm['p_one']:.4f}  p_two={perm['p_two']:.4f}")

    boot = weekly_block_bootstrap(yhat, ytrue, te['published_date'].values, B=1000, seed=42)
    print(f"  weekly block-bootstrap 95%CI: [{boot['lo']:+.4f}, {boot['hi']:+.4f}]  (mean {boot['mean']:+.4f}, {boot['n_blocks']} weeks)")

    te['month'] = te['published_date'].dt.to_period('M').astype(str)
    by_month = []
    for m, sub in te.groupby('month'):
        idx = sub.index.values - te.index.min()
        # safer: recompute indices in te frame
        mask = te['month'].values == m
        m_mcc = safe_mcc(ytrue[mask], yhat[mask])
        n = int(mask.sum())
        by_month.append({"month": m, "n": n, "mcc": m_mcc,
                         "pred_up_rate": float(yhat[mask].mean()),
                         "label_up_rate": float(ytrue[mask].mean())})
        print(f"  {m}: n={n:>4}  MCC={m_mcc:+.4f}  pred_up={float(yhat[mask].mean()):.2f}")

    return {
        "label": label,
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_train_up": int(tr['y'].sum()),
        "n_test_up": int(te['y'].sum()),
        "test_period_min": str(te['published_date'].min()),
        "test_period_max": str(te['published_date'].max()),
        "mcc": mcc,
        "balacc": bacc,
        "pred_up_rate": float(yhat.mean()),
        "label_up_rate": float(ytrue.mean()),
        "perm_test": perm,
        "block_bootstrap": boot,
        "per_month": by_month,
        "hp": MA_HP,
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("W3 extended-window M&A locked-test")
    print("=" * 60)
    ma = load_ma()
    print(f"Total M&A articles (UP/DOWN only): {len(ma)}")
    print(f"Date range: {ma['published_date'].min()} .. {ma['published_date'].max()}")
    print(f"Per-year: {ma['year'].value_counts().sort_index().to_dict()}")

    results = {}
    # Scenario 1: extended Mar-Aug 2025 window
    results['extended_mar_aug_2025'] = evaluate(
        ma, train_end=EXTENDED_TEST_START, test_start=EXTENDED_TEST_START,
        label='extended_mar_aug_2025')
    # Scenario 2: paper's original Jun-Aug 2025 (sanity check; train+val merge analog)
    results['paper_jun_aug_2025_trainvalmerge'] = evaluate(
        ma, train_end=PAPER_TEST_START, test_start=PAPER_TEST_START,
        label='paper_jun_aug_2025_trainvalmerge')

    out = {
        "meta": {
            "timestamp": pd.Timestamp.now().isoformat(),
            "elapsed_s": float(time.time() - t0),
            "hp": MA_HP,
            "data_file": DATA,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUT}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
