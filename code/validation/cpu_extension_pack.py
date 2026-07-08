"""
CPU Extension Pack — 6 experiments (B1-B6) to run in parallel with GPU pkg #2.

B1 ROC/PR/calibration for TF-IDF M&A specialist + FinBERT[CLS]+LR baseline
B2 Per-firm fairness audit (bucketed by acquirer frequency)
B3 Top-N informative n-grams + counterfactual removal
B4 EDT cross-year robustness (train 2020 / test 2021)
B5 Lexical-cue ablation (TitleCase ORG-like tokens -> [ORG])
B6 Audit robustness — extra HP cells (ngram, max_features, depth)

All outputs go to results/validation/cpu_pack_*.json. No GPU needed.
"""
import sys, io, os, json, time, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from itertools import product

BASE = r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package"
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
EDT  = os.path.join(BASE, "data", "external", "edt_evaluate_slim.parquet")
FINBERT_CACHE = os.path.join(BASE, "data", "embeddings_cache", "finbert_title.npy")
OUTDIR = os.path.join(BASE, "results", "validation")
os.makedirs(OUTDIR, exist_ok=True)

TRAIN_END = pd.Timestamp('2025-04-01')
VAL_END   = pd.Timestamp('2025-06-01')

# Authoritative M&A specialist HP (from ma_hp_sweep_extended.json)
MA_HP = dict(max_features=100, C=5.0, sublinear_tf=False, min_df=2, ngram_range=(1, 1))


def safe_mcc(y, yp):
    if len(np.unique(y)) < 2 or len(np.unique(yp)) < 2:
        return 0.0
    return float(matthews_corrcoef(y, yp))


def save(name, obj):
    out = os.path.join(OUTDIR, name)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  -> {out}")


def load_df():
    df = pd.read_parquet(DATA)
    df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
    df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy().reset_index(drop=True)
    df['y'] = (df['actual_side'].str.lower() == 'up').astype(int)
    df['title_en'] = df['title_en'].fillna('').astype(str)
    return df


def fit_tfidf_lr(tr_text, y_tr, te_text, hp=MA_HP):
    tf = TfidfVectorizer(max_features=hp['max_features'], stop_words='english',
                         min_df=hp['min_df'], sublinear_tf=hp['sublinear_tf'],
                         ngram_range=hp['ngram_range'])
    Xtr = tf.fit_transform(tr_text)
    Xte = tf.transform(te_text)
    clf = LogisticRegression(max_iter=2000, C=hp['C'], random_state=42)
    clf.fit(Xtr, y_tr)
    return tf, clf, clf.predict(Xte), clf.predict_proba(Xte)[:, 1]


# ====================================================================
# B1 - ROC/PR/calibration on M&A locked test
# ====================================================================

def b1_calibration():
    print("\n[B1] ROC + PR + calibration")
    df = load_df()
    ma = df[df['event'] == 'mergers_acquisitions'].copy()
    ma_tr = ma[ma['published_date'] < VAL_END]
    ma_te = ma[ma['published_date'] >= VAL_END]
    y_tr = ma_tr['y'].values; y_te = ma_te['y'].values

    out = {'meta': {'n_train': int(len(ma_tr)), 'n_test': int(len(ma_te))}, 'models': {}}

    # TF-IDF + LR
    _, clf, yp, prob = fit_tfidf_lr(ma_tr['title_en'].values, y_tr, ma_te['title_en'].values)
    fpr, tpr, _ = roc_curve(y_te, prob)
    prec, rec, _ = precision_recall_curve(y_te, prob)
    cal_true, cal_pred = calibration_curve(y_te, prob, n_bins=10, strategy='quantile')
    out['models']['tfidf_lr'] = {
        'mcc': safe_mcc(y_te, yp),
        'balacc': float(balanced_accuracy_score(y_te, yp)),
        'roc_auc': float(roc_auc_score(y_te, prob)),
        'pr_auc': float(average_precision_score(y_te, prob)),
        'brier': float(brier_score_loss(y_te, prob)),
        'roc': {'fpr': fpr.tolist()[::max(1, len(fpr)//50)], 'tpr': tpr.tolist()[::max(1, len(tpr)//50)]},
        'pr':  {'precision': prec.tolist()[::max(1, len(prec)//50)], 'recall': rec.tolist()[::max(1, len(rec)//50)]},
        'calibration': {'frac_pos': cal_true.tolist(), 'mean_pred': cal_pred.tolist()},
    }

    # FinBERT [CLS] + LR using cached embeddings (rows aligned with df)
    if os.path.exists(FINBERT_CACHE):
        emb = np.load(FINBERT_CACHE)
        if emb.shape[0] == len(df):
            ma_idx = np.where(df['event'].values == 'mergers_acquisitions')[0]
            tr_mask = df.loc[ma_idx, 'published_date'].values < np.datetime64(VAL_END)
            tr_emb = emb[ma_idx[tr_mask]]; te_emb = emb[ma_idx[~tr_mask]]
            ytr_f = df.loc[ma_idx[tr_mask], 'y'].values; yte_f = df.loc[ma_idx[~tr_mask], 'y'].values
            clf2 = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(tr_emb, ytr_f)
            yp2 = clf2.predict(te_emb); prob2 = clf2.predict_proba(te_emb)[:, 1]
            fpr2, tpr2, _ = roc_curve(yte_f, prob2)
            prec2, rec2, _ = precision_recall_curve(yte_f, prob2)
            ct2, cp2 = calibration_curve(yte_f, prob2, n_bins=10, strategy='quantile')
            out['models']['finbert_cls_lr'] = {
                'mcc': safe_mcc(yte_f, yp2),
                'balacc': float(balanced_accuracy_score(yte_f, yp2)),
                'roc_auc': float(roc_auc_score(yte_f, prob2)),
                'pr_auc': float(average_precision_score(yte_f, prob2)),
                'brier': float(brier_score_loss(yte_f, prob2)),
                'roc': {'fpr': fpr2.tolist()[::max(1, len(fpr2)//50)], 'tpr': tpr2.tolist()[::max(1, len(tpr2)//50)]},
                'pr':  {'precision': prec2.tolist()[::max(1, len(prec2)//50)], 'recall': rec2.tolist()[::max(1, len(rec2)//50)]},
                'calibration': {'frac_pos': ct2.tolist(), 'mean_pred': cp2.tolist()},
            }
        else:
            out['models']['finbert_cls_lr'] = {'skipped': f'cache size {emb.shape[0]} != df {len(df)}'}
    save('cpu_pack_b1_calibration.json', out)


# ====================================================================
# B2 - Per-firm fairness
# ====================================================================

def b2_fairness():
    print("\n[B2] Per-firm fairness audit")
    df = load_df()
    ma = df[df['event'] == 'mergers_acquisitions'].copy()
    ma_tr = ma[ma['published_date'] < VAL_END]
    ma_te = ma[ma['published_date'] >= VAL_END].copy()
    y_tr = ma_tr['y'].values; y_te = ma_te['y'].values

    _, _, yp, prob = fit_tfidf_lr(ma_tr['title_en'].values, y_tr, ma_te['title_en'].values)
    ma_te['pred'] = yp; ma_te['prob'] = prob

    out = {'meta': {'n_train': int(len(ma_tr)), 'n_test': int(len(ma_te))},
           'overall_mcc': safe_mcc(y_te, yp), 'buckets': {}}

    # By ticker frequency in train+val
    freq = ma_tr['yf_ticker'].value_counts().to_dict()
    def bucket(t):
        f = freq.get(t, 0)
        if f >= 10: return 'head'
        if f >= 3:  return 'torso'
        if f >= 1:  return 'tail_seen'
        return 'unseen'
    ma_te['bucket'] = ma_te['yf_ticker'].map(bucket)
    for b, grp in ma_te.groupby('bucket'):
        out['buckets'][f'ticker_freq:{b}'] = {
            'n': int(len(grp)),
            'mcc': safe_mcc(grp['y'].values, grp['pred'].values),
            'balacc': float(balanced_accuracy_score(grp['y'].values, grp['pred'].values))
                       if len(np.unique(grp['y'])) >= 2 else None,
            'up_rate_true': float(grp['y'].mean()),
        }

    # By industry (top categories)
    if 'industry' in ma_te.columns:
        ind_counts = ma_te['industry'].value_counts()
        for ind in ind_counts.index[:6]:
            grp = ma_te[ma_te['industry'] == ind]
            if len(grp) < 30: continue
            out['buckets'][f'industry:{ind}'] = {
                'n': int(len(grp)),
                'mcc': safe_mcc(grp['y'].values, grp['pred'].values),
                'up_rate_true': float(grp['y'].mean()),
            }

    # By exchange (top)
    if 'exchange' in ma_te.columns:
        for exc in ma_te['exchange'].value_counts().index[:5]:
            grp = ma_te[ma_te['exchange'] == exc]
            if len(grp) < 30: continue
            out['buckets'][f'exchange:{exc}'] = {
                'n': int(len(grp)),
                'mcc': safe_mcc(grp['y'].values, grp['pred'].values),
                'up_rate_true': float(grp['y'].mean()),
            }
    save('cpu_pack_b2_fairness.json', out)


# ====================================================================
# B3 - Top-N coefs + counterfactual ablation
# ====================================================================

def b3_topgrams():
    print("\n[B3] Top-N informative n-grams + counterfactual removal")
    df = load_df()
    ma = df[df['event'] == 'mergers_acquisitions'].copy()
    ma_tr = ma[ma['published_date'] < VAL_END]
    ma_te = ma[ma['published_date'] >= VAL_END]
    y_tr = ma_tr['y'].values; y_te = ma_te['y'].values

    tf, clf, yp, _ = fit_tfidf_lr(ma_tr['title_en'].values, y_tr, ma_te['title_en'].values)
    vocab = tf.get_feature_names_out()
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    top_neg = [(vocab[i], float(coefs[i])) for i in order[:20]]
    top_pos = [(vocab[i], float(coefs[i])) for i in order[::-1][:20]]
    out = {'baseline_mcc': safe_mcc(y_te, yp),
           'top_positive_ngrams_UP': top_pos,
           'top_negative_ngrams_DOWN': top_neg,
           'ablation': []}

    # Counterfactual: remove top-K |coef| tokens from BOTH classes from input text
    abs_order = np.argsort(-np.abs(coefs))
    for K in [5, 10, 20, 40]:
        kill = set(vocab[i] for i in abs_order[:K])
        pat = re.compile(r'\b(' + '|'.join(re.escape(t) for t in kill) + r')\b', re.I)
        tr_a = [pat.sub(' ', t) for t in ma_tr['title_en'].values]
        te_a = [pat.sub(' ', t) for t in ma_te['title_en'].values]
        _, _, yp_a, _ = fit_tfidf_lr(tr_a, y_tr, te_a)
        out['ablation'].append({'K_removed': K, 'mcc': safe_mcc(y_te, yp_a),
                                'killed_tokens_sample': sorted(kill)[:10]})
    save('cpu_pack_b3_topgrams.json', out)


# ====================================================================
# B4 - EDT cross-year robustness
# ====================================================================

def b4_edt_crossyear():
    print("\n[B4] EDT cross-year (train 2020 / test 2021)")
    if not os.path.exists(EDT):
        print("  EDT parquet missing, skipping"); return
    edt = pd.read_parquet(EDT)
    edt['pub_time'] = pd.to_datetime(edt['pub_time'])
    edt['title'] = edt['title'].fillna('').astype(str)
    # Narrow M&A keyword
    pat = re.compile(r'\b(acquir|acquisition|merger|takeover|buyout)\b', re.I)
    ma = edt[edt['title'].str.contains(pat, na=False)].copy()
    ma['year'] = ma['pub_time'].dt.year
    out = {'meta': {'n_total_edt': int(len(edt)), 'n_narrow_ma': int(len(ma))},
           'experiments': []}

    # Cross-year temporal
    tr = ma[ma['year'] == 2020]; te = ma[ma['year'] == 2021]
    if len(tr) >= 100 and len(te) >= 100:
        _, _, yp, _ = fit_tfidf_lr(tr['title'].values, tr['y'].values, te['title'].values)
        out['experiments'].append({
            'name': 'edt_narrow_ma_train2020_test2021', 'n_train': int(len(tr)),
            'n_test': int(len(te)), 'mcc': safe_mcc(te['y'].values, yp),
            'balacc': float(balanced_accuracy_score(te['y'].values, yp))})

    # Within-year random for comparison (test 2021 size, 5 seeds)
    rng = np.random.default_rng(42)
    mccs_rand = []
    yr2021 = ma[ma['year'] == 2021].reset_index(drop=True)
    if len(yr2021) >= 200:
        n_te = min(len(te) if len(te) else 200, len(yr2021)//3)
        for s in range(5):
            idx = rng.permutation(len(yr2021))
            te_i = idx[:n_te]; tr_i = idx[n_te:]
            _, _, yp_r, _ = fit_tfidf_lr(yr2021.iloc[tr_i]['title'].values,
                                          yr2021.iloc[tr_i]['y'].values,
                                          yr2021.iloc[te_i]['title'].values)
            mccs_rand.append(safe_mcc(yr2021.iloc[te_i]['y'].values, yp_r))
        out['experiments'].append({
            'name': 'edt_narrow_ma_2021_within_year_random_5seed',
            'n_train': int(len(yr2021)-n_te), 'n_test': int(n_te),
            'mcc_mean': float(np.mean(mccs_rand)), 'mcc_std': float(np.std(mccs_rand)),
            'mccs': [float(x) for x in mccs_rand]})
    save('cpu_pack_b4_edt_crossyear.json', out)


# ====================================================================
# B5 - Lexical-cue ablation (regex-based ORG replacement)
# ====================================================================

def b5_org_ablation():
    print("\n[B5] ORG-token ablation on M&A specialist")
    df = load_df()
    ma = df[df['event'] == 'mergers_acquisitions'].copy()
    ma_tr = ma[ma['published_date'] < VAL_END]
    ma_te = ma[ma['published_date'] >= VAL_END]
    y_tr = ma_tr['y'].values; y_te = ma_te['y'].values

    # Heuristic: capitalised tokens or ALL-CAPS acronyms (not at sentence start)
    cap_pat = re.compile(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}|[A-Z]{2,})\b')
    def mask_orgs(text):
        return cap_pat.sub('[ORG]', text)

    # Baseline (original)
    _, _, yp_b, _ = fit_tfidf_lr(ma_tr['title_en'].values, y_tr, ma_te['title_en'].values)
    base = safe_mcc(y_te, yp_b)

    # Masked
    tr_m = [mask_orgs(t) for t in ma_tr['title_en'].values]
    te_m = [mask_orgs(t) for t in ma_te['title_en'].values]
    _, _, yp_m, _ = fit_tfidf_lr(tr_m, y_tr, te_m)
    mask = safe_mcc(y_te, yp_m)

    out = {'meta': {'n_train': int(len(ma_tr)), 'n_test': int(len(ma_te))},
           'baseline_mcc': base, 'org_masked_mcc': mask,
           'delta_due_to_org_removal': base - mask,
           'interpretation': ('positive delta => org/ID tokens contribute; '
                              'small delta => signal is in verbs / non-ORG tokens'),
           'examples_baseline_vs_masked': [
               {'orig': ma_tr['title_en'].iloc[i],
                'masked': mask_orgs(ma_tr['title_en'].iloc[i])}
               for i in range(5)]}
    save('cpu_pack_b5_org_ablation.json', out)


# ====================================================================
# B6 - Audit robustness (extra HP cells for App D)
# ====================================================================

def b6_audit_robustness():
    print("\n[B6] Extra audit HP cells (TF-IDF/RF/GB on full corpus)")
    df = load_df()
    tr = df[df['published_date'] < TRAIN_END]
    te = df[df['published_date'] >= VAL_END]
    y_tr = tr['y'].values; y_te = te['y'].values

    out = {'meta': {'n_train': int(len(tr)), 'n_test': int(len(te))}, 'cells': []}

    # Multiple TF-IDF cells
    grid = [
        ('TFIDF50_uni',   dict(max_features=50,   ngram_range=(1, 1))),
        ('TFIDF50_bi',    dict(max_features=50,   ngram_range=(1, 2))),
        ('TFIDF200_uni',  dict(max_features=200,  ngram_range=(1, 1))),
        ('TFIDF200_bi',   dict(max_features=200,  ngram_range=(1, 2))),
        ('TFIDF1000_uni', dict(max_features=1000, ngram_range=(1, 1))),
        ('TFIDF2000_bi',  dict(max_features=2000, ngram_range=(1, 2))),
    ]
    for name, params in grid:
        tf = TfidfVectorizer(stop_words='english', min_df=2, **params)
        Xtr = tf.fit_transform(tr['title_en']); Xte = tf.transform(te['title_en'])
        # LR
        clf = LogisticRegression(max_iter=2000, C=0.5, random_state=42).fit(Xtr, y_tr)
        ypl = clf.predict(Xte)
        out['cells'].append({'features': name, 'model': 'LR', 'mcc': safe_mcc(y_te, ypl),
                             'balacc': float(balanced_accuracy_score(y_te, ypl))})
        # RF (only on smaller features for speed)
        if params['max_features'] <= 500:
            rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1,
                                        min_samples_leaf=2, random_state=42).fit(Xtr.toarray(), y_tr)
            ypr = rf.predict(Xte.toarray())
            out['cells'].append({'features': name, 'model': 'RF', 'mcc': safe_mcc(y_te, ypr),
                                 'balacc': float(balanced_accuracy_score(y_te, ypr))})

    # Sentence length / RF depth sweep on TFIDF50_bi
    tf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words='english', min_df=2)
    Xtr = tf.fit_transform(tr['title_en']).toarray(); Xte = tf.transform(te['title_en']).toarray()
    for d in [5, 10, 15, 25, 40]:
        rf = RandomForestClassifier(n_estimators=200, max_depth=d, n_jobs=-1,
                                    min_samples_leaf=2, random_state=42).fit(Xtr, y_tr)
        yp = rf.predict(Xte)
        out['cells'].append({'features': 'TFIDF50_bi', 'model': f'RF_depth{d}',
                             'mcc': safe_mcc(y_te, yp),
                             'balacc': float(balanced_accuracy_score(y_te, yp))})
    save('cpu_pack_b6_audit_robustness.json', out)


def main():
    t0 = time.time()
    for fn in [b1_calibration, b2_fairness, b3_topgrams,
               b4_edt_crossyear, b5_org_ablation, b6_audit_robustness]:
        ts = time.time()
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} failed: {e}")
        print(f"  {fn.__name__} elapsed {time.time()-ts:.1f}s")
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
