"""
CPU Extension Pack #2 — B7-B11 (resubmission-ready appendix material).

B7  Decision-threshold sweep on calibrated probabilities (M&A specialist)
B8  Per-month rolling MCC on the M&A locked test
B9  TRUE spaCy NER entity blanking (vs B5 regex TitleCase)
B10 Per-headline LR-coefficient attribution on top-10 ACQUIRER articles
B11 Publisher-level fairness audit on M&A locked test

All outputs go to results/validation/cpu_pack_b{7..11}_*.json. CPU only, ~2 min total.
"""
import sys, io, os, json, time, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV

BASE = r"."
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUTDIR = os.path.join(BASE, "results", "validation")
os.makedirs(OUTDIR, exist_ok=True)

TRAIN_END = pd.Timestamp('2025-04-01')
VAL_END   = pd.Timestamp('2025-06-01')

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
    df['publisher'] = df['publisher'].fillna('UNKNOWN').astype(str)
    return df


def fit_tfidf_lr(tr_text, y_tr, hp=MA_HP, return_proba_fn=False):
    tf = TfidfVectorizer(max_features=hp['max_features'], stop_words='english',
                         min_df=hp['min_df'], sublinear_tf=hp['sublinear_tf'],
                         ngram_range=hp['ngram_range'])
    Xtr = tf.fit_transform(tr_text)
    clf = LogisticRegression(max_iter=2000, C=hp['C'], random_state=42)
    clf.fit(Xtr, y_tr)
    if return_proba_fn:
        def proba(text):
            return clf.predict_proba(tf.transform(text))[:, 1]
        return tf, clf, proba
    return tf, clf


# ====================================================================
# B7 - Decision-threshold sweep
# ====================================================================
def b7_threshold_sweep():
    print("\n[B7] Decision-threshold sweep on M&A specialist")
    df = load_df()
    ma = df[df['event'] == 'mergers_acquisitions'].copy()
    tr = ma[ma['published_date'] < TRAIN_END]
    va = ma[(ma['published_date'] >= TRAIN_END) & (ma['published_date'] < VAL_END)]
    te = ma[ma['published_date'] >= VAL_END]
    print(f"  splits: train={len(tr)} val={len(va)} test={len(te)}")

    tf, clf, proba_fn = fit_tfidf_lr(tr['title_en'], tr['y'].values, return_proba_fn=True)
    p_va = proba_fn(va['title_en'])
    p_te = proba_fn(te['title_en'])
    y_va = va['y'].values
    y_te = te['y'].values

    thresholds = np.arange(0.30, 0.71, 0.02)
    rows = []
    for t in thresholds:
        yp_va = (p_va >= t).astype(int)
        yp_te = (p_te >= t).astype(int)
        rows.append({
            'threshold': float(round(t, 3)),
            'val_mcc': safe_mcc(y_va, yp_va),
            'val_balacc': float(balanced_accuracy_score(y_va, yp_va)) if len(np.unique(yp_va)) >= 2 else 0.5,
            'val_pred_up_rate': float(yp_va.mean()),
            'test_mcc': safe_mcc(y_te, yp_te),
            'test_balacc': float(balanced_accuracy_score(y_te, yp_te)) if len(np.unique(yp_te)) >= 2 else 0.5,
            'test_f1_up': float(f1_score(y_te, yp_te, pos_label=1)) if len(np.unique(yp_te)) >= 2 else 0.0,
            'test_f1_down': float(f1_score(y_te, yp_te, pos_label=0)) if len(np.unique(yp_te)) >= 2 else 0.0,
            'test_prec_up': float(precision_score(y_te, yp_te, pos_label=1)) if (yp_te == 1).any() else 0.0,
            'test_rec_up': float(recall_score(y_te, yp_te, pos_label=1)) if (y_te == 1).any() else 0.0,
            'test_pred_up_rate': float(yp_te.mean()),
        })

    val_best = max(rows, key=lambda r: r['val_mcc'])
    val_best_t = val_best['threshold']
    val_best_row = next(r for r in rows if r['threshold'] == val_best_t)
    default_row = min(rows, key=lambda r: abs(r['threshold'] - 0.50))

    out = {
        'description': 'B7: Decision-threshold sweep (train -> val for threshold selection; report on locked test)',
        'hp': MA_HP,
        'n_tr': int(len(tr)), 'n_va': int(len(va)), 'n_te': int(len(te)),
        'true_up_rate_val': float(y_va.mean()), 'true_up_rate_test': float(y_te.mean()),
        'sweep': rows,
        'val_optimal_threshold': float(val_best_t),
        'val_optimal_test_mcc': float(val_best_row['test_mcc']),
        'default_threshold_test_mcc': float(default_row['test_mcc']),
        'delta_vs_default_05': float(val_best_row['test_mcc'] - default_row['test_mcc']),
    }
    save('cpu_pack_b7_threshold_sweep.json', out)
    print(f"  val-optimal t={val_best_t:.2f}: test MCC {val_best_row['test_mcc']:.4f} "
          f"(vs default 0.50 -> {default_row['test_mcc']:.4f}; delta {val_best_row['test_mcc'] - default_row['test_mcc']:+.4f})")
    return out


# ====================================================================
# B8 - Per-month rolling M&A MCC on locked test
# ====================================================================
def b8_per_month_rolling():
    print("\n[B8] Per-month rolling MCC on M&A locked test")
    df = load_df()
    ma = df[df['event'] == 'mergers_acquisitions'].copy()
    tr = ma[ma['published_date'] < TRAIN_END]
    va = ma[(ma['published_date'] >= TRAIN_END) & (ma['published_date'] < VAL_END)]
    te = ma[ma['published_date'] >= VAL_END].copy()

    tf_all = pd.concat([tr, va], ignore_index=True)
    tf, clf, proba_fn = fit_tfidf_lr(tf_all['title_en'], tf_all['y'].values, return_proba_fn=True)
    p_te = proba_fn(te['title_en'])
    yp_te = (p_te >= 0.5).astype(int)
    te = te.assign(y_pred=yp_te, p_up=p_te)
    te['period'] = te['published_date'].dt.to_period('M')

    rows = []
    for period, g in te.groupby('period'):
        if len(g) < 5: continue
        rows.append({
            'period': str(period),
            'n': int(len(g)),
            'mcc': safe_mcc(g['y'].values, g['y_pred'].values),
            'true_up_rate': float(g['y'].mean()),
            'pred_up_rate': float(g['y_pred'].mean()),
            'mean_proba_up': float(g['p_up'].mean()),
        })

    pooled = {
        'n': int(len(te)),
        'mcc': safe_mcc(te['y'].values, te['y_pred'].values),
        'true_up_rate': float(te['y'].mean()),
        'pred_up_rate': float(te['y_pred'].mean()),
    }
    out = {
        'description': 'B8: Per-month rolling MCC on M&A locked test',
        'hp': MA_HP,
        'pooled_test': pooled,
        'per_month': rows,
    }
    save('cpu_pack_b8_rolling_mcc.json', out)
    print(f"  pooled MCC {pooled['mcc']:.4f} ({pooled['n']} articles); "
          f"per-month range [{min(r['mcc'] for r in rows):.3f}, {max(r['mcc'] for r in rows):.3f}] across {len(rows)} months")
    return out


# ====================================================================
# B9 - spaCy NER entity blanking
# ====================================================================
def b9_ner_blanking():
    print("\n[B9] spaCy NER entity blanking on M&A specialist")
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['lemmatizer', 'tagger', 'parser'])

    df = load_df()
    ma = df[df['event'] == 'mergers_acquisitions'].copy()
    tr = ma[ma['published_date'] < TRAIN_END]
    te = ma[ma['published_date'] >= VAL_END]
    print(f"  splits: train={len(tr)} test={len(te)}; running NER...")
    t0 = time.time()
    target_labels = {'ORG': '[ORG]', 'PERSON': '[PERSON]', 'MONEY': '[MONEY]',
                     'GPE': '[GPE]', 'PERCENT': '[PCT]', 'CARDINAL': '[NUM]'}
    def mask(text):
        if not text: return text
        doc = nlp(text)
        spans = sorted([(e.start_char, e.end_char, target_labels.get(e.label_, None))
                        for e in doc.ents], key=lambda x: -x[0])
        out = text
        for s, e, lab in spans:
            if lab: out = out[:s] + lab + out[e:]
        return out

    tr_txt_mask = [mask(t) for t in tr['title_en'].tolist()]
    te_txt_mask = [mask(t) for t in te['title_en'].tolist()]
    print(f"  NER done in {time.time()-t0:.1f}s")

    # Conditions: original, full mask (ORG+PERSON+MONEY+GPE+PCT+NUM), ORG-only mask
    def mask_subset(text, allowed_labels):
        if not text: return text
        doc = nlp(text)
        spans = sorted([(e.start_char, e.end_char, e.label_) for e in doc.ents], key=lambda x: -x[0])
        out = text
        for s, e, lab in spans:
            if lab in allowed_labels:
                out = out[:s] + f"[{lab}]" + out[e:]
        return out

    tr_org = [mask_subset(t, {'ORG'}) for t in tr['title_en'].tolist()]
    te_org = [mask_subset(t, {'ORG'}) for t in te['title_en'].tolist()]
    tr_orgper = [mask_subset(t, {'ORG', 'PERSON'}) for t in tr['title_en'].tolist()]
    te_orgper = [mask_subset(t, {'ORG', 'PERSON'}) for t in te['title_en'].tolist()]

    results = {}
    for name, tr_t, te_t in [
        ('original', tr['title_en'].tolist(), te['title_en'].tolist()),
        ('mask_ORG', tr_org, te_org),
        ('mask_ORG_PERSON', tr_orgper, te_orgper),
        ('mask_ALL', tr_txt_mask, te_txt_mask),
    ]:
        tf, clf = fit_tfidf_lr(tr_t, tr['y'].values)
        yp = clf.predict(tf.transform(te_t))
        results[name] = {
            'mcc': safe_mcc(te['y'].values, yp),
            'pred_up_rate': float(yp.mean()),
        }
        print(f"    {name:18s}: MCC {results[name]['mcc']:.4f}, pred_UP {results[name]['pred_up_rate']:.3f}")

    out = {
        'description': 'B9: spaCy NER entity blanking on M&A specialist (4 conditions)',
        'hp': MA_HP, 'n_tr': int(len(tr)), 'n_te': int(len(te)),
        'true_up_rate_test': float(te['y'].mean()),
        'results': results,
        'delta_mask_ORG_vs_original': float(results['mask_ORG']['mcc'] - results['original']['mcc']),
        'delta_mask_ALL_vs_original': float(results['mask_ALL']['mcc'] - results['original']['mcc']),
    }
    save('cpu_pack_b9_ner_blanking.json', out)
    return out


# ====================================================================
# B10 - Top-10 ACQUIRER headline coefficient attribution
# ====================================================================
def b10_per_headline_attribution():
    print("\n[B10] Per-headline LR-coefficient attribution on top-10 ACQUIRER headlines")
    df = load_df()
    ma = df[df['event'] == 'mergers_acquisitions'].copy()
    tr = ma[ma['published_date'] < TRAIN_END]
    te = ma[ma['published_date'] >= VAL_END].copy()
    tf, clf, proba_fn = fit_tfidf_lr(tr['title_en'], tr['y'].values, return_proba_fn=True)
    p_te = proba_fn(te['title_en'])
    te = te.assign(p_up=p_te)
    vocab = np.array(tf.get_feature_names_out())
    beta = clf.coef_[0]

    # Regex acquirer set (matches the body §8 definition)
    acq_pat = re.compile(
        r"\b(acquires?|acquired|acquiring|acquisition\s+of|to\s+acquire|"
        r"buys?\s+\w+|purchases?|to\s+purchase|takeover\s+of|tender\s+offer\s+for)\b",
        re.I)
    te['is_acquirer'] = te['title_en'].apply(lambda s: bool(acq_pat.search(s or '')))
    acq = te[te['is_acquirer']].copy()
    print(f"  test set: {len(te)} articles, acquirer-side: {len(acq)}")

    if len(acq) == 0:
        save('cpu_pack_b10_attribution.json', {'description': 'B10: empty acquirer set'})
        return {}

    # Sort by predicted prob (most confident UP first)
    acq_top = acq.sort_values('p_up', ascending=False).head(10).reset_index(drop=True)
    rows = []
    for i, row in acq_top.iterrows():
        title = row['title_en']
        x_vec = tf.transform([title]).toarray()[0]
        nz = np.where(x_vec > 0)[0]
        contribs = [(vocab[j], float(beta[j] * x_vec[j]), float(beta[j])) for j in nz]
        contribs.sort(key=lambda x: -abs(x[1]))
        rows.append({
            'rank': int(i + 1),
            'title': title,
            'true_label_up': bool(row['y']),
            'predicted_proba_up': float(row['p_up']),
            'predicted_label_up': bool(row['p_up'] >= 0.5),
            'token_contribs_top5': contribs[:5],
            'sum_contribs': float(sum(c[1] for c in contribs)),
        })

    out = {
        'description': 'B10: Per-headline LR-coefficient attribution on top-10 ACQUIRER articles by p(UP)',
        'hp': MA_HP, 'n_tr': int(len(tr)), 'n_te': int(len(te)), 'n_acq_te': int(len(acq)),
        'intercept': float(clf.intercept_[0]),
        'top10_acquirer_headlines': rows,
    }
    save('cpu_pack_b10_attribution.json', out)
    print(f"  -> 10 acquirer headlines with token-level attributions saved")
    return out


# ====================================================================
# B11 - Publisher-level fairness audit
# ====================================================================
def b11_publisher_fairness():
    print("\n[B11] Publisher-level fairness audit on M&A locked test")
    df = load_df()
    ma = df[df['event'] == 'mergers_acquisitions'].copy()
    tr = ma[ma['published_date'] < TRAIN_END]
    va = ma[(ma['published_date'] >= TRAIN_END) & (ma['published_date'] < VAL_END)]
    te = ma[ma['published_date'] >= VAL_END].copy()
    tf_all = pd.concat([tr, va], ignore_index=True)
    tf, clf, proba_fn = fit_tfidf_lr(tf_all['title_en'], tf_all['y'].values, return_proba_fn=True)
    p_te = proba_fn(te['title_en'])
    yp_te = (p_te >= 0.5).astype(int)
    te = te.assign(y_pred=yp_te, p_up=p_te)

    # Bucket by train-side publisher frequency
    pub_freq = tf_all['publisher'].value_counts().to_dict()
    def bucket(p):
        n = pub_freq.get(p, 0)
        if n == 0: return 'unseen_at_train'
        if n >= 1000: return 'major (>=1000 articles)'
        if n >= 200:  return 'mid (200-999)'
        if n >= 50:   return 'small (50-199)'
        return 'tail (<50)'
    te['pub_bucket'] = te['publisher'].apply(bucket)

    rows = []
    for b, g in te.groupby('pub_bucket'):
        rows.append({
            'bucket': b,
            'n': int(len(g)),
            'n_distinct_publishers': int(g['publisher'].nunique()),
            'mcc': safe_mcc(g['y'].values, g['y_pred'].values),
            'true_up_rate': float(g['y'].mean()),
            'pred_up_rate': float(g['y_pred'].mean()),
        })

    # Per-publisher (top 8 by test frequency)
    top_pubs = te['publisher'].value_counts().head(8).index.tolist()
    per_pub = []
    for p in top_pubs:
        g = te[te['publisher'] == p]
        if len(g) < 10: continue
        per_pub.append({
            'publisher': p,
            'n_test': int(len(g)),
            'n_train': int(pub_freq.get(p, 0)),
            'mcc': safe_mcc(g['y'].values, g['y_pred'].values),
            'true_up_rate': float(g['y'].mean()),
            'pred_up_rate': float(g['y_pred'].mean()),
        })

    out = {
        'description': 'B11: Publisher-level fairness audit',
        'hp': MA_HP, 'n_te': int(len(te)),
        'pooled_test_mcc': safe_mcc(te['y'].values, te['y_pred'].values),
        'per_bucket': rows,
        'per_publisher_top8': per_pub,
    }
    save('cpu_pack_b11_publisher_fairness.json', out)
    print(f"  {len(rows)} buckets, top publisher MCC range "
          f"[{min(r['mcc'] for r in rows):+.3f}, {max(r['mcc'] for r in rows):+.3f}]")
    return out


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    t0 = time.time()
    print(f"CPU extension pack #2 (B7-B11)")
    print(f"  data: {DATA}")
    print(f"  out:  {OUTDIR}")
    results = {}
    for fn in [b7_threshold_sweep, b8_per_month_rolling, b9_ner_blanking,
               b10_per_headline_attribution, b11_publisher_fairness]:
        try:
            results[fn.__name__] = fn()
        except Exception as e:
            import traceback
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[fn.__name__] = {'error': str(e)}
    print(f"\nDone in {time.time()-t0:.1f}s")
