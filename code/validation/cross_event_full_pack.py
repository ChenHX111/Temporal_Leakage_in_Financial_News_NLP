"""
Cross-Event Full Pack — applies the M&A CPU pack experiments (B1, B2, B3, B5,
B6, B7, B8, B9, B11) to two non-M&A events: clinical_study and law_legal_issues.

B4 (EDT cross-year) is M&A-specific and skipped.
B10 (acquirer-attribution) is M&A-specific and skipped; we substitute a generic
top-10 most-confident article attribution.

Conventions match the M&A pack as closely as possible:
- TF-IDF HP = paper-authoritative M&A HP (max_features=100, C=5.0,
  sublinear_tf=False, min_df=2, ngram=(1,1)) for cross-event headline numbers
  so test MCC is directly comparable to cross_event_audit.json.
- Cutoffs are adapted per event because legal_issues is sparse pre-2025-06:
    M&A and CLN: train_end=2025-04-01, val_end=2025-06-01
    LGL:        train_end=2025-06-01, val_end=2025-07-15

Outputs: results/validation/cross_event_pack_{event}_{B}.json (one per
event x experiment), wall time ~3-5 min on CPU.
"""
import sys, io, os, json, time, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score,
                              roc_auc_score, average_precision_score, brier_score_loss,
                              roc_curve, precision_recall_curve, precision_score,
                              recall_score, f1_score)
from sklearn.calibration import calibration_curve

BASE = r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package"
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUTDIR = os.path.join(BASE, "results", "validation")
os.makedirs(OUTDIR, exist_ok=True)

PAPER_HP = dict(max_features=100, C=5.0, sublinear_tf=False, min_df=2, ngram_range=(1, 1))

# Event -> (train_end, val_end) cutoffs
EVENTS = {
    'clinical_study':                          (pd.Timestamp('2025-04-01'), pd.Timestamp('2025-06-01')),
    'law_legal_issues':                        (pd.Timestamp('2025-06-01'), pd.Timestamp('2025-07-15')),
    'earnings_releases_and_operating_results': (pd.Timestamp('2025-04-01'), pd.Timestamp('2025-06-01')),
}


def safe_mcc(y, yp):
    if len(np.unique(y)) < 2 or len(np.unique(yp)) < 2:
        return 0.0
    return float(matthews_corrcoef(y, yp))


def save(name, obj):
    out = os.path.join(OUTDIR, name)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  -> {out}")


def load_event(event):
    df = pd.read_parquet(DATA)
    df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
    df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy().reset_index(drop=True)
    df['y'] = (df['actual_side'].str.lower() == 'up').astype(int)
    df['title_en'] = df['title_en'].fillna('').astype(str)
    for col in ('publisher', 'yf_ticker', 'industry', 'exchange'):
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN').astype(str)
    return df[df['event'] == event].copy().reset_index(drop=True)


def fit_tfidf_lr(tr_text, y_tr, hp=PAPER_HP, return_proba_fn=False):
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
# B1 - ROC + PR + calibration
# ====================================================================
def b1_calibration(event, tr_end, va_end):
    print(f"\n[B1 / {event}] ROC + PR + calibration")
    df = load_event(event)
    tr = df[df['published_date'] < va_end]
    te = df[df['published_date'] >= va_end]
    y_tr = tr['y'].values; y_te = te['y'].values
    out = {'meta': {'event': event, 'n_train_plus_val': int(len(tr)), 'n_test': int(len(te))},
           'models': {}}
    _, clf = fit_tfidf_lr(tr['title_en'].values, y_tr)
    tf, _, proba_fn = fit_tfidf_lr(tr['title_en'].values, y_tr, return_proba_fn=True)
    prob = proba_fn(te['title_en'].values)
    yp = (prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, prob)
    prec, rec, _ = precision_recall_curve(y_te, prob)
    try:
        ct, cp = calibration_curve(y_te, prob, n_bins=10, strategy='quantile')
    except ValueError:
        ct, cp = np.array([]), np.array([])
    out['models']['tfidf_lr'] = {
        'mcc': safe_mcc(y_te, yp),
        'balacc': float(balanced_accuracy_score(y_te, yp)),
        'roc_auc': float(roc_auc_score(y_te, prob)) if len(np.unique(y_te)) >= 2 else None,
        'pr_auc': float(average_precision_score(y_te, prob)) if len(np.unique(y_te)) >= 2 else None,
        'brier': float(brier_score_loss(y_te, prob)),
        'roc': {'fpr': fpr.tolist()[::max(1, len(fpr)//50)],
                'tpr': tpr.tolist()[::max(1, len(tpr)//50)]},
        'pr':  {'precision': prec.tolist()[::max(1, len(prec)//50)],
                'recall':    rec.tolist()[::max(1, len(rec)//50)]},
        'calibration': {'frac_pos': ct.tolist(), 'mean_pred': cp.tolist()},
    }
    save(f'cross_event_pack_{event}_b1_calibration.json', out)


# ====================================================================
# B2 - Fairness (ticker freq, industry, exchange)
# ====================================================================
def b2_fairness(event, tr_end, va_end):
    print(f"\n[B2 / {event}] Fairness (ticker freq, industry, exchange)")
    df = load_event(event)
    tr = df[df['published_date'] < va_end]
    te = df[df['published_date'] >= va_end].copy()
    y_tr = tr['y'].values; y_te = te['y'].values
    tf, clf = fit_tfidf_lr(tr['title_en'].values, y_tr)
    yp = clf.predict(tf.transform(te['title_en'].values))
    te['pred'] = yp
    out = {'meta': {'event': event, 'n_train_plus_val': int(len(tr)), 'n_test': int(len(te))},
           'overall_mcc': safe_mcc(y_te, yp), 'buckets': {}}
    if 'yf_ticker' in tr.columns:
        freq = tr['yf_ticker'].value_counts().to_dict()
        def bk(t):
            f = freq.get(t, 0)
            if f >= 10: return 'head'
            if f >= 3: return 'torso'
            if f >= 1: return 'tail_seen'
            return 'unseen'
        te['bucket'] = te['yf_ticker'].map(bk)
        for b, g in te.groupby('bucket'):
            out['buckets'][f'ticker_freq:{b}'] = {
                'n': int(len(g)),
                'mcc': safe_mcc(g['y'].values, g['pred'].values),
                'balacc': float(balanced_accuracy_score(g['y'].values, g['pred'].values))
                          if len(np.unique(g['y'])) >= 2 else None,
                'up_rate_true': float(g['y'].mean())}
    if 'industry' in te.columns:
        for ind in te['industry'].value_counts().index[:6]:
            g = te[te['industry'] == ind]
            if len(g) < 30: continue
            out['buckets'][f'industry:{ind}'] = {
                'n': int(len(g)), 'mcc': safe_mcc(g['y'].values, g['pred'].values),
                'up_rate_true': float(g['y'].mean())}
    if 'exchange' in te.columns:
        for ex in te['exchange'].value_counts().index[:5]:
            g = te[te['exchange'] == ex]
            if len(g) < 30: continue
            out['buckets'][f'exchange:{ex}'] = {
                'n': int(len(g)), 'mcc': safe_mcc(g['y'].values, g['pred'].values),
                'up_rate_true': float(g['y'].mean())}
    save(f'cross_event_pack_{event}_b2_fairness.json', out)


# ====================================================================
# B3 - Top-N informative n-grams + counterfactual removal
# ====================================================================
def b3_topgrams(event, tr_end, va_end):
    print(f"\n[B3 / {event}] Top-N informative n-grams + counterfactual removal")
    df = load_event(event)
    tr = df[df['published_date'] < va_end]
    te = df[df['published_date'] >= va_end]
    y_tr = tr['y'].values; y_te = te['y'].values
    tf, clf = fit_tfidf_lr(tr['title_en'].values, y_tr)
    yp = clf.predict(tf.transform(te['title_en'].values))
    vocab = tf.get_feature_names_out()
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    top_neg = [(vocab[i], float(coefs[i])) for i in order[:20]]
    top_pos = [(vocab[i], float(coefs[i])) for i in order[::-1][:20]]
    out = {'event': event, 'baseline_mcc': safe_mcc(y_te, yp),
           'top_positive_ngrams_UP': top_pos, 'top_negative_ngrams_DOWN': top_neg,
           'ablation': []}
    abs_order = np.argsort(-np.abs(coefs))
    for K in [5, 10, 20, 40]:
        if K > len(vocab): continue
        kill = set(vocab[i] for i in abs_order[:K])
        pat = re.compile(r'\b(' + '|'.join(re.escape(t) for t in kill) + r')\b', re.I)
        tr_a = [pat.sub(' ', t) for t in tr['title_en'].values]
        te_a = [pat.sub(' ', t) for t in te['title_en'].values]
        tf2, clf2 = fit_tfidf_lr(tr_a, y_tr)
        yp_a = clf2.predict(tf2.transform(te_a))
        out['ablation'].append({'K_removed': K, 'mcc': safe_mcc(y_te, yp_a),
                                'killed_tokens_sample': sorted(kill)[:10]})
    save(f'cross_event_pack_{event}_b3_topgrams.json', out)


# ====================================================================
# B5 - Regex ORG-token ablation
# ====================================================================
def b5_org_ablation(event, tr_end, va_end):
    print(f"\n[B5 / {event}] Regex ORG-token ablation")
    df = load_event(event)
    tr = df[df['published_date'] < va_end]
    te = df[df['published_date'] >= va_end]
    y_tr = tr['y'].values; y_te = te['y'].values
    cap_pat = re.compile(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}|[A-Z]{2,})\b')
    def mask_orgs(t): return cap_pat.sub('[ORG]', t)
    tf, clf = fit_tfidf_lr(tr['title_en'].values, y_tr)
    base = safe_mcc(y_te, clf.predict(tf.transform(te['title_en'].values)))
    tr_m = [mask_orgs(t) for t in tr['title_en'].values]
    te_m = [mask_orgs(t) for t in te['title_en'].values]
    tf2, clf2 = fit_tfidf_lr(tr_m, y_tr)
    mask = safe_mcc(y_te, clf2.predict(tf2.transform(te_m)))
    out = {'event': event, 'n_train_plus_val': int(len(tr)), 'n_test': int(len(te)),
           'baseline_mcc': base, 'org_masked_mcc': mask,
           'delta_due_to_org_removal': base - mask,
           'examples_baseline_vs_masked':
               [{'orig': tr['title_en'].iloc[i], 'masked': mask_orgs(tr['title_en'].iloc[i])}
                for i in range(min(5, len(tr)))]}
    save(f'cross_event_pack_{event}_b5_org_ablation.json', out)


# ====================================================================
# B6 - Audit robustness (extra HP cells)
# ====================================================================
def b6_audit_robustness(event, tr_end, va_end):
    print(f"\n[B6 / {event}] Extra audit HP cells")
    df = load_event(event)
    tr = df[df['published_date'] < tr_end]
    te = df[df['published_date'] >= va_end]
    y_tr = tr['y'].values; y_te = te['y'].values
    if len(tr) < 50 or len(te) < 30:
        save(f'cross_event_pack_{event}_b6_audit_robustness.json',
             {'event': event, 'skipped': f'too small (tr={len(tr)} te={len(te)})'})
        return
    out = {'event': event, 'meta': {'n_train': int(len(tr)), 'n_test': int(len(te))},
           'cells': []}
    grid = [
        ('TFIDF50_uni',   dict(max_features=50,   ngram_range=(1, 1))),
        ('TFIDF50_bi',    dict(max_features=50,   ngram_range=(1, 2))),
        ('TFIDF200_uni',  dict(max_features=200,  ngram_range=(1, 1))),
        ('TFIDF200_bi',   dict(max_features=200,  ngram_range=(1, 2))),
        ('TFIDF1000_uni', dict(max_features=1000, ngram_range=(1, 1))),
        ('TFIDF2000_bi', dict(max_features=2000,  ngram_range=(1, 2))),
    ]
    for name, params in grid:
        tf = TfidfVectorizer(stop_words='english', min_df=2, **params)
        Xtr = tf.fit_transform(tr['title_en']); Xte = tf.transform(te['title_en'])
        clf = LogisticRegression(max_iter=2000, C=0.5, random_state=42).fit(Xtr, y_tr)
        ypl = clf.predict(Xte)
        out['cells'].append({'features': name, 'model': 'LR', 'mcc': safe_mcc(y_te, ypl),
                             'balacc': float(balanced_accuracy_score(y_te, ypl))})
        if params['max_features'] <= 500:
            rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1,
                                        min_samples_leaf=2, random_state=42).fit(Xtr.toarray(), y_tr)
            ypr = rf.predict(Xte.toarray())
            out['cells'].append({'features': name, 'model': 'RF', 'mcc': safe_mcc(y_te, ypr),
                                 'balacc': float(balanced_accuracy_score(y_te, ypr))})
    save(f'cross_event_pack_{event}_b6_audit_robustness.json', out)


# ====================================================================
# B7 - Decision-threshold sweep
# ====================================================================
def b7_threshold_sweep(event, tr_end, va_end):
    print(f"\n[B7 / {event}] Decision-threshold sweep")
    df = load_event(event)
    tr = df[df['published_date'] < tr_end]
    va = df[(df['published_date'] >= tr_end) & (df['published_date'] < va_end)]
    te = df[df['published_date'] >= va_end]
    print(f"  splits: train={len(tr)} val={len(va)} test={len(te)}")
    if len(tr) < 50 or len(va) < 20 or len(te) < 30:
        save(f'cross_event_pack_{event}_b7_threshold_sweep.json',
             {'event': event, 'skipped': f'too small (tr={len(tr)} va={len(va)} te={len(te)})'})
        return
    tf, clf, proba_fn = fit_tfidf_lr(tr['title_en'], tr['y'].values, return_proba_fn=True)
    p_va = proba_fn(va['title_en']); p_te = proba_fn(te['title_en'])
    y_va = va['y'].values; y_te = te['y'].values
    rows = []
    for t in np.arange(0.30, 0.71, 0.02):
        yp_va = (p_va >= t).astype(int); yp_te = (p_te >= t).astype(int)
        rows.append({
            'threshold': float(round(t, 3)),
            'val_mcc': safe_mcc(y_va, yp_va),
            'val_pred_up_rate': float(yp_va.mean()),
            'test_mcc': safe_mcc(y_te, yp_te),
            'test_balacc': float(balanced_accuracy_score(y_te, yp_te)) if len(np.unique(yp_te)) >= 2 else 0.5,
            'test_f1_up': float(f1_score(y_te, yp_te, pos_label=1)) if len(np.unique(yp_te)) >= 2 else 0.0,
            'test_pred_up_rate': float(yp_te.mean()),
        })
    val_best = max(rows, key=lambda r: r['val_mcc'])
    default_row = min(rows, key=lambda r: abs(r['threshold'] - 0.50))
    out = {'event': event, 'hp': PAPER_HP,
           'n_tr': int(len(tr)), 'n_va': int(len(va)), 'n_te': int(len(te)),
           'true_up_rate_val': float(y_va.mean()), 'true_up_rate_test': float(y_te.mean()),
           'sweep': rows,
           'val_optimal_threshold': float(val_best['threshold']),
           'val_optimal_test_mcc': float(val_best['test_mcc']),
           'default_threshold_test_mcc': float(default_row['test_mcc']),
           'delta_vs_default_05': float(val_best['test_mcc'] - default_row['test_mcc'])}
    save(f'cross_event_pack_{event}_b7_threshold_sweep.json', out)
    print(f"  val-opt t={val_best['threshold']:.2f} -> test MCC={val_best['test_mcc']:+.4f}; "
          f"default t=0.50 -> {default_row['test_mcc']:+.4f}")


# ====================================================================
# B8 - Per-month rolling MCC on locked test
# ====================================================================
def b8_per_month_rolling(event, tr_end, va_end):
    print(f"\n[B8 / {event}] Per-month rolling MCC on locked test")
    df = load_event(event)
    tr = df[df['published_date'] < tr_end]
    va = df[(df['published_date'] >= tr_end) & (df['published_date'] < va_end)]
    te = df[df['published_date'] >= va_end].copy()
    if len(te) < 30:
        save(f'cross_event_pack_{event}_b8_rolling_mcc.json',
             {'event': event, 'skipped': f'te too small ({len(te)})'})
        return
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
            'period': str(period), 'n': int(len(g)),
            'mcc': safe_mcc(g['y'].values, g['y_pred'].values),
            'true_up_rate': float(g['y'].mean()),
            'pred_up_rate': float(g['y_pred'].mean()),
            'mean_proba_up': float(g['p_up'].mean()),
        })
    pooled = {'n': int(len(te)),
              'mcc': safe_mcc(te['y'].values, te['y_pred'].values),
              'true_up_rate': float(te['y'].mean()),
              'pred_up_rate': float(te['y_pred'].mean())}
    out = {'event': event, 'hp': PAPER_HP, 'pooled_test': pooled, 'per_month': rows}
    save(f'cross_event_pack_{event}_b8_rolling_mcc.json', out)
    print(f"  pooled MCC={pooled['mcc']:+.4f} ({pooled['n']} articles); per-month n={len(rows)}")


# ====================================================================
# B9 - spaCy NER blanking
# ====================================================================
def b9_ner_blanking(event, tr_end, va_end):
    print(f"\n[B9 / {event}] spaCy NER blanking")
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm', disable=['lemmatizer', 'tagger', 'parser'])
    except Exception as e:
        save(f'cross_event_pack_{event}_b9_ner_blanking.json',
             {'event': event, 'skipped': f'spaCy unavailable: {e}'})
        return
    df = load_event(event)
    tr = df[df['published_date'] < tr_end]
    te = df[df['published_date'] >= va_end]
    if len(tr) < 50 or len(te) < 30:
        save(f'cross_event_pack_{event}_b9_ner_blanking.json',
             {'event': event, 'skipped': f'tr={len(tr)} te={len(te)}'})
        return
    print(f"  splits: train={len(tr)} test={len(te)}")
    t0 = time.time()
    def mask_subset(text, allowed):
        if not text: return text
        doc = nlp(text)
        spans = sorted([(e.start_char, e.end_char, e.label_) for e in doc.ents],
                       key=lambda x: -x[0])
        out = text
        for s, e, lab in spans:
            if lab in allowed:
                out = out[:s] + f"[{lab}]" + out[e:]
        return out
    full_set = {'ORG', 'PERSON', 'MONEY', 'GPE', 'PERCENT', 'CARDINAL'}
    cache_full_tr = [mask_subset(t, full_set) for t in tr['title_en'].tolist()]
    cache_full_te = [mask_subset(t, full_set) for t in te['title_en'].tolist()]
    cache_org_tr = [mask_subset(t, {'ORG'}) for t in tr['title_en'].tolist()]
    cache_org_te = [mask_subset(t, {'ORG'}) for t in te['title_en'].tolist()]
    cache_op_tr = [mask_subset(t, {'ORG', 'PERSON'}) for t in tr['title_en'].tolist()]
    cache_op_te = [mask_subset(t, {'ORG', 'PERSON'}) for t in te['title_en'].tolist()]
    print(f"  NER done in {time.time()-t0:.1f}s")
    results = {}
    for name, tr_t, te_t in [
        ('original', tr['title_en'].tolist(), te['title_en'].tolist()),
        ('mask_ORG', cache_org_tr, cache_org_te),
        ('mask_ORG_PERSON', cache_op_tr, cache_op_te),
        ('mask_ALL', cache_full_tr, cache_full_te),
    ]:
        tf, clf = fit_tfidf_lr(tr_t, tr['y'].values)
        yp = clf.predict(tf.transform(te_t))
        results[name] = {'mcc': safe_mcc(te['y'].values, yp),
                         'pred_up_rate': float(yp.mean())}
        print(f"    {name:18s}: MCC {results[name]['mcc']:+.4f} pred_up {results[name]['pred_up_rate']:.3f}")
    out = {'event': event, 'hp': PAPER_HP,
           'n_tr': int(len(tr)), 'n_te': int(len(te)),
           'true_up_rate_test': float(te['y'].mean()),
           'results': results,
           'delta_mask_ORG_vs_original': float(results['mask_ORG']['mcc'] - results['original']['mcc']),
           'delta_mask_ALL_vs_original': float(results['mask_ALL']['mcc'] - results['original']['mcc'])}
    save(f'cross_event_pack_{event}_b9_ner_blanking.json', out)


# ====================================================================
# B11 - Publisher-level fairness
# ====================================================================
def b11_publisher_fairness(event, tr_end, va_end):
    print(f"\n[B11 / {event}] Publisher-level fairness")
    df = load_event(event)
    if 'publisher' not in df.columns:
        save(f'cross_event_pack_{event}_b11_publisher_fairness.json',
             {'event': event, 'skipped': 'no publisher column'})
        return
    tr = df[df['published_date'] < tr_end]
    va = df[(df['published_date'] >= tr_end) & (df['published_date'] < va_end)]
    te = df[df['published_date'] >= va_end].copy()
    tf_all = pd.concat([tr, va], ignore_index=True)
    tf, clf, proba_fn = fit_tfidf_lr(tf_all['title_en'], tf_all['y'].values, return_proba_fn=True)
    p_te = proba_fn(te['title_en'])
    yp_te = (p_te >= 0.5).astype(int)
    te = te.assign(y_pred=yp_te, p_up=p_te)
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
            'bucket': b, 'n': int(len(g)),
            'n_distinct_publishers': int(g['publisher'].nunique()),
            'mcc': safe_mcc(g['y'].values, g['y_pred'].values),
            'true_up_rate': float(g['y'].mean()),
            'pred_up_rate': float(g['y_pred'].mean())})
    top_pubs = te['publisher'].value_counts().head(8).index.tolist()
    per_pub = []
    for p in top_pubs:
        g = te[te['publisher'] == p]
        if len(g) < 10: continue
        per_pub.append({'publisher': p, 'n_test': int(len(g)),
                        'n_train': int(pub_freq.get(p, 0)),
                        'mcc': safe_mcc(g['y'].values, g['y_pred'].values),
                        'true_up_rate': float(g['y'].mean()),
                        'pred_up_rate': float(g['y_pred'].mean())})
    out = {'event': event, 'hp': PAPER_HP, 'n_te': int(len(te)),
           'pooled_test_mcc': safe_mcc(te['y'].values, te['y_pred'].values),
           'per_bucket': rows, 'per_publisher_top8': per_pub}
    save(f'cross_event_pack_{event}_b11_publisher_fairness.json', out)


# ====================================================================
# Main
# ====================================================================
def main():
    t0 = time.time()
    print(f"Cross-Event Full Pack (B1, B2, B3, B5, B6, B7, B8, B9, B11)")
    print(f"  data: {DATA}")
    print(f"  out:  {OUTDIR}")
    for event, (tr_end, va_end) in EVENTS.items():
        print(f"\n\n==================== EVENT: {event} ====================")
        print(f"  cutoffs: train_end={tr_end.date()}  val_end={va_end.date()}")
        for fn in [b1_calibration, b2_fairness, b3_topgrams,
                   b5_org_ablation, b6_audit_robustness,
                   b7_threshold_sweep, b8_per_month_rolling,
                   b9_ner_blanking, b11_publisher_fairness]:
            ts = time.time()
            try:
                fn(event, tr_end, va_end)
            except Exception as e:
                import traceback
                print(f"  !! {fn.__name__} ({event}) failed: {e}")
                traceback.print_exc()
            print(f"  {fn.__name__} elapsed {time.time()-ts:.1f}s")
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
