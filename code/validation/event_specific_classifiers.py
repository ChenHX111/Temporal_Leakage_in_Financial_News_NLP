"""
Focused Event-Specific Text Classifiers with Rolling Validation.

For the 3 events where text adds signal:
1. mergers_acquisitions (most stable: mean MCC=0.08)
2. exchange_announcement (highest peak but unstable)
3. shares_issue

Builds per-event TF-IDF + LogReg classifiers with rolling 8-month validation,
plus a combined "text-predictive events" selective classifier.
Also builds risk-coverage curves at varying thresholds.
"""
import sys, io, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, accuracy_score
from scipy.sparse import hstack, csr_matrix

BASE_DIR = r'.'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'classifier_training_v2.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'validation')

df = pd.read_parquet(DATA_PATH)
df['published_date'] = pd.to_datetime(df['published_date'])
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['label'] = (df['actual_side'].str.lower() == 'up').astype(int)
df['year_month'] = df['published_date'].dt.to_period('M')

EXCLUDE_COLS = {
    'news_id', 'yf_ticker', 'exchange', 'etf_ticker', 'title_en', 'content_en',
    'event', 'publisher', 'published_date', 'industry', 'publisher_topic',
    'actual_side', 'nextday_side', 'created_at',
    'price_change', 'price_change_percentage',
    'index_price_change', 'index_price_change_percentage',
    'nextday_price_change_percentage', 'label', 'year_month'
}
num_cols = [c for c in df.columns if c not in EXCLUDE_COLS
            and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

TEXT_PREDICTIVE = ['mergers_acquisitions', 'exchange_announcement', 'shares_issue']

# ============================================================
# Part 1: Per-Event Specialized Models with Rolling Validation
# ============================================================
print("=" * 80)
print("PART 1: PER-EVENT SPECIALIZED TEXT CLASSIFIERS")
print("=" * 80)

val_months = [pd.Period(f'2025-{m:02d}', 'M') for m in range(1, 9)]

for event in TEXT_PREDICTIVE:
    print(f"\n--- {event} ---")
    event_df = df[df['event'] == event].copy()
    print(f"  Total samples: {len(event_df)}")
    
    # Try multiple model configs
    configs = [
        ("TF-IDF title C=0.1", "title", 0.1, 200),
        ("TF-IDF title C=1.0", "title", 1.0, 200),
        ("TF-IDF title C=0.01", "title", 0.01, 200),
        ("TF-IDF title 500feat", "title", 0.1, 500),
        ("TF-IDF content C=0.1", "content", 0.1, 200),
        ("TF-IDF title+content C=0.1", "both", 0.1, 200),
    ]
    
    for config_name, text_src, C, max_feat in configs:
        mccs = []
        for vm in val_months:
            tr = event_df[event_df['year_month'] < vm]
            vl = event_df[event_df['year_month'] == vm]
            if len(tr) < 15 or len(vl) < 5:
                continue
            try:
                if text_src == "title":
                    text_tr = tr['title_en'].fillna('').astype(str)
                    text_vl = vl['title_en'].fillna('').astype(str)
                elif text_src == "content":
                    text_tr = tr['content_en'].fillna('').astype(str)
                    text_vl = vl['content_en'].fillna('').astype(str)
                else:
                    text_tr = (tr['title_en'].fillna('') + ' ' + tr['content_en'].fillna('')).astype(str)
                    text_vl = (vl['title_en'].fillna('') + ' ' + vl['content_en'].fillna('')).astype(str)
                
                tfidf = TfidfVectorizer(max_features=max_feat, stop_words='english', min_df=2, sublinear_tf=True)
                Xt = tfidf.fit_transform(text_tr)
                Xv = tfidf.transform(text_vl)
                lr = LogisticRegression(max_iter=2000, random_state=42, C=C)
                lr.fit(Xt, tr['label'])
                mcc = matthews_corrcoef(vl['label'], lr.predict(Xv))
                mccs.append(mcc)
            except:
                pass
        
        if mccs:
            print(f"  {config_name:<30} mean={np.mean(mccs):>7.4f} std={np.std(mccs):>6.4f} n_windows={len(mccs)}")

# ============================================================
# Part 2: Selective Prediction — Risk-Coverage Curves
# ============================================================
print(f"\n{'='*80}")
print("PART 2: SELECTIVE PREDICTION — RISK-COVERAGE CURVES")
print(f"{'='*80}")

train = df[df['published_date'] < '2025-04-01'].copy()
val = df[(df['published_date'] >= '2025-04-01') & (df['published_date'] < '2025-06-01')].copy()

# Build global model
tfidf = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, sublinear_tf=True)
X_tr = tfidf.fit_transform(train['title_en'].fillna('').astype(str))
X_vl = tfidf.transform(val['title_en'].fillna('').astype(str))
lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr.fit(X_tr, train['label'])
proba = lr.predict_proba(X_vl)[:, 1]

print(f"\n  {'Coverage':>10} {'N':>6} {'MCC':>8} {'BalAcc':>8} {'Acc':>8} {'Strategy':>25}")
print("  " + "-" * 75)

# Strategy 1: Confidence-based selection (high confidence predictions)
for coverage_pct in [100, 80, 60, 40, 30, 20, 10, 5]:
    n_select = max(1, int(len(val) * coverage_pct / 100))
    confidence = np.abs(proba - 0.5)
    top_idx = np.argsort(confidence)[-n_select:]
    y_true = val['label'].values[top_idx]
    y_pred = (proba[top_idx] > 0.5).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"  {coverage_pct:>9}% {n_select:>6} {mcc:>8.4f} {bacc:>8.4f} {acc:>8.4f} {'confidence':>25}")

print()

# Strategy 2: Event-based selection (text-predictive events only)
for events_set, set_name in [
    (TEXT_PREDICTIVE, "3 text-predictive"),
    (['mergers_acquisitions'], "M&A only"),
    (['mergers_acquisitions', 'shares_issue'], "M&A + shares_issue"),
]:
    mask = val['event'].isin(events_set)
    if mask.sum() < 10:
        continue
    y_true = val.loc[mask, 'label'].values
    y_pred = (proba[mask.values] > 0.5).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    coverage = mask.sum() / len(val) * 100
    print(f"  {coverage:>9.1f}% {mask.sum():>6} {mcc:>8.4f} {bacc:>8.4f} {acc:>8.4f} {set_name:>25}")

# Strategy 3: Event + confidence combined
print()
for events_set, set_name in [
    (TEXT_PREDICTIVE, "3 text-pred + conf"),
    (['mergers_acquisitions'], "M&A + conf"),
]:
    mask = val['event'].isin(events_set)
    if mask.sum() < 10:
        continue
    subset_proba = proba[mask.values]
    subset_conf = np.abs(subset_proba - 0.5)
    for conf_pct in [100, 75, 50]:
        n_select = max(1, int(mask.sum() * conf_pct / 100))
        top_idx = np.argsort(subset_conf)[-n_select:]
        y_true_sub = val.loc[mask, 'label'].values[top_idx]
        y_pred_sub = (subset_proba[top_idx] > 0.5).astype(int)
        mcc = matthews_corrcoef(y_true_sub, y_pred_sub)
        bacc = balanced_accuracy_score(y_true_sub, y_pred_sub)
        acc = accuracy_score(y_true_sub, y_pred_sub)
        coverage = n_select / len(val) * 100
        print(f"  {coverage:>9.1f}% {n_select:>6} {mcc:>8.4f} {bacc:>8.4f} {acc:>8.4f} {f'{set_name} top{conf_pct}%':>25}")

# ============================================================
# Part 3: M&A Deep Dive — What text features matter?
# ============================================================
print(f"\n{'='*80}")
print("PART 3: M&A DEEP DIVE — TOP PREDICTIVE TEXT FEATURES")
print(f"{'='*80}")

ma_train = train[train['event'] == 'mergers_acquisitions']
ma_val = val[val['event'] == 'mergers_acquisitions']
print(f"  M&A train: {len(ma_train)}, val: {len(ma_val)}")

tfidf_ma = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
Xt_ma = tfidf_ma.fit_transform(ma_train['title_en'].fillna('').astype(str))
Xv_ma = tfidf_ma.transform(ma_val['title_en'].fillna('').astype(str))
lr_ma = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_ma.fit(Xt_ma, ma_train['label'])
preds_ma = lr_ma.predict(Xv_ma)

mcc_ma = matthews_corrcoef(ma_val['label'], preds_ma)
bacc_ma = balanced_accuracy_score(ma_val['label'], preds_ma)
print(f"  M&A specialized model: MCC={mcc_ma:.4f}, BalAcc={bacc_ma:.4f}")

# Top features
feature_names = tfidf_ma.get_feature_names_out()
coef = lr_ma.coef_[0]
top_up = np.argsort(coef)[-15:][::-1]
top_down = np.argsort(coef)[:15]

print(f"\n  Top 15 features predicting UP (M&A):")
for idx in top_up:
    print(f"    {feature_names[idx]:<25} coef={coef[idx]:>8.4f}")

print(f"\n  Top 15 features predicting DOWN (M&A):")
for idx in top_down:
    print(f"    {feature_names[idx]:<25} coef={coef[idx]:>8.4f}")

# Rolling validation for M&A specialized model
print(f"\n  M&A Rolling Validation:")
ma_rolling = []
for vm in val_months:
    tr = df[(df['year_month'] < vm) & (df['event'] == 'mergers_acquisitions')]
    vl = df[(df['year_month'] == vm) & (df['event'] == 'mergers_acquisitions')]
    if len(tr) < 15 or len(vl) < 5:
        continue
    tfidf_r = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
    Xt_r = tfidf_r.fit_transform(tr['title_en'].fillna('').astype(str))
    Xv_r = tfidf_r.transform(vl['title_en'].fillna('').astype(str))
    lr_r = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_r.fit(Xt_r, tr['label'])
    p = lr_r.predict(Xv_r)
    mcc = matthews_corrcoef(vl['label'], p)
    bacc = balanced_accuracy_score(vl['label'], p)
    up_pct = vl['label'].mean() * 100
    print(f"    {str(vm)}: n_tr={len(tr):>5} n_vl={len(vl):>4} UP%={up_pct:>5.1f} MCC={mcc:>7.4f} BalAcc={bacc:>7.4f}")
    ma_rolling.append({"month": str(vm), "train_n": len(tr), "val_n": len(vl),
                        "mcc": round(mcc,4), "bacc": round(bacc,4), "up_pct": round(up_pct,1)})

# Save
results = {
    "ma_specialized": {"mcc": round(mcc_ma,4), "bacc": round(bacc_ma,4),
                        "top_up_features": [feature_names[i] for i in top_up],
                        "top_down_features": [feature_names[i] for i in top_down]},
    "ma_rolling": ma_rolling
}
out_path = os.path.join(RESULTS_DIR, 'event_specific_classifiers.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")
