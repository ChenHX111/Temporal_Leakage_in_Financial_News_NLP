"""
Leave-one-exchange-out analysis for M&A robustness.
Tests whether M&A signal is driven by specific exchanges.
"""
import sys, io, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

BASE_DIR = r'.'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'classifier_training_v2.parquet')

df = pd.read_parquet(DATA_PATH)
df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['label'] = (df['actual_side'].str.lower() == 'up').astype(int)

ma = df[df['event'] == 'mergers_acquisitions'].copy()
train = ma[ma['published_date'] < '2025-06-01']
test = ma[ma['published_date'] >= '2025-06-01']

# Per-exchange M&A counts
print("M&A articles by exchange (train+val / test):")
exchanges = ma['exchange'].value_counts()
top_exchanges = exchanges[exchanges >= 20].index.tolist()

print(f"\n{'Exchange':<10} {'Train+Val':>10} {'Test':>6} {'Train UP%':>10} {'Test UP%':>9}")
print("-" * 50)
for ex in top_exchanges[:15]:
    tr_n = len(train[train['exchange'] == ex])
    ts_n = len(test[test['exchange'] == ex])
    tr_up = train[train['exchange'] == ex]['label'].mean() * 100 if tr_n > 0 else 0
    ts_up = test[test['exchange'] == ex]['label'].mean() * 100 if ts_n > 0 else 0
    print(f"{ex:<10} {tr_n:>10} {ts_n:>6} {tr_up:>10.1f} {ts_up:>9.1f}")

# Leave-one-exchange-out
print(f"\n{'='*70}")
print("LEAVE-ONE-EXCHANGE-OUT (M&A text model)")
print(f"{'='*70}")
print(f"\n{'Excluded':10} {'Tr':>5} {'Ts':>5} {'MCC_val':>8} {'MCC_test':>9} {'Delta':>8}")
print("-" * 50)

base_tfidf = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
base_tr = ma[ma['published_date'] < '2025-04-01']
base_vl = ma[(ma['published_date'] >= '2025-04-01') & (ma['published_date'] < '2025-06-01')]
base_ts = test

X_base_tr = base_tfidf.fit_transform(base_tr['title_en'].fillna('').astype(str))
X_base_vl = base_tfidf.transform(base_vl['title_en'].fillna('').astype(str))
X_base_ts = base_tfidf.transform(base_ts['title_en'].fillna('').astype(str))
lr_base = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_base.fit(X_base_tr, base_tr['label'])
base_mcc_val = matthews_corrcoef(base_vl['label'], lr_base.predict(X_base_vl))
base_mcc_test = matthews_corrcoef(base_ts['label'], lr_base.predict(X_base_ts))
print(f"{'(none)':10} {len(base_tr):>5} {len(base_ts):>5} {base_mcc_val:>8.4f} {base_mcc_test:>9.4f} {'(base)':>8}")

results = []
for ex in top_exchanges[:10]:
    tr_loo = base_tr[base_tr['exchange'] != ex]
    vl_loo = base_vl[base_vl['exchange'] != ex]
    ts_loo = base_ts[base_ts['exchange'] != ex]
    
    if len(tr_loo) < 30 or len(vl_loo) < 10 or len(ts_loo) < 10:
        continue
    
    tfidf_loo = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
    X_tr_loo = tfidf_loo.fit_transform(tr_loo['title_en'].fillna('').astype(str))
    X_vl_loo = tfidf_loo.transform(vl_loo['title_en'].fillna('').astype(str))
    X_ts_loo = tfidf_loo.transform(ts_loo['title_en'].fillna('').astype(str))
    
    lr_loo = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_loo.fit(X_tr_loo, tr_loo['label'])
    mcc_val = matthews_corrcoef(vl_loo['label'], lr_loo.predict(X_vl_loo))
    mcc_test = matthews_corrcoef(ts_loo['label'], lr_loo.predict(X_ts_loo))
    delta = mcc_test - base_mcc_test
    
    print(f"{ex:10} {len(tr_loo):>5} {len(ts_loo):>5} {mcc_val:>8.4f} {mcc_test:>9.4f} {delta:>+8.4f}")
    results.append({"exchange": ex, "train_n": len(tr_loo), "test_n": len(ts_loo),
                     "mcc_val": round(mcc_val, 4), "mcc_test": round(mcc_test, 4),
                     "delta": round(delta, 4)})

# Per-exchange M&A performance (test set)
print(f"\n{'='*70}")
print("PER-EXCHANGE M&A PERFORMANCE (test set, global M&A model)")
print(f"{'='*70}")
for ex in top_exchanges[:10]:
    ex_mask = test['exchange'] == ex
    if ex_mask.sum() >= 10:
        preds = lr_base.predict(X_base_ts)
        ex_preds = preds[ex_mask.values]
        ex_labels = test.loc[ex_mask, 'label'].values
        mcc = matthews_corrcoef(ex_labels, ex_preds)
        print(f"  {ex:<10} n={ex_mask.sum():>4} MCC={mcc:.4f}")

out_path = os.path.join(BASE_DIR, 'results', 'validation', 'exchange_robustness.json')
with open(out_path, 'w') as f:
    json.dump({"base_mcc_test": round(base_mcc_test, 4), "loo_results": results}, f, indent=2)
print(f"\nSaved to: {out_path}")
