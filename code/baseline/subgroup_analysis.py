"""Subgroup predictability analysis: per-exchange and per-event-type."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
import json, os

# Load data
df = pd.read_parquet(r'.\data\classifier_training_v2.parquet')
df['published_date'] = pd.to_datetime(df['published_date'])
df = df[df['actual_side'].str.lower().isin(['up','down'])].copy()
df['label'] = (df['actual_side'].str.lower()=='up').astype(int)

# Temporal split
train = df[df['published_date'] < '2025-04-01']
val = df[(df['published_date'] >= '2025-04-01') & (df['published_date'] < '2025-06-01')]

# Shared TF-IDF (fitted on all train)
tfidf = TfidfVectorizer(max_features=500, stop_words='english', min_df=2)
train_titles = train['title_en'].fillna('').astype(str)
val_titles = val['title_en'].fillna('').astype(str)
tfidf.fit(train_titles)

results = {"per_exchange": [], "per_event": []}

# === PER-EXCHANGE ===
print("=== PER-EXCHANGE ANALYSIS (LogReg, title TF-IDF) ===")
header = f"{'Exchange':<10} {'Train':>6} {'Val':>6} {'UP%':>5} {'MCC':>7} {'BalAcc':>7}"
print(header)
print("-" * len(header))

for exch in val['exchange'].value_counts().head(20).index:
    t = train[train['exchange'] == exch]
    v = val[val['exchange'] == exch]
    if len(t) < 50 or len(v) < 30:
        continue
    Xt = tfidf.transform(t['title_en'].fillna('').astype(str))
    Xv = tfidf.transform(v['title_en'].fillna('').astype(str))
    try:
        m = LogisticRegression(max_iter=1000, random_state=42)
        m.fit(Xt, t['label'])
        p = m.predict(Xv)
        mcc = matthews_corrcoef(v['label'], p)
        ba = balanced_accuracy_score(v['label'], p)
        up_pct = v['label'].mean() * 100
        print(f"{exch:<10} {len(t):>6} {len(v):>6} {up_pct:>5.1f} {mcc:>7.4f} {ba:>7.4f}")
        results["per_exchange"].append({
            "exchange": exch, "train_n": len(t), "val_n": len(v),
            "up_pct": round(up_pct, 1), "mcc": round(mcc, 4), "bal_acc": round(ba, 4)
        })
    except Exception as e:
        print(f"{exch:<10} {len(t):>6} {len(v):>6} ERROR: {e}")

# === PER-EVENT ===
print("\n=== PER-EVENT-TYPE ANALYSIS (LogReg, title TF-IDF) ===")
header2 = f"{'Event':<45} {'Train':>6} {'Val':>6} {'MCC':>7} {'BalAcc':>7}"
print(header2)
print("-" * len(header2))

for ev in val['event'].value_counts().head(30).index:
    if pd.isna(ev):
        continue
    t = train[train['event'] == ev]
    v = val[val['event'] == ev]
    if len(t) < 30 or len(v) < 20:
        continue
    Xt = tfidf.transform(t['title_en'].fillna('').astype(str))
    Xv = tfidf.transform(v['title_en'].fillna('').astype(str))
    try:
        m = LogisticRegression(max_iter=1000, random_state=42)
        m.fit(Xt, t['label'])
        p = m.predict(Xv)
        mcc = matthews_corrcoef(v['label'], p)
        ba = balanced_accuracy_score(v['label'], p)
        print(f"{str(ev)[:44]:<45} {len(t):>6} {len(v):>6} {mcc:>7.4f} {ba:>7.4f}")
        results["per_event"].append({
            "event": str(ev), "train_n": len(t), "val_n": len(v),
            "mcc": round(mcc, 4), "bal_acc": round(ba, 4)
        })
    except Exception as e:
        print(f"{str(ev)[:44]:<45} ERROR: {e}")

# Also try: what if we add exchange as a feature?
print("\n=== EXCHANGE-AWARE GLOBAL MODEL ===")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_exch = le.fit_transform(train['exchange'].fillna('UNK'))
val_exch = le.transform(val['exchange'].apply(lambda x: x if x in le.classes_ else 'UNK'))

from scipy.sparse import hstack, csr_matrix
X_train_full = hstack([tfidf.transform(train_titles), csr_matrix(train_exch.reshape(-1,1))])
X_val_full = hstack([tfidf.transform(val_titles), csr_matrix(val_exch.reshape(-1,1))])

m = LogisticRegression(max_iter=1000, random_state=42)
m.fit(X_train_full, train['label'])
p = m.predict(X_val_full)
mcc = matthews_corrcoef(val['label'], p)
ba = balanced_accuracy_score(val['label'], p)
print(f"TF-IDF + exchange_id: MCC={mcc:.4f} BalAcc={ba:.4f}")

# Save results
out_path = r'.\results\baseline\subgroup_analysis.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")
