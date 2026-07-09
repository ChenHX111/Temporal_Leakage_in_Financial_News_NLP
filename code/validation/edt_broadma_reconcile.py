"""
EDT BroadMA Reconciliation
==========================
The original EDT validation used two M&A definitions:
- Narrow MA keyword: MCC = 0.066 (positive, replicates)
- Broad MA keyword: MCC = -0.0096 (negative, does NOT replicate)

This script investigates WHY broad keywords kill the signal:
1. What additional articles does "broad" include?
2. Are they noise (different event types)?
3. What is the per-keyword breakdown?

Saved to: results/validation/edt_broadma_reconcile.json
"""
import sys, io, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

BASE = r'.'
EDT_DIR = os.path.join(BASE, 'data', 'external')
OUT  = os.path.join(BASE, 'results', 'validation', 'edt_broadma_reconcile.json')

# Find EDT data
print("Looking for EDT files in:", EDT_DIR)
candidates = []
if os.path.exists(EDT_DIR):
    for root, _, files in os.walk(EDT_DIR):
        for f in files:
            if f.endswith(('.csv', '.parquet', '.json')):
                candidates.append(os.path.join(root, f))
print("Found:", candidates[:20])

if not candidates:
    print("No EDT files found. Skipping.")
    sys.exit(0)

# Try to load — prefer parquet
edt_path = next((p for p in candidates if p.endswith('.parquet')), None)
if not edt_path:
    edt_path = next((p for p in candidates if p.endswith('.csv')), None)
if not edt_path:
    print("No usable EDT file format")
    sys.exit(0)

print(f"Loading {edt_path} ...")
if edt_path.endswith('.parquet'):
    edt = pd.read_parquet(edt_path)
else:
    edt = pd.read_csv(edt_path)
print(f"EDT shape: {edt.shape}")
print(f"EDT columns: {list(edt.columns)}")

# Heuristic: find title and timestamp columns
title_col = next((c for c in edt.columns if c.lower() in ['title','headline','title_en']), None)
date_col  = next((c for c in edt.columns if c.lower() in ['date','timestamp','published_date','published','time']), None)
ret_col   = next((c for c in edt.columns if 'price' in c.lower() or 'return' in c.lower() or 'change' in c.lower()), None)

print(f"Detected: title={title_col}, date={date_col}, return={ret_col}")

if not (title_col and date_col):
    print("Missing required columns. Aborting.")
    sys.exit(0)

# Build labels if needed
if 'actual_side' in edt.columns:
    label_col = 'actual_side'
    edt['y'] = (edt[label_col].astype(str).str.lower() == 'up').astype(int)
elif 'label' in edt.columns:
    label_col = 'label'
    edt['y'] = (edt[label_col].astype(int) > 0).astype(int)
elif ret_col:
    edt['y'] = (edt[ret_col] > 0).astype(int)
else:
    print("No label info found")
    sys.exit(0)

edt[title_col] = edt[title_col].fillna('').astype(str)
edt[date_col] = pd.to_datetime(edt[date_col], errors='coerce').dt.tz_localize(None)
edt = edt.dropna(subset=[date_col, title_col]).copy()
edt = edt[edt[title_col].str.len() > 5].copy()
print(f"After cleaning: {len(edt)}")

# ----- Define narrow vs broad M&A keywords -----
NARROW = re._compile_str = None  # placeholder
import re as _re
NARROW_PAT = _re.compile(r'\b(merger|acquisition|acquir|to be acquired|takeover|tender offer)\b', _re.IGNORECASE)
BROAD_PAT  = _re.compile(r'\b(merger|acquisition|acquir|takeover|tender|buyout|deal|stake|combin|consolidat|partnership|joint venture)\b', _re.IGNORECASE)

edt['has_narrow_ma'] = edt[title_col].apply(lambda t: bool(NARROW_PAT.search(t)))
edt['has_broad_ma']  = edt[title_col].apply(lambda t: bool(BROAD_PAT.search(t)))
edt['only_broad']    = edt['has_broad_ma'] & ~edt['has_narrow_ma']

print(f"\nNarrow M&A: {edt['has_narrow_ma'].sum()}")
print(f"Broad M&A:  {edt['has_broad_ma'].sum()}")
print(f"ONLY broad (added by broad def): {edt['only_broad'].sum()}")

# Show what kind of articles "only_broad" picks up
print("\nSample 'only_broad' titles:")
sample = edt[edt['only_broad']][title_col].sample(min(15, edt['only_broad'].sum()), random_state=42).tolist()
for t in sample:
    print(f"  - {t[:140]}")

# ----- Run TF-IDF + LR on each subset under temporal split -----
edt = edt.sort_values(date_col).reset_index(drop=True)
n = len(edt)
tr_end = int(n * 0.6)
va_end = int(n * 0.8)

def run_subset(mask, label):
    sub = edt[mask].copy()
    if len(sub) < 200:
        return {'label': label, 'n': len(sub), 'mcc': None, 'note': 'too few'}
    sub = sub.sort_values(date_col).reset_index(drop=True)
    n_sub = len(sub)
    tr_end_s = int(n_sub * 0.6)
    va_end_s = int(n_sub * 0.8)
    tr = sub.iloc[:tr_end_s]
    te = sub.iloc[va_end_s:]
    if tr['y'].nunique() < 2 or te['y'].nunique() < 2:
        return {'label': label, 'n': len(sub), 'mcc': None, 'note': 'monoclass'}
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', min_df=2)
    Xtr = tfidf.fit_transform(tr[title_col])
    Xte = tfidf.transform(te[title_col])
    clf = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
    clf.fit(Xtr, tr['y'].values)
    yp = clf.predict(Xte)
    mcc = matthews_corrcoef(te['y'].values, yp)
    return {
        'label': label,
        'n_total': int(len(sub)),
        'n_train': int(tr_end_s),
        'n_test':  int(n_sub - va_end_s),
        'mcc':     float(mcc),
        'bacc':    float(balanced_accuracy_score(te['y'].values, yp)),
        'up_rate': float(te['y'].mean()),
    }

print("\n--- Subset evaluation ---")
results = {}
for label, mask in [
    ('narrow_ma', edt['has_narrow_ma']),
    ('broad_ma',  edt['has_broad_ma']),
    ('only_broad_added', edt['only_broad']),
]:
    r = run_subset(mask, label)
    results[label] = r
    print(f"  {label:>20}: n={r.get('n_total','?'):>5} MCC={r.get('mcc')} bacc={r.get('bacc')}")

# ----- Per-keyword breakdown for narrow vs broad -----
print("\n--- Per-keyword breakdown (narrow vs broad) ---")
keywords = ['merger', 'acquisition', 'acquir', 'takeover', 'tender', 'buyout',
            'deal', 'stake', 'combin', 'consolidat', 'partnership', 'joint venture']
kw_breakdown = {}
for kw in keywords:
    mask = edt[title_col].str.contains(kw, case=False, regex=False, na=False)
    n = int(mask.sum())
    if n >= 100:
        up = float(edt.loc[mask, 'y'].mean())
        kw_breakdown[kw] = {'n': n, 'up_rate': up}
        print(f"  '{kw}': n={n}, UP rate={up:.3f}")

result = {
    'edt_path': edt_path,
    'total_articles': int(len(edt)),
    'narrow_ma_count': int(edt['has_narrow_ma'].sum()),
    'broad_ma_count':  int(edt['has_broad_ma'].sum()),
    'only_broad_count': int(edt['only_broad'].sum()),
    'subset_results':  results,
    'keyword_breakdown': kw_breakdown,
    'sample_only_broad_titles': sample,
    'timestamp': pd.Timestamp.now().isoformat(),
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {OUT}")
