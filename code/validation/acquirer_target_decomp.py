"""
Acquirer vs Target Decomposition for M&A Articles
==================================================
Tests whether the M&A signal comes primarily from the acquirer-side or
target-side of deals (or both). Uses keyword detection on title to
classify each M&A article as ACQUIRER, TARGET, BOTH, or NEITHER.

For each subgroup, computes MCC on locked TEST set using the same
pipeline as the main M&A specialist (TF-IDF title + LogReg).

Saved to: results/validation/acquirer_target_decomp.json
"""
import sys, io, os, json, time, warnings, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

BASE = r'C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package'
DATA = os.path.join(BASE, 'data', 'classifier_training_v2.parquet')
OUT  = os.path.join(BASE, 'results', 'validation', 'acquirer_target_decomp.json')

print("Loading data ...")
df = pd.read_parquet(DATA)
df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['y'] = (df['actual_side'].str.lower() == 'up').astype(int)
df['title_en'] = df['title_en'].fillna('').astype(str)

ma = df[df['event'] == 'mergers_acquisitions'].copy()
print(f"Total M&A: {len(ma)}")

# ----- Role keyword detection -----
ACQ_PAT = re.compile(
    r'\b(acqui|acquir|takeover|buyer|buy[- ]?out|purchas|to acquire|will acquire|'
    r'completes acquisition|launches offer|tender offer|bid for|offers? to buy)\b',
    re.IGNORECASE)
TGT_PAT = re.compile(
    r'\b(target|to be acquir|being acquir|to be sold|sold to|sale of|divest|'
    r'subject of (an? )?bid|merger with|to merge with|received offer|'
    r'agree(s|d)? to be acquir)\b',
    re.IGNORECASE)
SELLER_PAT = re.compile(r'\b(seller|sells|to sell|divest|spin[- ]?off|spin[- ]?out)\b',
                        re.IGNORECASE)

def classify_role(title):
    is_acq = bool(ACQ_PAT.search(title))
    is_tgt = bool(TGT_PAT.search(title))
    is_sel = bool(SELLER_PAT.search(title))
    if is_acq and is_tgt:    return 'BOTH'
    if is_acq:               return 'ACQUIRER'
    if is_tgt or is_sel:     return 'TARGET'
    return 'NEITHER'

ma['role'] = ma['title_en'].apply(classify_role)
print("Role distribution:")
print(ma['role'].value_counts())

# ----- Splits -----
TRAIN_END = pd.Timestamp('2025-04-01')
VAL_END   = pd.Timestamp('2025-06-01')
ma_tr = ma[ma['published_date'] < VAL_END]
ma_te = ma[ma['published_date'] >= VAL_END]
print(f"\nTrain+val: {len(ma_tr)}, Test: {len(ma_te)}")

# ----- Train one M&A specialist on full M&A train+val -----
tfidf = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
Xtr = tfidf.fit_transform(ma_tr['title_en'])
Xte = tfidf.transform(ma_te['title_en'])
ytr = ma_tr['y'].values
yte = ma_te['y'].values
clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
clf.fit(Xtr, ytr)
yp = clf.predict(Xte)

overall_mcc = matthews_corrcoef(yte, yp)
print(f"\nOverall M&A test MCC: {overall_mcc:+.4f}")

# ----- Per-role evaluation -----
per_role = {}
for role in ['ACQUIRER', 'TARGET', 'BOTH', 'NEITHER']:
    mask = (ma_te['role'] == role).values
    n = int(mask.sum())
    if n < 5:
        per_role[role] = {'n_test': n, 'mcc': None, 'note': 'too few samples'}
        continue
    yt = yte[mask]
    yh = yp[mask]
    if len(np.unique(yt)) < 2 or len(np.unique(yh)) < 2:
        mcc = 0.0
    else:
        mcc = float(matthews_corrcoef(yt, yh))
    bacc = float(balanced_accuracy_score(yt, yh)) if len(np.unique(yt)) >= 2 else None
    up_rate = float(yt.mean())
    pred_up = float(yh.mean())
    per_role[role] = {
        'n_test': n,
        'mcc':    mcc,
        'bacc':   bacc,
        'up_rate_true': up_rate,
        'up_rate_pred': pred_up,
    }
    bacc_str = f"{bacc:.4f}" if bacc is not None else "n/a"
    print(f"  {role:>10}: n={n:>4}  MCC={mcc:+.4f}  BalAcc={bacc_str}  UP_true={up_rate:.3f}  UP_pred={pred_up:.3f}")

# ----- Train role-specific models -----
print("\n--- Role-specific models (train on role subset, test on role subset) ---")
role_specific = {}
for role in ['ACQUIRER', 'TARGET']:
    tr_mask = (ma_tr['role'] == role).values
    te_mask = (ma_te['role'] == role).values
    n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
    if n_tr < 50 or n_te < 20:
        role_specific[role] = {'n_train': n_tr, 'n_test': n_te, 'mcc': None,
                               'note': 'insufficient'}
        continue
    tf = TfidfVectorizer(max_features=200, stop_words='english', min_df=2, sublinear_tf=True)
    Xtr_r = tf.fit_transform(ma_tr.loc[tr_mask, 'title_en'])
    Xte_r = tf.transform(ma_te.loc[te_mask, 'title_en'])
    ytr_r = ma_tr.loc[tr_mask, 'y'].values
    yte_r = ma_te.loc[te_mask, 'y'].values
    cl = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    cl.fit(Xtr_r, ytr_r)
    yp_r = cl.predict(Xte_r)
    if len(np.unique(yte_r)) < 2 or len(np.unique(yp_r)) < 2:
        mcc_r = 0.0
    else:
        mcc_r = float(matthews_corrcoef(yte_r, yp_r))
    role_specific[role] = {
        'n_train': n_tr,
        'n_test':  n_te,
        'mcc':     mcc_r,
        'up_rate_true': float(yte_r.mean()),
        'up_rate_pred': float(yp_r.mean()),
    }
    print(f"  {role}: train={n_tr} test={n_te} MCC={mcc_r:+.4f}")

# ----- UP rate analysis: are TARGETs more likely to go UP? -----
print("\n--- UP rate by role (test set) ---")
up_by_role = {}
for role in ['ACQUIRER', 'TARGET', 'BOTH', 'NEITHER']:
    mask = ma_te['role'] == role
    if mask.sum() == 0: continue
    up_rate = float(ma_te.loc[mask, 'y'].mean())
    up_by_role[role] = {'n': int(mask.sum()), 'up_rate': up_rate}
    print(f"  {role}: n={mask.sum()}, UP rate={up_rate:.3f}")

# Heuristic: predict UP for TARGET, DOWN for ACQUIRER (Jensen-Ruback)
heur_pred = np.where(ma_te['role'].isin(['TARGET', 'BOTH']).values, 1, 0)
heur_mcc = matthews_corrcoef(yte, heur_pred) if len(np.unique(heur_pred)) >= 2 else 0.0
print(f"\nJensen-Ruback heuristic (TARGET=UP, ACQUIRER=DOWN): MCC={heur_mcc:+.4f}")

result = {
    'role_distribution': ma['role'].value_counts().to_dict(),
    'overall_ma_test_mcc': float(overall_mcc),
    'per_role_eval':       per_role,
    'role_specific_models': role_specific,
    'up_rate_by_role':     up_by_role,
    'jensen_ruback_heuristic_mcc': float(heur_mcc),
    'n_train': len(ma_tr),
    'n_test':  len(ma_te),
    'timestamp': pd.Timestamp.now().isoformat(),
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {OUT}")
