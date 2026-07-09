"""
Leakage Audit V2 — with TF-IDF + numerical metadata (mimicking original v2 baseline)
======================================================================================
The original baseline_temporal_results.json reported same-day RF MCC=0.0283 on temporal.
The paper claims "Random MCC=0.205" but that exact run is missing. This script
reproduces the full original baseline (TF-IDF title + content + numerical metadata)
under both random and temporal splits, with ALL non-leaky features.
Saved to: results/validation/leakage_audit_full_features.json
"""
import sys, io, os, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, accuracy_score, roc_auc_score

BASE = r'.'
DATA = os.path.join(BASE, 'data', 'classifier_training_v2.parquet')
OUT  = os.path.join(BASE, 'results', 'validation', 'leakage_audit_full_features.json')

print("Loading data ...")
df = pd.read_parquet(DATA)
df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['y'] = (df['actual_side'].str.lower() == 'up').astype(int)
df['title_en'] = df['title_en'].fillna('').astype(str)
df['content_en'] = df['content_en'].fillna('').astype(str)
df = df.sort_values('published_date').reset_index(drop=True)
print(f"Total: {len(df)}, UP rate {df['y'].mean():.3f}")

# Same exclusions as reproduce_baseline_temporal.py
EXCLUDE_COLS = [
    'news_id', 'yf_ticker', 'exchange', 'etf_ticker', 'title_en', 'content_en',
    'event', 'publisher', 'published_date', 'industry', 'publisher_topic',
    'actual_side', 'nextday_side', 'created_at',
    # Same-day labels: MUST exclude all price-change columns to avoid target leakage
    'price_change', 'price_change_percentage',
    'index_price_change', 'index_price_change_percentage',
    'nextday_price_change_percentage',
]
CAT_COLS = ['market_status']
num_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c not in CAT_COLS]
print(f"Numerical features ({len(num_cols)}): {num_cols[:15]} ...")

# Encode categorical (fit on full data so encoders are fixed)
df_enc = df.copy()
for col in CAT_COLS:
    if col in df_enc.columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

feature_cols = num_cols + CAT_COLS
X_num_full = df_enc[feature_cols].fillna(0).values

# Splits
TRAIN_END = pd.Timestamp('2025-04-01')
VAL_END   = pd.Timestamp('2025-06-01')
m_tr = df['published_date'] < TRAIN_END
m_va = (df['published_date'] >= TRAIN_END) & (df['published_date'] < VAL_END)
m_te = df['published_date'] >= VAL_END
N_TR, N_VA, N_TE = int(m_tr.sum()), int(m_va.sum()), int(m_te.sum())

splits = {
    'temporal': {'train': df.index[m_tr].tolist(),
                 'val':   df.index[m_va].tolist(),
                 'test':  df.index[m_te].tolist()},
}
y_arr = df['y'].values
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=N_TE, random_state=42)
remain_idx, test_idx = next(sss1.split(np.zeros(len(df)), y_arr))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=N_VA, random_state=42)
train_idx, val_idx = next(sss2.split(np.zeros(len(remain_idx)), y_arr[remain_idx]))
train_idx = remain_idx[train_idx]; val_idx = remain_idx[val_idx]
splits['random'] = {'train': train_idx.tolist(), 'val': val_idx.tolist(), 'test': test_idx.tolist()}
print(f"Train={N_TR} Val={N_VA} Test={N_TE}")

def eval_arch(X_tr, y_tr, X_te, y_te, clf):
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    try:
        proba = clf.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, proba))
    except: auc = None
    return {
        'mcc':  float(matthews_corrcoef(y_te, yp)),
        'bacc': float(balanced_accuracy_score(y_te, yp)),
        'acc':  float(accuracy_score(y_te, yp)),
        'auc':  auc,
        'time_s': round(time.time()-t0, 1),
    }

def build_features(tr_idx, te_idx):
    """Build full feature matrix matching original baseline."""
    # Title TF-IDF (50 features, ngram 1-2)
    tf_t = TfidfVectorizer(max_features=50, ngram_range=(1,2), stop_words='english')
    Xt_tr = tf_t.fit_transform(df.loc[tr_idx, 'title_en'])
    Xt_te = tf_t.transform(df.loc[te_idx, 'title_en'])
    # Content TF-IDF (100 features, ngram 1-2)
    tf_c = TfidfVectorizer(max_features=100, ngram_range=(1,2), stop_words='english')
    Xc_tr = tf_c.fit_transform(df.loc[tr_idx, 'content_en'])
    Xc_te = tf_c.transform(df.loc[te_idx, 'content_en'])
    # Numerical
    Xn_tr = X_num_full[tr_idx]
    Xn_te = X_num_full[te_idx]
    # Stack: TF-IDF as dense (matching original)
    X_tr = np.hstack([Xn_tr, Xt_tr.toarray(), Xc_tr.toarray()])
    X_te = np.hstack([Xn_te, Xt_te.toarray(), Xc_te.toarray()])
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    return X_tr_s, X_te_s

results = {'splits': {k: {'n_train':len(v['train']),'n_test':len(v['test'])} for k,v in splits.items()},
           'archs': {}}

for split_name, ix in splits.items():
    print(f"\n========== {split_name.upper()} ==========")
    tr_idx = np.array(ix['train']); te_idx = np.array(ix['test'])
    y_tr = df.loc[tr_idx, 'y'].values; y_te = df.loc[te_idx, 'y'].values
    X_tr, X_te = build_features(tr_idx, te_idx)
    print(f"  Feature shape: {X_tr.shape}")

    arch_results = {}
    arch_results['rf_full'] = eval_arch(X_tr, y_tr, X_te, y_te,
        RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=2,
                               max_features='sqrt', n_jobs=-1, random_state=42))
    print(f"  RF (TF-IDF+num)     : MCC={arch_results['rf_full']['mcc']:+.4f}  AUC={arch_results['rf_full']['auc']:.4f}")

    arch_results['logreg_full'] = eval_arch(X_tr, y_tr, X_te, y_te,
        LogisticRegression(max_iter=2000, C=0.5, random_state=42, n_jobs=-1))
    print(f"  LR (TF-IDF+num)     : MCC={arch_results['logreg_full']['mcc']:+.4f}  AUC={arch_results['logreg_full']['auc']:.4f}")

    arch_results['gb_full'] = eval_arch(X_tr, y_tr, X_te, y_te,
        GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
    print(f"  GB (TF-IDF+num)     : MCC={arch_results['gb_full']['mcc']:+.4f}  AUC={arch_results['gb_full']['auc']:.4f}")

    results['archs'][split_name] = arch_results

print("\n========== LEAKAGE DELTA (Random - Temporal, full features) ==========")
leakage = {}
for arch in results['archs']['temporal']:
    R = results['archs']['random'][arch]['mcc']
    T = results['archs']['temporal'][arch]['mcc']
    leakage[arch] = {
        'random_mcc': R, 'temporal_mcc': T, 'delta_mcc': R - T,
        'inflation_ratio': (R/T) if abs(T) > 0.005 else None,
    }
    ratio = f"{R/T:.1f}x" if abs(T) > 0.005 else "n/a"
    print(f"  {arch:>16}  R={R:+.4f}  T={T:+.4f}  Δ={R-T:+.4f}  ratio={ratio}")

results['leakage'] = leakage
results['feature_count'] = X_tr.shape[1]
results['timestamp'] = pd.Timestamp.now().isoformat()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT}")
