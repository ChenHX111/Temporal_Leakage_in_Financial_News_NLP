"""
FAST Leakage Audit (TF-IDF based, no embeddings)
Reproduces the headline 'random vs temporal' comparison without slow embeddings.
Saved to: results/validation/leakage_audit_fast.json
"""
import sys, io, os, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, accuracy_score, roc_auc_score

BASE = r'.'
DATA = os.path.join(BASE, 'data', 'classifier_training_v2.parquet')
OUT  = os.path.join(BASE, 'results', 'validation', 'leakage_audit_fast.json')

print("Loading data ...")
df = pd.read_parquet(DATA)
df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['y'] = (df['actual_side'].str.lower() == 'up').astype(int)
df['title_en'] = df['title_en'].fillna('').astype(str)
df = df.sort_values('published_date').reset_index(drop=True)
print(f"Total: {len(df)}, UP rate {df['y'].mean():.3f}")

# Splits (same sizes for both)
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

# Random split: same sizes
rng = np.random.RandomState(42)
y_arr = df['y'].values
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=N_TE, random_state=42)
remain_idx, test_idx = next(sss1.split(np.zeros(len(df)), y_arr))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=N_VA, random_state=42)
train_idx, val_idx = next(sss2.split(np.zeros(len(remain_idx)), y_arr[remain_idx]))
train_idx = remain_idx[train_idx]; val_idx = remain_idx[val_idx]
splits['random'] = {'train': train_idx.tolist(), 'val': val_idx.tolist(), 'test': test_idx.tolist()}
print(f"Train={N_TR} Val={N_VA} Test={N_TE}")

def eval_arch(name, X_tr, y_tr, X_te, y_te, clf):
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

results = {'splits': {k: {'n_train':len(v['train']),'n_val':len(v['val']),'n_test':len(v['test'])} for k,v in splits.items()},
           'archs': {}}

for split_name, ix in splits.items():
    print(f"\n========== {split_name.upper()} ==========")
    tr_idx = np.array(ix['train']); te_idx = np.array(ix['test'])
    y_tr = df.loc[tr_idx, 'y'].values; y_te = df.loc[te_idx, 'y'].values
    arch_results = {}

    # Title TF-IDF
    tf_t = TfidfVectorizer(max_features=2000, stop_words='english', min_df=2, sublinear_tf=True)
    Xtr = tf_t.fit_transform(df.loc[tr_idx, 'title_en'])
    Xte = tf_t.transform(df.loc[te_idx, 'title_en'])

    arch_results['tfidf_title_logreg'] = eval_arch('LR', Xtr, y_tr, Xte, y_te,
        LogisticRegression(max_iter=2000, C=0.5, random_state=42, n_jobs=-1))
    print(f"  TF-IDF(title)+LR : MCC={arch_results['tfidf_title_logreg']['mcc']:+.4f}")

    arch_results['tfidf_title_rf'] = eval_arch('RF', Xtr, y_tr, Xte, y_te,
        RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2,
                               n_jobs=-1, random_state=42))
    print(f"  TF-IDF(title)+RF : MCC={arch_results['tfidf_title_rf']['mcc']:+.4f}")

    arch_results['tfidf_title_gradboost'] = eval_arch('GB', Xtr.toarray(), y_tr, Xte.toarray(), y_te,
        GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
    print(f"  TF-IDF(title)+GB : MCC={arch_results['tfidf_title_gradboost']['mcc']:+.4f}")

    results['archs'][split_name] = arch_results

print("\n========== LEAKAGE DELTA ==========")
leakage = {}
for arch in results['archs']['temporal']:
    R = results['archs']['random'][arch]['mcc']
    T = results['archs']['temporal'][arch]['mcc']
    leakage[arch] = {
        'random_mcc': R, 'temporal_mcc': T, 'delta_mcc': R - T,
        'random_bacc': results['archs']['random'][arch]['bacc'],
        'temporal_bacc': results['archs']['temporal'][arch]['bacc'],
        'inflation_ratio': (R/T) if abs(T) > 0.005 else None,
    }
    ratio = f"{R/T:.1f}x" if abs(T) > 0.005 else "n/a"
    print(f"  {arch:>32}  R={R:+.4f}  T={T:+.4f}  Δ={R-T:+.4f}  ratio={ratio}")

results['leakage'] = leakage
results['timestamp'] = pd.Timestamp.now().isoformat()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT}")
