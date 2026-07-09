"""
10K Permutation Test on M&A Locked Test Set
============================================
Currently we only have 500 permutations -> p=0.068 (marginal).
With 10K, we get a precise p-value to claim significance or not.

Uses the SAME locked-test M&A specialist setup from purged_and_final_test.py
to keep results comparable.

Saved to: results/validation/permutation_test_ma_10k.json
"""
import sys, io, os, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from joblib import Parallel, delayed

BASE = r'.'
DATA = os.path.join(BASE, 'data', 'classifier_training_v2.parquet')
OUT  = os.path.join(BASE, 'results', 'validation', 'permutation_test_ma_10k.json')

print("Loading data ...")
df = pd.read_parquet(DATA)
df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['y'] = (df['actual_side'].str.lower() == 'up').astype(int)
df['title_en'] = df['title_en'].fillna('').astype(str)

# Same locked-test setup
TRAIN_END = pd.Timestamp('2025-04-01')
VAL_END   = pd.Timestamp('2025-06-01')
ma = df[df['event'] == 'mergers_acquisitions'].copy()

ma_tr = ma[ma['published_date'] < VAL_END]   # train+val together for final
ma_te = ma[ma['published_date'] >= VAL_END]
print(f"M&A train+val: {len(ma_tr)}, test: {len(ma_te)}")

tfidf = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
Xtr = tfidf.fit_transform(ma_tr['title_en'])
Xte = tfidf.transform(ma_te['title_en'])
ytr = ma_tr['y'].values
yte = ma_te['y'].values

clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
clf.fit(Xtr, ytr)
yp = clf.predict(Xte)
observed_mcc = matthews_corrcoef(yte, yp)
print(f"Observed MCC: {observed_mcc:.6f}")

# ----- Permutation: shuffle TEST labels and recompute MCC -----
N_PERM = 10_000
print(f"Running {N_PERM:,} permutations (parallel) ...")

def one_perm(seed):
    rng = np.random.RandomState(seed)
    yperm = rng.permutation(yte)
    return matthews_corrcoef(yperm, yp)

t0 = time.time()
perm_mccs = Parallel(n_jobs=-1, verbose=1, batch_size=200)(
    delayed(one_perm)(s) for s in range(N_PERM))
perm_mccs = np.array(perm_mccs)
print(f"Done in {time.time()-t0:.1f}s")

p_value = float(np.mean(perm_mccs >= observed_mcc))
ci = np.percentile(perm_mccs, [2.5, 97.5])
zscore = (observed_mcc - perm_mccs.mean()) / perm_mccs.std()

result = {
    'observed_mcc':  float(observed_mcc),
    'n_permutations': N_PERM,
    'perm_mean':     float(perm_mccs.mean()),
    'perm_std':      float(perm_mccs.std()),
    'perm_2_5':      float(ci[0]),
    'perm_97_5':     float(ci[1]),
    'perm_99th':     float(np.percentile(perm_mccs, 99)),
    'perm_max':      float(perm_mccs.max()),
    'p_value':       p_value,
    'p_value_two_sided': float(np.mean(np.abs(perm_mccs) >= abs(observed_mcc))),
    'z_score':       float(zscore),
    'n_train':       len(ytr),
    'n_test':        len(yte),
    'test_up_rate':  float(yte.mean()),
    'pred_up_rate':  float(yp.mean()),
    'timestamp':     pd.Timestamp.now().isoformat(),
}
print(json.dumps(result, indent=2))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved: {OUT}")
