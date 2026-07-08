"""
DEFINITIVE Leakage Audit (rerun 2026-05-15)
============================================
Compares Random vs Temporal splits across multiple architectures using
IDENTICAL train/val/test sizes so the comparison is apples-to-apples.

Architectures:
  - TF-IDF(title) + LogisticRegression  (linear baseline)
  - TF-IDF(title+content) + GradientBoosting (high capacity)
  - TF-IDF(title) + RandomForest (high capacity, RF was original baseline)
  - all-MiniLM-L6-v2 embeddings + LogisticRegression (sentence transformer)
  - FinBERT [CLS] embeddings + LogisticRegression (domain pre-trained)

Splits (same SIZES under both regimes):
  - Temporal: < 2025-04 / 2025-04-2025-05 / >= 2025-06
  - Random:   60/20/20 stratified-by-label, fixed seed=42

Saved to: results/validation/leakage_audit_clean.json
"""
import sys, io, os, json, time, hashlib, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    matthews_corrcoef, balanced_accuracy_score, accuracy_score, roc_auc_score
)

BASE = r'C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package'
DATA = os.path.join(BASE, 'data', 'classifier_training_v2.parquet')
OUT  = os.path.join(BASE, 'results', 'validation', 'leakage_audit_clean.json')

print("Loading data ...")
df = pd.read_parquet(DATA)
df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['y'] = (df['actual_side'].str.lower() == 'up').astype(int)
df['title_en'] = df['title_en'].fillna('').astype(str)
df['content_en'] = df['content_en'].fillna('').astype(str)
df = df.sort_values('published_date').reset_index(drop=True)

print(f"Total binary: {len(df)}, UP rate {df['y'].mean():.3f}")

# ----- Temporal split -----
TRAIN_END = pd.Timestamp('2025-04-01')
VAL_END   = pd.Timestamp('2025-06-01')
m_tr = df['published_date'] < TRAIN_END
m_va = (df['published_date'] >= TRAIN_END) & (df['published_date'] < VAL_END)
m_te = df['published_date'] >= VAL_END
N_TR, N_VA, N_TE = int(m_tr.sum()), int(m_va.sum()), int(m_te.sum())
print(f"Temporal split: train={N_TR} val={N_VA} test={N_TE}")

splits = {}
splits['temporal'] = {
    'train': df.index[m_tr].tolist(),
    'val':   df.index[m_va].tolist(),
    'test':  df.index[m_te].tolist(),
}

# ----- Random split (stratified, same TRAIN/TEST sizes as temporal) -----
# Use stratified shuffle: first carve out test of size N_TE, then val of size N_VA, rest is train
rng = np.random.RandomState(42)
N_TOTAL = len(df)
y_arr = df['y'].values

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=N_TE, random_state=42)
remain_idx, test_idx = next(sss1.split(np.zeros(N_TOTAL), y_arr))

# val of size N_VA from remain
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=N_VA, random_state=42)
train_idx, val_idx = next(sss2.split(np.zeros(len(remain_idx)), y_arr[remain_idx]))
train_idx = remain_idx[train_idx]
val_idx   = remain_idx[val_idx]

splits['random'] = {
    'train': train_idx.tolist(),
    'val':   val_idx.tolist(),
    'test':  test_idx.tolist(),
}
print(f"Random split:   train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

# Sanity: same sizes
assert len(splits['random']['train']) == N_TR
assert len(splits['random']['val'])   == N_VA
assert len(splits['random']['test'])  == N_TE

# ----- Helper: evaluate a configuration -----
def eval_config(name, X_tr, y_tr, X_te, y_te, model):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    try:
        proba = model.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, proba))
    except Exception:
        auc = None
    out = {
        'mcc':  float(matthews_corrcoef(y_te, yp)),
        'bacc': float(balanced_accuracy_score(y_te, yp)),
        'acc':  float(accuracy_score(y_te, yp)),
        'auc':  auc,
        'n_train': int(len(y_tr)),
        'n_test':  int(len(y_te)),
        'up_pred': float(yp.mean()),
        'time_s':  round(time.time() - t0, 2),
    }
    print(f"  [{name:>40}] MCC={out['mcc']:+.4f} BalAcc={out['bacc']:.4f} "
          f"Acc={out['acc']:.4f} AUC={out['auc']:.4f}" if out['auc'] is not None else
          f"  [{name:>40}] MCC={out['mcc']:+.4f} BalAcc={out['bacc']:.4f} Acc={out['acc']:.4f}")
    return out

# ----- Run ALL architectures under BOTH splits -----
results = {'splits': {k: {kk: len(vv) for kk, vv in v.items()} for k, v in splits.items()},
           'archs': {}}

# Pre-compute embeddings once (cache) — only if available
def maybe_load_minilm():
    cache = os.path.join(BASE, 'data', 'embeddings_cache', 'minilm_title.npy')
    if os.path.exists(cache):
        emb = np.load(cache)
        if len(emb) == len(df):
            print(f"  Loaded cached MiniLM embeddings ({emb.shape})")
            return emb
    try:
        from sentence_transformers import SentenceTransformer
        print("  Computing MiniLM embeddings (one-shot, may take a few minutes) ...")
        m = SentenceTransformer('all-MiniLM-L6-v2')
        emb = m.encode(df['title_en'].tolist(), batch_size=64, show_progress_bar=True,
                       convert_to_numpy=True)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, emb)
        return emb
    except Exception as e:
        print(f"  MiniLM unavailable: {e}")
        return None

def maybe_load_finbert():
    cache = os.path.join(BASE, 'data', 'embeddings_cache', 'finbert_title.npy')
    if os.path.exists(cache):
        emb = np.load(cache)
        if len(emb) == len(df):
            print(f"  Loaded cached FinBERT embeddings ({emb.shape})")
            return emb
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        print("  Computing FinBERT [CLS] embeddings (this may take a long time on CPU) ...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        mdl = AutoModel.from_pretrained('ProsusAI/finbert').to(device).eval()
        embs = []
        BATCH = 32
        titles = df['title_en'].tolist()
        with torch.no_grad():
            for i in range(0, len(titles), BATCH):
                batch = titles[i:i+BATCH]
                enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
                out = mdl(**enc)
                cls = out.last_hidden_state[:, 0, :].cpu().numpy()
                embs.append(cls)
                if (i // BATCH) % 50 == 0:
                    print(f"    FinBERT {i}/{len(titles)}")
        emb = np.vstack(embs)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, emb)
        return emb
    except Exception as e:
        print(f"  FinBERT unavailable: {e}")
        return None

minilm_emb  = maybe_load_minilm()
finbert_emb = maybe_load_finbert()

for split_name, idx_set in splits.items():
    print(f"\n========== SPLIT: {split_name} ==========")
    tr_idx = np.array(idx_set['train'])
    te_idx = np.array(idx_set['test'])  # use TEST for headline numbers
    y_tr   = df.loc[tr_idx, 'y'].values
    y_te   = df.loc[te_idx, 'y'].values
    arch_results = {}

    # 1. TF-IDF(title) + LogisticRegression
    tfidf_t = TfidfVectorizer(max_features=2000, stop_words='english', min_df=2, sublinear_tf=True)
    Xtr = tfidf_t.fit_transform(df.loc[tr_idx, 'title_en'])
    Xte = tfidf_t.transform(df.loc[te_idx, 'title_en'])
    arch_results['tfidf_title_logreg'] = eval_config(
        'TFIDF(title)+LogReg', Xtr, y_tr, Xte, y_te,
        LogisticRegression(max_iter=2000, C=0.5, random_state=42, n_jobs=-1))

    # 2. TF-IDF(title) + RandomForest
    arch_results['tfidf_title_rf'] = eval_config(
        'TFIDF(title)+RandomForest', Xtr, y_tr, Xte, y_te,
        RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2,
                               n_jobs=-1, random_state=42))

    # 3. TF-IDF(title+content) + GradientBoosting
    tfidf_tc = TfidfVectorizer(max_features=2000, stop_words='english', min_df=2,
                               sublinear_tf=True, ngram_range=(1, 1))
    txt_tc_tr = (df.loc[tr_idx, 'title_en'] + ' ' + df.loc[tr_idx, 'content_en'].str[:2000])
    txt_tc_te = (df.loc[te_idx, 'title_en'] + ' ' + df.loc[te_idx, 'content_en'].str[:2000])
    Xtr2 = tfidf_tc.fit_transform(txt_tc_tr)
    Xte2 = tfidf_tc.transform(txt_tc_te)
    arch_results['tfidf_titlecontent_gradboost'] = eval_config(
        'TFIDF(title+content)+GradBoost', Xtr2, y_tr, Xte2, y_te,
        GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))

    # 4. MiniLM embeddings + LogReg
    if minilm_emb is not None:
        Xe_tr = minilm_emb[tr_idx]; Xe_te = minilm_emb[te_idx]
        arch_results['minilm_logreg'] = eval_config(
            'MiniLM(title)+LogReg', Xe_tr, y_tr, Xe_te, y_te,
            LogisticRegression(max_iter=2000, C=1.0, random_state=42, n_jobs=-1))

    # 5. FinBERT embeddings + LogReg
    if finbert_emb is not None:
        Xf_tr = finbert_emb[tr_idx]; Xf_te = finbert_emb[te_idx]
        arch_results['finbert_logreg'] = eval_config(
            'FinBERT(title)+LogReg', Xf_tr, y_tr, Xf_te, y_te,
            LogisticRegression(max_iter=2000, C=1.0, random_state=42, n_jobs=-1))

    results['archs'][split_name] = arch_results

# ----- Compute leakage delta per architecture -----
print("\n========== LEAKAGE DELTA (Random - Temporal) ==========")
leakage = {}
for arch in results['archs']['temporal'].keys():
    if arch in results['archs']['random']:
        d_mcc  = results['archs']['random'][arch]['mcc']  - results['archs']['temporal'][arch]['mcc']
        d_bacc = results['archs']['random'][arch]['bacc'] - results['archs']['temporal'][arch]['bacc']
        leakage[arch] = {
            'random_mcc':   results['archs']['random'][arch]['mcc'],
            'temporal_mcc': results['archs']['temporal'][arch]['mcc'],
            'delta_mcc':    d_mcc,
            'random_bacc':  results['archs']['random'][arch]['bacc'],
            'temporal_bacc': results['archs']['temporal'][arch]['bacc'],
            'delta_bacc':   d_bacc,
            'inflation_ratio': (results['archs']['random'][arch]['mcc'] /
                                results['archs']['temporal'][arch]['mcc']
                                if abs(results['archs']['temporal'][arch]['mcc']) > 0.001 else None),
        }
        print(f"  {arch:>32}  R={leakage[arch]['random_mcc']:+.4f}  "
              f"T={leakage[arch]['temporal_mcc']:+.4f}  "
              f"Delta={d_mcc:+.4f}")

results['leakage'] = leakage
results['data_hash'] = hashlib.md5(open(DATA, 'rb').read(1_000_000)).hexdigest()
results['timestamp'] = pd.Timestamp.now().isoformat()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {OUT}")
