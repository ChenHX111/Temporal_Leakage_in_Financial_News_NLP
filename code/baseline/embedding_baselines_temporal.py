"""
Sentence-Transformer Embedding Baseline
========================================
Replace TF-IDF with dense embeddings from a pre-trained sentence transformer.
This is the modern NLP baseline needed for EMNLP comparison.
"""

import pandas as pd
import numpy as np
import sys, io, time, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, roc_auc_score
)

DATA_PATH = r'C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package\data\classifier_training_v2.parquet'
CACHE_DIR = r'C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package\data\embeddings_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# Load and split
# ============================================================
df = pd.read_parquet(DATA_PATH)
dates = pd.to_datetime(df['published_date'], errors='coerce')
binary = df[df['actual_side'].isin(['up','down'])].copy()
dates_b = dates[binary.index]

train = binary[dates_b < '2025-04-01'].copy()
val = binary[(dates_b >= '2025-04-01') & (dates_b < '2025-06-01')].copy()
print(f'Train: {len(train):,} | Val: {len(val):,}')

y_train = (train['actual_side'] == 'up').astype(int).values
y_val = (val['actual_side'] == 'up').astype(int).values

# ============================================================
# Generate embeddings (cache for reuse)
# ============================================================
train_cache = os.path.join(CACHE_DIR, 'train_embeddings.npy')
val_cache = os.path.join(CACHE_DIR, 'val_embeddings.npy')

if os.path.exists(train_cache) and os.path.exists(val_cache):
    print('Loading cached embeddings...')
    X_train_emb = np.load(train_cache)
    X_val_emb = np.load(val_cache)
else:
    print('Generating sentence-transformer embeddings...')
    from sentence_transformers import SentenceTransformer
    
    # Use a lightweight model that runs on CPU
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Combine title + content for embedding
    train_texts = (train['title_en'].fillna('').astype(str) + ' ' + 
                   train['content_en'].fillna('').astype(str)).tolist()
    val_texts = (val['title_en'].fillna('').astype(str) + ' ' + 
                 val['content_en'].fillna('').astype(str)).tolist()
    
    print(f'Encoding {len(train_texts)} train texts...')
    t0 = time.time()
    X_train_emb = model.encode(train_texts, batch_size=128, show_progress_bar=True)
    print(f'Train embeddings: {X_train_emb.shape} in {time.time()-t0:.0f}s')
    
    print(f'Encoding {len(val_texts)} val texts...')
    t0 = time.time()
    X_val_emb = model.encode(val_texts, batch_size=128, show_progress_bar=True)
    print(f'Val embeddings: {X_val_emb.shape} in {time.time()-t0:.0f}s')
    
    # Cache
    np.save(train_cache, X_train_emb)
    np.save(val_cache, X_val_emb)
    print('Embeddings cached for reuse')

print(f'Embedding dim: {X_train_emb.shape[1]}')

# ============================================================
# Also prepare numerical features
# ============================================================
EXCLUDE = [
    'news_id', 'yf_ticker', 'exchange', 'etf_ticker', 'title_en', 'content_en',
    'event', 'publisher', 'published_date', 'industry', 'publisher_topic',
    'actual_side', 'nextday_side', 'created_at',
    'price_change', 'price_change_percentage',
    'index_price_change', 'index_price_change_percentage',
    'nextday_price_change_percentage',
]
num_cols = [c for c in train.columns if c not in EXCLUDE and c != 'market_status']
le = LabelEncoder()
train['market_status_enc'] = le.fit_transform(train['market_status'].astype(str))
val['market_status_enc'] = val['market_status'].astype(str).map(
    lambda x: le.transform([x])[0] if x in le.classes_ else -1)
feature_cols = num_cols + ['market_status_enc']
X_train_num = train[feature_cols].fillna(0).values
X_val_num = val[feature_cols].fillna(0).values

# ============================================================
# Evaluate models
# ============================================================
def evaluate(name, X_tr, y_tr, X_va, y_va, model):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    t0 = time.time()
    model.fit(X_tr_s, y_tr)
    elapsed = time.time() - t0
    y_pred = model.predict(X_va_s)
    y_proba = model.predict_proba(X_va_s)[:, 1]
    m = {
        'accuracy': accuracy_score(y_va, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_va, y_pred),
        'f1_macro': f1_score(y_va, y_pred, average='macro'),
        'mcc': matthews_corrcoef(y_va, y_pred),
        'roc_auc': roc_auc_score(y_va, y_proba),
        'time_s': elapsed,
        'n_features': X_tr.shape[1],
    }
    print(f'{name:<35} Acc={m["accuracy"]:.4f} BalAcc={m["balanced_accuracy"]:.4f} MCC={m["mcc"]:.4f} AUC={m["roc_auc"]:.4f}')
    return m

results = {}
print('\n=== EMBEDDING-ONLY BASELINES ===')

# Embeddings only
results['emb_logreg'] = evaluate('Embeddings → LogReg', 
    X_train_emb, y_train, X_val_emb, y_val,
    LogisticRegression(max_iter=1000, random_state=42))

results['emb_rf'] = evaluate('Embeddings → RF',
    X_train_emb, y_train, X_val_emb, y_val,
    RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1))

# Embeddings + numerical features
X_train_combo = np.hstack([X_train_emb, X_train_num])
X_val_combo = np.hstack([X_val_emb, X_val_num])

print('\n=== EMBEDDINGS + NUMERICAL FEATURES ===')
results['emb_num_logreg'] = evaluate('Emb+Num → LogReg',
    X_train_combo, y_train, X_val_combo, y_val,
    LogisticRegression(max_iter=1000, random_state=42))

results['emb_num_rf'] = evaluate('Emb+Num → RF',
    X_train_combo, y_train, X_val_combo, y_val,
    RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1))

try:
    from lightgbm import LGBMClassifier
    results['emb_num_lgbm'] = evaluate('Emb+Num → LightGBM',
        X_train_combo, y_train, X_val_combo, y_val,
        LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                        random_state=42, n_jobs=-1, verbose=-1))
except:
    pass

# Summary
print('\n' + '='*80)
print('ALL EMBEDDING BASELINES SUMMARY')
print('='*80)
print(f'{"Model":<35} {"Acc":>7} {"BalAcc":>7} {"MCC":>7} {"AUC":>7} {"Feats":>6}')
print('-'*80)
for name, m in sorted(results.items(), key=lambda x: x[1]['mcc'], reverse=True):
    print(f'{name:<35} {m["accuracy"]:>7.4f} {m["balanced_accuracy"]:>7.4f} {m["mcc"]:>7.4f} {m["roc_auc"]:>7.4f} {m["n_features"]:>6}')

# Save
results_path = r'C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package\results\baseline\embedding_baselines_temporal.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to: {results_path}')
