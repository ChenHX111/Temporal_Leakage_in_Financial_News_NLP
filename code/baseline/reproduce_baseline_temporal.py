"""
Baseline Reproduction Script for Fin_NLP Project
=================================================
Reproduces the Random Forest + TF-IDF baseline using canonical temporal splits.
Reports: accuracy, balanced accuracy, F1, MCC, ROC-AUC on validation set.
"""

import pandas as pd
import numpy as np
import sys, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    matthews_corrcoef, roc_auc_score, classification_report,
    confusion_matrix
)

# ============================================================
# 1. Load data
# ============================================================
DATA_PATH = r'C:\Users\a-chenhaoxue\OneDrive - Microsoft\Documents\Fin_NLP\autoresearch_package\data\classifier_training_v2.parquet'
df = pd.read_parquet(DATA_PATH)
print(f'Loaded {len(df):,} rows')

# ============================================================
# 2. Apply temporal split
# ============================================================
dates = pd.to_datetime(df['published_date'], errors='coerce')

# Binary filter
binary_mask = df['actual_side'].isin(['up', 'down'])
df_binary = df[binary_mask].copy()
dates_binary = dates[binary_mask]

train = df_binary[dates_binary < '2025-04-01'].copy()
val = df_binary[(dates_binary >= '2025-04-01') & (dates_binary < '2025-06-01')].copy()
test = df_binary[dates_binary >= '2025-06-01'].copy()

print(f'Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}')
print(f'Train UP%: {(train.actual_side=="up").mean()*100:.1f}%')
print(f'Val UP%: {(val.actual_side=="up").mean()*100:.1f}%')

# ============================================================
# 3. Feature engineering (matching original v2 pipeline)
# ============================================================
# Numerical features (exclude identifiers, text, targets, date)
EXCLUDE_COLS = [
    'news_id', 'yf_ticker', 'exchange', 'etf_ticker', 'title_en', 'content_en',
    'event', 'publisher', 'published_date', 'industry', 'publisher_topic',
    'actual_side', 'nextday_side', 'created_at',
    # For same-day model, exclude ALL price features (leakage)
    'price_change', 'price_change_percentage', 
    'index_price_change', 'index_price_change_percentage',
    'nextday_price_change_percentage',
]

# Categorical features to encode
CAT_COLS = ['market_status']  # Only market_status is safe for same-day

# Numerical feature columns
num_cols = [c for c in train.columns if c not in EXCLUDE_COLS and c not in CAT_COLS]
print(f'\nNumerical features ({len(num_cols)}): {num_cols}')

# Encode categoricals
le_dict = {}
for col in CAT_COLS:
    if col in train.columns:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        val[col] = val[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        le_dict[col] = le

# Combine numerical features
feature_cols = num_cols + CAT_COLS
X_train_num = train[feature_cols].fillna(0).values
X_val_num = val[feature_cols].fillna(0).values

# TF-IDF on title
tfidf_title = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words='english')
X_train_title = tfidf_title.fit_transform(train['title_en'].fillna('').astype(str))
X_val_title = tfidf_title.transform(val['title_en'].fillna('').astype(str))

# TF-IDF on content
tfidf_content = TfidfVectorizer(max_features=100, ngram_range=(1, 2), stop_words='english')
X_train_content = tfidf_content.fit_transform(train['content_en'].fillna('').astype(str))
X_val_content = tfidf_content.transform(val['content_en'].fillna('').astype(str))

# Combine all features
from scipy.sparse import hstack as sparse_hstack
X_train_sparse = sparse_hstack([X_train_title, X_train_content])
X_val_sparse = sparse_hstack([X_val_title, X_val_content])

X_train_all = np.hstack([X_train_num, X_train_sparse.toarray()])
X_val_all = np.hstack([X_val_num, X_val_sparse.toarray()])

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)
X_val_scaled = scaler.transform(X_val_all)

# Target
y_train = (train['actual_side'] == 'up').astype(int).values
y_val = (val['actual_side'] == 'up').astype(int).values

total_features = X_train_scaled.shape[1]
print(f'Total features: {total_features} ({len(num_cols)} num + {len(CAT_COLS)} cat + 50 title_tfidf + 100 content_tfidf)')

# ============================================================
# 4. Train Random Forest
# ============================================================
print('\n=== TRAINING RANDOM FOREST ===')
start = time.time()

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
elapsed = time.time() - start
print(f'Training time: {elapsed:.1f}s')

# ============================================================
# 5. Evaluate on validation set
# ============================================================
y_pred = rf.predict(X_val_scaled)
y_proba = rf.predict_proba(X_val_scaled)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_val, y_pred),
    'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
    'f1_weighted': f1_score(y_val, y_pred, average='weighted'),
    'f1_macro': f1_score(y_val, y_pred, average='macro'),
    'mcc': matthews_corrcoef(y_val, y_pred),
    'roc_auc': roc_auc_score(y_val, y_proba),
}

print('\n=== VALIDATION RESULTS (Same-Day Model) ===')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')

print(f'\n{classification_report(y_val, y_pred, target_names=["DOWN", "UP"])}')
print(f'Confusion matrix:\n{confusion_matrix(y_val, y_pred)}')

# ============================================================
# 6. Also train next-day model (can use price features)
# ============================================================
print('\n\n=== NEXT-DAY MODEL ===')

# For next-day, we CAN use same-day price features
EXCLUDE_NEXTDAY = [
    'news_id', 'yf_ticker', 'exchange', 'etf_ticker', 'title_en', 'content_en',
    'event', 'publisher', 'published_date', 'industry', 'publisher_topic',
    'actual_side', 'nextday_side', 'created_at',
    'nextday_price_change_percentage',  # target-related
]

# Filter for nextday binary
nd_binary = df['nextday_side'].isin(['up', 'down'])
train_nd = df[nd_binary & (dates < '2025-04-01')].copy()
val_nd = df[nd_binary & (dates >= '2025-04-01') & (dates < '2025-06-01')].copy()

num_cols_nd = [c for c in train_nd.columns if c not in EXCLUDE_NEXTDAY and c not in CAT_COLS]
print(f'Next-day features ({len(num_cols_nd)}): includes price_change, index features')

for col in CAT_COLS:
    if col in train_nd.columns:
        le = le_dict[col]
        train_nd[col] = train_nd[col].astype(str).map(lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1)
        val_nd[col] = val_nd[col].astype(str).map(lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1)

X_train_nd_num = train_nd[num_cols_nd + CAT_COLS].fillna(0).values
X_val_nd_num = val_nd[num_cols_nd + CAT_COLS].fillna(0).values

# TF-IDF
X_train_nd_title = tfidf_title.transform(train_nd['title_en'].fillna('').astype(str))
X_val_nd_title = tfidf_title.transform(val_nd['title_en'].fillna('').astype(str))
X_train_nd_content = tfidf_content.transform(train_nd['content_en'].fillna('').astype(str))
X_val_nd_content = tfidf_content.transform(val_nd['content_en'].fillna('').astype(str))

X_train_nd_all = np.hstack([X_train_nd_num, X_train_nd_title.toarray(), X_train_nd_content.toarray()])
X_val_nd_all = np.hstack([X_val_nd_num, X_val_nd_title.toarray(), X_val_nd_content.toarray()])

scaler_nd = StandardScaler()
X_train_nd_scaled = scaler_nd.fit_transform(X_train_nd_all)
X_val_nd_scaled = scaler_nd.transform(X_val_nd_all)

y_train_nd = (train_nd['nextday_side'] == 'up').astype(int).values
y_val_nd = (val_nd['nextday_side'] == 'up').astype(int).values

rf_nd = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt', bootstrap=True,
    class_weight=None, random_state=42, n_jobs=-1
)
rf_nd.fit(X_train_nd_scaled, y_train_nd)

y_pred_nd = rf_nd.predict(X_val_nd_scaled)
y_proba_nd = rf_nd.predict_proba(X_val_nd_scaled)[:, 1]

metrics_nd = {
    'accuracy': accuracy_score(y_val_nd, y_pred_nd),
    'balanced_accuracy': balanced_accuracy_score(y_val_nd, y_pred_nd),
    'f1_weighted': f1_score(y_val_nd, y_pred_nd, average='weighted'),
    'f1_macro': f1_score(y_val_nd, y_pred_nd, average='macro'),
    'mcc': matthews_corrcoef(y_val_nd, y_pred_nd),
    'roc_auc': roc_auc_score(y_val_nd, y_proba_nd),
}

print('\n=== VALIDATION RESULTS (Next-Day Model) ===')
for k, v in metrics_nd.items():
    print(f'  {k}: {v:.4f}')

print(f'\n{classification_report(y_val_nd, y_pred_nd, target_names=["DOWN", "UP"])}')

# ============================================================
# 7. Summary
# ============================================================
print('\n\n' + '='*60)
print('BASELINE SUMMARY')
print('='*60)
print(f'Same-day: Acc={metrics["accuracy"]:.4f} BalAcc={metrics["balanced_accuracy"]:.4f} MCC={metrics["mcc"]:.4f} AUC={metrics["roc_auc"]:.4f}')
print(f'Next-day: Acc={metrics_nd["accuracy"]:.4f} BalAcc={metrics_nd["balanced_accuracy"]:.4f} MCC={metrics_nd["mcc"]:.4f} AUC={metrics_nd["roc_auc"]:.4f}')

# Save results as JSON
results = {
    'same_day': metrics,
    'next_day': metrics_nd,
    'split': {'train': len(train), 'val': len(val), 'test': len(test)},
    'features': {'same_day': total_features, 'next_day': X_train_nd_scaled.shape[1]},
}
results_path = r'C:\Users\a-chenhaoxue\OneDrive - Microsoft\Documents\Fin_NLP\autoresearch_package\results\baseline\baseline_temporal_results.json'
import os
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to: {results_path}')
