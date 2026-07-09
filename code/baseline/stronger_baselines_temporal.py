"""
Stronger Baselines for EMNLP: XGBoost, LightGBM, Sentence-Transformer Embeddings
==================================================================================
All evaluated on canonical temporal splits. Same-day prediction (actual_side).
"""

import pandas as pd
import numpy as np
import sys, io, time, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, roc_auc_score, classification_report
)

DATA_PATH = r'.\data\classifier_training_v2.parquet'

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

# ============================================================
# Feature preparation
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

# TF-IDF
tfidf_title = TfidfVectorizer(max_features=50, ngram_range=(1,2), stop_words='english')
tfidf_content = TfidfVectorizer(max_features=100, ngram_range=(1,2), stop_words='english')

X_tr_title = tfidf_title.fit_transform(train['title_en'].fillna('').astype(str)).toarray()
X_va_title = tfidf_title.transform(val['title_en'].fillna('').astype(str)).toarray()
X_tr_content = tfidf_content.fit_transform(train['content_en'].fillna('').astype(str)).toarray()
X_va_content = tfidf_content.transform(val['content_en'].fillna('').astype(str)).toarray()

X_train_tfidf = np.hstack([X_train_num, X_tr_title, X_tr_content])
X_val_tfidf = np.hstack([X_val_num, X_va_title, X_va_content])

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_tfidf)
X_val_s = scaler.transform(X_val_tfidf)

y_train = (train['actual_side'] == 'up').astype(int).values
y_val = (val['actual_side'] == 'up').astype(int).values

print(f'Features: {X_train_s.shape[1]}')

# ============================================================
# Models
# ============================================================
def evaluate(name, model, X_tr, y_tr, X_va, y_va):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    y_pred = model.predict(X_va)
    y_proba = model.predict_proba(X_va)[:, 1]
    m = {
        'accuracy': accuracy_score(y_va, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_va, y_pred),
        'f1_weighted': f1_score(y_va, y_pred, average='weighted'),
        'f1_macro': f1_score(y_va, y_pred, average='macro'),
        'mcc': matthews_corrcoef(y_va, y_pred),
        'roc_auc': roc_auc_score(y_va, y_proba),
        'time_s': elapsed,
    }
    print(f'\n--- {name} ---')
    for k, v in m.items():
        print(f'  {k}: {v:.4f}')
    return m

results = {}

# 1. Logistic Regression
print('\n=== LOGISTIC REGRESSION ===')
results['LogisticRegression'] = evaluate('LogReg', 
    LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    X_train_s, y_train, X_val_s, y_val)

# 2. XGBoost
print('\n=== XGBOOST ===')
try:
    from xgboost import XGBClassifier
    results['XGBoost'] = evaluate('XGBoost',
        XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                      random_state=42, n_jobs=-1, eval_metric='logloss'),
        X_train_s, y_train, X_val_s, y_val)
except Exception as e:
    print(f'XGBoost failed: {e}')

# 3. LightGBM
print('\n=== LIGHTGBM ===')
try:
    from lightgbm import LGBMClassifier
    results['LightGBM'] = evaluate('LightGBM',
        LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                        random_state=42, n_jobs=-1, verbose=-1),
        X_train_s, y_train, X_val_s, y_val)
except Exception as e:
    print(f'LightGBM failed: {e}')

# 4. Gradient Boosting (sklearn)
print('\n=== GRADIENT BOOSTING ===')
results['GradientBoosting'] = evaluate('GradBoost',
    GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                random_state=42),
    X_train_s, y_train, X_val_s, y_val)

# 5. Random Forest (for comparison)
print('\n=== RANDOM FOREST (baseline) ===')
results['RandomForest'] = evaluate('RF',
    RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5,
                            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1),
    X_train_s, y_train, X_val_s, y_val)

# ============================================================
# Summary
# ============================================================
print('\n\n' + '='*80)
print('BASELINE COMPARISON (Same-Day, Temporal Split, TF-IDF+NumFeatures)')
print('='*80)
print(f'{"Model":<20} {"Acc":>7} {"BalAcc":>7} {"MCC":>7} {"AUC":>7} {"F1_m":>7} {"Time":>7}')
print('-'*80)
for name, m in sorted(results.items(), key=lambda x: x[1]['mcc'], reverse=True):
    print(f'{name:<20} {m["accuracy"]:>7.4f} {m["balanced_accuracy"]:>7.4f} {m["mcc"]:>7.4f} {m["roc_auc"]:>7.4f} {m["f1_macro"]:>7.4f} {m["time_s"]:>6.1f}s')

# Save results
results_path = r'.\results\baseline\stronger_baselines_temporal.json'
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to: {results_path}')
