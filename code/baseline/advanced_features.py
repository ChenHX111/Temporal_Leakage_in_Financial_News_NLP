"""
Advanced feature engineering: content TF-IDF, title+content, feature importance analysis.
Also tests: temporal proximity features, day-of-week effects, seasonal patterns.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

BASE_DIR = r'C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'classifier_training_v2.parquet')

df = pd.read_parquet(DATA_PATH)
df['published_date'] = pd.to_datetime(df['published_date'])
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['label'] = (df['actual_side'].str.lower() == 'up').astype(int)

train = df[df['published_date'] < '2025-04-01'].copy()
val = df[(df['published_date'] >= '2025-04-01') & (df['published_date'] < '2025-06-01')].copy()
print(f"Train: {len(train)} | Val: {len(val)}")

# Safe numerical features
EXCLUDE_COLS = {
    'news_id', 'yf_ticker', 'exchange', 'etf_ticker', 'title_en', 'content_en',
    'event', 'publisher', 'published_date', 'industry', 'publisher_topic',
    'actual_side', 'nextday_side', 'created_at',
    'price_change', 'price_change_percentage',
    'index_price_change', 'index_price_change_percentage',
    'nextday_price_change_percentage', 'label'
}
num_cols = [c for c in train.columns if c not in EXCLUDE_COLS
            and train[c].dtype in ['float64', 'int64', 'float32', 'int32']]

def evaluate(name, y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0
    print(f"  {name:<50} Acc={acc:.4f} BalAcc={ba:.4f} MCC={mcc:.4f} AUC={auc:.4f}")
    return {"acc": round(acc,4), "bal_acc": round(ba,4), "mcc": round(mcc,4), "auc": round(auc,4)}

results = {}

# ============================================================
# Experiment 1: Content TF-IDF (not just title)
# ============================================================
print("\n=== CONTENT TF-IDF ===")
content_tfidf = TfidfVectorizer(max_features=1000, stop_words='english', min_df=5,
                                 ngram_range=(1,2), sublinear_tf=True)
X_train_content = content_tfidf.fit_transform(train['content_en'].fillna('').astype(str))
X_val_content = content_tfidf.transform(val['content_en'].fillna('').astype(str))

lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr.fit(X_train_content, train['label'])
results['content_tfidf_logreg'] = evaluate("content_tfidf_logreg", val['label'],
    lr.predict(X_val_content), lr.predict_proba(X_val_content)[:,1])

# ============================================================
# Experiment 2: Title + Content TF-IDF combined
# ============================================================
print("\n=== TITLE + CONTENT TF-IDF ===")
title_tfidf = TfidfVectorizer(max_features=500, stop_words='english', min_df=2,
                               sublinear_tf=True)
X_train_title = title_tfidf.fit_transform(train['title_en'].fillna('').astype(str))
X_val_title = title_tfidf.transform(val['title_en'].fillna('').astype(str))

X_train_tc = hstack([X_train_title, X_train_content, csr_matrix(train[num_cols].fillna(0).values)])
X_val_tc = hstack([X_val_title, X_val_content, csr_matrix(val[num_cols].fillna(0).values)])

lr_tc = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_tc.fit(X_train_tc, train['label'])
results['title_content_num_logreg'] = evaluate("title+content+num_logreg", val['label'],
    lr_tc.predict(X_val_tc), lr_tc.predict_proba(X_val_tc)[:,1])

# ============================================================
# Experiment 3: Numerical features only (are any useful?)
# ============================================================
print("\n=== NUMERICAL FEATURES ONLY ===")
scaler = StandardScaler()
X_train_ns = scaler.fit_transform(train[num_cols].fillna(0))
X_val_ns = scaler.transform(val[num_cols].fillna(0))

lr_num = LogisticRegression(max_iter=1000, random_state=42)
lr_num.fit(X_train_ns, train['label'])
results['num_only_logreg'] = evaluate("num_only_logreg", val['label'],
    lr_num.predict(X_val_ns), lr_num.predict_proba(X_val_ns)[:,1])

# Feature importance from numerical-only model
print("\n  Top numerical feature importances:")
coefs = pd.Series(lr_num.coef_[0], index=num_cols).abs().sort_values(ascending=False)
for feat, coef in coefs.head(15).items():
    direction = "UP" if lr_num.coef_[0][num_cols.index(feat)] > 0 else "DOWN"
    print(f"    {feat:<35} |coef|={coef:.4f} ({direction})")

# ============================================================
# Experiment 4: LightGBM on title+content+num+event
# ============================================================
print("\n=== LIGHTGBM: Title+Content+Num+Event ===")
# Add event one-hot
from sklearn.preprocessing import OneHotEncoder
event_counts = train['event'].value_counts()
top_events = set(event_counts.head(30).index)
train_ev = train['event'].apply(lambda x: x if x in top_events else 'OTHER').fillna('MISSING')
val_ev = val['event'].apply(lambda x: x if x in top_events else 'OTHER').fillna('MISSING')
ev_enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_train_ev = ev_enc.fit_transform(np.array(train_ev).reshape(-1,1))
X_val_ev = ev_enc.transform(np.array(val_ev).reshape(-1,1))

X_train_all = hstack([X_train_title, X_train_content, csr_matrix(train[num_cols].fillna(0).values), X_train_ev])
X_val_all = hstack([X_val_title, X_val_content, csr_matrix(val[num_cols].fillna(0).values), X_val_ev])

lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.3,
                           random_state=42, verbose=-1, n_jobs=-1)
lgbm.fit(X_train_all.toarray(), train['label'])
p = lgbm.predict(X_val_all.toarray())
pp = lgbm.predict_proba(X_val_all.toarray())[:,1]
results['full_lgbm'] = evaluate("full_lgbm (title+content+num+event)", val['label'], p, pp)

# Feature importance from LightGBM
importance = lgbm.feature_importances_
n_title = X_train_title.shape[1]
n_content = X_train_content.shape[1]
n_num = len(num_cols)
n_event = X_train_ev.shape[1]

title_imp = importance[:n_title].sum()
content_imp = importance[n_title:n_title+n_content].sum()
num_imp = importance[n_title+n_content:n_title+n_content+n_num].sum()
event_imp = importance[n_title+n_content+n_num:].sum()

total_imp = importance.sum()
print(f"\n  Feature group importances (LightGBM):")
print(f"    Title TF-IDF:   {title_imp/total_imp*100:>5.1f}% ({n_title} features)")
print(f"    Content TF-IDF: {content_imp/total_imp*100:>5.1f}% ({n_content} features)")
print(f"    Numerical:      {num_imp/total_imp*100:>5.1f}% ({n_num} features)")
print(f"    Event type:     {event_imp/total_imp*100:>5.1f}% ({n_event} features)")

# Top individual features
feat_names = (list(title_tfidf.get_feature_names_out()) +
              list(content_tfidf.get_feature_names_out()) +
              num_cols +
              list(ev_enc.get_feature_names_out()))
top_feats = pd.Series(importance, index=feat_names[:len(importance)]).sort_values(ascending=False)
print(f"\n  Top 20 individual features:")
for feat, imp in top_feats.head(20).items():
    print(f"    {str(feat)[:40]:<42} importance={imp}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*80)
print("ADVANCED FEATURE ENGINEERING SUMMARY")
print("="*80)
print(f"{'Model':<50} {'MCC':>7} {'BalAcc':>7} {'AUC':>7}")
print("-"*75)
for name, r in sorted(results.items(), key=lambda x: -x[1]['mcc']):
    print(f"{name:<50} {r['mcc']:>7.4f} {r['bal_acc']:>7.4f} {r['auc']:>7.4f}")

out_path = os.path.join(BASE_DIR, 'results', 'baseline', 'advanced_features_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")
