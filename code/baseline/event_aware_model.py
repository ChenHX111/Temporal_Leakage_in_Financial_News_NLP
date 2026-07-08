"""
Event-Aware Routing Model for Financial News Direction Prediction.

Key insight: Different event types have very different predictability.
Some (exchange_announcement, share_capital_increase) have MCC>0.2,
while others (management_changes, regulatory_filings) are anti-correlated.

Approach:
1. Event-type as a feature (one-hot) → let the model learn event-specific patterns
2. Event-type routing → train separate models for each event group
3. Event-type + exchange interaction features
4. Confidence-weighted: only predict when model is confident, abstain otherwise
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                              roc_auc_score, accuracy_score, classification_report)
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

BASE_DIR = r'C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'classifier_training_v2.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'baseline')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load and split
df = pd.read_parquet(DATA_PATH)
df['published_date'] = pd.to_datetime(df['published_date'])
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['label'] = (df['actual_side'].str.lower() == 'up').astype(int)

train = df[df['published_date'] < '2025-04-01'].copy()
val = df[(df['published_date'] >= '2025-04-01') & (df['published_date'] < '2025-06-01')].copy()

print(f"Train: {len(train)} | Val: {len(val)}")

# Shared TF-IDF on title
tfidf = TfidfVectorizer(max_features=500, stop_words='english', min_df=2)
X_train_tfidf = tfidf.fit_transform(train['title_en'].fillna('').astype(str))
X_val_tfidf = tfidf.transform(val['title_en'].fillna('').astype(str))

# Numerical features (safe ones only)
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
print(f"Numerical features ({len(num_cols)}): {num_cols[:10]}...")

X_train_num = train[num_cols].fillna(0).values
X_val_num = val[num_cols].fillna(0).values

results = {}

def evaluate(name, y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0
    print(f"  {name:<45} Acc={acc:.4f} BalAcc={ba:.4f} MCC={mcc:.4f} AUC={auc:.4f}")
    results[name] = {"acc": round(acc,4), "bal_acc": round(ba,4),
                     "mcc": round(mcc,4), "auc": round(auc,4)}
    return mcc

# ============================================================
# Model 1: Baseline (TF-IDF + num, no event info)
# ============================================================
print("\n=== MODEL 1: Baseline (TF-IDF + num, no event) ===")
X_train_base = hstack([X_train_tfidf, csr_matrix(X_train_num)])
X_val_base = hstack([X_val_tfidf, csr_matrix(X_val_num)])

lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
lr.fit(X_train_base, train['label'])
evaluate("baseline_logreg", val['label'], lr.predict(X_val_base), lr.predict_proba(X_val_base)[:,1])

# ============================================================
# Model 2: + Event type as one-hot feature
# ============================================================
print("\n=== MODEL 2: + Event type one-hot ===")
# Top 30 events + "other"
event_counts = train['event'].value_counts()
top_events = set(event_counts.head(30).index)

def encode_event(series, top_events):
    return series.apply(lambda x: x if x in top_events else 'OTHER').fillna('MISSING')

train_event = encode_event(train['event'], top_events)
val_event = encode_event(val['event'], top_events)

event_enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_train_event = event_enc.fit_transform(np.array(train_event).reshape(-1,1))
X_val_event = event_enc.transform(np.array(val_event).reshape(-1,1))

X_train_ev = hstack([X_train_tfidf, csr_matrix(X_train_num), X_train_event])
X_val_ev = hstack([X_val_tfidf, csr_matrix(X_val_num), X_val_event])

lr_ev = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
lr_ev.fit(X_train_ev, train['label'])
evaluate("event_onehot_logreg", val['label'], lr_ev.predict(X_val_ev), lr_ev.predict_proba(X_val_ev)[:,1])

# ============================================================
# Model 3: + Exchange as one-hot feature
# ============================================================
print("\n=== MODEL 3: + Event + Exchange one-hot ===")
top_exchanges = set(train['exchange'].value_counts().head(20).index)

def encode_exchange(series, top_exchanges):
    return series.apply(lambda x: x if x in top_exchanges else 'OTHER').fillna('MISSING')

train_exch = encode_exchange(train['exchange'], top_exchanges)
val_exch = encode_exchange(val['exchange'], top_exchanges)

exch_enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_train_exch = exch_enc.fit_transform(np.array(train_exch).reshape(-1,1))
X_val_exch = exch_enc.transform(np.array(val_exch).reshape(-1,1))

X_train_full = hstack([X_train_tfidf, csr_matrix(X_train_num), X_train_event, X_train_exch])
X_val_full = hstack([X_val_tfidf, csr_matrix(X_val_num), X_val_event, X_val_exch])

lr_full = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
lr_full.fit(X_train_full, train['label'])
evaluate("event+exchange_logreg", val['label'], lr_full.predict(X_val_full), lr_full.predict_proba(X_val_full)[:,1])

# Also with LightGBM
lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                           random_state=42, verbose=-1, n_jobs=-1)
lgbm.fit(X_train_full.toarray(), train['label'])
p_lgbm = lgbm.predict(X_val_full.toarray())
pp_lgbm = lgbm.predict_proba(X_val_full.toarray())[:,1]
evaluate("event+exchange_lgbm", val['label'], p_lgbm, pp_lgbm)

# ============================================================
# Model 4: Event-Routing (separate model per event group)
# ============================================================
print("\n=== MODEL 4: Event-Routing (separate models per event group) ===")

# Group events by predictability
HIGH_SIGNAL = {'exchange_announcement', 'share_capital_increase', 'interim_information',
               'shares_issue', 'corporate_action', 'clinical_study'}
ANTI_SIGNAL = {'management_changes', 'press_releases', 'company_regulatory_filings',
               'changes_in_share_capital_and_votes'}

def event_group(e):
    if pd.isna(e): return 'missing'
    if e in HIGH_SIGNAL: return 'high'
    if e in ANTI_SIGNAL: return 'anti'
    return 'other'

train['event_group'] = train['event'].apply(event_group)
val['event_group'] = val['event'].apply(event_group)

val_preds = np.zeros(len(val))
val_probas = np.zeros(len(val))

for group in ['high', 'anti', 'other', 'missing']:
    t_mask = train['event_group'] == group
    v_mask = val['event_group'] == group
    if t_mask.sum() < 50 or v_mask.sum() < 10:
        # Fall back to majority class
        val_preds[v_mask.values] = train.loc[t_mask, 'label'].mode().iloc[0]
        val_probas[v_mask.values] = 0.5
        continue

    Xt = hstack([tfidf.transform(train.loc[t_mask, 'title_en'].fillna('').astype(str)),
                 csr_matrix(X_train_num[t_mask.values])])
    Xv = hstack([tfidf.transform(val.loc[v_mask, 'title_en'].fillna('').astype(str)),
                 csr_matrix(X_val_num[v_mask.values])])

    m = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    m.fit(Xt, train.loc[t_mask, 'label'])
    val_preds[v_mask.values] = m.predict(Xv)
    val_probas[v_mask.values] = m.predict_proba(Xv)[:,1]

    sub_mcc = matthews_corrcoef(val.loc[v_mask, 'label'], m.predict(Xv))
    print(f"  Group '{group}': n_train={t_mask.sum()}, n_val={v_mask.sum()}, MCC={sub_mcc:.4f}")

evaluate("event_routing_logreg", val['label'], val_preds.astype(int), val_probas)

# ============================================================
# Model 5: Confidence-weighted (abstain on low confidence)
# ============================================================
print("\n=== MODEL 5: Confidence-weighted (abstain when uncertain) ===")
# Use the best global model (event+exchange LightGBM)
confidence = np.abs(pp_lgbm - 0.5)
for threshold in [0.0, 0.05, 0.10, 0.15, 0.20]:
    mask = confidence >= threshold
    if mask.sum() < 100:
        continue
    y_sub = val['label'].values[mask]
    p_sub = p_lgbm[mask]
    pp_sub = pp_lgbm[mask]
    mcc = matthews_corrcoef(y_sub, p_sub)
    ba = balanced_accuracy_score(y_sub, p_sub)
    coverage = mask.mean() * 100
    print(f"  Threshold={threshold:.2f}: Coverage={coverage:.1f}% ({mask.sum()} docs) "
          f"MCC={mcc:.4f} BalAcc={ba:.4f}")
    results[f"confidence_t{threshold}"] = {
        "threshold": threshold, "coverage": round(coverage,1),
        "n": int(mask.sum()), "mcc": round(mcc,4), "bal_acc": round(ba,4)
    }

# ============================================================
# Model 6: Event×Exchange interaction features
# ============================================================
print("\n=== MODEL 6: Event×Exchange interaction ===")
train['event_exchange'] = train['event'].fillna('MISSING') + '_' + train['exchange'].fillna('MISSING')
val['event_exchange'] = val['event'].fillna('MISSING') + '_' + val['exchange'].fillna('MISSING')

# Keep only interactions seen enough times
ee_counts = train['event_exchange'].value_counts()
top_ee = set(ee_counts[ee_counts >= 20].index)

def encode_ee(series):
    return series.apply(lambda x: x if x in top_ee else 'OTHER_EE')

train_ee = encode_ee(train['event_exchange'])
val_ee = encode_ee(val['event_exchange'])

ee_enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_train_ee = ee_enc.fit_transform(np.array(train_ee).reshape(-1,1))
X_val_ee = ee_enc.transform(np.array(val_ee).reshape(-1,1))

X_train_interact = hstack([X_train_tfidf, csr_matrix(X_train_num), X_train_event, X_train_exch, X_train_ee])
X_val_interact = hstack([X_val_tfidf, csr_matrix(X_val_num), X_val_event, X_val_exch, X_val_ee])

lr_interact = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
lr_interact.fit(X_train_interact, train['label'])
evaluate("event_exchange_interact_logreg", val['label'],
         lr_interact.predict(X_val_interact), lr_interact.predict_proba(X_val_interact)[:,1])

# ============================================================
# Summary
# ============================================================
print("\n" + "="*80)
print("EVENT-AWARE MODEL SUMMARY")
print("="*80)
print(f"{'Model':<45} {'MCC':>7} {'BalAcc':>7} {'AUC':>7}")
print("-"*70)
for name, r in sorted(results.items(), key=lambda x: -x[1].get('mcc', 0)):
    if 'confidence' in name:
        continue
    print(f"{name:<45} {r['mcc']:>7.4f} {r['bal_acc']:>7.4f} {r['auc']:>7.4f}")

# Save
out_path = os.path.join(RESULTS_DIR, 'event_aware_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")
