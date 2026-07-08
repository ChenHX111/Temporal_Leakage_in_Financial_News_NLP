"""
M&A Entity-Role Analysis — What role does the focal company play?

The rubber-duck critique suggests the text signal may partly be entity-role signal:
- "sells" → focal company is SELLER → typically bullish (unlocking value)
- "acquisition" → focal company is ACQUIRER → typically bearish (overpayment premium)
- "target" → focal company is TARGET → typically bullish (premium from acquirer)

This analysis:
1. Infers entity roles from title text (rule-based)
2. Checks if role predicts direction independent of text model
3. Tests whether text model learns beyond role information
"""
import sys, io, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from scipy.sparse import hstack, csr_matrix

BASE_DIR = r'C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'classifier_training_v2.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'validation')

df = pd.read_parquet(DATA_PATH)
df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['label'] = (df['actual_side'].str.lower() == 'up').astype(int)

ma = df[df['event'] == 'mergers_acquisitions'].copy()
print(f"Total M&A articles: {len(ma)}")

# Rule-based role inference from title text
def infer_role(title):
    """Classify the likely role of the focal company from the title."""
    if pd.isna(title):
        return 'unknown'
    t = title.lower()
    
    # Seller indicators (focal company is selling/divesting)
    seller_terms = ['sells', 'divests', 'disposes', 'disposal', 'divestiture',
                    'sale of', 'sold', 'divestment', 'exit from', 'exits']
    # Acquirer indicators (focal company is buying/acquiring)
    acquirer_terms = ['acquires', 'acquisition', 'to acquire', 'completes acquisition',
                      'purchases', 'bought', 'buying', 'to buy', 'takeover bid',
                      'completed acquisition', 'completed the acquisition']
    # Target indicators (focal company is being acquired)
    target_terms = ['received offer', 'receives offer', 'approached', 'bid for',
                    'takeover target', 'to be acquired', 'merger agreement']
    # Stake/investment
    stake_terms = ['stake in', 'investment in', 'enters', 'invests in', 
                   'takes stake', 'strategic investment', 'minority stake']
    # Joint/partnership
    joint_terms = ['joint venture', 'partnership', 'collaboration', 'alliance',
                   'merger of equals', 'combines with']
    
    for term in seller_terms:
        if term in t:
            return 'seller'
    for term in acquirer_terms:
        if term in t:
            return 'acquirer'
    for term in target_terms:
        if term in t:
            return 'target'
    for term in stake_terms:
        if term in t:
            return 'stake_investor'
    for term in joint_terms:
        if term in t:
            return 'joint_partner'
    return 'unclear'

ma['role'] = ma['title_en'].apply(infer_role)

# Role distribution
print("\nRole distribution:")
role_counts = ma['role'].value_counts()
for role, count in role_counts.items():
    up_pct = ma[ma['role'] == role]['label'].mean() * 100
    print(f"  {role:<20} n={count:>5} ({count/len(ma)*100:.1f}%) UP%={up_pct:.1f}%")

# Role as predictor
print("\nRole as predictor (UP% by role):")
train_ma = ma[ma['published_date'] < '2025-04-01']
val_ma = ma[(ma['published_date'] >= '2025-04-01') & (ma['published_date'] < '2025-06-01')]
test_ma = ma[ma['published_date'] >= '2025-06-01']

print(f"\n  Train: {len(train_ma)}, Val: {len(val_ma)}, Test: {len(test_ma)}")

# Check if role predicts direction on val and test
for name, dataset in [("Val", val_ma), ("Test", test_ma)]:
    print(f"\n  --- {name} set ---")
    for role in ['seller', 'acquirer', 'stake_investor', 'unclear', 'target', 'joint_partner']:
        subset = dataset[dataset['role'] == role]
        if len(subset) >= 5:
            up_pct = subset['label'].mean() * 100
            print(f"    {role:<20} n={len(subset):>4} UP%={up_pct:.1f}%")

# Role-based prediction model
print("\n--- Role-based prediction models ---")
# Encode roles
role_enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_tr_role = role_enc.fit_transform(np.array(train_ma['role']).reshape(-1,1))
X_vl_role = role_enc.transform(np.array(val_ma['role']).reshape(-1,1))
X_ts_role = role_enc.transform(np.array(test_ma['role']).reshape(-1,1))

# Role-only model
lr_role = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_role.fit(X_tr_role, train_ma['label'])
for name, X, y in [("Val", X_vl_role, val_ma['label']), ("Test", X_ts_role, test_ma['label'])]:
    preds = lr_role.predict(X)
    mcc = matthews_corrcoef(y, preds)
    bacc = balanced_accuracy_score(y, preds)
    print(f"  Role-only LogReg ({name}): MCC={mcc:.4f}, BalAcc={bacc:.4f}")

# Text-only model
tfidf = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
X_tr_text = tfidf.fit_transform(train_ma['title_en'].fillna('').astype(str))
X_vl_text = tfidf.transform(val_ma['title_en'].fillna('').astype(str))
X_ts_text = tfidf.transform(test_ma['title_en'].fillna('').astype(str))

lr_text = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_text.fit(X_tr_text, train_ma['label'])
for name, X, y in [("Val", X_vl_text, val_ma['label']), ("Test", X_ts_text, test_ma['label'])]:
    preds = lr_text.predict(X)
    mcc = matthews_corrcoef(y, preds)
    bacc = balanced_accuracy_score(y, preds)
    print(f"  Text-only LogReg ({name}): MCC={mcc:.4f}, BalAcc={bacc:.4f}")

# Role + text model
X_tr_both = hstack([X_tr_text, X_tr_role])
X_vl_both = hstack([X_vl_text, X_vl_role])
X_ts_both = hstack([X_ts_text, X_ts_role])

lr_both = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_both.fit(X_tr_both, train_ma['label'])
for name, X, y in [("Val", X_vl_both, val_ma['label']), ("Test", X_ts_both, test_ma['label'])]:
    preds = lr_both.predict(X)
    mcc = matthews_corrcoef(y, preds)
    bacc = balanced_accuracy_score(y, preds)
    print(f"  Text+Role LogReg ({name}): MCC={mcc:.4f}, BalAcc={bacc:.4f}")

# Text with role removed (ablation - remove role-indicative words)
print("\n--- Ablation: Text with role terms removed ---")
role_words = {'sells', 'divests', 'disposal', 'sale', 'sold', 'acquires', 'acquisition', 
              'acquire', 'purchases', 'bought', 'buying', 'enters', 'stake', 'invests',
              'investment', 'completes', 'completed', 'joint', 'venture', 'partnership',
              'divestiture', 'divestment', 'takeover', 'merger', 'bid'}

def remove_role_words(title):
    if pd.isna(title):
        return ''
    words = str(title).lower().split()
    return ' '.join(w for w in words if w not in role_words)

train_ma_norole = train_ma['title_en'].apply(remove_role_words)
val_ma_norole = val_ma['title_en'].apply(remove_role_words)
test_ma_norole = test_ma['title_en'].apply(remove_role_words)

tfidf_nr = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
X_tr_nr = tfidf_nr.fit_transform(train_ma_norole)
X_vl_nr = tfidf_nr.transform(val_ma_norole)
X_ts_nr = tfidf_nr.transform(test_ma_norole)

lr_nr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_nr.fit(X_tr_nr, train_ma['label'])
for name, X, y in [("Val", X_vl_nr, val_ma['label']), ("Test", X_ts_nr, test_ma['label'])]:
    preds = lr_nr.predict(X)
    mcc = matthews_corrcoef(y, preds)
    bacc = balanced_accuracy_score(y, preds)
    print(f"  Text-no-role-words ({name}): MCC={mcc:.4f}, BalAcc={bacc:.4f}")

# Save results
results = {
    "role_distribution": {role: {"n": int(count), "pct": round(count/len(ma)*100,1)} 
                          for role, count in role_counts.items()},
    "role_up_pct": {role: round(ma[ma['role']==role]['label'].mean()*100, 1) 
                     for role in role_counts.index}
}
out_path = os.path.join(RESULTS_DIR, 'entity_role_analysis.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")
