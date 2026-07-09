"""
Non-Text Control Baselines — Critical for isolating text signal.

Tests whether the M&A text signal is actually from:
1. Event type alone (already tested)
2. Company/ticker identity
3. Exchange/sector
4. Publisher/source
5. Temporal patterns (day of week, month)
6. Market regime (prior index return)
7. All non-text features combined

If text doesn't improve over the best non-text control, the "text signal" claim is invalid.
"""
import sys, io, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from scipy.sparse import hstack, csr_matrix

BASE_DIR = r'.'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'classifier_training_v2.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'validation')

df = pd.read_parquet(DATA_PATH)
df['published_date'] = pd.to_datetime(df['published_date'])
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['label'] = (df['actual_side'].str.lower() == 'up').astype(int)

train = df[df['published_date'] < '2025-04-01'].copy()
val = df[(df['published_date'] >= '2025-04-01') & (df['published_date'] < '2025-06-01')].copy()

# Extract metadata features
for d in [train, val]:
    d['dow'] = d['published_date'].dt.dayofweek
    d['month'] = d['published_date'].dt.month
    d['hour'] = d['published_date'].dt.hour

EXCLUDE_COLS = {
    'news_id', 'yf_ticker', 'exchange', 'etf_ticker', 'title_en', 'content_en',
    'event', 'publisher', 'published_date', 'industry', 'publisher_topic',
    'actual_side', 'nextday_side', 'created_at',
    'price_change', 'price_change_percentage',
    'index_price_change', 'index_price_change_percentage',
    'nextday_price_change_percentage', 'label', 'dow', 'month', 'hour'
}
num_cols = [c for c in df.columns if c not in EXCLUDE_COLS
            and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

def encode_cat(train_series, val_series, min_count=10):
    """One-hot encode categorical with frequency cutoff."""
    counts = train_series.value_counts()
    valid = set(counts[counts >= min_count].index)
    tr = train_series.apply(lambda x: x if x in valid else '_OTHER_').fillna('_MISSING_')
    vl = val_series.apply(lambda x: x if x in valid else '_OTHER_').fillna('_MISSING_')
    enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_tr = enc.fit_transform(np.array(tr).reshape(-1, 1))
    X_vl = enc.transform(np.array(vl).reshape(-1, 1))
    return X_tr, X_vl

def eval_model(X_tr, y_tr, X_vl, y_vl, C=0.1):
    """Train LogReg and return MCC."""
    lr = LogisticRegression(max_iter=2000, random_state=42, C=C)
    lr.fit(X_tr, y_tr)
    preds = lr.predict(X_vl)
    return matthews_corrcoef(y_vl, preds), balanced_accuracy_score(y_vl, preds)

y_tr = train['label'].values
y_vl = val['label'].values

print("=" * 80)
print("NON-TEXT CONTROL BASELINES — FULL DATASET")
print("=" * 80)
print(f"Train: {len(train)}, Val: {len(val)}")
print(f"\n{'Baseline':<45} {'MCC':>8} {'BalAcc':>8}")
print("-" * 65)

results_full = {}

# 1. Majority
majority = int(y_tr.mean() > 0.5)
mcc_maj = matthews_corrcoef(y_vl, np.full(len(y_vl), majority))
print(f"{'Majority class (UP)':45} {mcc_maj:>8.4f} {0.5:>8.4f}")
results_full['majority'] = {'mcc': round(mcc_maj, 4)}

# 2. Event type only
X_tr_ev, X_vl_ev = encode_cat(train['event'], val['event'])
mcc, bacc = eval_model(X_tr_ev, y_tr, X_vl_ev, y_vl)
print(f"{'Event type (one-hot)':45} {mcc:>8.4f} {bacc:>8.4f}")
results_full['event_type'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# 3. Exchange only
X_tr_ex, X_vl_ex = encode_cat(train['exchange'], val['exchange'])
mcc, bacc = eval_model(X_tr_ex, y_tr, X_vl_ex, y_vl)
print(f"{'Exchange (one-hot)':45} {mcc:>8.4f} {bacc:>8.4f}")
results_full['exchange'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# 4. Publisher only
X_tr_pub, X_vl_pub = encode_cat(train['publisher'].fillna('unknown'), val['publisher'].fillna('unknown'))
mcc, bacc = eval_model(X_tr_pub, y_tr, X_vl_pub, y_vl)
print(f"{'Publisher (one-hot)':45} {mcc:>8.4f} {bacc:>8.4f}")
results_full['publisher'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# 5. Ticker (company identity)
X_tr_tk, X_vl_tk = encode_cat(train['yf_ticker'].fillna('unknown'), val['yf_ticker'].fillna('unknown'), min_count=5)
mcc, bacc = eval_model(X_tr_tk, y_tr, X_vl_tk, y_vl)
print(f"{'Ticker/company (one-hot)':45} {mcc:>8.4f} {bacc:>8.4f}")
results_full['ticker'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# 6. Industry/sector
X_tr_ind, X_vl_ind = encode_cat(train['industry'].fillna('unknown'), val['industry'].fillna('unknown'))
mcc, bacc = eval_model(X_tr_ind, y_tr, X_vl_ind, y_vl)
print(f"{'Industry/sector (one-hot)':45} {mcc:>8.4f} {bacc:>8.4f}")
results_full['industry'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# 7. Temporal (day of week + hour + month)
X_tr_time = np.column_stack([train['dow'].values, train['hour'].values, train['month'].values])
X_vl_time = np.column_stack([val['dow'].values, val['hour'].values, val['month'].values])
mcc, bacc = eval_model(csr_matrix(X_tr_time), y_tr, csr_matrix(X_vl_time), y_vl)
print(f"{'Temporal (dow + hour + month)':45} {mcc:>8.4f} {bacc:>8.4f}")
results_full['temporal'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# 8. Numerical features only (non-leaking)
X_tr_num = train[num_cols].fillna(0).values
X_vl_num = val[num_cols].fillna(0).values
mcc, bacc = eval_model(csr_matrix(X_tr_num), y_tr, csr_matrix(X_vl_num), y_vl)
print(f"{'Numerical features only':45} {mcc:>8.4f} {bacc:>8.4f}")
results_full['numerical'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# 9. All non-text combined
X_tr_all_notext = hstack([X_tr_ev, X_tr_ex, X_tr_pub, X_tr_ind, csr_matrix(X_tr_time), csr_matrix(X_tr_num)])
X_vl_all_notext = hstack([X_vl_ev, X_vl_ex, X_vl_pub, X_vl_ind, csr_matrix(X_vl_time), csr_matrix(X_vl_num)])
mcc, bacc = eval_model(X_tr_all_notext, y_tr, X_vl_all_notext, y_vl)
print(f"{'All non-text combined':45} {mcc:>8.4f} {bacc:>8.4f}")
results_full['all_nontext'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# 10. Text (TF-IDF title) only
tfidf = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, sublinear_tf=True)
X_tr_text = tfidf.fit_transform(train['title_en'].fillna('').astype(str))
X_vl_text = tfidf.transform(val['title_en'].fillna('').astype(str))
mcc, bacc = eval_model(X_tr_text, y_tr, X_vl_text, y_vl)
print(f"{'Text (TF-IDF title) only':45} {mcc:>8.4f} {bacc:>8.4f}")
results_full['text_only'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# 11. Text + all non-text
X_tr_all = hstack([X_tr_text, X_tr_all_notext])
X_vl_all = hstack([X_vl_text, X_vl_all_notext])
mcc, bacc = eval_model(X_tr_all, y_tr, X_vl_all, y_vl)
print(f"{'Text + all non-text':45} {mcc:>8.4f} {bacc:>8.4f}")
results_full['text_plus_nontext'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# ============================================================
# M&A SUBSET — Does text help beyond non-text controls?
# ============================================================
print(f"\n{'='*80}")
print("NON-TEXT CONTROL BASELINES — M&A SUBSET ONLY")
print(f"{'='*80}")

ma_tr = train[train['event'] == 'mergers_acquisitions'].copy()
ma_vl = val[val['event'] == 'mergers_acquisitions'].copy()
print(f"M&A train: {len(ma_tr)}, M&A val: {len(ma_vl)}")
y_ma_tr = ma_tr['label'].values
y_ma_vl = ma_vl['label'].values

print(f"\n{'Baseline':<45} {'MCC':>8} {'BalAcc':>8}")
print("-" * 65)

results_ma = {}

# Majority
majority_ma = int(y_ma_tr.mean() > 0.5)
mcc_maj = matthews_corrcoef(y_ma_vl, np.full(len(y_ma_vl), majority_ma))
print(f"{'Majority class':45} {mcc_maj:>8.4f} {0.5:>8.4f}")
results_ma['majority'] = {'mcc': round(mcc_maj, 4)}

# Exchange
X_tr_ex_ma, X_vl_ex_ma = encode_cat(ma_tr['exchange'], ma_vl['exchange'], min_count=3)
mcc, bacc = eval_model(X_tr_ex_ma, y_ma_tr, X_vl_ex_ma, y_ma_vl)
print(f"{'Exchange':45} {mcc:>8.4f} {bacc:>8.4f}")
results_ma['exchange'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# Publisher
X_tr_pub_ma, X_vl_pub_ma = encode_cat(ma_tr['publisher'].fillna('unk'), ma_vl['publisher'].fillna('unk'), min_count=3)
mcc, bacc = eval_model(X_tr_pub_ma, y_ma_tr, X_vl_pub_ma, y_ma_vl)
print(f"{'Publisher':45} {mcc:>8.4f} {bacc:>8.4f}")
results_ma['publisher'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# Ticker
X_tr_tk_ma, X_vl_tk_ma = encode_cat(ma_tr['yf_ticker'].fillna('unk'), ma_vl['yf_ticker'].fillna('unk'), min_count=3)
mcc, bacc = eval_model(X_tr_tk_ma, y_ma_tr, X_vl_tk_ma, y_ma_vl)
print(f"{'Ticker/company':45} {mcc:>8.4f} {bacc:>8.4f}")
results_ma['ticker'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# Temporal
X_tr_time_ma = np.column_stack([ma_tr['dow'].values, ma_tr['hour'].values, ma_tr['month'].values])
X_vl_time_ma = np.column_stack([ma_vl['dow'].values, ma_vl['hour'].values, ma_vl['month'].values])
mcc, bacc = eval_model(csr_matrix(X_tr_time_ma), y_ma_tr, csr_matrix(X_vl_time_ma), y_ma_vl)
print(f"{'Temporal (dow + hour + month)':45} {mcc:>8.4f} {bacc:>8.4f}")
results_ma['temporal'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# Numerical
X_tr_num_ma = ma_tr[num_cols].fillna(0).values
X_vl_num_ma = ma_vl[num_cols].fillna(0).values
mcc, bacc = eval_model(csr_matrix(X_tr_num_ma), y_ma_tr, csr_matrix(X_vl_num_ma), y_ma_vl)
print(f"{'Numerical features':45} {mcc:>8.4f} {bacc:>8.4f}")
results_ma['numerical'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# All non-text for M&A
X_tr_ma_notext = hstack([X_tr_ex_ma, X_tr_pub_ma, X_tr_tk_ma, csr_matrix(X_tr_time_ma), csr_matrix(X_tr_num_ma)])
X_vl_ma_notext = hstack([X_vl_ex_ma, X_vl_pub_ma, X_vl_tk_ma, csr_matrix(X_vl_time_ma), csr_matrix(X_vl_num_ma)])
mcc, bacc = eval_model(X_tr_ma_notext, y_ma_tr, X_vl_ma_notext, y_ma_vl)
print(f"{'All non-text combined':45} {mcc:>8.4f} {bacc:>8.4f}")
results_ma['all_nontext'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# Text only (TF-IDF title)
tfidf_ma = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
X_tr_text_ma = tfidf_ma.fit_transform(ma_tr['title_en'].fillna('').astype(str))
X_vl_text_ma = tfidf_ma.transform(ma_vl['title_en'].fillna('').astype(str))
mcc, bacc = eval_model(X_tr_text_ma, y_ma_tr, X_vl_text_ma, y_ma_vl)
print(f"{'Text (TF-IDF title) only':45} {mcc:>8.4f} {bacc:>8.4f}")
results_ma['text_only'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# Text + all non-text
X_tr_ma_all = hstack([X_tr_text_ma, X_tr_ma_notext])
X_vl_ma_all = hstack([X_vl_text_ma, X_vl_ma_notext])
mcc, bacc = eval_model(X_tr_ma_all, y_ma_tr, X_vl_ma_all, y_ma_vl)
print(f"{'Text + all non-text':45} {mcc:>8.4f} {bacc:>8.4f}")
results_ma['text_plus_nontext'] = {'mcc': round(mcc, 4), 'bacc': round(bacc, 4)}

# Text + non-text vs non-text incremental gain
print(f"\n--- INCREMENTAL TEXT GAIN ---")
best_notext_full = results_full['all_nontext']['mcc']
text_plus_full = results_full['text_plus_nontext']['mcc']
best_notext_ma = results_ma['all_nontext']['mcc']
text_plus_ma = results_ma['text_plus_nontext']['mcc']
print(f"  Full dataset: non-text MCC={best_notext_full:.4f} → +text MCC={text_plus_full:.4f} (gain={text_plus_full-best_notext_full:+.4f})")
print(f"  M&A subset:   non-text MCC={best_notext_ma:.4f} → +text MCC={text_plus_ma:.4f} (gain={text_plus_ma-best_notext_ma:+.4f})")

all_results = {"full_dataset": results_full, "ma_subset": results_ma}
out_path = os.path.join(RESULTS_DIR, 'nontext_controls.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to: {out_path}")
