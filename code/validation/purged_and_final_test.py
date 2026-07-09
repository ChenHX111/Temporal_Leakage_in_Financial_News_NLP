"""
Purged Temporal Validation + Final Locked Test Evaluation.

1. Purged splits: remove articles near split boundary (same-event/firm/date)
2. Block bootstrap for temporal correlation
3. FINAL locked test set evaluation (one-shot)
"""
import sys, io, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, accuracy_score
from scipy.sparse import hstack, csr_matrix

BASE_DIR = r'.'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'classifier_training_v2.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'validation')

df = pd.read_parquet(DATA_PATH)
df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['label'] = (df['actual_side'].str.lower() == 'up').astype(int)
df['date_only'] = df['published_date'].dt.date

# ============================================================
# EXPERIMENT 1: PURGED TEMPORAL SPLITS
# ============================================================
print("=" * 80)
print("EXPERIMENT 1: PURGED TEMPORAL SPLITS")
print("=" * 80)

TRAIN_END = pd.Timestamp('2025-04-01')
VAL_END = pd.Timestamp('2025-06-01')

purge_results = {}
for purge_days in [0, 1, 3, 7]:
    train_cutoff = TRAIN_END - pd.Timedelta(days=purge_days)
    val_start = TRAIN_END + pd.Timedelta(days=purge_days)
    
    tr = df[df['published_date'] < train_cutoff]
    vl = df[(df['published_date'] >= val_start) & (df['published_date'] < VAL_END)]
    
    if len(tr) < 1000 or len(vl) < 100:
        continue
    
    tfidf = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, sublinear_tf=True)
    X_tr = tfidf.fit_transform(tr['title_en'].fillna('').astype(str))
    X_vl = tfidf.transform(vl['title_en'].fillna('').astype(str))
    
    lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr.fit(X_tr, tr['label'])
    preds = lr.predict(X_vl)
    mcc = matthews_corrcoef(vl['label'], preds)
    bacc = balanced_accuracy_score(vl['label'], preds)
    
    # M&A subset
    ma_vl_mask = vl['event'] == 'mergers_acquisitions'
    mcc_ma = matthews_corrcoef(vl.loc[ma_vl_mask, 'label'], preds[ma_vl_mask.values]) if ma_vl_mask.sum() >= 15 else None
    
    mcc_ma_str = f"{mcc_ma:.4f}" if mcc_ma is not None else "N/A"
    print(f"  Purge={purge_days}d: train={len(tr):>6} val={len(vl):>5} MCC={mcc:.4f} BalAcc={bacc:.4f} MCC_MA={mcc_ma_str}")
    purge_results[f"purge_{purge_days}d"] = {
        "train_n": len(tr), "val_n": len(vl), "mcc": round(mcc, 4),
        "bacc": round(bacc, 4), "mcc_ma": round(mcc_ma, 4) if mcc_ma else None
    }

# Deduplicated split (remove articles with same title appearing in both periods)
print("\n  --- Deduplication purge ---")
train_base = df[df['published_date'] < TRAIN_END]
val_base = df[(df['published_date'] >= TRAIN_END) & (df['published_date'] < VAL_END)]
shared_titles = set(train_base['title_en'].dropna()) & set(val_base['title_en'].dropna())
tr_dedup = train_base[~train_base['title_en'].isin(shared_titles)]
vl_dedup = val_base[~val_base['title_en'].isin(shared_titles)]
print(f"  Shared titles removed: {len(shared_titles)}")
print(f"  Dedup train: {len(tr_dedup)} (removed {len(train_base)-len(tr_dedup)})")
print(f"  Dedup val: {len(vl_dedup)} (removed {len(val_base)-len(vl_dedup)})")

tfidf_d = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, sublinear_tf=True)
X_tr_d = tfidf_d.fit_transform(tr_dedup['title_en'].fillna('').astype(str))
X_vl_d = tfidf_d.transform(vl_dedup['title_en'].fillna('').astype(str))
lr_d = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_d.fit(X_tr_d, tr_dedup['label'])
preds_d = lr_d.predict(X_vl_d)
mcc_d = matthews_corrcoef(vl_dedup['label'], preds_d)
print(f"  Dedup MCC={mcc_d:.4f}")
purge_results['dedup'] = {"mcc": round(mcc_d, 4), "shared_titles": len(shared_titles)}

# ============================================================
# EXPERIMENT 2: BLOCK BOOTSTRAP (weekly blocks)
# ============================================================
print(f"\n{'='*80}")
print("EXPERIMENT 2: BLOCK BOOTSTRAP (weekly blocks)")
print(f"{'='*80}")

train_full = df[df['published_date'] < TRAIN_END]
val_full = df[(df['published_date'] >= TRAIN_END) & (df['published_date'] < VAL_END)].reset_index(drop=True)

tfidf_b = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, sublinear_tf=True)
X_tr_b = tfidf_b.fit_transform(train_full['title_en'].fillna('').astype(str))
X_vl_b = tfidf_b.transform(val_full['title_en'].fillna('').astype(str))
lr_b = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_b.fit(X_tr_b, train_full['label'])
preds_b = lr_b.predict(X_vl_b)

# Assign week blocks
val_full['week'] = val_full['published_date'].dt.isocalendar().week.astype(int)
weeks = val_full['week'].unique()
# Pre-build week-to-index mapping
week_to_idx = {w: val_full.index[val_full['week'] == w].tolist() for w in weeks}

N_BOOT = 1000
np.random.seed(42)
block_mccs = []
for _ in range(N_BOOT):
    sampled_weeks = np.random.choice(weeks, size=len(weeks), replace=True)
    idx = []
    for w in sampled_weeks:
        idx.extend(week_to_idx[w])
    y_true = val_full['label'].values[idx]
    y_pred = preds_b[idx]
    try:
        block_mccs.append(matthews_corrcoef(y_true, y_pred))
    except:
        pass

lo, hi = np.percentile(block_mccs, [2.5, 97.5])
print(f"  Full coverage (block bootstrap, {len(weeks)} weeks): MCC={np.mean(block_mccs):.4f} [{lo:.4f}, {hi:.4f}] 95% CI")

# M&A block bootstrap
ma_mask = val_full['event'] == 'mergers_acquisitions'
ma_val_block = val_full[ma_mask].copy()
ma_positions = ma_val_block.index.tolist()
preds_ma_block = preds_b[ma_positions]

if len(ma_val_block) >= 30:
    ma_weeks = ma_val_block['week'].unique()
    ma_week_to_idx = {}
    for w in ma_weeks:
        local_idx = np.where((ma_val_block['week'] == w).values)[0]
        ma_week_to_idx[w] = local_idx.tolist()
    
    ma_block_mccs = []
    for _ in range(N_BOOT):
        sampled_weeks = np.random.choice(ma_weeks, size=len(ma_weeks), replace=True)
        idx = []
        for w in sampled_weeks:
            idx.extend(ma_week_to_idx[w])
        y_true = ma_val_block['label'].values[idx]
        y_pred = preds_ma_block[idx]
        try:
            ma_block_mccs.append(matthews_corrcoef(y_true, y_pred))
        except:
            pass
    
    lo_ma, hi_ma = np.percentile(ma_block_mccs, [2.5, 97.5])
    print(f"  M&A subset (block bootstrap, {len(ma_weeks)} weeks): MCC={np.mean(ma_block_mccs):.4f} [{lo_ma:.4f}, {hi_ma:.4f}] 95% CI")

block_boot_results = {
    "full": {"mcc_mean": round(np.mean(block_mccs), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n_weeks": int(len(weeks))},
    "ma": {"mcc_mean": round(np.mean(ma_block_mccs), 4), "ci_lo": round(lo_ma, 4), "ci_hi": round(hi_ma, 4), "n_weeks": int(len(ma_weeks))} if len(ma_val_block) >= 30 else None
}

# ============================================================
# EXPERIMENT 3: FINAL LOCKED TEST SET EVALUATION
# ============================================================
print(f"\n{'='*80}")
print("EXPERIMENT 3: FINAL LOCKED TEST SET (>= 2025-06-01)")
print("WARNING: This is a one-shot evaluation. Results are final.")
print(f"{'='*80}")

# Train on everything before test
train_final = df[df['published_date'] < VAL_END].reset_index(drop=True)
test = df[df['published_date'] >= VAL_END].reset_index(drop=True)
print(f"\n  Train+Val: {len(train_final)}, Test: {len(test)}")
print(f"  Test date range: {test['published_date'].min()} to {test['published_date'].max()}")
print(f"  Test UP%: {test['label'].mean()*100:.1f}%")

# Model 1: Global TF-IDF + LogReg
tfidf_final = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, sublinear_tf=True)
X_tr_f = tfidf_final.fit_transform(train_final['title_en'].fillna('').astype(str))
X_test = tfidf_final.transform(test['title_en'].fillna('').astype(str))
lr_f = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_f.fit(X_tr_f, train_final['label'])
preds_global = lr_f.predict(X_test)
mcc_global = matthews_corrcoef(test['label'], preds_global)
bacc_global = balanced_accuracy_score(test['label'], preds_global)
print(f"\n  Global text model: MCC={mcc_global:.4f}, BalAcc={bacc_global:.4f}")

# Model 2: M&A specialized
ma_tr_f = train_final[train_final['event'] == 'mergers_acquisitions']
ma_test = test[test['event'] == 'mergers_acquisitions']
print(f"  M&A test: {len(ma_test)} samples")

tfidf_ma_f = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
X_tr_ma_f = tfidf_ma_f.fit_transform(ma_tr_f['title_en'].fillna('').astype(str))
X_test_ma = tfidf_ma_f.transform(ma_test['title_en'].fillna('').astype(str))
lr_ma_f = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_ma_f.fit(X_tr_ma_f, ma_tr_f['label'])
preds_ma = lr_ma_f.predict(X_test_ma)
mcc_ma = matthews_corrcoef(ma_test['label'], preds_ma)
bacc_ma = balanced_accuracy_score(ma_test['label'], preds_ma)
print(f"  M&A specialized text model: MCC={mcc_ma:.4f}, BalAcc={bacc_ma:.4f}")

# Model 3: Event-routing (separate per event group)
# High-signal: use text model. Low-signal: use majority
ma_test_mask = test['event'] == 'mergers_acquisitions'
preds_routed = preds_global.copy()  # default global for everything
preds_routed[ma_test_mask.values] = preds_ma  # replace M&A with specialized

# Model 4: Selective (M&A only, abstain on rest)
print(f"\n  --- Selective prediction (M&A only, {len(ma_test)} of {len(test)} = {len(ma_test)/len(test)*100:.1f}% coverage) ---")
print(f"  M&A test MCC={mcc_ma:.4f}, BalAcc={bacc_ma:.4f}")

# Per-event performance on test set
print(f"\n  --- Per-event test performance (global model) ---")
test_events = test['event'].value_counts().head(10).index
for ev in test_events:
    ev_mask = test['event'] == ev
    if ev_mask.sum() >= 20:
        mcc_ev = matthews_corrcoef(test.loc[ev_mask, 'label'], preds_global[ev_mask.values])
        print(f"    {ev:<40} n={ev_mask.sum():>5} MCC={mcc_ev:.4f}")

# M&A on test with non-text controls
ma_test_majority = int(ma_tr_f['label'].mean() > 0.5)
mcc_ma_maj = matthews_corrcoef(ma_test['label'], np.full(len(ma_test), ma_test_majority))
print(f"\n  M&A majority baseline on test: MCC={mcc_ma_maj:.4f}")

# Sign test for rolling windows (using test months)
test_months = test['published_date'].dt.to_period('M').unique()
test_rolling = []
for tm in sorted(test_months):
    tr_rm = df[df['published_date'].dt.to_period('M') < tm]
    ts_rm = df[(df['published_date'].dt.to_period('M') == tm) & (df['event'] == 'mergers_acquisitions')]
    if len(tr_rm[tr_rm['event']=='mergers_acquisitions']) < 30 or len(ts_rm) < 10:
        continue
    ma_tr_rm = tr_rm[tr_rm['event'] == 'mergers_acquisitions']
    tfidf_rm = TfidfVectorizer(max_features=300, stop_words='english', min_df=2, sublinear_tf=True)
    X_tr_rm = tfidf_rm.fit_transform(ma_tr_rm['title_en'].fillna('').astype(str))
    X_ts_rm = tfidf_rm.transform(ts_rm['title_en'].fillna('').astype(str))
    lr_rm = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_rm.fit(X_tr_rm, ma_tr_rm['label'])
    mcc_rm = matthews_corrcoef(ts_rm['label'], lr_rm.predict(X_ts_rm))
    test_rolling.append({"month": str(tm), "mcc": round(mcc_rm, 4), "n": len(ts_rm)})
    print(f"    Test M&A {str(tm)}: n={len(ts_rm):>4} MCC={mcc_rm:.4f}")

n_positive = sum(1 for r in test_rolling if r['mcc'] > 0)
print(f"\n  M&A test rolling: {n_positive}/{len(test_rolling)} months positive")

# Save all results
final_results = {
    "purged_splits": purge_results,
    "block_bootstrap": block_boot_results,
    "final_test": {
        "global_text": {"mcc": round(mcc_global, 4), "bacc": round(bacc_global, 4), "n": len(test)},
        "ma_specialized": {"mcc": round(mcc_ma, 4), "bacc": round(bacc_ma, 4), "n": len(ma_test)},
        "ma_majority": {"mcc": round(mcc_ma_maj, 4)},
        "ma_rolling_test": test_rolling
    }
}
out_path = os.path.join(RESULTS_DIR, 'purged_and_final_test.json')
with open(out_path, 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"\nSaved to: {out_path}")
