"""
Rolling Temporal Validation & Event-Signal Stability Check.

Critical experiments to validate findings and address overfitting risk:
1. Rolling temporal validation (train up to month M, validate month M+1)
2. Per-event-type stability across rolling windows
3. Event-only vs text+event to determine signal source
4. Bootstrap confidence intervals
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

BASE_DIR = r'.'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'classifier_training_v2.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'validation')

df = pd.read_parquet(DATA_PATH)
df['published_date'] = pd.to_datetime(df['published_date'])
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['label'] = (df['actual_side'].str.lower() == 'up').astype(int)
df['year_month'] = df['published_date'].dt.to_period('M')

EXCLUDE_COLS = {
    'news_id', 'yf_ticker', 'exchange', 'etf_ticker', 'title_en', 'content_en',
    'event', 'publisher', 'published_date', 'industry', 'publisher_topic',
    'actual_side', 'nextday_side', 'created_at',
    'price_change', 'price_change_percentage',
    'index_price_change', 'index_price_change_percentage',
    'nextday_price_change_percentage', 'label', 'year_month', 'event_group'
}
num_cols = [c for c in df.columns if c not in EXCLUDE_COLS
            and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

HIGH_SIGNAL = {'exchange_announcement', 'share_capital_increase', 'interim_information',
               'shares_issue', 'corporate_action', 'clinical_study'}

# ============================================================
# Experiment 1: Rolling Temporal Validation
# ============================================================
print("=" * 80)
print("EXPERIMENT 1: ROLLING TEMPORAL VALIDATION")
print("=" * 80)

month_counts = df['year_month'].value_counts().sort_index()
valid_months = month_counts[month_counts >= 100].index.tolist()
print(f"\nMonths with >= 100 samples: {len(valid_months)}")
for m in valid_months:
    print(f"  {m}: {month_counts[m]} samples")

rolling_results = []
header = f"{'Window':<15} {'Train':>6} {'Val':>5} {'UP%':>5} {'MCC_full':>9} {'MCC_high':>9} {'MCC_evonly':>10} {'n_high':>7}"
print(f"\n{header}")
print("-" * len(header))

for val_month in valid_months:
    if val_month < pd.Period('2025-01', 'M'):
        continue

    train_mask = df['year_month'] < val_month
    val_mask = df['year_month'] == val_month
    tr = df[train_mask]
    vl = df[val_mask]

    if len(tr) < 500 or len(vl) < 50:
        continue

    tfidf = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, sublinear_tf=True)
    X_tr_tfidf = tfidf.fit_transform(tr['title_en'].fillna('').astype(str))
    X_vl_tfidf = tfidf.transform(vl['title_en'].fillna('').astype(str))
    X_tr_num = tr[num_cols].fillna(0).values
    X_vl_num = vl[num_cols].fillna(0).values

    event_counts_tr = tr['event'].value_counts()
    top_events = set(event_counts_tr.head(30).index)
    tr_ev = tr['event'].apply(lambda x: x if x in top_events else 'OTHER').fillna('MISSING')
    vl_ev = vl['event'].apply(lambda x: x if x in top_events else 'OTHER').fillna('MISSING')
    ev_enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_tr_ev = ev_enc.fit_transform(np.array(tr_ev).reshape(-1,1))
    X_vl_ev = ev_enc.transform(np.array(vl_ev).reshape(-1,1))

    # Full model
    X_tr_full = hstack([X_tr_tfidf, csr_matrix(X_tr_num), X_tr_ev])
    X_vl_full = hstack([X_vl_tfidf, csr_matrix(X_vl_num), X_vl_ev])
    lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr.fit(X_tr_full, tr['label'])
    preds_full = lr.predict(X_vl_full)
    mcc_global = matthews_corrcoef(vl['label'], preds_full)

    # High-signal events
    high_mask = vl['event'].isin(HIGH_SIGNAL)
    n_high = int(high_mask.sum())
    if n_high >= 15:
        mcc_high = matthews_corrcoef(vl.loc[high_mask, 'label'], preds_full[high_mask.values])
    else:
        mcc_high = float('nan')

    # Event-only model
    lr_ev = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_ev.fit(X_tr_ev, tr['label'])
    mcc_evonly = matthews_corrcoef(vl['label'], lr_ev.predict(X_vl_ev))

    up_pct = vl['label'].mean() * 100
    mcc_high_str = f"{mcc_high:>9.4f}" if not np.isnan(mcc_high) else "      N/A"
    print(f"  <{str(val_month):<12} {len(tr):>6} {len(vl):>5} {up_pct:>5.1f} {mcc_global:>9.4f} {mcc_high_str} {mcc_evonly:>10.4f} {n_high:>7}")

    rolling_results.append({
        "val_month": str(val_month), "train_n": len(tr), "val_n": len(vl),
        "up_pct": round(up_pct, 1),
        "mcc_full": round(mcc_global, 4),
        "mcc_high_signal": round(mcc_high, 4) if not np.isnan(mcc_high) else None,
        "mcc_event_only": round(mcc_evonly, 4),
        "n_high": n_high
    })

# ============================================================
# Experiment 2: Per-Event Stability
# ============================================================
print(f"\n{'='*80}")
print("EXPERIMENT 2: PER-EVENT-TYPE STABILITY ACROSS WINDOWS")
print(f"{'='*80}")

target_events = list(HIGH_SIGNAL) + ['management_changes', 'press_releases',
                                      'financial_results', 'earnings_releases_and_operating_results',
                                      'annual_general_meeting', 'mergers_acquisitions']

event_stability = {}
for event in sorted(target_events):
    event_mccs = []
    for val_month in valid_months:
        if val_month < pd.Period('2025-01', 'M'):
            continue
        tr = df[(df['year_month'] < val_month) & (df['event'] == event)]
        vl = df[(df['year_month'] == val_month) & (df['event'] == event)]
        if len(tr) < 20 or len(vl) < 10:
            event_mccs.append({"month": str(val_month), "mcc": None, "n": len(vl)})
            continue
        try:
            tfidf_e = TfidfVectorizer(max_features=200, stop_words='english', min_df=2)
            Xt = tfidf_e.fit_transform(tr['title_en'].fillna('').astype(str))
            Xv = tfidf_e.transform(vl['title_en'].fillna('').astype(str))
            lr_e = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
            lr_e.fit(Xt, tr['label'])
            mcc = matthews_corrcoef(vl['label'], lr_e.predict(Xv))
        except:
            mcc = None
        event_mccs.append({"month": str(val_month), "mcc": round(mcc, 4) if mcc is not None else None, "n": len(vl)})

    event_stability[event] = event_mccs
    valid_mccs = [r['mcc'] for r in event_mccs if r['mcc'] is not None]
    if valid_mccs:
        mean_mcc = np.mean(valid_mccs)
        std_mcc = np.std(valid_mccs)
        per_month = ", ".join([f"{r['month'][-2:]}:{r['mcc']:.3f}" for r in event_mccs if r['mcc'] is not None])
        print(f"  {event:<45} mean={mean_mcc:>7.4f} std={std_mcc:>6.4f} [{per_month}]")
    else:
        print(f"  {event:<45} insufficient data")

# ============================================================
# Experiment 3: Signal Source
# ============================================================
print(f"\n{'='*80}")
print("EXPERIMENT 3: SIGNAL SOURCE — EVENT-ONLY vs TEXT+EVENT")
print(f"{'='*80}")

train = df[df['published_date'] < '2025-04-01'].copy()
val = df[(df['published_date'] >= '2025-04-01') & (df['published_date'] < '2025-06-01')].copy()

tfidf_main = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, sublinear_tf=True)
tfidf_main.fit(train['title_en'].fillna('').astype(str))
X_tr_num_main = train[num_cols].fillna(0).values
X_vl_num_main = val[num_cols].fillna(0).values

print(f"\n  {'Event':<38} {'Majority':>8} {'Text':>8} {'Text+Meta':>10} {'TextGain':>9} {'N':>5}")
print("  " + "-" * 85)

signal_source = {}
for event in sorted(target_events):
    t = train[train['event'] == event]
    v = val[val['event'] == event]
    if len(t) < 30 or len(v) < 15:
        continue

    majority = t['label'].mode().iloc[0]
    mcc_maj = matthews_corrcoef(v['label'], np.full(len(v), majority))

    try:
        tfidf_e = TfidfVectorizer(max_features=200, stop_words='english', min_df=2)
        Xt = tfidf_e.fit_transform(t['title_en'].fillna('').astype(str))
        Xv = tfidf_e.transform(v['title_en'].fillna('').astype(str))
        lr_text = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        lr_text.fit(Xt, t['label'])
        mcc_text = matthews_corrcoef(v['label'], lr_text.predict(Xv))
    except:
        mcc_text = float('nan')

    try:
        t_mask = train['event'] == event
        v_mask = val['event'] == event
        Xt_f = hstack([Xt, csr_matrix(X_tr_num_main[t_mask.values])])
        Xv_f = hstack([Xv, csr_matrix(X_vl_num_main[v_mask.values])])
        lr_f = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        lr_f.fit(Xt_f, t['label'])
        mcc_full = matthews_corrcoef(v['label'], lr_f.predict(Xv_f))
    except:
        mcc_full = float('nan')

    text_gain = mcc_text - mcc_maj if not np.isnan(mcc_text) else float('nan')
    tg_str = f"{text_gain:>9.4f}" if not np.isnan(text_gain) else "      N/A"
    mt_str = f"{mcc_text:>8.4f}" if not np.isnan(mcc_text) else "     N/A"
    mf_str = f"{mcc_full:>10.4f}" if not np.isnan(mcc_full) else "       N/A"

    print(f"  {event:<38} {mcc_maj:>8.4f} {mt_str} {mf_str} {tg_str} {len(v):>5}")

    signal_source[event] = {
        "majority_mcc": round(mcc_maj, 4),
        "text_mcc": round(mcc_text, 4) if not np.isnan(mcc_text) else None,
        "text_meta_mcc": round(mcc_full, 4) if not np.isnan(mcc_full) else None,
        "text_gain": round(text_gain, 4) if not np.isnan(text_gain) else None,
        "val_n": len(v)
    }

# ============================================================
# Experiment 4: Bootstrap CIs
# ============================================================
print(f"\n{'='*80}")
print("EXPERIMENT 4: BOOTSTRAP CONFIDENCE INTERVALS (1000 samples)")
print(f"{'='*80}")

N_BOOTSTRAP = 1000
np.random.seed(42)

X_tr_all = hstack([tfidf_main.transform(train['title_en'].fillna('').astype(str)), csr_matrix(X_tr_num_main)])
X_vl_all = hstack([tfidf_main.transform(val['title_en'].fillna('').astype(str)), csr_matrix(X_vl_num_main)])
lr_all = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_all.fit(X_tr_all, train['label'])
preds_all = lr_all.predict(X_vl_all)

bootstrap_results = {}
for name, mask in [
    ("full_coverage", np.ones(len(val), dtype=bool)),
    ("high_signal_events", val['event'].isin(HIGH_SIGNAL).values),
]:
    if mask.sum() < 30:
        continue
    y_true = val['label'].values[mask]
    y_pred = preds_all[mask]

    boot_mccs = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        try:
            boot_mccs.append(matthews_corrcoef(y_true[idx], y_pred[idx]))
        except:
            pass

    lo, hi = np.percentile(boot_mccs, [2.5, 97.5])
    print(f"  {name} (n={mask.sum()}): MCC={np.mean(boot_mccs):.4f} [{lo:.4f}, {hi:.4f}] 95% CI")
    bootstrap_results[name] = {
        "n": int(mask.sum()), "mcc_mean": round(np.mean(boot_mccs),4),
        "mcc_lo": round(lo,4), "mcc_hi": round(hi,4)
    }

# Save
all_results = {
    "rolling_validation": rolling_results,
    "event_stability": event_stability,
    "signal_source": signal_source,
    "bootstrap_ci": bootstrap_results
}
out_path = os.path.join(RESULTS_DIR, 'rolling_validation_results.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to: {out_path}")
