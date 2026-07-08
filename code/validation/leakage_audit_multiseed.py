"""
Multi-seed extension of leakage_audit_definitive.py.

For each (features, model) combination:
- TEMPORAL split: deterministic, single number (unchanged).
- RANDOM split: run with seeds 0..N_SEEDS-1, report mean+/-std MCC and BalAcc.

Output: leakage_audit_multiseed.json with per-seed results and aggregated summary.
This produces error bars for Table 2 in the EMNLP paper.
"""

import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, hstack as sphstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package")
DATA = ROOT / "data" / "classifier_training_v2.parquet"
EMB_CACHE = ROOT / "data" / "embeddings_cache" / "minilm_title.npy"
FINBERT_CACHE = ROOT / "data" / "embeddings_cache" / "finbert_title.npy"
OUT = ROOT / "results" / "validation" / "leakage_audit_multiseed.json"

TRAIN_END = pd.Timestamp("2025-04-01")
VAL_START = TRAIN_END
VAL_END = pd.Timestamp("2025-06-01")
TEST_START = VAL_END
MODEL_SEED = 42
N_SEEDS = 10  # data-shuffle seeds for random split

EXCLUDE_COLS = {
    "news_id", "yf_ticker", "exchange", "etf_ticker",
    "title_en", "content_en", "event", "publisher",
    "published_date", "industry", "publisher_topic",
    "actual_side", "nextday_side", "created_at",
    "price_change", "price_change_percentage",
    "index_price_change", "index_price_change_percentage",
    "nextday_price_change_percentage",
    "market_status",
    "y",
}


def load_data():
    df = pd.read_parquet(DATA)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    df["content_en"] = df["content_en"].fillna("").astype(str)
    df = df.sort_values("published_date").reset_index(drop=True)
    return df


def temporal_split(df):
    m_tr = (df["published_date"] < TRAIN_END).values
    m_va = ((df["published_date"] >= VAL_START) & (df["published_date"] < VAL_END)).values
    m_te = (df["published_date"] >= TEST_START).values
    return (np.where(m_tr)[0], np.where(m_va)[0], np.where(m_te)[0])


def random_split(n, n_tr, n_va, n_te, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return (perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:n_tr + n_va + n_te])


def get_numerical_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype in
            ["int64", "int32", "float64", "float32", "bool"]]


def make_text_features(df, tr_idx, te_idx, mode):
    if mode == "title":
        tf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words="english")
        Xtr = tf.fit_transform(df.loc[tr_idx, "title_en"])
        Xte = tf.transform(df.loc[te_idx, "title_en"])
        return Xtr, Xte
    elif mode == "title+content":
        tf_t = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words="english")
        Xt_tr = tf_t.fit_transform(df.loc[tr_idx, "title_en"])
        Xt_te = tf_t.transform(df.loc[te_idx, "title_en"])
        tf_c = TfidfVectorizer(max_features=100, ngram_range=(1, 2), stop_words="english")
        Xc_tr = tf_c.fit_transform(df.loc[tr_idx, "content_en"])
        Xc_te = tf_c.transform(df.loc[te_idx, "content_en"])
        return sphstack([Xt_tr, Xc_tr]).tocsr(), sphstack([Xt_te, Xc_te]).tocsr()


def make_num_features(df, tr_idx, te_idx, num_cols):
    Xtr = df.loc[tr_idx, num_cols].fillna(0).astype(np.float32).values
    Xte = df.loc[te_idx, num_cols].fillna(0).astype(np.float32).values
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def fit_eval(X_tr, y_tr, X_te, y_te, model_name):
    if model_name == "lr":
        clf = LogisticRegression(max_iter=2000, C=0.5, random_state=MODEL_SEED, n_jobs=-1)
    elif model_name == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=2,
                                     n_jobs=-1, random_state=MODEL_SEED)
    elif model_name == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                         random_state=MODEL_SEED)
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    return float(matthews_corrcoef(y_te, yp)), float(balanced_accuracy_score(y_te, yp))


def evaluate_one_split(df, tr_idx, te_idx, num_cols, minilm, finbert):
    """Run all 13 (features, model) configs on a single (tr, te) split.
    Returns list of dicts."""
    y = df["y"].values
    y_tr, y_te = y[tr_idx], y[te_idx]
    rec = []

    # TFIDF title
    Xt_tr, Xt_te = make_text_features(df, tr_idx, te_idx, "title")
    for arch in ["lr", "rf", "gb"]:
        mcc, ba = fit_eval(Xt_tr, y_tr, Xt_te, y_te, arch)
        rec.append({"features": "tfidf_title", "model": arch, "mcc": mcc, "balacc": ba})

    # TFIDF title+content
    Xtc_tr, Xtc_te = make_text_features(df, tr_idx, te_idx, "title+content")
    for arch in ["lr", "rf", "gb"]:
        mcc, ba = fit_eval(Xtc_tr, y_tr, Xtc_te, y_te, arch)
        rec.append({"features": "tfidf_title_content", "model": arch, "mcc": mcc, "balacc": ba})

    # TFIDF + numerical
    Xn_tr, Xn_te = make_num_features(df, tr_idx, te_idx, num_cols)
    Xfull_tr = sphstack([csr_matrix(Xn_tr), Xtc_tr]).toarray()
    Xfull_te = sphstack([csr_matrix(Xn_te), Xtc_te]).toarray()
    for arch in ["lr", "rf", "gb"]:
        mcc, ba = fit_eval(Xfull_tr, y_tr, Xfull_te, y_te, arch)
        rec.append({"features": "tfidf_full+num", "model": arch, "mcc": mcc, "balacc": ba})

    # MiniLM
    Xe_tr = minilm[tr_idx]; Xe_te = minilm[te_idx]
    sc = StandardScaler()
    Xe_tr_s = sc.fit_transform(Xe_tr); Xe_te_s = sc.transform(Xe_te)
    for arch in ["lr", "rf"]:
        mcc, ba = fit_eval(Xe_tr_s, y_tr, Xe_te_s, y_te, arch)
        rec.append({"features": "minilm_title", "model": arch, "mcc": mcc, "balacc": ba})

    # FinBERT
    Xf_tr = finbert[tr_idx]; Xf_te = finbert[te_idx]
    sc2 = StandardScaler()
    Xf_tr_s = sc2.fit_transform(Xf_tr); Xf_te_s = sc2.transform(Xf_te)
    for arch in ["lr", "rf"]:
        mcc, ba = fit_eval(Xf_tr_s, y_tr, Xf_te_s, y_te, arch)
        rec.append({"features": "finbert_title", "model": arch, "mcc": mcc, "balacc": ba})

    return rec


def run():
    t0 = time.time()
    print(f"[multiseed audit] N_SEEDS={N_SEEDS}", flush=True)
    df = load_data()
    n = len(df)
    print(f"  binary articles: {n}", flush=True)

    minilm = np.load(EMB_CACHE)
    finbert = np.load(FINBERT_CACHE)
    assert minilm.shape[0] == n and finbert.shape[0] == n

    num_cols = get_numerical_cols(df)
    print(f"  numerical features: {len(num_cols)}", flush=True)

    # Temporal split: run once
    tr_t, va_t, te_t = temporal_split(df)
    n_tr, n_va, n_te = len(tr_t), len(va_t), len(te_t)
    print(f"  sizes: train={n_tr} val={n_va} test={n_te}", flush=True)

    print("\n[TEMPORAL split (deterministic)]", flush=True)
    t1 = time.time()
    temporal_rec = evaluate_one_split(df, tr_t, te_t, num_cols, minilm, finbert)
    print(f"  done in {time.time()-t1:.1f}s", flush=True)
    for r in temporal_rec:
        print(f"    {r['features']:22s} {r['model']:3s}  MCC={r['mcc']:+.4f}", flush=True)

    # Random split: run N_SEEDS times
    print(f"\n[RANDOM splits, {N_SEEDS} seeds]", flush=True)
    random_rec_per_seed = {}
    for seed in range(N_SEEDS):
        t2 = time.time()
        tr_r, va_r, te_r = random_split(n, n_tr, n_va, n_te, seed)
        recs = evaluate_one_split(df, tr_r, te_r, num_cols, minilm, finbert)
        random_rec_per_seed[seed] = recs
        elapsed = time.time() - t2
        print(f"  seed {seed:2d}: done in {elapsed:.1f}s", flush=True)
        # Save partial results after each seed (resilience)
        partial = {
            "metadata": {"n_total": n, "n_train": n_tr, "n_test": n_te,
                         "train_end": str(TRAIN_END.date()),
                         "model_seed": MODEL_SEED, "n_seeds_done": seed + 1,
                         "elapsed_s": float(time.time() - t0)},
            "temporal": temporal_rec,
            "random_per_seed": {str(s): r for s, r in random_rec_per_seed.items()},
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(partial, f, indent=2)

    # Aggregate
    summary = []
    keys = [(r["features"], r["model"]) for r in temporal_rec]
    for feats, model in keys:
        t_rec = next(r for r in temporal_rec if r["features"] == feats and r["model"] == model)
        r_mccs = []
        r_baccs = []
        for s in range(N_SEEDS):
            recs = random_rec_per_seed[s]
            rr = next(r for r in recs if r["features"] == feats and r["model"] == model)
            r_mccs.append(rr["mcc"])
            r_baccs.append(rr["balacc"])
        r_mccs = np.array(r_mccs)
        r_baccs = np.array(r_baccs)
        ratio = r_mccs.mean() / t_rec["mcc"] if t_rec["mcc"] != 0 else float("inf")
        summary.append({
            "features": feats, "model": model,
            "temporal_mcc": t_rec["mcc"],
            "temporal_balacc": t_rec["balacc"],
            "random_mcc_mean": float(r_mccs.mean()),
            "random_mcc_std": float(r_mccs.std(ddof=1)),
            "random_mcc_min": float(r_mccs.min()),
            "random_mcc_max": float(r_mccs.max()),
            "random_balacc_mean": float(r_baccs.mean()),
            "random_balacc_std": float(r_baccs.std(ddof=1)),
            "leakage_ratio_mean": float(ratio),
            "diff_mcc": float(r_mccs.mean() - t_rec["mcc"]),
        })

    print("\n=== AGGREGATE SUMMARY ===", flush=True)
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))

    out = {
        "metadata": {
            "n_total": n, "n_train": n_tr, "n_val": n_va, "n_test": n_te,
            "train_end": str(TRAIN_END.date()),
            "val_end": str(VAL_END.date()),
            "model_seed": MODEL_SEED, "n_seeds": N_SEEDS,
            "elapsed_s": float(time.time() - t0),
        },
        "temporal": temporal_rec,
        "random_per_seed": {str(s): r for s, r in random_rec_per_seed.items()},
        "summary": summary,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run()
