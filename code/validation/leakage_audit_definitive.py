"""
DEFINITIVE leakage audit: random vs temporal split across architectures.

Produces a clean comparison table for the EMNLP paper.
- Architectures: TF-IDF+LR, TF-IDF+RF, TF-IDF+GradBoost, MiniLM+LR, MiniLM+RF
- Splits: random (size-matched, seeded) vs temporal (canonical)
- Tasks: TITLE-only and TITLE+CONTENT+NUM (full-feature) variants
- Metric: MCC primary, also balanced accuracy

This replaces the bogus hardcoded "0.205" claim from generate_figures.py.
"""

import json
import os
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
OUT = ROOT / "results" / "validation" / "leakage_audit_definitive.json"

TRAIN_END = pd.Timestamp("2025-04-01")
VAL_START = TRAIN_END
VAL_END = pd.Timestamp("2025-06-01")
TEST_START = VAL_END
SEED = 42

EXCLUDE_COLS = {
    "news_id", "yf_ticker", "exchange", "etf_ticker",
    "title_en", "content_en", "event", "publisher",
    "published_date", "industry", "publisher_topic",
    "actual_side", "nextday_side", "created_at",
    # KNOWN target leakage:
    "price_change", "price_change_percentage",
    "index_price_change", "index_price_change_percentage",
    "nextday_price_change_percentage",
    # market_status causes mysterious MCC=1.0 leakage when LabelEncoded -> excluded
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


def build_splits(df, rng_seed=SEED):
    """Return temporal and random splits as (train_idx, val_idx, test_idx) tuples.
    Sizes are matched between random and temporal splits.
    """
    n = len(df)
    m_tr = (df["published_date"] < TRAIN_END).values
    m_va = ((df["published_date"] >= VAL_START) & (df["published_date"] < VAL_END)).values
    m_te = (df["published_date"] >= TEST_START).values
    idx_t = (np.where(m_tr)[0], np.where(m_va)[0], np.where(m_te)[0])

    n_tr, n_va, n_te = m_tr.sum(), m_va.sum(), m_te.sum()
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n)
    idx_r = (perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:n_tr + n_va + n_te])
    return {"temporal": idx_t, "random": idx_r}


def get_numerical_cols(df):
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if df[c].dtype in ["int64", "int32", "float64", "float32", "bool"]:
            cols.append(c)
    return cols


def make_text_features(df, tr_idx, te_idx, mode="title"):
    """Build TF-IDF features matching the original baseline configuration."""
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
    else:
        raise ValueError(mode)


def make_num_features(df, tr_idx, te_idx, num_cols):
    Xtr = df.loc[tr_idx, num_cols].fillna(0).astype(np.float32).values
    Xte = df.loc[te_idx, num_cols].fillna(0).astype(np.float32).values
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def fit_eval(X_tr, y_tr, X_te, y_te, model_name):
    if model_name == "lr":
        clf = LogisticRegression(max_iter=2000, C=0.5, random_state=SEED, n_jobs=-1)
    elif model_name == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                     min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    elif model_name == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         learning_rate=0.05, random_state=SEED)
    else:
        raise ValueError(model_name)
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    return float(matthews_corrcoef(y_te, yp)), float(balanced_accuracy_score(y_te, yp))


def run_audit():
    t0 = time.time()
    print("Loading data ...")
    df = load_data()
    print(f"Total binary articles: {len(df)} | UP rate: {df['y'].mean():.3f}")

    print("Loading cached MiniLM embeddings ...")
    minilm = np.load(EMB_CACHE)
    assert minilm.shape[0] == len(df), f"MiniLM cache mismatch: {minilm.shape[0]} vs {len(df)}"
    print(f"  shape: {minilm.shape}")

    print("Loading cached FinBERT embeddings ...")
    finbert = np.load(FINBERT_CACHE)
    assert finbert.shape[0] == len(df), f"FinBERT cache mismatch: {finbert.shape[0]} vs {len(df)}"
    print(f"  shape: {finbert.shape}")

    splits = build_splits(df)
    num_cols = get_numerical_cols(df)
    print(f"Numerical features: {len(num_cols)}")

    results = []
    y = df["y"].values

    # We evaluate train -> test (the "leakage test")
    for split_name, (tr_idx, va_idx, te_idx) in splits.items():
        y_tr, y_te = y[tr_idx], y[te_idx]
        print(f"\n=== {split_name.upper()} SPLIT === (train={len(tr_idx)}, test={len(te_idx)})")
        print(f"  Train UP: {y_tr.mean():.3f} | Test UP: {y_te.mean():.3f}")

        # ---- TF-IDF title only ----
        Xt_tr, Xt_te = make_text_features(df, tr_idx, te_idx, "title")
        for arch in ["lr", "rf", "gb"]:
            mcc, ba = fit_eval(Xt_tr, y_tr, Xt_te, y_te, arch)
            results.append({"split": split_name, "features": "tfidf_title", "model": arch,
                           "mcc": mcc, "balanced_accuracy": ba, "n_train": int(len(tr_idx)),
                           "n_test": int(len(te_idx))})
            print(f"  TFIDF-title + {arch.upper():3s} : MCC={mcc:+.4f}  BalAcc={ba:.4f}")

        # ---- TF-IDF title+content ----
        Xtc_tr, Xtc_te = make_text_features(df, tr_idx, te_idx, "title+content")
        for arch in ["lr", "rf", "gb"]:
            mcc, ba = fit_eval(Xtc_tr, y_tr, Xtc_te, y_te, arch)
            results.append({"split": split_name, "features": "tfidf_title_content", "model": arch,
                           "mcc": mcc, "balanced_accuracy": ba, "n_train": int(len(tr_idx)),
                           "n_test": int(len(te_idx))})
            print(f"  TFIDF-title+ctx + {arch.upper():3s} : MCC={mcc:+.4f}  BalAcc={ba:.4f}")

        # ---- TF-IDF + Numerical (full feature stack matching original baseline) ----
        Xn_tr, Xn_te = make_num_features(df, tr_idx, te_idx, num_cols)
        Xfull_tr = sphstack([csr_matrix(Xn_tr), Xtc_tr]).toarray()
        Xfull_te = sphstack([csr_matrix(Xn_te), Xtc_te]).toarray()
        for arch in ["lr", "rf", "gb"]:
            mcc, ba = fit_eval(Xfull_tr, y_tr, Xfull_te, y_te, arch)
            results.append({"split": split_name, "features": "tfidf_full+num", "model": arch,
                           "mcc": mcc, "balanced_accuracy": ba, "n_train": int(len(tr_idx)),
                           "n_test": int(len(te_idx))})
            print(f"  TFIDF-full+num + {arch.upper():3s} : MCC={mcc:+.4f}  BalAcc={ba:.4f}")

        # ---- MiniLM embedding (title) ----
        Xe_tr = minilm[tr_idx]
        Xe_te = minilm[te_idx]
        sc = StandardScaler()
        Xe_tr_s = sc.fit_transform(Xe_tr)
        Xe_te_s = sc.transform(Xe_te)
        for arch in ["lr", "rf"]:
            mcc, ba = fit_eval(Xe_tr_s, y_tr, Xe_te_s, y_te, arch)
            results.append({"split": split_name, "features": "minilm_title", "model": arch,
                           "mcc": mcc, "balanced_accuracy": ba, "n_train": int(len(tr_idx)),
                           "n_test": int(len(te_idx))})
            print(f"  MiniLM-title + {arch.upper():3s} : MCC={mcc:+.4f}  BalAcc={ba:.4f}")

        # ---- FinBERT embedding (title CLS) ----
        Xf_tr = finbert[tr_idx]
        Xf_te = finbert[te_idx]
        sc2 = StandardScaler()
        Xf_tr_s = sc2.fit_transform(Xf_tr)
        Xf_te_s = sc2.transform(Xf_te)
        for arch in ["lr", "rf"]:
            mcc, ba = fit_eval(Xf_tr_s, y_tr, Xf_te_s, y_te, arch)
            results.append({"split": split_name, "features": "finbert_title", "model": arch,
                           "mcc": mcc, "balanced_accuracy": ba, "n_train": int(len(tr_idx)),
                           "n_test": int(len(te_idx))})
            print(f"  FinBERT-title + {arch.upper():3s} : MCC={mcc:+.4f}  BalAcc={ba:.4f}")

    # Summary: compute leakage ratio (random_mcc / temporal_mcc) for each (features, model)
    df_r = pd.DataFrame(results)
    summary = []
    for (feats, arch), grp in df_r.groupby(["features", "model"]):
        rd = grp[grp.split == "random"].iloc[0]
        td = grp[grp.split == "temporal"].iloc[0]
        ratio = rd["mcc"] / td["mcc"] if td["mcc"] != 0 else float("inf")
        diff = rd["mcc"] - td["mcc"]
        summary.append({
            "features": feats, "model": arch,
            "random_mcc": rd["mcc"], "temporal_mcc": td["mcc"],
            "random_balacc": rd["balanced_accuracy"], "temporal_balacc": td["balanced_accuracy"],
            "diff_mcc": diff, "leakage_ratio": ratio,
        })
    summary_df = pd.DataFrame(summary)
    print("\n=== LEAKAGE SUMMARY ===")
    print(summary_df.to_string(index=False))

    out = {
        "metadata": {
            "n_total_binary": int(len(df)),
            "n_train": int(len(splits["temporal"][0])),
            "n_val": int(len(splits["temporal"][1])),
            "n_test": int(len(splits["temporal"][2])),
            "train_end": str(TRAIN_END.date()),
            "val_end": str(VAL_END.date()),
            "seed": SEED,
            "elapsed_s": float(time.time() - t0),
        },
        "results": results,
        "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")
    return out


if __name__ == "__main__":
    run_audit()
