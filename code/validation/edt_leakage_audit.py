"""
EDT full-corpus leakage audit.

Replicates our Table 2 (random vs temporal split inflation) on the EDT
dataset (Zhou et al. ACL 2021, ~106K US news articles 2020-2021).

Why: addresses the "single-regime" objection. If EDT shows the same
1.2x-6x inflation pattern, the leakage finding is a CROSS-CORPUS,
CROSS-REGIME phenomenon, not specific to our 2025 European data.

Output: results/validation/edt_leakage_audit.json
"""
import os
import sys
import io
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

BASE = r"."
EDT = os.path.join(BASE, "data", "external", "edt_evaluate_slim.parquet")
OUT = os.path.join(BASE, "results", "validation", "edt_leakage_audit.json")

SEEDS = [42, 0, 1, 2, 3]


def safe_mcc(y, p):
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2: return 0.0
    return float(matthews_corrcoef(y, p))


def fit_eval(X_tr, y_tr, X_te, y_te, model_kind):
    if model_kind == "lr":
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, n_jobs=-1)
    elif model_kind == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                     n_jobs=-1, min_samples_leaf=10)
    elif model_kind == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    else:
        raise ValueError(model_kind)
    clf.fit(X_tr, y_tr)
    p = clf.predict(X_te)
    return safe_mcc(y_te, p), float(balanced_accuracy_score(y_te, p))


def build_features(titles_tr, titles_te, max_features=2000):
    tf = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=2)
    return tf.fit_transform(titles_tr), tf.transform(titles_te)


def main():
    t0 = time.time()
    print("Loading EDT slim parquet ...", flush=True)
    df = pd.read_parquet(EDT)
    df["pub_time"] = pd.to_datetime(df["pub_time"]).dt.tz_localize(None)
    df["title"] = df["title"].fillna("").astype(str)
    df = df.sort_values("pub_time").reset_index(drop=True)
    print(f"  EDT total: {len(df)} rows, {df['pub_time'].min()} -> {df['pub_time'].max()}",
          flush=True)
    print(f"  y distribution: {df['y'].value_counts().to_dict()}", flush=True)

    # Temporal split: 70% train, 15% val, 15% test (chronological)
    n = len(df)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    tr_t = df.iloc[:n_tr]
    va_t = df.iloc[n_tr:n_tr + n_va]
    te_t = df.iloc[n_tr + n_va:]
    print(f"  Temporal: train={len(tr_t)}  val={len(va_t)}  test={len(te_t)}",
          flush=True)
    print(f"  Train period: {tr_t['pub_time'].min().date()} .. {tr_t['pub_time'].max().date()}")
    print(f"  Test period:  {te_t['pub_time'].min().date()} .. {te_t['pub_time'].max().date()}")

    rows_temporal = []
    print("\n[TEMPORAL split (deterministic)]", flush=True)
    Xtr, Xte = build_features(tr_t["title"].tolist(), te_t["title"].tolist())
    for kind in ["lr", "rf", "gb"]:
        mcc, bacc = fit_eval(Xtr, tr_t["y"].values, Xte, te_t["y"].values, kind)
        print(f"  tfidf_title  {kind:>2}   MCC={mcc:+.4f}  BalAcc={bacc:.4f}", flush=True)
        rows_temporal.append({"split": "temporal", "model": kind,
                              "mcc": mcc, "balacc": bacc})

    print(f"\n[RANDOM splits, {len(SEEDS)} seeds]", flush=True)
    rows_random = []
    for seed in SEEDS:
        # 70/30 random split (no val needed for audit)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n_train_rand = int(0.85 * len(df))  # match temporal proportion
        tr_idx = idx[:n_train_rand]; te_idx = idx[n_train_rand:]
        tr_r = df.iloc[tr_idx]; te_r = df.iloc[te_idx]
        Xtr, Xte = build_features(tr_r["title"].tolist(), te_r["title"].tolist())
        seed_rows = []
        for kind in ["lr", "rf", "gb"]:
            mcc, bacc = fit_eval(Xtr, tr_r["y"].values, Xte, te_r["y"].values, kind)
            seed_rows.append({"seed": int(seed), "model": kind,
                              "mcc": mcc, "balacc": bacc})
        rows_random.extend(seed_rows)
        agg = {kind: np.mean([r["mcc"] for r in seed_rows if r["model"] == kind])
               for kind in ["lr", "rf", "gb"]}
        print(f"  seed {seed}: lr={agg['lr']:+.4f}  rf={agg['rf']:+.4f}  gb={agg['gb']:+.4f}",
              flush=True)

    # Aggregate
    print("\n[AGGREGATED RESULTS]", flush=True)
    print(f"{'Model':<8} {'Random (mean+/-std)':<22} {'Temporal':<12} {'Inflation':<10}")
    summary = []
    for kind in ["lr", "rf", "gb"]:
        mccs = np.array([r["mcc"] for r in rows_random if r["model"] == kind])
        rand_mean = float(mccs.mean()); rand_std = float(mccs.std())
        temp_mcc = next(r["mcc"] for r in rows_temporal if r["model"] == kind)
        infl = rand_mean / temp_mcc if abs(temp_mcc) > 1e-6 else float("nan")
        print(f"{kind:<8} {rand_mean:+.4f}+/-{rand_std:.4f}      {temp_mcc:+.4f}     "
              f"{infl:+.2f}x", flush=True)
        summary.append({"model": kind, "random_mean_mcc": rand_mean,
                        "random_std_mcc": rand_std, "temporal_mcc": temp_mcc,
                        "inflation_ratio": float(infl) if not np.isnan(infl) else None})

    out = {
        "meta": {"timestamp": pd.Timestamp.now().isoformat(),
                 "n_total": int(len(df)), "n_train_temporal": int(len(tr_t)),
                 "n_test_temporal": int(len(te_t)), "n_seeds_random": len(SEEDS),
                 "elapsed_s": float(time.time() - t0),
                 "dataset": "EDT (Zhou et al ACL 2021)",
                 "date_min": str(df["pub_time"].min().date()),
                 "date_max": str(df["pub_time"].max().date())},
        "temporal": rows_temporal,
        "random_per_seed": rows_random,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)
    print(f"Elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
