"""
M&A specialist hyperparameter sweep on validation set.

DRIVEN BY: Discovery that HP_EDT (max_features=1000, C=1.0, no sublinear_tf)
gives MCC=0.129 on our test M&A vs paper's HP_OUR which gave 0.071.

We need to:
  1. Define a hyperparameter grid.
  2. Train on (our M&A train), select best HP on (our M&A val), then evaluate
     ONCE on (our M&A test).
  3. Run permutation test on the chosen HP.
  4. Re-run acquirer-target decomposition with chosen HP.

If validation selects HP_EDT-like settings, the paper headline updates.
If validation selects HP_OUR-like, paper stays.

Splits used:
  Train: ours M&A pre-2025-04-01    (n=731)
  Val:   ours M&A 2025-04 to 2025-05 (n=369)
  Test:  ours M&A from 2025-06       (n=786, locked)

Output: results/validation/ma_hp_sweep.json
"""
import os
import sys
import io
import json
import re
import time
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from itertools import product

BASE = r"."
OUR = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(BASE, "results", "validation", "ma_hp_sweep.json")


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def fit_eval(tr_titles, y_tr, te_titles, y_te, max_features, C, sublinear_tf, min_df):
    tf = TfidfVectorizer(max_features=max_features, stop_words="english",
                         min_df=min_df, sublinear_tf=sublinear_tf)
    Xtr = tf.fit_transform(tr_titles)
    Xte = tf.transform(te_titles)
    clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
    clf.fit(Xtr, y_tr)
    yp = clf.predict(Xte)
    return safe_mcc(y_te, yp), float(balanced_accuracy_score(y_te, yp)), tf, clf, yp


def main():
    t0 = time.time()
    rng = np.random.default_rng(42)
    print("Loading data ...", flush=True)
    df = pd.read_parquet(OUR)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["title_en"] = df["title_en"].fillna("").astype(str)

    ma = df[df["event"] == "mergers_acquisitions"].copy()
    TRAIN_END = pd.Timestamp("2025-04-01")
    VAL_END = pd.Timestamp("2025-06-01")
    ma_tr = ma[ma["published_date"] < TRAIN_END].copy()
    ma_va = ma[(ma["published_date"] >= TRAIN_END) & (ma["published_date"] < VAL_END)].copy()
    ma_te = ma[ma["published_date"] >= VAL_END].copy()
    print(f"  M&A train={len(ma_tr)}  val={len(ma_va)}  test={len(ma_te)}", flush=True)

    # Train on ma_tr, select on ma_va
    grid = {
        "max_features": [200, 500, 1000, 2000],
        "C": [0.05, 0.1, 0.5, 1.0, 5.0],
        "sublinear_tf": [True, False],
        "min_df": [1, 2],
    }

    print("\n[Validation sweep on val set]", flush=True)
    sweep = []
    for mf, C, sl, md in product(grid["max_features"], grid["C"],
                                  grid["sublinear_tf"], grid["min_df"]):
        try:
            val_mcc, val_bacc, _, _, _ = fit_eval(
                ma_tr["title_en"].tolist(), ma_tr["y"].values,
                ma_va["title_en"].tolist(), ma_va["y"].values,
                mf, C, sl, md)
        except Exception as exc:
            val_mcc = -2.0; val_bacc = 0.0
        sweep.append({"max_features": mf, "C": C, "sublinear_tf": sl, "min_df": md,
                      "val_mcc": float(val_mcc), "val_balacc": float(val_bacc)})
    sweep_df = pd.DataFrame(sweep).sort_values("val_mcc", ascending=False)
    print("\n  Top-15 by val MCC:")
    print(sweep_df.head(15).to_string(index=False), flush=True)
    print("\n  Bottom-5:")
    print(sweep_df.tail(5).to_string(index=False), flush=True)

    # Select top HP
    best = sweep_df.iloc[0].to_dict()
    print(f"\n[Best HP] {best}", flush=True)

    # Evaluate on test
    test_mcc, test_bacc, tf_best, cl_best, yp_te = fit_eval(
        ma_tr["title_en"].tolist(), ma_tr["y"].values,
        ma_te["title_en"].tolist(), ma_te["y"].values,
        int(best["max_features"]), float(best["C"]), bool(best["sublinear_tf"]),
        int(best["min_df"]))
    print(f"\n[Test on locked test set with best HP]", flush=True)
    print(f"  test MCC={test_mcc:+.4f}  BalAcc={test_bacc:.4f}", flush=True)

    # Permutation test 10K
    print(f"\n[10K permutation on test]", flush=True)
    yte = ma_te["y"].values
    n_perm = 10000
    null = np.zeros(n_perm)
    for k in range(n_perm):
        null[k] = safe_mcc(rng.permutation(yte), yp_te)
        if (k + 1) % 2000 == 0:
            print(f"  perm {k+1}/{n_perm}", flush=True)
    p_one = float((null >= test_mcc).mean())
    p_two = float((np.abs(null) >= abs(test_mcc)).mean())
    z = (test_mcc - null.mean()) / (null.std() + 1e-12)
    print(f"  observed={test_mcc:+.4f}  null_mean={null.mean():+.4f}  null_std={null.std():.4f}",
          flush=True)
    print(f"  z={z:.3f}  p_one={p_one:.4f}  p_two={p_two:.4f}", flush=True)

    # Block-bootstrap by week (1000 resamples)
    print(f"\n[Weekly block bootstrap (1000 resamples)]", flush=True)
    ma_te2 = ma_te.copy()
    ma_te2["week"] = ma_te2["published_date"].dt.to_period("W").astype(str)
    weeks = ma_te2["week"].unique()
    print(f"  {len(weeks)} weekly clusters", flush=True)
    boot_mccs = []
    for b in range(1000):
        sampled_weeks = rng.choice(weeks, size=len(weeks), replace=True)
        idx = []
        for w in sampled_weeks:
            idx.extend(ma_te2[ma_te2["week"] == w].index.tolist())
        idx = np.array(idx)
        if len(np.unique(yte[ma_te.index.get_indexer(idx)])) < 2:
            continue
        # Map original positions to relative
        pos = ma_te.index.get_indexer(idx)
        sub_y = yte[pos]
        sub_p = yp_te[pos]
        boot_mccs.append(safe_mcc(sub_y, sub_p))
    boot_mccs = np.array(boot_mccs)
    ci_lo = float(np.percentile(boot_mccs, 2.5))
    ci_hi = float(np.percentile(boot_mccs, 97.5))
    print(f"  bootstrap mean={boot_mccs.mean():+.4f}  CI95=[{ci_lo:+.4f}, {ci_hi:+.4f}]",
          flush=True)

    # Per-month decomposition
    print(f"\n[Monthly decomposition of test set with best HP]", flush=True)
    ma_te2["month"] = ma_te2["published_date"].dt.to_period("M").astype(str)
    monthly = []
    for m, sub in ma_te2.groupby("month"):
        if len(sub) < 20: continue
        idx = ma_te.index.get_indexer(sub.index)
        sub_y = yte[idx]; sub_p = yp_te[idx]
        mc = safe_mcc(sub_y, sub_p)
        monthly.append({"month": str(m), "n": int(len(sub)),
                        "mcc": float(mc), "true_up": float(sub_y.mean()),
                        "pred_up": float(sub_p.mean())})
        print(f"  {m}: n={len(sub):>4}  MCC={mc:+.4f}", flush=True)

    out = {
        "meta": {"timestamp": pd.Timestamp.now().isoformat(),
                 "n_train": int(len(ma_tr)), "n_val": int(len(ma_va)),
                 "n_test": int(len(ma_te)), "elapsed_s": float(time.time() - t0)},
        "best_hp": best,
        "sweep_top15": sweep_df.head(15).to_dict("records"),
        "sweep_bottom5": sweep_df.tail(5).to_dict("records"),
        "test_result": {
            "mcc": float(test_mcc), "balacc": float(test_bacc),
            "true_up": float(yte.mean()), "pred_up": float(yp_te.mean()),
        },
        "permutation_test": {
            "n_perm": n_perm, "observed_mcc": float(test_mcc),
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "z_score": float(z), "p_one_sided": p_one, "p_two_sided": p_two,
        },
        "block_bootstrap": {
            "n_resamples": 1000, "n_weekly_clusters": int(len(weeks)),
            "mean": float(boot_mccs.mean()),
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        },
        "monthly": monthly,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
