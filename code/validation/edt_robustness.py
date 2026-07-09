"""
EDT signal robustness check.

CRITICAL audit triggered by combined_ma_bigger.py finding that EDT MCC
collapsed from paper's 0.097 to 0.011 when using M&A-specialist settings.

We test the EDT M&A signal across:
  - 3 split fractions: 60/20/20, 70/15/15, 80/10/10 (chronological)
  - 4 hyperparameter settings:
      H1: max_features=1000, C=1.0, no sublinear_tf      (paper's EDT setting)
      H2: max_features=300,  C=0.1, sublinear_tf=True    (our M&A specialist setting)
      H3: max_features=500,  C=0.5, sublinear_tf=True    (intermediate)
      H4: max_features=2000, C=1.0, sublinear_tf=False   (more features)
  - 5 random seeds where applicable

This tells us whether the EDT replication claim survives sensitivity analysis.

Output: results/validation/edt_robustness.json
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

BASE = r"."
EDT = os.path.join(BASE, "data", "external", "edt_evaluate_slim.parquet")
OUT = os.path.join(BASE, "results", "validation", "edt_robustness.json")

NARROW_MA = re.compile(r"\b(merger|acquisition|acquir|takeover|tender offer)\b", re.IGNORECASE)


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


HYPER_SETS = {
    "H1_paper_edt": {"max_features": 1000, "C": 1.0, "sublinear_tf": False, "min_df": 1},
    "H2_our_specialist": {"max_features": 300, "C": 0.1, "sublinear_tf": True, "min_df": 2},
    "H3_intermediate": {"max_features": 500, "C": 0.5, "sublinear_tf": True, "min_df": 2},
    "H4_more_features": {"max_features": 2000, "C": 1.0, "sublinear_tf": False, "min_df": 1},
}

SPLITS = {"60_20_20": (0.6, 0.2), "70_15_15": (0.7, 0.15), "80_10_10": (0.8, 0.1)}


def fit_eval(tr_titles, y_tr, te_titles, y_te, hp):
    tf = TfidfVectorizer(max_features=hp["max_features"], stop_words="english",
                         min_df=hp["min_df"], sublinear_tf=hp["sublinear_tf"])
    Xtr = tf.fit_transform(tr_titles)
    Xte = tf.transform(te_titles)
    clf = LogisticRegression(max_iter=2000, C=hp["C"], random_state=42)
    clf.fit(Xtr, y_tr)
    yp = clf.predict(Xte)
    return safe_mcc(y_te, yp), float(balanced_accuracy_score(y_te, yp))


def main():
    t0 = time.time()
    print("Loading EDT ...", flush=True)
    df = pd.read_parquet(EDT)
    df["pub_time"] = pd.to_datetime(df["pub_time"]).dt.tz_localize(None)
    df["title"] = df["title"].fillna("").astype(str)
    ma = df[df["title"].apply(lambda s: bool(NARROW_MA.search(s)))].copy()
    ma = ma.sort_values("pub_time").reset_index(drop=True)
    print(f"  EDT narrow MA: {len(ma)}", flush=True)

    out = {"meta": {"timestamp": pd.Timestamp.now().isoformat(),
                    "n_total": int(len(ma)),
                    "split_fractions": SPLITS,
                    "hyperparameter_sets": HYPER_SETS},
           "results": []}

    # === Sensitivity matrix: 4 hyperparameter sets x 3 split fractions ===
    for split_name, (frac_tr, frac_va) in SPLITS.items():
        n = len(ma)
        a = int(n * frac_tr)
        b = int(n * (frac_tr + frac_va))
        tr = ma.iloc[:a]; te = ma.iloc[b:]
        print(f"\n[{split_name}] train={len(tr)}, test={len(te)}", flush=True)
        for hp_name, hp in HYPER_SETS.items():
            mcc, bacc = fit_eval(tr["title"].tolist(), tr["y"].values,
                                 te["title"].tolist(), te["y"].values, hp)
            print(f"  {hp_name:>20s} : MCC={mcc:+.4f}  BalAcc={bacc:.4f}", flush=True)
            out["results"].append({
                "split": split_name, "hyperparams": hp_name,
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "mcc": float(mcc), "balacc": float(bacc),
                "true_up": float(te["y"].mean()),
            })

    # === Random-seed robustness for one cell (60/20/20, H1 paper settings) ===
    print("\n[seed sensitivity for paper setting 60/20/20 H1]", flush=True)
    n = len(ma); a = int(n * 0.6); b = int(n * 0.8)
    tr = ma.iloc[:a]; te = ma.iloc[b:]
    seed_mccs = []
    for seed in range(5):
        # Vary the random_state of LR; data split is chronological (deterministic)
        tf = TfidfVectorizer(max_features=1000, stop_words="english")
        Xtr = tf.fit_transform(tr["title"]); Xte = tf.transform(te["title"])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(Xtr, tr["y"].values)
        yp = clf.predict(Xte)
        mcc = safe_mcc(te["y"].values, yp)
        seed_mccs.append(mcc)
        print(f"  seed={seed}: MCC={mcc:+.4f}", flush=True)
    out["seed_robustness_paper_60_20_20"] = {
        "seed_mccs": [float(m) for m in seed_mccs],
        "mean": float(np.mean(seed_mccs)),
        "std": float(np.std(seed_mccs, ddof=1)),
    }

    # === Test month-by-month robustness within paper's test slice ===
    print("\n[monthly decomposition of paper test slice]", flush=True)
    te_with_month = te.copy()
    te_with_month["month"] = te_with_month["pub_time"].dt.to_period("M").astype(str)

    # Use H1 (paper) settings
    tf = TfidfVectorizer(max_features=1000, stop_words="english")
    Xtr = tf.fit_transform(tr["title"])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(Xtr, tr["y"].values)

    monthly = []
    for m, sub in te_with_month.groupby("month"):
        if len(sub) < 20:
            continue
        Xs = tf.transform(sub["title"])
        ys = sub["y"].values
        ps = clf.predict(Xs)
        mc = safe_mcc(ys, ps)
        monthly.append({"month": m, "n": int(len(sub)),
                        "mcc": float(mc), "true_up": float(ys.mean())})
        print(f"  {m}: n={len(sub):>4}  MCC={mc:+.4f}", flush=True)
    out["monthly_paper_test"] = monthly

    out["meta"]["elapsed_s"] = float(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
