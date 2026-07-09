"""
M&A specialist HP grid — LEAVE-ONE-AXIS-OUT robustness.

Audit weakness W2 (STRICT_REVIEWER_AUDIT_19May.md):
    The 360-cell grid was extended after a 40-cell grid gave marginal p=0.039.
    Headline MCC jumped from 0.075 to 0.138. Reviewer 2 will write: "researcher
    degrees of freedom — show me this is not a single lucky cell."

This script computes the robustness panel by collapsing each axis at a time to
its modal val-winning value, re-running val-selected HP -> test exactly once
per panel.

Output: results/validation/ma_hp_grid_robustness.json
        + a small markdown table to paste into the paper's appendix.

For each of 5 axes:
    1. Drop that axis (collapse to the single value that wins most often on val).
    2. Sweep the reduced 4-D grid on val.
    3. Pick val-best on the reduced grid.
    4. Eval on test ONCE.

Pre-condition: the full 360-cell sweep result must exist (ma_hp_sweep_extended.json).
"""
import os
import sys
import io
import json
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
from collections import Counter

BASE = r"."
OUR = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(BASE, "results", "validation", "ma_hp_grid_robustness.json")
SWEEP_IN = os.path.join(BASE, "results", "validation", "ma_hp_sweep_extended.json")


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def fit_eval(tr_titles, y_tr, te_titles, y_te,
             max_features, C, sublinear_tf, min_df, ngram_range):
    tf = TfidfVectorizer(max_features=max_features, stop_words="english",
                         min_df=min_df, sublinear_tf=sublinear_tf,
                         ngram_range=ngram_range)
    Xtr = tf.fit_transform(tr_titles)
    Xte = tf.transform(te_titles)
    clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
    clf.fit(Xtr, y_tr)
    yp = clf.predict(Xte)
    return safe_mcc(y_te, yp), float(balanced_accuracy_score(y_te, yp)), yp


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

    full_grid = {
        "max_features": [50, 100, 200, 500, 1000, 2000],
        "C": [0.05, 0.1, 0.5, 1.0, 5.0],
        "sublinear_tf": [True, False],
        "min_df": [1, 2],
        "ngram_range": [(1, 1), (1, 2), (1, 3)],
    }

    # Compute modal val-winning values from the existing 360-cell sweep
    if not os.path.exists(SWEEP_IN):
        print(f"FATAL: need {SWEEP_IN} to exist; run ma_hp_sweep_extended.py first", flush=True)
        sys.exit(1)
    sweep_in = json.load(open(SWEEP_IN))
    top15 = sweep_in["sweep_top15"]
    # Mode = the single most frequent value in top-N val winners
    TOPN = 15
    modes = {}
    for axis in ["max_features", "C", "sublinear_tf", "min_df"]:
        vals = [r[axis] for r in top15[:TOPN]]
        modes[axis] = Counter(vals).most_common(1)[0][0]
    ng_vals = [tuple(r["ngram_range"]) for r in top15[:TOPN]]
    modes["ngram_range"] = Counter(ng_vals).most_common(1)[0][0]
    print(f"\n[Modal val-winning values across top-{TOPN}]: {modes}", flush=True)

    # Verify global best on the full grid for baseline reference
    best_global = sweep_in["best_hp"]
    test_global = sweep_in["test_result"]
    print(f"[Global best HP across full 360-cell grid]: {best_global}", flush=True)
    print(f"  -> test MCC = {test_global['mcc']:+.4f}", flush=True)

    panels = []  # one entry per dropped axis

    for drop_axis in full_grid.keys():
        sub_grid = {k: ([modes[k]] if k == drop_axis else v) for k, v in full_grid.items()}
        n_cells = (len(sub_grid["max_features"]) * len(sub_grid["C"]) *
                   len(sub_grid["sublinear_tf"]) * len(sub_grid["min_df"]) *
                   len(sub_grid["ngram_range"]))
        print(f"\n[Drop axis = {drop_axis}; collapse to {modes[drop_axis]}]"
              f"  reduced grid = {n_cells} cells", flush=True)

        local_sweep = []
        for mf, C, sl, md, ng in product(sub_grid["max_features"], sub_grid["C"],
                                          sub_grid["sublinear_tf"], sub_grid["min_df"],
                                          sub_grid["ngram_range"]):
            try:
                val_mcc, val_bacc, _ = fit_eval(
                    ma_tr["title_en"].tolist(), ma_tr["y"].values,
                    ma_va["title_en"].tolist(), ma_va["y"].values,
                    mf, C, sl, md, ng)
            except Exception:
                val_mcc = -2.0; val_bacc = 0.0
            local_sweep.append({"max_features": mf, "C": C, "sublinear_tf": sl,
                                "min_df": md, "ngram_range": list(ng),
                                "val_mcc": float(val_mcc), "val_balacc": float(val_bacc)})
        local_df = pd.DataFrame(local_sweep).sort_values("val_mcc", ascending=False)
        best = local_df.iloc[0].to_dict()
        print(f"  val-best on reduced grid: {best}", flush=True)

        test_mcc, test_bacc, _ = fit_eval(
            ma_tr["title_en"].tolist(), ma_tr["y"].values,
            ma_te["title_en"].tolist(), ma_te["y"].values,
            int(best["max_features"]), float(best["C"]), bool(best["sublinear_tf"]),
            int(best["min_df"]), tuple(best["ngram_range"]))
        print(f"  -> test MCC = {test_mcc:+.4f}, BalAcc = {test_bacc:.4f}", flush=True)

        panels.append({
            "dropped_axis": drop_axis,
            "collapsed_to": modes[drop_axis] if drop_axis != "ngram_range" else list(modes[drop_axis]),
            "reduced_grid_size": n_cells,
            "val_best_hp": best,
            "val_best_mcc": float(best["val_mcc"]),
            "test_mcc": float(test_mcc),
            "test_balacc": float(test_bacc),
        })

    # Compute summary stats over the 5 panels
    test_mccs = [p["test_mcc"] for p in panels]
    summary = {
        "n_panels": len(panels),
        "test_mcc_min": float(min(test_mccs)),
        "test_mcc_max": float(max(test_mccs)),
        "test_mcc_mean": float(np.mean(test_mccs)),
        "test_mcc_std": float(np.std(test_mccs, ddof=1)),
        "full_grid_test_mcc": float(test_global["mcc"]),
    }
    print(f"\n[Summary across 5 leave-one-axis-out panels]", flush=True)
    print(f"  test MCC range: [{summary['test_mcc_min']:+.4f}, {summary['test_mcc_max']:+.4f}]",
          flush=True)
    print(f"  mean: {summary['test_mcc_mean']:+.4f} +- {summary['test_mcc_std']:.4f}",
          flush=True)
    print(f"  full-grid headline: {summary['full_grid_test_mcc']:+.4f}", flush=True)

    out = {
        "meta": {"timestamp": pd.Timestamp.now().isoformat(),
                 "elapsed_s": float(time.time() - t0),
                 "n_train": int(len(ma_tr)), "n_val": int(len(ma_va)),
                 "n_test": int(len(ma_te))},
        "full_grid_size": 360,
        "topN_for_mode": TOPN,
        "modal_values": {k: (v if k != "ngram_range" else list(v))
                         for k, v in modes.items()},
        "panels": panels,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)

    # Markdown table for paper appendix
    md = ["| Dropped axis | Collapsed value | Reduced grid | Val MCC | Test MCC |",
          "|---|---|---|---|---|"]
    for p in panels:
        md.append(f"| {p['dropped_axis']} | {p['collapsed_to']} | "
                  f"{p['reduced_grid_size']} | "
                  f"{p['val_best_mcc']:+.3f} | {p['test_mcc']:+.3f} |")
    md.append(f"| (full 360-cell grid headline) | -- | 360 | "
              f"-- | {summary['full_grid_test_mcc']:+.3f} |")
    md_path = OUT.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# M&A HP grid leave-one-axis-out robustness\n\n")
        f.write("\n".join(md))
        f.write(f"\n\n**Summary**: test MCC ranges in "
                f"[{summary['test_mcc_min']:+.3f}, {summary['test_mcc_max']:+.3f}], "
                f"mean {summary['test_mcc_mean']:+.3f} ± "
                f"{summary['test_mcc_std']:.3f}.\n")
    print(f"Saved markdown: {md_path}", flush=True)


if __name__ == "__main__":
    main()
