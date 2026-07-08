"""
A4 v2: Combined-corpus M&A with PROPER hyperparameters.

Lessons from edt_robustness.py: the EDT signal is real but is sensitive
to hyperparameter choice (max_features, C, sublinear_tf). The original
M&A specialist settings (max_features=300, C=0.1, sublinear_tf=True) are
suboptimal for the larger combined corpus.

This v2 uses the paper-EDT settings (max_features=1000, C=1.0,
sublinear_tf=False, min_df=1) which are appropriate for the combined-corpus
data scale.

Tests:
  E1: ours-only baseline (paper replication, our M&A specialist settings)
  E2: EDT-only baseline (paper EDT replication)
  E3a: combined train -> our test (paper-EDT HP)
  E3b: combined train -> EDT test (paper-EDT HP)
  E4: cross-corpus transfer (with paper-EDT HP)
  E7: ours-only with paper-EDT HP (control: are HP differences alone
      explaining the gap?)

Output: results/validation/combined_ma_bigger_v2.json
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

BASE = r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package"
OUR = os.path.join(BASE, "data", "classifier_training_v2.parquet")
EDT = os.path.join(BASE, "data", "external", "edt_evaluate_slim.parquet")
OUT = os.path.join(BASE, "results", "validation", "combined_ma_bigger_v2.json")

NARROW_MA = re.compile(r"\b(merger|acquisition|acquir|takeover|tender offer)\b", re.IGNORECASE)


# Paper-EDT hyperparameters (H1)
HP_EDT = {"max_features": 1000, "C": 1.0, "sublinear_tf": False, "min_df": 1}
# Our M&A specialist hyperparameters (H2)
HP_OUR = {"max_features": 300, "C": 0.1, "sublinear_tf": True, "min_df": 2}


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def fit_eval(tr_titles, y_tr, te_titles, y_te, hp):
    tf = TfidfVectorizer(max_features=hp["max_features"], stop_words="english",
                         min_df=hp["min_df"], sublinear_tf=hp["sublinear_tf"])
    Xtr = tf.fit_transform(tr_titles)
    Xte = tf.transform(te_titles)
    clf = LogisticRegression(max_iter=2000, C=hp["C"], random_state=42)
    clf.fit(Xtr, y_tr)
    yp = clf.predict(Xte)
    return safe_mcc(y_te, yp), float(balanced_accuracy_score(y_te, yp)), tf, clf


def main():
    t0 = time.time()
    rng = np.random.default_rng(42)
    print("Loading our data ...", flush=True)
    ours = pd.read_parquet(OUR)
    ours = ours[ours["actual_side"].str.lower().isin(["up", "down"])].copy()
    ours["y"] = (ours["actual_side"].str.lower() == "up").astype(int)
    ours["published_date"] = pd.to_datetime(ours["published_date"]).dt.tz_localize(None)
    ours["title_en"] = ours["title_en"].fillna("").astype(str)
    our_ma = ours[ours["event"] == "mergers_acquisitions"].copy()

    OUR_VAL_END = pd.Timestamp("2025-06-01")
    our_ma_tr = our_ma[our_ma["published_date"] < OUR_VAL_END].copy()
    our_ma_te = our_ma[our_ma["published_date"] >= OUR_VAL_END].copy()
    print(f"  our M&A: train+val={len(our_ma_tr)}  test={len(our_ma_te)}", flush=True)

    print("Loading EDT ...", flush=True)
    edt = pd.read_parquet(EDT)
    edt["pub_time"] = pd.to_datetime(edt["pub_time"]).dt.tz_localize(None)
    edt["title"] = edt["title"].fillna("").astype(str)
    edt_ma = edt[edt["title"].apply(lambda s: bool(NARROW_MA.search(s)))].copy()
    edt_ma = edt_ma.sort_values("pub_time").reset_index(drop=True)
    n_edt = len(edt_ma)

    # 60/20/20 chronological split (paper convention)
    a = int(n_edt * 0.6); b = int(n_edt * 0.8)
    edt_ma_tr = edt_ma.iloc[:a].copy()
    edt_ma_va = edt_ma.iloc[a:b].copy()
    edt_ma_te = edt_ma.iloc[b:].copy()
    print(f"  EDT narrow MA: train={len(edt_ma_tr)}  val={len(edt_ma_va)}  test={len(edt_ma_te)}",
          flush=True)

    out = {"meta": {"timestamp": pd.Timestamp.now().isoformat(),
                    "hp_paper_edt": HP_EDT, "hp_our_specialist": HP_OUR},
           "experiments": {}}

    # === E1: ours-only with our M&A specialist HP (paper baseline) ===
    print("\n[E1] ours-only with HP_OUR (paper baseline)", flush=True)
    mcc, bacc, _, _ = fit_eval(our_ma_tr["title_en"].tolist(), our_ma_tr["y"].values,
                                our_ma_te["title_en"].tolist(), our_ma_te["y"].values, HP_OUR)
    print(f"  E1: MCC={mcc:+.4f}", flush=True)
    out["experiments"]["E1_our_only_HP_OUR"] = {
        "n_train": int(len(our_ma_tr)), "n_test": int(len(our_ma_te)),
        "mcc": float(mcc), "balacc": float(bacc), "hp": "HP_OUR"}

    # === E1b: ours-only with HP_EDT (control - did we underspec?) ===
    print("\n[E1b] ours-only with HP_EDT (HP control)", flush=True)
    mcc, bacc, _, _ = fit_eval(our_ma_tr["title_en"].tolist(), our_ma_tr["y"].values,
                                our_ma_te["title_en"].tolist(), our_ma_te["y"].values, HP_EDT)
    print(f"  E1b: MCC={mcc:+.4f}", flush=True)
    out["experiments"]["E1b_our_only_HP_EDT"] = {
        "n_train": int(len(our_ma_tr)), "n_test": int(len(our_ma_te)),
        "mcc": float(mcc), "balacc": float(bacc), "hp": "HP_EDT"}

    # === E2: EDT-only with HP_EDT (paper EDT result) ===
    print("\n[E2] EDT-only with HP_EDT (paper EDT result)", flush=True)
    mcc, bacc, _, _ = fit_eval(edt_ma_tr["title"].tolist(), edt_ma_tr["y"].values,
                                edt_ma_te["title"].tolist(), edt_ma_te["y"].values, HP_EDT)
    print(f"  E2: MCC={mcc:+.4f}", flush=True)
    out["experiments"]["E2_edt_only_HP_EDT"] = {
        "n_train": int(len(edt_ma_tr)), "n_test": int(len(edt_ma_te)),
        "mcc": float(mcc), "balacc": float(bacc), "hp": "HP_EDT"}

    # === E2b: EDT-only with HP_OUR ===
    print("\n[E2b] EDT-only with HP_OUR", flush=True)
    mcc, bacc, _, _ = fit_eval(edt_ma_tr["title"].tolist(), edt_ma_tr["y"].values,
                                edt_ma_te["title"].tolist(), edt_ma_te["y"].values, HP_OUR)
    print(f"  E2b: MCC={mcc:+.4f}", flush=True)
    out["experiments"]["E2b_edt_only_HP_OUR"] = {
        "n_train": int(len(edt_ma_tr)), "n_test": int(len(edt_ma_te)),
        "mcc": float(mcc), "balacc": float(bacc), "hp": "HP_OUR"}

    # === E3: COMBINED with HP_EDT, two test sets ===
    print("\n[E3] COMBINED training with HP_EDT, two test sets", flush=True)
    combined_titles = pd.concat([our_ma_tr["title_en"], edt_ma_tr["title"]],
                                ignore_index=True).values
    combined_y = np.concatenate([our_ma_tr["y"].values, edt_ma_tr["y"].values])
    print(f"  combined train: {len(combined_titles)}", flush=True)
    tf = TfidfVectorizer(max_features=HP_EDT["max_features"], stop_words="english",
                         min_df=HP_EDT["min_df"], sublinear_tf=HP_EDT["sublinear_tf"])
    Xtr = tf.fit_transform(combined_titles)
    Xte_our = tf.transform(our_ma_te["title_en"])
    Xte_edt = tf.transform(edt_ma_te["title"])
    cl = LogisticRegression(max_iter=2000, C=HP_EDT["C"], random_state=42)
    cl.fit(Xtr, combined_y)
    yp_our = cl.predict(Xte_our); yp_edt = cl.predict(Xte_edt)
    e3_our_mcc = safe_mcc(our_ma_te["y"].values, yp_our)
    e3_edt_mcc = safe_mcc(edt_ma_te["y"].values, yp_edt)
    print(f"  E3 -> our test: MCC={e3_our_mcc:+.4f}", flush=True)
    print(f"  E3 -> edt test: MCC={e3_edt_mcc:+.4f}", flush=True)

    # Permutation tests for E3
    print("  permutation tests (5K)...", flush=True)
    n_perm = 5000
    null_our = np.zeros(n_perm); null_edt = np.zeros(n_perm)
    yte_our = our_ma_te["y"].values; yte_edt = edt_ma_te["y"].values
    for k in range(n_perm):
        null_our[k] = safe_mcc(rng.permutation(yte_our), yp_our)
        null_edt[k] = safe_mcc(rng.permutation(yte_edt), yp_edt)
    p_our = float((null_our >= e3_our_mcc).mean())
    p_edt = float((null_edt >= e3_edt_mcc).mean())
    print(f"  E3 our p_one={p_our:.4f}  E3 edt p_one={p_edt:.4f}", flush=True)
    out["experiments"]["E3_combined_HP_EDT"] = {
        "n_train": int(len(combined_titles)),
        "test_our": {"n": int(len(yte_our)), "mcc": float(e3_our_mcc),
                     "p_one_sided": p_our,
                     "balacc": float(balanced_accuracy_score(yte_our, yp_our))},
        "test_edt": {"n": int(len(yte_edt)), "mcc": float(e3_edt_mcc),
                     "p_one_sided": p_edt,
                     "balacc": float(balanced_accuracy_score(yte_edt, yp_edt))},
        "hp": "HP_EDT",
    }

    # === E4: Cross-corpus transfer (with HP_EDT) ===
    print("\n[E4] cross-corpus transfer (with HP_EDT)", flush=True)
    # Train EDT-only -> evaluate on our test
    tf2 = TfidfVectorizer(max_features=HP_EDT["max_features"], stop_words="english",
                          min_df=HP_EDT["min_df"], sublinear_tf=HP_EDT["sublinear_tf"])
    Xtr2 = tf2.fit_transform(edt_ma_tr["title"])
    Xte2 = tf2.transform(our_ma_te["title_en"])
    cl2 = LogisticRegression(max_iter=2000, C=HP_EDT["C"], random_state=42)
    cl2.fit(Xtr2, edt_ma_tr["y"].values)
    yp = cl2.predict(Xte2)
    e4_edt_to_our = safe_mcc(yte_our, yp)

    # Train OUR-only -> evaluate on EDT test
    tf3 = TfidfVectorizer(max_features=HP_EDT["max_features"], stop_words="english",
                          min_df=HP_EDT["min_df"], sublinear_tf=HP_EDT["sublinear_tf"])
    Xtr3 = tf3.fit_transform(our_ma_tr["title_en"])
    Xte3 = tf3.transform(edt_ma_te["title"])
    cl3 = LogisticRegression(max_iter=2000, C=HP_EDT["C"], random_state=42)
    cl3.fit(Xtr3, our_ma_tr["y"].values)
    yp3 = cl3.predict(Xte3)
    e4_our_to_edt = safe_mcc(yte_edt, yp3)
    print(f"  EDT -> our: MCC={e4_edt_to_our:+.4f}", flush=True)
    print(f"  our -> EDT: MCC={e4_our_to_edt:+.4f}", flush=True)
    out["experiments"]["E4_transfer_HP_EDT"] = {
        "edt_to_our": {"n": int(len(yte_our)), "mcc": float(e4_edt_to_our)},
        "our_to_edt": {"n": int(len(yte_edt)), "mcc": float(e4_our_to_edt)},
    }

    # === Vocabulary overlap diagnostic ===
    print("\n[Vocab overlap]", flush=True)
    tf_our = TfidfVectorizer(max_features=1000, stop_words="english")
    tf_our.fit(our_ma_tr["title_en"])
    tf_edt = TfidfVectorizer(max_features=1000, stop_words="english")
    tf_edt.fit(edt_ma_tr["title"])
    vocab_our = set(tf_our.vocabulary_.keys())
    vocab_edt = set(tf_edt.vocabulary_.keys())
    overlap = vocab_our & vocab_edt
    out["vocab"] = {
        "our_top1000": len(vocab_our),
        "edt_top1000": len(vocab_edt),
        "overlap": len(overlap),
        "overlap_frac_of_our": float(len(overlap) / len(vocab_our)),
    }
    print(f"  our top-1000 vocab: {len(vocab_our)}", flush=True)
    print(f"  EDT top-1000 vocab: {len(vocab_edt)}", flush=True)
    print(f"  overlap: {len(overlap)} ({100*len(overlap)/len(vocab_our):.1f}% of our)",
          flush=True)
    print(f"  example overlap words: "
          f"{sorted(list(overlap))[:20]}", flush=True)
    print(f"  example our-only words: "
          f"{sorted(list(vocab_our - vocab_edt))[:20]}", flush=True)
    print(f"  example EDT-only words: "
          f"{sorted(list(vocab_edt - vocab_our))[:20]}", flush=True)

    # === E5: COMBINED with HP_OUR for completeness ===
    print("\n[E5] COMBINED with HP_OUR", flush=True)
    tf5 = TfidfVectorizer(max_features=HP_OUR["max_features"], stop_words="english",
                          min_df=HP_OUR["min_df"], sublinear_tf=HP_OUR["sublinear_tf"])
    Xtr5 = tf5.fit_transform(combined_titles)
    Xte5_our = tf5.transform(our_ma_te["title_en"])
    Xte5_edt = tf5.transform(edt_ma_te["title"])
    cl5 = LogisticRegression(max_iter=2000, C=HP_OUR["C"], random_state=42)
    cl5.fit(Xtr5, combined_y)
    yp5_our = cl5.predict(Xte5_our); yp5_edt = cl5.predict(Xte5_edt)
    e5_our = safe_mcc(yte_our, yp5_our); e5_edt = safe_mcc(yte_edt, yp5_edt)
    print(f"  E5 our: MCC={e5_our:+.4f}  EDT: MCC={e5_edt:+.4f}", flush=True)
    out["experiments"]["E5_combined_HP_OUR"] = {
        "n_train": int(len(combined_titles)),
        "test_our": {"n": int(len(yte_our)), "mcc": float(e5_our)},
        "test_edt": {"n": int(len(yte_edt)), "mcc": float(e5_edt)},
        "hp": "HP_OUR",
    }

    out["meta"]["elapsed_s"] = float(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
