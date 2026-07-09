"""
A2 sanity tests for the acquirer/target asymmetry finding.

Key questions to answer:
  Q1. Is the ACQUIRER MCC=0.154 (n=125) statistically significant?
      -> 10K permutation test on the 125 acquirer test labels.
  Q2. Is the TARGET MCC=0.000 (predict UP for all 116) actually degenerate,
      or just consistent with a noisy model?
      -> Inspect predict_proba distribution; threshold-shift sensitivity;
         random-feature null baseline.
  Q3. Could the asymmetry be explained by article length (info-density
      hypothesis from the paper)?
      -> Train length-only and length+log(n_words) models on each role.
  Q4. Could the asymmetry be explained by capitalised-token (proper-noun)
      density?
      -> Same protocol with cap-token-count feature.

Output: results/validation/acquirer_target_sanity.json
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
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(BASE, "results", "validation", "acquirer_target_sanity.json")

ACQ_PAT = re.compile(
    r"\b(acqui|acquir|takeover|buyer|buy[- ]?out|purchas|to acquire|will acquire|"
    r"completes acquisition|launches offer|tender offer|bid for|offers? to buy)\b",
    re.IGNORECASE)
TGT_PAT = re.compile(
    r"\b(target|to be acquir|being acquir|to be sold|sold to|sale of|divest|"
    r"subject of (an? )?bid|merger with|to merge with|received offer|"
    r"agree(s|d)? to be acquir)\b",
    re.IGNORECASE)
SELLER_PAT = re.compile(r"\b(seller|sells|to sell|divest|spin[- ]?off|spin[- ]?out)\b",
                        re.IGNORECASE)


def classify_role(title):
    is_acq = bool(ACQ_PAT.search(title))
    is_tgt = bool(TGT_PAT.search(title))
    is_sel = bool(SELLER_PAT.search(title))
    if is_acq and is_tgt:
        return "BOTH"
    if is_acq:
        return "ACQUIRER"
    if is_tgt or is_sel:
        return "TARGET"
    return "NEITHER"


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def main():
    rng = np.random.default_rng(42)
    print("Loading data ...", flush=True)
    df = pd.read_parquet(DATA)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    df["content_en"] = df["content_en"].fillna("").astype(str)

    ma = df[df["event"] == "mergers_acquisitions"].copy()
    ma["role"] = ma["title_en"].apply(classify_role)

    TRAIN_END = pd.Timestamp("2025-04-01")
    VAL_END = pd.Timestamp("2025-06-01")
    ma_tr = ma[ma["published_date"] < VAL_END].copy()
    ma_te = ma[ma["published_date"] >= VAL_END].copy()

    print(f"Total M&A: {len(ma)}  train+val={len(ma_tr)}  test={len(ma_te)}", flush=True)
    print("Role counts (test):", ma_te["role"].value_counts().to_dict(), flush=True)

    # === Train an overall M&A specialist (matches paper) ===
    tfidf = TfidfVectorizer(max_features=300, stop_words="english", min_df=2, sublinear_tf=True)
    Xtr = tfidf.fit_transform(ma_tr["title_en"])
    Xte = tfidf.transform(ma_te["title_en"])
    ytr, yte = ma_tr["y"].values, ma_te["y"].values
    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    clf.fit(Xtr, ytr)
    yp_te = clf.predict(Xte)
    yp_te_proba = clf.predict_proba(Xte)[:, 1]
    overall_mcc = safe_mcc(yte, yp_te)
    print(f"\nOverall M&A test MCC (using overall model): {overall_mcc:+.4f}", flush=True)

    out = {"meta": {"timestamp": pd.Timestamp.now().isoformat()},
           "overall_ma_test_mcc": overall_mcc, "tests": {}}

    # === Q1: Permutation null for ACQUIRER MCC=0.154 ===
    print("\n[Q1] Permutation null for ACQUIRER MCC ...", flush=True)
    acq_mask_te = (ma_te["role"] == "ACQUIRER").values
    acq_mask_tr = (ma_tr["role"] == "ACQUIRER").values
    n_acq_te = int(acq_mask_te.sum())
    print(f"  acquirer test n={n_acq_te}", flush=True)

    # Train a role-specific ACQUIRER model
    tf_acq = TfidfVectorizer(max_features=200, stop_words="english", min_df=2, sublinear_tf=True)
    Xtr_acq = tf_acq.fit_transform(ma_tr.loc[acq_mask_tr, "title_en"])
    Xte_acq = tf_acq.transform(ma_te.loc[acq_mask_te, "title_en"])
    ytr_acq = ma_tr.loc[acq_mask_tr, "y"].values
    yte_acq = ma_te.loc[acq_mask_te, "y"].values
    cl_acq = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    cl_acq.fit(Xtr_acq, ytr_acq)
    yp_acq = cl_acq.predict(Xte_acq)
    yp_acq_proba = cl_acq.predict_proba(Xte_acq)[:, 1]
    acq_mcc_obs = safe_mcc(yte_acq, yp_acq)
    print(f"  observed ACQUIRER MCC (role-specific model): {acq_mcc_obs:+.4f}", flush=True)

    n_perm = 10000
    null_mccs = np.zeros(n_perm)
    t0 = time.time()
    for k in range(n_perm):
        y_shuf = rng.permutation(yte_acq)
        null_mccs[k] = safe_mcc(y_shuf, yp_acq)
        if (k + 1) % 2000 == 0:
            print(f"  perm {k+1}/{n_perm}  elapsed {time.time()-t0:.1f}s", flush=True)
    p_one = float((null_mccs >= acq_mcc_obs).mean())
    p_two = float((np.abs(null_mccs) >= abs(acq_mcc_obs)).mean())
    z = (acq_mcc_obs - null_mccs.mean()) / (null_mccs.std() + 1e-12)
    out["tests"]["q1_acquirer_permutation"] = {
        "n_test": n_acq_te, "observed_mcc": float(acq_mcc_obs),
        "null_mean": float(null_mccs.mean()), "null_std": float(null_mccs.std()),
        "z_score": float(z), "p_one_sided": p_one, "p_two_sided": p_two,
        "n_perm": n_perm,
    }
    print(f"  z={z:.3f}  p_one={p_one:.4f}  p_two={p_two:.4f}", flush=True)

    # === Q2: Is TARGET MCC=0 a degenerate prediction or a learned null? ===
    print("\n[Q2] TARGET sanity ...", flush=True)
    tgt_mask_te = (ma_te["role"] == "TARGET").values
    tgt_mask_tr = (ma_tr["role"] == "TARGET").values
    n_tgt_te = int(tgt_mask_te.sum())
    n_tgt_tr = int(tgt_mask_tr.sum())
    print(f"  target train={n_tgt_tr}  test={n_tgt_te}", flush=True)

    tf_tgt = TfidfVectorizer(max_features=200, stop_words="english", min_df=2, sublinear_tf=True)
    Xtr_tgt = tf_tgt.fit_transform(ma_tr.loc[tgt_mask_tr, "title_en"])
    Xte_tgt = tf_tgt.transform(ma_te.loc[tgt_mask_te, "title_en"])
    ytr_tgt = ma_tr.loc[tgt_mask_tr, "y"].values
    yte_tgt = ma_te.loc[tgt_mask_te, "y"].values
    cl_tgt = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    cl_tgt.fit(Xtr_tgt, ytr_tgt)
    yp_tgt = cl_tgt.predict(Xte_tgt)
    yp_tgt_proba = cl_tgt.predict_proba(Xte_tgt)[:, 1]
    tgt_mcc_obs = safe_mcc(yte_tgt, yp_tgt)
    print(f"  observed TARGET MCC: {tgt_mcc_obs:+.4f}", flush=True)
    print(f"  predict UP rate: {yp_tgt.mean():.3f}  (true UP rate: {yte_tgt.mean():.3f})", flush=True)
    print(f"  proba [min, mean, median, max]: "
          f"{yp_tgt_proba.min():.3f}, {yp_tgt_proba.mean():.3f}, "
          f"{np.median(yp_tgt_proba):.3f}, {yp_tgt_proba.max():.3f}", flush=True)

    # Threshold sweep: maybe the issue is just threshold
    thresh_results = []
    for t in [0.4, 0.45, 0.5, np.median(yp_tgt_proba), 0.55, 0.6]:
        yp_t = (yp_tgt_proba >= t).astype(int)
        mcc = safe_mcc(yte_tgt, yp_t)
        thresh_results.append({"threshold": float(t), "mcc": float(mcc),
                               "pred_up_rate": float(yp_t.mean())})
    print("  threshold sweep:")
    for r in thresh_results:
        print(f"    t={r['threshold']:.3f}  pred_up={r['pred_up_rate']:.3f}  MCC={r['mcc']:+.4f}",
              flush=True)

    # Random-feature null: is the role-conditional MCC consistent with random
    # features?
    print("  random-feature null on TARGET ...", flush=True)
    n_rand_trials = 200
    rand_mccs = []
    for trial in range(n_rand_trials):
        Xrand_tr = rng.normal(0, 1, (n_tgt_tr, 50)).astype(np.float32)
        Xrand_te = rng.normal(0, 1, (n_tgt_te, 50)).astype(np.float32)
        cl = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        cl.fit(Xrand_tr, ytr_tgt)
        yp = cl.predict(Xrand_te)
        rand_mccs.append(safe_mcc(yte_tgt, yp))
    rand_mccs = np.array(rand_mccs)
    print(f"  random-feature TARGET MCC mean={rand_mccs.mean():+.4f} std={rand_mccs.std():.4f}",
          flush=True)
    print(f"  fraction degenerate (predicts >=99% UP): "
          f"{((rand_mccs == 0).mean()):.3f}", flush=True)

    out["tests"]["q2_target_sanity"] = {
        "n_train": n_tgt_tr, "n_test": n_tgt_te,
        "observed_mcc": float(tgt_mcc_obs),
        "predict_up_rate": float(yp_tgt.mean()),
        "true_up_rate": float(yte_tgt.mean()),
        "proba_stats": {
            "min": float(yp_tgt_proba.min()),
            "mean": float(yp_tgt_proba.mean()),
            "median": float(np.median(yp_tgt_proba)),
            "max": float(yp_tgt_proba.max()),
            "std": float(yp_tgt_proba.std()),
        },
        "threshold_sweep": thresh_results,
        "random_feature_null": {
            "n_trials": n_rand_trials,
            "mean_mcc": float(rand_mccs.mean()),
            "std_mcc": float(rand_mccs.std()),
            "frac_zero_mcc": float((rand_mccs == 0).mean()),
        },
    }

    # Same random-feature null on ACQUIRER for comparison
    print("\n  random-feature null on ACQUIRER ...", flush=True)
    n_acq_tr = int(acq_mask_tr.sum())
    rand_mccs_acq = []
    for trial in range(n_rand_trials):
        Xrand_tr = rng.normal(0, 1, (n_acq_tr, 50)).astype(np.float32)
        Xrand_te = rng.normal(0, 1, (n_acq_te, 50)).astype(np.float32)
        cl = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        cl.fit(Xrand_tr, ytr_acq)
        yp = cl.predict(Xrand_te)
        rand_mccs_acq.append(safe_mcc(yte_acq, yp))
    rand_mccs_acq = np.array(rand_mccs_acq)
    print(f"  random-feature ACQUIRER MCC mean={rand_mccs_acq.mean():+.4f} "
          f"std={rand_mccs_acq.std():.4f}", flush=True)
    print(f"  observed acquirer MCC z-score vs random-feature null: "
          f"{(acq_mcc_obs - rand_mccs_acq.mean()) / (rand_mccs_acq.std() + 1e-9):.3f}",
          flush=True)
    out["tests"]["q1b_acquirer_random_feature_null"] = {
        "n_trials": n_rand_trials,
        "mean_mcc": float(rand_mccs_acq.mean()),
        "std_mcc": float(rand_mccs_acq.std()),
        "z_observed_vs_null": float((acq_mcc_obs - rand_mccs_acq.mean()) /
                                    (rand_mccs_acq.std() + 1e-9)),
    }

    # === Q3: Length-only ablation ===
    print("\n[Q3] Length-only ablation ...", flush=True)
    ma_tr["title_len"] = ma_tr["title_en"].str.len()
    ma_tr["content_len"] = ma_tr["content_en"].str.len()
    ma_te["title_len"] = ma_te["title_en"].str.len()
    ma_te["content_len"] = ma_te["content_en"].str.len()
    ma_tr["title_words"] = ma_tr["title_en"].str.split().str.len()
    ma_te["title_words"] = ma_te["title_en"].str.split().str.len()

    for role, m_tr_, m_te_ in [("ACQUIRER", acq_mask_tr, acq_mask_te),
                                ("TARGET", tgt_mask_tr, tgt_mask_te)]:
        Xtr_l = ma_tr.loc[m_tr_, ["title_len", "content_len", "title_words"]].values
        Xte_l = ma_te.loc[m_te_, ["title_len", "content_len", "title_words"]].values
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_l)
        Xte_s = sc.transform(Xte_l)
        ytr_ = ma_tr.loc[m_tr_, "y"].values
        yte_ = ma_te.loc[m_te_, "y"].values
        cl = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
        cl.fit(Xtr_s, ytr_)
        yp = cl.predict(Xte_s)
        mcc_len = safe_mcc(yte_, yp)
        print(f"  {role:>10} length-only MCC: {mcc_len:+.4f}  "
              f"(true_up={yte_.mean():.3f} pred_up={yp.mean():.3f})", flush=True)
        out["tests"].setdefault("q3_length_only", {})[role] = {
            "mcc": float(mcc_len),
            "pred_up_rate": float(yp.mean()),
            "true_up_rate": float(yte_.mean()),
            "n_train": int(m_tr_.sum()), "n_test": int(m_te_.sum()),
        }

    # Length distribution differences
    for role, m_te_ in [("ACQUIRER", acq_mask_te), ("TARGET", tgt_mask_te)]:
        title_lens = ma_te.loc[m_te_, "title_len"].values
        content_lens = ma_te.loc[m_te_, "content_len"].values
        print(f"  {role:>10}: title_len mean={title_lens.mean():.1f}  "
              f"content_len mean={content_lens.mean():.1f}", flush=True)
        out["tests"].setdefault("q3_length_only", {}).setdefault(role, {}).update({
            "avg_title_len": float(title_lens.mean()),
            "avg_content_len": float(content_lens.mean()),
        })

    # === Q4: Capitalised-token (proper-noun proxy) ablation ===
    print("\n[Q4] Capitalised-token-count ablation ...", flush=True)
    cap_re = re.compile(r"\b[A-Z][A-Za-z0-9]+\b")
    ma_tr["cap_count"] = ma_tr["title_en"].apply(lambda s: len(cap_re.findall(s)))
    ma_te["cap_count"] = ma_te["title_en"].apply(lambda s: len(cap_re.findall(s)))

    for role, m_tr_, m_te_ in [("ACQUIRER", acq_mask_tr, acq_mask_te),
                                ("TARGET", tgt_mask_tr, tgt_mask_te)]:
        Xtr_c = ma_tr.loc[m_tr_, ["cap_count"]].values
        Xte_c = ma_te.loc[m_te_, ["cap_count"]].values
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_c)
        Xte_s = sc.transform(Xte_c)
        ytr_ = ma_tr.loc[m_tr_, "y"].values
        yte_ = ma_te.loc[m_te_, "y"].values
        cl = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
        cl.fit(Xtr_s, ytr_)
        yp = cl.predict(Xte_s)
        mcc_cap = safe_mcc(yte_, yp)
        avg_cap = ma_te.loc[m_te_, "cap_count"].mean()
        print(f"  {role:>10} capcount-only MCC: {mcc_cap:+.4f}  "
              f"(avg cap_count={avg_cap:.2f})", flush=True)
        out["tests"].setdefault("q4_cap_count", {})[role] = {
            "mcc": float(mcc_cap),
            "avg_cap_count": float(avg_cap),
        }

    # === Save ===
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
