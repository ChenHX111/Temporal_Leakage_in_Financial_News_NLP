"""
Paired statistical tests for the acquirer-vs-target asymmetry (reviewer W3).

Background
----------
The paper presents a *qualitative* convergent finding across three independent
role labellers (regex; NER+dep-parse; FinBERT role specialists) that the
acquirer-side {M&A} signal exceeds the target-side signal. The reviewer asked
for a paired statistical test (McNemar / Diebold-Mariano).

Caveat
------
Classical McNemar requires the SAME items to be classified by two competing
classifiers, and DM requires two competing forecasts on the SAME series.
The acquirer and target test subsets are DISJOINT (an article labelled
ACQUIRER cannot also be labelled TARGET, except in the small "BOTH" cell).
Therefore the strictly-correct tests for the asymmetry on disjoint subsamples
are:

  Test A. Two-proportion z-test on accuracy(acquirer) vs accuracy(target)
          [independent samples, classical Fisher-style asymmetric test].
  Test B. Weekly block-bootstrap CI on Delta = MCC_acquirer - MCC_target,
          with 2-sided percentile p-value at H0: Delta = 0.

We additionally provide a *paired* McNemar variant (Test C):

  Test C. Train an ACQ-specialist on regex_role==ACQUIRER training rows and
          a TGT-specialist on regex_role==TARGET training rows. Apply BOTH
          specialists to the FULL locked test set. McNemar on the per-row
          correctness vectors tests "are the two role-trained specialists
          statistically distinguishable as classifiers?".

We report all three tests for both REGEX and NER-v2 role labellers.

Output: results/validation/mcnemar_acquirer_target.json
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
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from scipy import stats

sys.path.insert(0, r".\code\validation")
from ner_role_attribution_v2 import classify_role_v2

BASE = r"."
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(BASE, "results", "validation", "mcnemar_acquirer_target.json")

ACQ_PAT = re.compile(
    r"\b(acqui|acquir|takeover|buyer|buy[- ]?out|purchas|to acquire|will acquire|"
    r"completes acquisition|launches offer|tender offer|bid for|offers? to buy)\b",
    re.IGNORECASE)
TGT_PAT = re.compile(
    r"\b(target|to be acquir|being acquir|to be sold|sold to|sale of|divest|"
    r"subject of (an? )?bid|merger with|to merge with|received offer|"
    r"agree(s|d)? to be acquir)\b",
    re.IGNORECASE)


def regex_role(t):
    a = bool(ACQ_PAT.search(t)); g = bool(TGT_PAT.search(t))
    if a and g: return "BOTH"
    if a: return "ACQUIRER"
    if g: return "TARGET"
    return "NEITHER"


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def two_proportion_z(p1, n1, p2, n2):
    """Two-sample two-proportion z-test (Wald). Returns (z, p_two_sided).
    p1, p2 are proportions, n1, n2 are sample sizes."""
    if n1 < 2 or n2 < 2:
        return None, None
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_two = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_two)


def mcnemar_test(b, c):
    """Exact mid-p McNemar: pair-counts where (b) classifier-1 right & 2 wrong,
    (c) classifier-1 wrong & 2 right. Returns chi2_continuity, p_chi2,
    p_exact_binomial (two-sided)."""
    if b + c == 0:
        return {"b": int(b), "c": int(c), "chi2": 0.0, "p_chi2": 1.0,
                "p_exact_binomial": 1.0}
    # Continuity-corrected chi-square (Edwards correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_chi2 = 1 - stats.chi2.cdf(chi2, df=1)
    # Two-sided exact binomial test at p=0.5
    p_exact = 2 * min(stats.binom.cdf(min(b, c), b + c, 0.5),
                       1 - stats.binom.cdf(min(b, c) - 1, b + c, 0.5))
    p_exact = min(1.0, p_exact)
    return {"b": int(b), "c": int(c), "chi2": float(chi2),
            "p_chi2": float(p_chi2), "p_exact_binomial": float(p_exact)}


def block_bootstrap_delta(y_full, p_full, role_full, dates_full,
                          role_a="ACQUIRER", role_b="TARGET",
                          n_boot=2000, seed=42):
    """Weekly block bootstrap on Delta = MCC(role_a) - MCC(role_b).

    For each bootstrap iteration: sample WEEKS (with replacement), compute
    role-subset MCCs on the resampled set, take their difference.

    Returns observed Delta, CI95, mean/std of bootstrap distribution,
    p_two_sided (percentile of |Delta_boot| >= |observed Delta|)."""
    weeks = pd.to_datetime(dates_full).to_period("W").astype(str).values
    unique_weeks = np.unique(weeks)
    rng = np.random.default_rng(seed)

    def delta_on_subset(idx):
        y_sub = y_full[idx]; p_sub = p_full[idx]; r_sub = role_full[idx]
        ma = (r_sub == role_a)
        mb = (r_sub == role_b)
        if ma.sum() < 10 or mb.sum() < 10:
            return None
        if len(np.unique(y_sub[ma])) < 2 or len(np.unique(y_sub[mb])) < 2:
            return None
        return safe_mcc(y_sub[ma], p_sub[ma]) - safe_mcc(y_sub[mb], p_sub[mb])

    # Observed
    obs_delta = delta_on_subset(np.arange(len(y_full)))
    if obs_delta is None:
        return {"observed_delta": None, "note": "insufficient data"}

    boot_deltas = []
    for b in range(n_boot):
        sample_weeks = rng.choice(unique_weeks, size=len(unique_weeks), replace=True)
        idx = np.concatenate([np.where(weeks == w)[0] for w in sample_weeks])
        d = delta_on_subset(idx)
        if d is not None:
            boot_deltas.append(d)

    boot_deltas = np.array(boot_deltas)
    if len(boot_deltas) < n_boot * 0.5:
        note = f"only {len(boot_deltas)}/{n_boot} valid bootstraps"
    else:
        note = "ok"
    # Two-sided percentile p: under H0 Delta=0, bootstrap is centred at obs
    # so we use the standard "bootstrap-recentred" p
    recentred = boot_deltas - obs_delta
    p_two = float((np.abs(recentred) >= abs(obs_delta)).mean())
    return {
        "observed_delta": float(obs_delta),
        "boot_mean": float(boot_deltas.mean()),
        "boot_std": float(boot_deltas.std()),
        "ci95_lower": float(np.percentile(boot_deltas, 2.5)),
        "ci95_upper": float(np.percentile(boot_deltas, 97.5)),
        "p_two_sided_recentred": p_two,
        "n_valid_bootstraps": int(len(boot_deltas)),
        "n_boot_requested": n_boot,
        "note": note,
    }


def main():
    t0 = time.time()
    print("Loading data ...", flush=True)
    df = pd.read_parquet(DATA)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    ma = df[df["event"] == "mergers_acquisitions"].copy()
    print(f"M&A total: {len(ma)}", flush=True)

    VAL_END = pd.Timestamp("2025-06-01")
    train_full = ma[ma["published_date"] < VAL_END].copy().reset_index(drop=True)
    test_full = ma[ma["published_date"] >= VAL_END].copy().reset_index(drop=True)
    print(f"Train+val: {len(train_full)}  Test: {len(test_full)}", flush=True)

    # Regex roles
    train_full["role_regex"] = train_full["title_en"].apply(regex_role)
    test_full["role_regex"] = test_full["title_en"].apply(regex_role)

    # NER roles
    print("Loading spaCy en_core_web_sm ...", flush=True)
    nlp = spacy.load("en_core_web_sm")
    def tag_ner(df_local):
        roles = []
        for doc in nlp.pipe(df_local["title_en"].tolist(), batch_size=128):
            r, _ = classify_role_v2(doc, "")
            roles.append(r)
        return roles
    train_full["role_ner"] = tag_ner(train_full)
    test_full["role_ner"] = tag_ner(test_full)

    # ---------- Global specialist (paper protocol) ----------
    print("\n[Training GLOBAL M&A specialist (paper protocol)]", flush=True)
    tfidf_g = TfidfVectorizer(max_features=300, stop_words="english",
                              min_df=2, sublinear_tf=True)
    Xtr_g = tfidf_g.fit_transform(train_full["title_en"])
    Xte_g = tfidf_g.transform(test_full["title_en"])
    ytr = train_full["y"].values; yte = test_full["y"].values
    clf_g = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    clf_g.fit(Xtr_g, ytr)
    yp_g = clf_g.predict(Xte_g)
    overall_mcc = safe_mcc(yte, yp_g)
    print(f"  Overall test MCC = {overall_mcc:+.4f}", flush=True)

    results = {
        "meta": {
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_train_val_ma": int(len(train_full)),
            "n_test_ma": int(len(test_full)),
            "test_window": "2025-06-01 .. end",
            "global_specialist_hp": "TF-IDF max_features=300 sublinear_tf min_df=2; LR C=0.1 seed=42",
        },
        "overall_test_mcc_global": float(overall_mcc),
    }

    # ---------- Test A + B per labeller ----------
    for label_col, label_name in [("role_regex", "REGEX"), ("role_ner", "NER_v2")]:
        print(f"\n[Labeller: {label_name}]", flush=True)
        role_arr = test_full[label_col].values
        acq_mask = role_arr == "ACQUIRER"
        tgt_mask = role_arr == "TARGET"
        n_acq = int(acq_mask.sum()); n_tgt = int(tgt_mask.sum())
        print(f"  n_acquirer={n_acq}  n_target={n_tgt}", flush=True)

        # Per-role MCC & accuracy
        if n_acq > 0:
            mcc_acq = safe_mcc(yte[acq_mask], yp_g[acq_mask])
            acc_acq = float((yte[acq_mask] == yp_g[acq_mask]).mean())
        else:
            mcc_acq = None; acc_acq = None
        if n_tgt > 0:
            mcc_tgt = safe_mcc(yte[tgt_mask], yp_g[tgt_mask])
            acc_tgt = float((yte[tgt_mask] == yp_g[tgt_mask]).mean())
        else:
            mcc_tgt = None; acc_tgt = None

        # Test A: two-proportion z on accuracy
        if n_acq >= 2 and n_tgt >= 2:
            z, p_two = two_proportion_z(acc_acq, n_acq, acc_tgt, n_tgt)
        else:
            z, p_two = None, None

        # Test B: weekly block bootstrap on Delta MCC
        boot = block_bootstrap_delta(yte, yp_g, role_arr,
                                     test_full["published_date"].values,
                                     n_boot=2000, seed=42)

        results[label_name] = {
            "n_acquirer": n_acq, "n_target": n_tgt,
            "mcc_acquirer": mcc_acq, "mcc_target": mcc_tgt,
            "acc_acquirer": acc_acq, "acc_target": acc_tgt,
            "test_A_two_proportion_z": {"z": z, "p_two_sided": p_two,
                "note": "independent-samples Wald test on accuracy"},
            "test_B_block_bootstrap_delta_mcc": boot,
        }
        print(f"  MCC_acq = {mcc_acq}  MCC_tgt = {mcc_tgt}", flush=True)
        print(f"  Test A  (two-prop z on acc): z={z}  p_two={p_two}", flush=True)
        print(f"  Test B  (block-boot Delta MCC): obs={boot.get('observed_delta')} "
              f"CI95=[{boot.get('ci95_lower')}, {boot.get('ci95_upper')}] "
              f"p_two={boot.get('p_two_sided_recentred')}", flush=True)

    # ---------- Test C: McNemar via role-specific specialists on full test ----------
    print("\n[Test C: Paired McNemar via role-trained specialists]", flush=True)
    for label_col, label_name in [("role_regex", "REGEX"), ("role_ner", "NER_v2")]:
        tr_acq = train_full[train_full[label_col] == "ACQUIRER"]
        tr_tgt = train_full[train_full[label_col] == "TARGET"]
        n_tr_acq = len(tr_acq); n_tr_tgt = len(tr_tgt)
        # Need enough training rows AND both classes present
        if (n_tr_acq < 30 or n_tr_tgt < 30
                or tr_acq["y"].nunique() < 2 or tr_tgt["y"].nunique() < 2):
            print(f"  {label_name}: insufficient training data for paired specialists "
                  f"(n_acq_train={n_tr_acq}, n_tgt_train={n_tr_tgt})", flush=True)
            results.setdefault(label_name, {})["test_C_mcnemar_paired"] = {
                "note": "skipped: insufficient training rows or single-class",
                "n_train_acq": n_tr_acq, "n_train_tgt": n_tr_tgt,
            }
            continue

        tfidf_a = TfidfVectorizer(max_features=300, stop_words="english",
                                  min_df=2, sublinear_tf=True)
        Xa_tr = tfidf_a.fit_transform(tr_acq["title_en"])
        Xa_te = tfidf_a.transform(test_full["title_en"])
        clf_a = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        clf_a.fit(Xa_tr, tr_acq["y"].values)
        yp_a = clf_a.predict(Xa_te)

        tfidf_t = TfidfVectorizer(max_features=300, stop_words="english",
                                  min_df=2, sublinear_tf=True)
        Xt_tr = tfidf_t.fit_transform(tr_tgt["title_en"])
        Xt_te = tfidf_t.transform(test_full["title_en"])
        clf_t = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        clf_t.fit(Xt_tr, tr_tgt["y"].values)
        yp_t = clf_t.predict(Xt_te)

        # Correctness vectors
        corr_a = (yp_a == yte).astype(int)
        corr_t = (yp_t == yte).astype(int)
        b = int(((corr_a == 1) & (corr_t == 0)).sum())  # acq-right, tgt-wrong
        c = int(((corr_a == 0) & (corr_t == 1)).sum())  # acq-wrong, tgt-right
        mn = mcnemar_test(b, c)
        mcc_a_full = safe_mcc(yte, yp_a)
        mcc_t_full = safe_mcc(yte, yp_t)
        results.setdefault(label_name, {})["test_C_mcnemar_paired"] = {
            "n_train_acq": n_tr_acq, "n_train_tgt": n_tr_tgt,
            "n_test": int(len(yte)),
            "mcc_acq_specialist_full_test": float(mcc_a_full),
            "mcc_tgt_specialist_full_test": float(mcc_t_full),
            "mcnemar": mn,
            "interpretation": (
                "b = #articles where acq-specialist correct AND tgt-specialist wrong; "
                "c = #articles where acq-specialist wrong AND tgt-specialist correct. "
                "p > 0.05 means the two role-trained specialists are not "
                "statistically distinguishable on the full test set."
            )
        }
        print(f"  {label_name}: MCC_acq_spec={mcc_a_full:.4f}  "
              f"MCC_tgt_spec={mcc_t_full:.4f}  "
              f"McNemar b={b} c={c} p_chi2={mn['p_chi2']:.4f} "
              f"p_exact={mn['p_exact_binomial']:.4f}", flush=True)

    # Overall interpretation
    results["interpretation_summary"] = {
        "test_A_summary": (
            "Two-proportion z-test on accuracy is the proper test for "
            "INDEPENDENT subsamples. Reports z and 2-sided p for "
            "acquirer-vs-target accuracy on the same locked test."
        ),
        "test_B_summary": (
            "Weekly block-bootstrap CI on Delta MCC is the proper test for "
            "INDEPENDENT subsamples that controls for within-week correlation."
        ),
        "test_C_summary": (
            "Classical paired McNemar via training role-specific specialists "
            "and comparing their per-row correctness on the FULL test set. "
            "This tests whether the two role-trained classifiers are themselves "
            "statistically distinguishable, NOT whether the global model "
            "shows asymmetry on the partitioned test set."
        ),
        "why_no_DM_test": (
            "Diebold-Mariano test requires two competing forecasts on the "
            "SAME time series. Acquirer and target are disjoint subsets of "
            "test articles, not competing forecasts on the same series; DM "
            "is not the appropriate test here."
        ),
        "elapsed_seconds": float(time.time() - t0),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
