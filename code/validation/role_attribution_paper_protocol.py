"""
NER role attribution comparison — using PAPER PROTOCOL.

The paper actually evaluates a single GLOBAL M&A specialist on per-role subsets
of the test set (acquirer_target_sanity.py line ~95-100). The paper text at
line 380 mis-describes this as "role-conditional specialists"; the underlying
code is "global model, per-role test split".

This script:
    1. Tags M&A articles with both REGEX and NER v2 partitions.
    2. Trains a SINGLE global M&A TF-IDF + LR specialist (same HP as
       acquirer_target_sanity).
    3. Evaluates that single model on test set partitioned BOTH ways.
    4. Reports MCC per role per partition.
    5. Runs 10K permutation test on each role's MCC.

Question answered: does ACQUIRER > TARGET asymmetry SURVIVE when we replace
regex role labels with NER+dep-parse labels?

Output: results/validation/role_attribution_paper_protocol.json
"""
import os
import sys
import io
import json
import time
import warnings
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

# Re-import the v2 classifier (DRY: would normally import from a module)
sys.path.insert(0, r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package\code\validation")
from ner_role_attribution_v2 import classify_role_v2, ACQ_VERBS, SELL_VERBS, DEAL_NOUNS

BASE = r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package"
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(BASE, "results", "validation", "role_attribution_paper_protocol.json")

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

    # Tag train and test with both regex and NER
    train_full["role_regex"] = train_full["title_en"].apply(regex_role)
    test_full["role_regex"] = test_full["title_en"].apply(regex_role)

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

    print("\n[Test set role counts]", flush=True)
    print(f"  regex: {test_full['role_regex'].value_counts().to_dict()}", flush=True)
    print(f"  NER:   {test_full['role_ner'].value_counts().to_dict()}", flush=True)

    # Train GLOBAL M&A specialist (paper's protocol)
    print("\n[Training GLOBAL M&A specialist (paper protocol)]", flush=True)
    tfidf = TfidfVectorizer(max_features=300, stop_words="english",
                            min_df=2, sublinear_tf=True)
    Xtr = tfidf.fit_transform(train_full["title_en"])
    Xte = tfidf.transform(test_full["title_en"])
    ytr = train_full["y"].values; yte = test_full["y"].values
    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    clf.fit(Xtr, ytr)
    yp_te = clf.predict(Xte); yp_proba = clf.predict_proba(Xte)[:, 1]
    overall_mcc = safe_mcc(yte, yp_te)
    print(f"  Overall test MCC = {overall_mcc:+.4f} (paper headline)", flush=True)

    def eval_partition(partition_col, role_label):
        mask = (test_full[partition_col] == role_label).values
        n = int(mask.sum())
        if n < 20:
            return {"n": n, "mcc": None, "balacc": None,
                    "true_up_frac": None, "pred_up_frac": None,
                    "proba_std": None}
        sub_y = yte[mask]; sub_p = yp_te[mask]; sub_pr = yp_proba[mask]
        if len(np.unique(sub_y)) < 2:
            return {"n": n, "mcc": None, "balacc": None,
                    "true_up_frac": float(sub_y.mean()),
                    "pred_up_frac": float(sub_p.mean()),
                    "proba_std": float(sub_pr.std()),
                    "note": "single-class test"}
        mcc = safe_mcc(sub_y, sub_p)
        bacc = balanced_accuracy_score(sub_y, sub_p)
        return {"n": n, "mcc": float(mcc), "balacc": float(bacc),
                "true_up_frac": float(sub_y.mean()),
                "pred_up_frac": float(sub_p.mean()),
                "proba_std": float(sub_pr.std())}

    def perm_test(partition_col, role_label, n_perm=10000):
        mask = (test_full[partition_col] == role_label).values
        sub_y = yte[mask]; sub_p = yp_te[mask]
        if mask.sum() < 30 or len(np.unique(sub_y)) < 2:
            return None
        obs = safe_mcc(sub_y, sub_p)
        rng = np.random.default_rng(42)
        null = np.array([safe_mcc(rng.permutation(sub_y), sub_p)
                         for _ in range(n_perm)])
        return {
            "n": int(mask.sum()), "observed_mcc": obs,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "z_score": float((obs - null.mean()) / (null.std() + 1e-12)),
            "p_one_sided": float((null >= obs).mean()),
            "p_two_sided": float((np.abs(null) >= abs(obs)).mean()),
            "n_perm": n_perm,
        }

    print("\n[REGEX partition]")
    res_regex = {}
    for r in ["ACQUIRER", "TARGET", "BOTH", "NEITHER"]:
        m = eval_partition("role_regex", r)
        p = perm_test("role_regex", r)
        res_regex[r] = {"eval": m, "perm": p}
        print(f"  {r}: {m}", flush=True)
        if p is not None:
            print(f"     perm p_two={p['p_two_sided']:.4f}  z={p['z_score']:.2f}", flush=True)

    print("\n[NER v2 partition]")
    res_ner = {}
    for r in ["ACQUIRER", "TARGET", "BOTH", "AMBIGUOUS",
              "UNCLEAR_NO_ANCHOR", "UNCLEAR_NO_ORG"]:
        m = eval_partition("role_ner", r)
        p = perm_test("role_ner", r)
        res_ner[r] = {"eval": m, "perm": p}
        print(f"  {r}: {m}", flush=True)
        if p is not None:
            print(f"     perm p_two={p['p_two_sided']:.4f}  z={p['z_score']:.2f}", flush=True)

    print("\n[ASYMMETRY CHECK]")
    print(f"  REGEX: ACQUIRER MCC = {res_regex['ACQUIRER']['eval']['mcc']}, "
          f"TARGET MCC = {res_regex['TARGET']['eval']['mcc']}", flush=True)
    print(f"  NER v2: ACQUIRER MCC = {res_ner['ACQUIRER']['eval']['mcc']}, "
          f"TARGET MCC = {res_ner['TARGET']['eval']['mcc']}", flush=True)

    # Cross-tab on test
    cross_test = pd.crosstab(test_full["role_regex"], test_full["role_ner"])
    print("\n[Test-set cross-tab: regex (rows) vs NER (cols)]")
    print(cross_test.to_string())

    out = {
        "meta": {"timestamp": pd.Timestamp.now().isoformat(),
                 "elapsed_s": float(time.time() - t0),
                 "n_train_val_ma": int(len(train_full)),
                 "n_test_ma": int(len(test_full)),
                 "method_global_specialist": "TF-IDF max_features=300, stop_words=english, min_df=2, sublinear_tf=True; LR C=0.1, seed=42"},
        "overall_test_mcc": float(overall_mcc),
        "regex_partition": res_regex,
        "ner_v2_partition": res_ner,
        "test_set_cross_tab": cross_test.to_dict(),
        "interpretation": {
            "asymmetry_under_regex": res_regex['ACQUIRER']['eval']['mcc'] != res_regex['TARGET']['eval']['mcc'],
            "asymmetry_under_ner": (res_ner['ACQUIRER']['eval']['mcc'] is not None and
                                     res_ner['TARGET']['eval']['mcc'] is not None and
                                     res_ner['ACQUIRER']['eval']['mcc'] != res_ner['TARGET']['eval']['mcc']),
            "regex_acq_minus_target": (res_regex['ACQUIRER']['eval']['mcc'] - res_regex['TARGET']['eval']['mcc'])
                if (res_regex['ACQUIRER']['eval']['mcc'] is not None and res_regex['TARGET']['eval']['mcc'] is not None) else None,
            "ner_acq_minus_target": (res_ner['ACQUIRER']['eval']['mcc'] - res_ner['TARGET']['eval']['mcc'])
                if (res_ner['ACQUIRER']['eval']['mcc'] is not None and res_ner['TARGET']['eval']['mcc'] is not None) else None,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
