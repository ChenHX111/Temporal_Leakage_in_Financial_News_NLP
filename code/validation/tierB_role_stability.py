"""Tier B local item B2: Acquirer-target role stability test.

Question: How sensitive is the acquirer/target asymmetry finding to the regex
patterns used to assign roles?
Method: Bootstrap role-assignment regex (vary acquirer/target keyword sets);
        for each pattern variant, recompute role-conditional MCC on locked test.

Original regexes (from acquirer_target_sanity.py):
  ACQUIRER: r"\b(acquir|buy|takeover|takes\s+over|to\s+acquire|"
            r"completes\s+(?:the\s+)?acquisition\s+of)\b"
  TARGET:   r"\b(to\s+be\s+acquired|sold\s+to|merger\s+agreement\s+with|"
            r"agree(?:s|d|ment)?\s+to\s+(?:be\s+)?(?:acqui|merg))\b"

Variants tested:
  V1 (original): exhaustive multi-pattern
  V2 (minimal):  just "acquir" / "to be acquired"
  V3 (broad):    add "purchase", "deal with", "merger"
  V4 (strict):   only with explicit deal-stage words ("completes", "definitive")
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "data" / "classifier_training_v2.parquet"
OUT_PATH = REPO / "results" / "validation" / "tierB_role_stability.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MA_KEYWORDS = (
    r"merger|acquisition|acquire[sd]?|acquiring|takeover|buyout|"
    r"acquisition agreement|definitive agreement"
)

ROLE_VARIANTS = {
    "V1_original": dict(
        ACQUIRER=r"\b(acquir|buy|takeover|takes\s+over|to\s+acquire|completes\s+(?:the\s+)?acquisition\s+of)\b",
        TARGET=r"\b(to\s+be\s+acquired|sold\s+to|merger\s+agreement\s+with|agree(?:s|d|ment)?\s+to\s+(?:be\s+)?(?:acqui|merg))\b",
    ),
    "V2_minimal": dict(
        ACQUIRER=r"\b(acquir|to\s+acquire)\b",
        TARGET=r"\b(to\s+be\s+acquired)\b",
    ),
    "V3_broad": dict(
        ACQUIRER=r"\b(acquir|buy|purchase|takeover|takes\s+over|to\s+acquire|deal\s+with|merger\s+with)\b",
        TARGET=r"\b(to\s+be\s+acquired|sold\s+to|merger\s+agreement\s+with|acquired\s+by|sells\s+itself)\b",
    ),
    "V4_strict": dict(
        ACQUIRER=r"\b(completes\s+(?:the\s+)?acquisition\s+of|definitive\s+(?:agreement|merger))\b",
        TARGET=r"\b(to\s+be\s+acquired|sold\s+to|definitive\s+(?:agreement|merger)\s+to\s+be\s+acquired)\b",
    ),
}

HP = dict(max_features=100, C=5.0, sublinear_tf=False, min_df=2, ngram_range=(1, 1))


def load_ma() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    return df[df["event"] == "mergers_acquisitions"].reset_index(drop=True)


def assign_roles(titles: pd.Series, acq_re: str, tgt_re: str) -> pd.Series:
    is_acq = titles.str.contains(acq_re, flags=re.IGNORECASE, regex=True, na=False)
    is_tgt = titles.str.contains(tgt_re, flags=re.IGNORECASE, regex=True, na=False)
    role = pd.Series("NEITHER", index=titles.index)
    role[is_acq & ~is_tgt] = "ACQUIRER"
    role[is_tgt & ~is_acq] = "TARGET"
    role[is_acq & is_tgt] = "BOTH"
    return role


def fit_eval_role(train_df: pd.DataFrame, test_df: pd.DataFrame, role: str) -> dict:
    train_sub = train_df[train_df["role"] == role]
    test_sub = test_df[test_df["role"] == role]
    if len(train_sub) < 30 or len(test_sub) < 30:
        return dict(role=role, n_train=int(len(train_sub)), n_test=int(len(test_sub)),
                    test_mcc=None, note="too few samples")
    vec = TfidfVectorizer(max_features=HP["max_features"], min_df=HP["min_df"],
                          sublinear_tf=HP["sublinear_tf"], ngram_range=HP["ngram_range"],
                          stop_words="english")
    Xtr = vec.fit_transform(train_sub["title_en"].astype(str))
    Xte = vec.transform(test_sub["title_en"].astype(str))
    ytr = train_sub["y"].to_numpy()
    yte = test_sub["y"].to_numpy()
    clf = LogisticRegression(C=HP["C"], max_iter=2000, random_state=42)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    return dict(role=role, n_train=int(len(ytr)), n_test=int(len(yte)),
                test_mcc=float(matthews_corrcoef(yte, yp)),
                pos_rate_test=float(yte.mean()),
                pred_pos_rate=float(yp.mean()))


def main() -> None:
    ma = load_ma()
    test_start = pd.Timestamp("2025-06-01")
    train_cut = pd.Timestamp("2025-04-01")
    train_df = ma[ma["published_date"] < train_cut].reset_index(drop=True)
    test_df = ma[ma["published_date"] >= test_start].reset_index(drop=True)

    print(f"M&A train n={len(train_df)}, test n={len(test_df)}")

    summary = {"hp": HP, "variants": {}}
    for variant_name, regexes in ROLE_VARIANTS.items():
        train_df["role"] = assign_roles(train_df["title_en"].astype(str),
                                          regexes["ACQUIRER"], regexes["TARGET"])
        test_df["role"] = assign_roles(test_df["title_en"].astype(str),
                                          regexes["ACQUIRER"], regexes["TARGET"])
        train_dist = train_df["role"].value_counts().to_dict()
        test_dist = test_df["role"].value_counts().to_dict()
        roles = {}
        for r in ["ACQUIRER", "TARGET", "NEITHER"]:
            roles[r] = fit_eval_role(train_df, test_df, r)
        summary["variants"][variant_name] = dict(
            regex=regexes,
            train_role_dist=train_dist,
            test_role_dist=test_dist,
            results=roles,
        )
        acq = roles["ACQUIRER"]
        tgt = roles["TARGET"]
        print(f"\n{variant_name}:")
        print(f"  train roles: {train_dist}")
        print(f"  test roles: {test_dist}")
        print(f"  ACQUIRER n_test={acq['n_test']:3d} mcc={acq.get('test_mcc')}")
        print(f"  TARGET   n_test={tgt['n_test']:3d} mcc={tgt.get('test_mcc')}")

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
