"""Tier B local item B3: Earnings/Guidance event negative control.

Question: Is the M&A signal special, or do other large-volume event types also
show a positive temporal MCC if we val-tune them as carefully?
Method: Take the largest non-M&A event types (earnings, guidance, dividend)
        and run the same val-then-test pipeline used for M&A. Honest negative
        controls bolster the claim that M&A is genuinely special.

Authoritative M&A HP for reference (val MCC=0.228, test MCC=0.138):
  max_features=100, C=5.0, sublinear_tf=False, min_df=2, ngram_range=(1,1)
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "data" / "classifier_training_v2.parquet"
OUT_PATH = REPO / "results" / "validation" / "tierB_event_neg_control.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

EVENT_TARGETS = [
    "mergers_acquisitions",
    "earnings_releases_and_operating_results",
    "financial_results",
    "changes_in_companys_own_shares",
    "partnerships",
    "regulatory_filings",
    "clinical_study",
    "management_changes",
]

HP_GRID = dict(
    max_features=[50, 100, 200, 500, 1000],
    C=[0.1, 0.5, 1.0, 5.0],
    sublinear_tf=[True, False],
    min_df=[1, 2],
    ngram_range=[(1, 1), (1, 2)],
)


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    return df.reset_index(drop=True)


def filter_event(df: pd.DataFrame, event: str) -> pd.DataFrame:
    return df[df["event"] == event].reset_index(drop=True)


def split_chrono(df: pd.DataFrame):
    train_cut = pd.Timestamp("2025-04-01")
    val_end = pd.Timestamp("2025-06-01")
    tr = df[df["published_date"] < train_cut].reset_index(drop=True)
    va = df[(df["published_date"] >= train_cut) & (df["published_date"] < val_end)].reset_index(drop=True)
    te = df[df["published_date"] >= val_end].reset_index(drop=True)
    return tr, va, te


def fit_eval(tr: pd.DataFrame, ev: pd.DataFrame, hp: dict) -> dict:
    if len(tr) < 50 or len(ev) < 30:
        return dict(mcc=None, n_train=int(len(tr)), n_eval=int(len(ev)),
                    note="insufficient samples")
    vec = TfidfVectorizer(
        max_features=hp["max_features"], min_df=hp["min_df"],
        sublinear_tf=hp["sublinear_tf"], ngram_range=hp["ngram_range"],
        stop_words="english",
    )
    Xtr = vec.fit_transform(tr["title_en"].astype(str))
    Xev = vec.transform(ev["title_en"].astype(str))
    ytr = tr["y"].to_numpy()
    yev = ev["y"].to_numpy()
    if len(np.unique(ytr)) < 2:
        return dict(mcc=0.0, n_train=int(len(tr)), n_eval=int(len(ev)),
                    note="single-class train")
    clf = LogisticRegression(C=hp["C"], max_iter=2000, random_state=42)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xev)
    return dict(
        mcc=float(matthews_corrcoef(yev, yp)),
        balacc=float(balanced_accuracy_score(yev, yp)),
        n_train=int(len(tr)), n_eval=int(len(ev)),
        pos_rate=float(yev.mean()),
        pred_pos_rate=float(yp.mean()),
    )


def val_select_then_test(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame) -> dict:
    best_val = None
    best_hp = None
    n_cells = 0
    for mf, c, sl, md, ng in product(
        HP_GRID["max_features"], HP_GRID["C"], HP_GRID["sublinear_tf"],
        HP_GRID["min_df"], HP_GRID["ngram_range"]
    ):
        n_cells += 1
        hp = dict(max_features=mf, C=c, sublinear_tf=sl, min_df=md, ngram_range=ng)
        v = fit_eval(tr, va, hp)
        if v.get("mcc") is None:
            continue
        if best_val is None or v["mcc"] > best_val["mcc"]:
            best_val = v
            best_hp = hp
    if best_hp is None:
        return dict(error="no valid val cell")
    test_result = fit_eval(tr, te, best_hp)
    return dict(
        n_grid_cells=n_cells,
        best_hp=best_hp,
        val_mcc=best_val["mcc"],
        val_balacc=best_val.get("balacc"),
        test_mcc=test_result.get("mcc"),
        test_balacc=test_result.get("balacc"),
        n_train=test_result.get("n_train"),
        n_test=test_result.get("n_eval"),
    )


def main() -> None:
    df = load_data()
    summary = {"hp_grid_size": (
        len(HP_GRID["max_features"]) * len(HP_GRID["C"]) *
        len(HP_GRID["sublinear_tf"]) * len(HP_GRID["min_df"]) *
        len(HP_GRID["ngram_range"])
    ), "events": {}}
    for ev_name in EVENT_TARGETS:
        sub = filter_event(df, ev_name)
        tr, va, te = split_chrono(sub)
        print(f"\n{ev_name}: total n={len(sub)}, train={len(tr)}, val={len(va)}, test={len(te)}")
        if len(va) < 30 or len(te) < 30:
            print(f"  skipped (insufficient val/test)")
            summary["events"][ev_name] = dict(error="insufficient", n_total=len(sub))
            continue
        result = val_select_then_test(tr, va, te)
        summary["events"][ev_name] = result
        print(f"  best_hp={result.get('best_hp')}")
        print(f"  val_mcc={result.get('val_mcc'):+.4f} -> test_mcc={result.get('test_mcc'):+.4f}")

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
