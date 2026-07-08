"""Tier B local item B1: Cutoff perturbation sensitivity test.

Question: How sensitive is the M&A test MCC to the choice of train/test cutoff?
Method: Vary the train cutoff date by +/- 2 weeks; for each cutoff, refit the
        val-best HP and report locked-test MCC. If MCC stays stable across
        cutoffs, the result is robust; if it swings wildly, it's an artifact.

Original cutoff: train < 2025-04-01, val 2025-04 to 2025-05, test >= 2025-06-01.
We perturb the train cutoff by +/- 7 and +/- 14 days, holding test fixed.

Authoritative HP (from ma_hp_sweep_extended.json):
  max_features=100, C=5.0, sublinear_tf=False, min_df=2, ngram_range=(1,1)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "data" / "classifier_training_v2.parquet"
OUT_PATH = REPO / "results" / "validation" / "tierB_cutoff_perturbation.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MA_KEYWORDS = (
    r"merger|acquisition|acquire[sd]?|acquiring|takeover|buyout|"
    r"acquisition agreement|definitive agreement"
)

HP = dict(max_features=100, C=5.0, sublinear_tf=False, min_df=2, ngram_range=(1, 1))


def load_ma_subset() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    return df[df["event"] == "mergers_acquisitions"].reset_index(drop=True)


def evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    vec = TfidfVectorizer(
        max_features=HP["max_features"],
        sublinear_tf=HP["sublinear_tf"],
        min_df=HP["min_df"],
        ngram_range=HP["ngram_range"],
        stop_words="english",
    )
    Xtr = vec.fit_transform(train_df["title_en"].astype(str))
    Xte = vec.transform(test_df["title_en"].astype(str))
    ytr = train_df["y"].to_numpy()
    yte = test_df["y"].to_numpy()
    clf = LogisticRegression(C=HP["C"], max_iter=2000, random_state=42)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    return dict(
        n_train=int(len(ytr)),
        n_test=int(len(yte)),
        test_mcc=float(matthews_corrcoef(yte, yp)),
        test_balacc=float(balanced_accuracy_score(yte, yp)),
        pos_rate_train=float(ytr.mean()),
        pos_rate_test=float(yte.mean()),
    )


def main() -> None:
    ma = load_ma_subset()
    print(f"Loaded M&A subset: {len(ma):,} rows")

    test_start = pd.Timestamp("2025-06-01")
    test_df = ma[ma["published_date"] >= test_start].reset_index(drop=True)
    print(f"Locked test (>=2025-06-01): n={len(test_df)}")

    perturbations_days = [-14, -7, 0, +7, +14]
    base_cutoff = pd.Timestamp("2025-04-01")

    results = []
    for d in perturbations_days:
        cutoff = base_cutoff + pd.Timedelta(days=d)
        train_df = ma[ma["published_date"] < cutoff].reset_index(drop=True)
        result = evaluate(train_df, test_df)
        result["cutoff_offset_days"] = d
        result["cutoff_date"] = str(cutoff.date())
        results.append(result)
        print(
            f"  offset={d:+3d}d cutoff={cutoff.date()} n_tr={result['n_train']:4d} "
            f"test_mcc={result['test_mcc']:+.4f} balacc={result['test_balacc']:.4f}"
        )

    mccs = [r["test_mcc"] for r in results]
    summary = dict(
        hp=HP,
        test_n=int(len(test_df)),
        results=results,
        mcc_min=float(min(mccs)),
        mcc_max=float(max(mccs)),
        mcc_mean=float(np.mean(mccs)),
        mcc_std=float(np.std(mccs, ddof=1)),
        mcc_range=float(max(mccs) - min(mccs)),
    )
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print(f"Test MCC range across cutoff perturbations: "
          f"[{summary['mcc_min']:+.4f}, {summary['mcc_max']:+.4f}] "
          f"(mean={summary['mcc_mean']:+.4f}, std={summary['mcc_std']:.4f})")


if __name__ == "__main__":
    main()
