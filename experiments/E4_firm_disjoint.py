"""
E4 — Firm-disjoint (entity-holdout) M&A robustness (answers tkJL-W1 "unknown artifact of private data" / firm memorization).

Two complementary tests, both faithful to the paper M&A specialist HP:
  (A) TEMPORAL + firm-disjoint: paper temporal split, but DROP test rows whose yf_ticker also appears in train
      -> the signal is measured only on UNSEEN firms. If MCC stays positive, it is not firm-identity memorization.
  (B) GROUPED firm-holdout CV: GroupKFold by yf_ticker (5 folds) -> no ticker in both train and test in any fold;
      report mean MCC (ignores time, isolates cross-firm generalization).
Complements the paper's existing ORG-masking result (+0.045: masking firm tokens HURTS M&A).
Out: out/E4_firm_disjoint.json
"""
import os, io, sys, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GroupKFold

BASE = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "E4_firm_disjoint.json")
PAPER_HP = dict(max_features=100, sublinear_tf=False, min_df=2, ngram_range=(1, 1), stop_words="english")
TRAIN_END, VAL_END = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-01")


def mcc(y, p):
    return 0.0 if (len(np.unique(y)) < 2 or len(np.unique(p)) < 2) else float(matthews_corrcoef(y, p))


def fit_eval(tr_txt, tr_y, te_txt, te_y):
    tf = TfidfVectorizer(**PAPER_HP)
    Xtr = tf.fit_transform(tr_txt); Xte = tf.transform(te_txt)
    return mcc(te_y, LogisticRegression(max_iter=2000, C=5.0, random_state=42).fit(Xtr, tr_y).predict(Xte))


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    df["yf_ticker"] = df["yf_ticker"].fillna("NA").astype(str)
    ma = df[df["event"] == "mergers_acquisitions"].copy()

    tr = ma[ma["published_date"] < TRAIN_END]
    te = ma[ma["published_date"] >= VAL_END].copy()
    base = fit_eval(tr["title_en"], tr["y"].values, te["title_en"], te["y"].values)

    # (A) temporal + firm-disjoint test (unseen tickers only)
    train_tickers = set(tr["yf_ticker"].unique())
    te_unseen = te[~te["yf_ticker"].isin(train_tickers)]
    overlap_frac = float(te["yf_ticker"].isin(train_tickers).mean())
    a = {"paper_temporal_mcc": base, "n_test_full": int(len(te)),
         "test_ticker_overlap_frac": overlap_frac,
         "n_test_unseen_firms": int(len(te_unseen))}
    if len(te_unseen) >= 40 and te_unseen["y"].nunique() == 2:
        a["temporal_firm_disjoint_mcc"] = fit_eval(tr["title_en"], tr["y"].values,
                                                   te_unseen["title_en"], te_unseen["y"].values)
    else:
        a["temporal_firm_disjoint_mcc"] = None; a["note"] = "too few unseen-firm test rows"
    res = {"A_temporal_firm_disjoint": a}
    print("A: paper_temporal=%.4f  firm-disjoint(unseen firms, n=%d)=%s  [test overlap %.2f]" %
          (base, a["n_test_unseen_firms"], a["temporal_firm_disjoint_mcc"], overlap_frac), flush=True)

    # (B) grouped firm-holdout CV (GroupKFold by ticker)
    g = ma.reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)
    fold_mcc = []
    for tri, tei in gkf.split(g["title_en"], g["y"], groups=g["yf_ticker"]):
        if g["y"].values[tri].sum() in (0, len(tri)) or g["y"].values[tei].sum() in (0, len(tei)):
            continue
        fold_mcc.append(fit_eval(g["title_en"].values[tri], g["y"].values[tri],
                                 g["title_en"].values[tei], g["y"].values[tei]))
    res["B_grouped_firm_holdout_cv"] = {"mean_mcc": float(np.mean(fold_mcc)), "std": float(np.std(fold_mcc)),
                                        "n_folds": len(fold_mcc), "per_fold": [round(x, 4) for x in fold_mcc]}
    print("B: grouped firm-holdout CV mean MCC = %.4f +/- %.4f (%d folds)" %
          (np.mean(fold_mcc), np.std(fold_mcc), len(fold_mcc)), flush=True)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
