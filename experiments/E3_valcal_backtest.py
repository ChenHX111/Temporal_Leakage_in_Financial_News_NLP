"""
E3 — Validation-calibrated backtest (answers tkJL-C1 / BCS9-C6).

The paper's top-quartile Sharpe uses an EX-POST test-set confidence quantile. Here we compute the confidence
threshold on the VALIDATION set (75th pctile of |p-0.5| on val) and APPLY it to test, which removes the ex-post
*selection* concern (val vs test threshold differ by only ~0.0004). The economic Sharpe (+2.33 @10bps vs all-trade
-0.60) is still an explicit UPPER BOUND because slippage/market-impact are unmodelled.

Paper-authoritative M&A specialist (cost_aware_backtest_paperHP.py). Out: out/E3_valcal_backtest.json
"""
import os, io, sys, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

BASE = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "E3_valcal_backtest.json")
PAPER_HP = dict(max_features=100, sublinear_tf=False, min_df=2, ngram_range=(1, 1), stop_words="english")
TRAIN_END, VAL_END = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-01")
COSTS = [0, 2, 5, 10, 20, 50]


def sharpe_of(sub_df, signed_ret, tc):
    r = signed_ret - 2 * tc / 10000.0
    d = sub_df.copy(); d["r"] = r; d["date"] = pd.to_datetime(d["published_date"]).dt.date
    s = d.groupby("date")["r"].mean().dropna()
    return 0.0 if s.std() < 1e-12 else float(s.mean() / s.std() * np.sqrt(252))


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["pcp"] = pd.to_numeric(df["price_change_percentage"], errors="coerce")
    ma = df[df["event"] == "mergers_acquisitions"].copy()
    tr = ma[ma["published_date"] < TRAIN_END]
    val = ma[(ma["published_date"] >= TRAIN_END) & (ma["published_date"] < VAL_END)]
    te = ma[(ma["published_date"] >= VAL_END) & ma["pcp"].notna()].copy()

    tf = TfidfVectorizer(**PAPER_HP)
    Xtr = tf.fit_transform(tr["title_en"].fillna(""))
    clf = LogisticRegression(max_iter=2000, C=5.0, random_state=42).fit(Xtr, tr["y"].values)
    p_val = clf.predict_proba(tf.transform(val["title_en"].fillna("")))[:, 1]
    p_te = clf.predict_proba(tf.transform(te["title_en"].fillna("")))[:, 1]
    yp_te = (p_te >= 0.5).astype(int)
    conf_te = np.abs(p_te - 0.5)

    thr_val = float(np.percentile(np.abs(p_val - 0.5), 75))     # deployable: calibrated on VAL
    thr_test = float(np.percentile(conf_te, 75))                # paper: ex-post on TEST
    ret = te["pcp"].values / 100.0
    signed = np.where(yp_te == 1, 1.0, -1.0) * ret

    def grid(mask):
        sub = te[mask]; sr = signed[mask]
        return {f"{c}bps": round(sharpe_of(sub, sr, c), 4) for c in COSTS} | {"n": int(mask.sum())}

    res = {
        "thresholds": {"val_calibrated_75pct": thr_val, "test_expost_75pct": thr_test,
                       "abs_diff": round(abs(thr_val - thr_test), 4)},
        "all_trade": grid(np.ones(len(te), bool)),
        "top25_val_calibrated": grid(conf_te >= thr_val),   # val-calibrated (removes ex-post selection; still upper bound)
        "top25_test_expost": grid(conf_te >= thr_test),     # paper ex-post upper bound
        "locked_test_mcc": float(matthews_corrcoef(te["y"].values, yp_te)),
        "n_train": int(len(tr)), "n_val": int(len(val)), "n_test": int(len(te)),
    }
    print("thresholds val=%.4f test=%.4f (|diff|=%.4f)" % (thr_val, thr_test, abs(thr_val - thr_test)), flush=True)
    print("all-trade  Sharpe:", res["all_trade"], flush=True)
    print("top25 VAL-calibrated (removes ex-post selection):", res["top25_val_calibrated"], flush=True)
    print("top25 TEST-expost (paper upper bd):", res["top25_test_expost"], flush=True)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
