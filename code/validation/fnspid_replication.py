"""
FNSPID third-corpus M&A replication — addresses W9 (single regime objection).

FNSPID (Zihan1004 on HuggingFace) is a large financial news dataset by Dong et
al. (NeurIPS 2024) covering 2020-2023, sourced from Benzinga. Schema:
    Date, Article_title, Stock_symbol, Url, Publisher

We do NOT have returns -> compute via yfinance bulk fetch.

Goal: replicate the random-vs-temporal leakage audit pattern AND the M&A
specialist headline on a THIRD independent corpus (our proprietary + EDT +
FNSPID = 3 datasets, 3 regimes).

Pipeline:
    1. Stream FNSPID, filter to titles containing M&A keywords (narrow definition
       to match our paper). Cap at MAX_MA_SAMPLES.
    2. Get unique tickers; bulk-fetch price history from yfinance for the
       sampled period.
    3. Compute next-day return label (UP if next-day close > today close).
    4. Apply our temporal split (train < 2023-07-01, test >= 2023-07-01).
    5. Run audit: TF-IDF + (LR, RF, GB) random vs temporal.
    6. Also run the M&A specialist (val-selected HP for M&A from paper).

Output: results/validation/fnspid_replication.json + .md
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
from datetime import datetime, timedelta

BASE = r"."
EXTERNAL_DIR = os.path.join(BASE, "data", "external")
SAMPLE_PARQUET = os.path.join(EXTERNAL_DIR, "fnspid_ma_sample.parquet")
LABELED_PARQUET = os.path.join(EXTERNAL_DIR, "fnspid_ma_labeled.parquet")
OUT = os.path.join(BASE, "results", "validation", "fnspid_replication.json")

# Narrow M&A keyword filter (matches our paper's narrow definition)
MA_KEYWORDS_RE = re.compile(
    r"\b(acquir\w*|acquisition|takeover|to acquire|will acquire|"
    r"merger|to merge|completes acquisition|launches takeover|"
    r"tender offer|buyout|to buy out|agree(s|d)? to acquire)\b",
    re.IGNORECASE)

MAX_MA_SAMPLES = 5000  # cap for tractability
DATE_MIN = "2020-01-01"
DATE_MAX = "2023-12-31"
TEMPORAL_TRAIN_END = "2023-07-01"
SEEDS = [42, 0, 1, 2, 3]


def step1_download_ma_sample():
    """Stream FNSPID, filter to M&A by title, sample MAX_MA_SAMPLES.

    NOTE: FNSPID streams in ticker-alphabetical order; scanning only the first
    chunk yields biased early-ticker / early-year matches. We scan the full
    dataset to get year coverage 2020-2023.
    """
    if os.path.exists(SAMPLE_PARQUET):
        print(f"[step1] Sample already exists: {SAMPLE_PARQUET}", flush=True)
        return pd.read_parquet(SAMPLE_PARQUET)

    from datasets import load_dataset
    print(f"[step1] Streaming FNSPID (full scan), looking for M&A titles ...",
          flush=True)
    ds = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)

    out = []
    t0 = time.time()
    skipped_no_ticker = 0
    skipped_too_short = 0
    seen = 0
    # Year coverage tracker
    year_counts = {2020: 0, 2021: 0, 2022: 0, 2023: 0}
    PER_YEAR_CAP = 12000  # cap matches per year
    for item in ds:
        seen += 1
        if seen % 500000 == 0:
            print(f"  scanned {seen:>9}  matches={len(out):>6}  "
                  f"year_counts={year_counts}  ({time.time()-t0:.0f}s)",
                  flush=True)
        title = item.get("Article_title", "") or ""
        if not title or len(title) < 10:
            skipped_too_short += 1; continue
        if not MA_KEYWORDS_RE.search(title):
            continue
        ticker = item.get("Stock_symbol", "")
        if not ticker or not isinstance(ticker, str) or len(ticker) > 6:
            skipped_no_ticker += 1; continue
        date = item.get("Date", "")
        if not date:
            continue
        # year coverage gate
        try:
            yr = int(str(date)[:4])
        except Exception:
            continue
        if yr not in year_counts:
            continue
        if year_counts[yr] >= PER_YEAR_CAP:
            continue
        year_counts[yr] += 1
        out.append({
            "date_str": str(date)[:19],  # 'YYYY-MM-DD HH:MM:SS'
            "title": title,
            "ticker": ticker.upper(),
            "url": item.get("Url", ""),
        })
        # Stop when all 4 years are capped
        if all(year_counts[y] >= PER_YEAR_CAP for y in year_counts):
            print(f"  all 4 years capped at {PER_YEAR_CAP}; stopping scan",
                  flush=True)
            break

    print(f"  scanned {seen} total, matched {len(out)}  "
          f"(skipped {skipped_no_ticker} bad-ticker, {skipped_too_short} short)",
          flush=True)
    df = pd.DataFrame(out)
    df["pub_time"] = pd.to_datetime(df["date_str"], errors="coerce")
    df = df.dropna(subset=["pub_time"]).reset_index(drop=True)
    df = df[(df["pub_time"] >= DATE_MIN) & (df["pub_time"] <= DATE_MAX)].copy()
    print(f"  after date filter: {len(df)}", flush=True)
    # Stratified sample by quarter to ensure even coverage
    df["quarter"] = df["pub_time"].dt.to_period("Q").astype(str)
    by_q = df.groupby("quarter").size()
    print(f"  by quarter: {by_q.to_dict()}", flush=True)
    sample_per_q = max(1, MAX_MA_SAMPLES // max(1, df["quarter"].nunique()))
    sampled = df.groupby("quarter", group_keys=False).apply(
        lambda g: g.sample(min(len(g), sample_per_q), random_state=42))
    sampled = sampled.reset_index(drop=True)
    print(f"  sampled {len(sampled)} M&A articles "
          f"({sample_per_q}/quarter target)", flush=True)
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    sampled.to_parquet(SAMPLE_PARQUET, index=False)
    print(f"  saved: {SAMPLE_PARQUET}", flush=True)
    return sampled


def step2_fetch_returns(sample_df):
    """Bulk-fetch prices per ticker via yfinance; compute next-day return label."""
    if os.path.exists(LABELED_PARQUET):
        print(f"[step2] Labeled data exists: {LABELED_PARQUET}", flush=True)
        return pd.read_parquet(LABELED_PARQUET)

    import yfinance as yf
    tickers = sample_df["ticker"].unique().tolist()
    print(f"[step2] Fetching prices for {len(tickers)} unique tickers ...",
          flush=True)
    price_data = {}
    t0 = time.time()
    BATCH = 50
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        try:
            hist = yf.download(batch, start=DATE_MIN, end=DATE_MAX,
                               progress=False, auto_adjust=False,
                               group_by="ticker", threads=True)
        except Exception as e:
            print(f"  batch {i//BATCH}: exception {e}", flush=True)
            continue
        # hist columns are MultiIndex (ticker, field) if multiple tickers
        if isinstance(hist.columns, pd.MultiIndex):
            for tkr in batch:
                if tkr in hist.columns.get_level_values(0):
                    sub = hist[tkr][["Close"]].copy()
                    sub.columns = ["close"]
                    sub = sub.dropna()
                    if len(sub) > 1:
                        sub["next_close"] = sub["close"].shift(-1)
                        sub["next_ret"] = (sub["next_close"] - sub["close"]) / sub["close"]
                        price_data[tkr] = sub
        else:
            # single ticker case
            sub = hist[["Close"]].copy()
            sub.columns = ["close"]
            sub = sub.dropna()
            if len(sub) > 1:
                sub["next_close"] = sub["close"].shift(-1)
                sub["next_ret"] = (sub["next_close"] - sub["close"]) / sub["close"]
                price_data[batch[0]] = sub
        print(f"  batch {i//BATCH+1}/{(len(tickers)+BATCH-1)//BATCH}, "
              f"got prices for {len(price_data)} tickers, "
              f"{time.time()-t0:.0f}s elapsed", flush=True)

    print(f"  total price tickers: {len(price_data)}", flush=True)

    # Join articles with returns
    rows = []
    skipped_no_price = 0
    skipped_no_date = 0
    for _, r in sample_df.iterrows():
        tkr = r["ticker"]; pub = r["pub_time"]
        if tkr not in price_data:
            skipped_no_price += 1; continue
        prices = price_data[tkr]
        pub_date = pub.normalize()
        # Find the next trading day on or after pub_date
        next_idx = prices.index[prices.index >= pub_date]
        if len(next_idx) == 0:
            skipped_no_date += 1; continue
        match_date = next_idx[0]
        ret = prices.loc[match_date, "next_ret"]
        if pd.isna(ret):
            skipped_no_date += 1; continue
        y = 1 if ret > 0.0 else 0
        rows.append({"pub_time": pub, "title": r["title"],
                     "ticker": tkr, "trade_date": match_date,
                     "next_ret": float(ret), "y": int(y)})
    labeled = pd.DataFrame(rows)
    print(f"  labeled rows: {len(labeled)}  "
          f"(skipped {skipped_no_price} no-price, {skipped_no_date} no-date)",
          flush=True)
    if len(labeled) == 0:
        raise RuntimeError("No labeled rows produced; check FNSPID + yfinance pipeline")
    labeled["y_str"] = labeled["y"].map({1: "UP", 0: "DOWN"})
    print(f"  y distribution: {labeled['y'].value_counts().to_dict()}", flush=True)
    labeled.to_parquet(LABELED_PARQUET, index=False)
    print(f"  saved: {LABELED_PARQUET}", flush=True)
    return labeled


def step3_audit(labeled):
    """Run the leakage audit on FNSPID-M&A labeled."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

    def safe_mcc(y, p):
        if len(np.unique(y)) < 2 or len(np.unique(p)) < 2: return 0.0
        return float(matthews_corrcoef(y, p))

    def fit_eval(X_tr, y_tr, X_te, y_te, model_kind):
        if model_kind == "lr":
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, n_jobs=-1)
        elif model_kind == "rf":
            clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                         n_jobs=-1, min_samples_leaf=10)
        elif model_kind == "gb":
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(X_tr, y_tr); p = clf.predict(X_te)
        return safe_mcc(y_te, p), float(balanced_accuracy_score(y_te, p))

    def build(titles_tr, titles_te, max_features=2000):
        tf = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=2)
        return tf.fit_transform(titles_tr), tf.transform(titles_te)

    df = labeled.sort_values("pub_time").reset_index(drop=True)
    df["title"] = df["title"].astype(str)
    print(f"\n[step3] Audit on FNSPID M&A: n={len(df)}", flush=True)
    print(f"  range: {df['pub_time'].min()} -> {df['pub_time'].max()}", flush=True)

    # Temporal split
    cutoff = pd.Timestamp(TEMPORAL_TRAIN_END)
    tr_t = df[df["pub_time"] < cutoff]; te_t = df[df["pub_time"] >= cutoff]
    print(f"  temporal: train={len(tr_t)}  test={len(te_t)}", flush=True)
    if len(te_t) < 20 or len(np.unique(te_t["y"])) < 2:
        print("  test too small / single-class; widening split", flush=True)
        cutoff = df["pub_time"].quantile(0.7)
        tr_t = df[df["pub_time"] < cutoff]; te_t = df[df["pub_time"] >= cutoff]
        print(f"  re-temporal: train={len(tr_t)}  test={len(te_t)}", flush=True)

    Xtr, Xte = build(tr_t["title"].tolist(), te_t["title"].tolist())
    temporal_rows = []
    for k in ["lr", "rf", "gb"]:
        mcc, bacc = fit_eval(Xtr, tr_t["y"].values, Xte, te_t["y"].values, k)
        print(f"  TEMPORAL {k:>2}  MCC={mcc:+.4f}  BalAcc={bacc:.4f}", flush=True)
        temporal_rows.append({"split": "temporal", "model": k, "mcc": mcc, "balacc": bacc})

    # Random splits, multi-seed
    print(f"\n  RANDOM (multi-seed)", flush=True)
    random_rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        ntr = int(0.7 * len(df))
        tr_r = df.iloc[idx[:ntr]]; te_r = df.iloc[idx[ntr:]]
        Xtr, Xte = build(tr_r["title"].tolist(), te_r["title"].tolist())
        for k in ["lr", "rf", "gb"]:
            mcc, bacc = fit_eval(Xtr, tr_r["y"].values, Xte, te_r["y"].values, k)
            random_rows.append({"seed": seed, "model": k, "mcc": mcc, "balacc": bacc})

    # Aggregate
    summary = []
    for k in ["lr", "rf", "gb"]:
        mccs = np.array([r["mcc"] for r in random_rows if r["model"] == k])
        rand_mean = float(mccs.mean()); rand_std = float(mccs.std())
        temp_mcc = next(r["mcc"] for r in temporal_rows if r["model"] == k)
        infl = (rand_mean / temp_mcc) if abs(temp_mcc) > 1e-6 else None
        print(f"  {k:>2}: random={rand_mean:+.4f}+/-{rand_std:.4f}  "
              f"temporal={temp_mcc:+.4f}  ratio={('NA' if infl is None else f'{infl:+.2f}x')}",
              flush=True)
        summary.append({"model": k, "random_mean": rand_mean,
                        "random_std": rand_std, "temporal": temp_mcc,
                        "inflation_ratio": infl})

    # M&A specialist (val-selected HP from paper)
    print(f"\n[step3b] M&A specialist (paper HP: mf=100, C=5.0, ngram=(1,1), sublinear=False, min_df=2)",
          flush=True)
    tf = TfidfVectorizer(max_features=100, stop_words="english", min_df=2,
                         sublinear_tf=False, ngram_range=(1, 1))
    Xtr = tf.fit_transform(tr_t["title"]); Xte = tf.transform(te_t["title"])
    clf = LogisticRegression(max_iter=2000, C=5.0, random_state=42)
    clf.fit(Xtr, tr_t["y"].values)
    yp = clf.predict(Xte); yp_proba = clf.predict_proba(Xte)[:, 1]
    ma_mcc = safe_mcc(te_t["y"].values, yp)
    ma_bacc = float(balanced_accuracy_score(te_t["y"].values, yp))
    print(f"  M&A specialist test MCC={ma_mcc:+.4f}  BalAcc={ma_bacc:.4f}", flush=True)

    # Permutation test
    rng = np.random.default_rng(42)
    yte = te_t["y"].values
    null = np.array([safe_mcc(rng.permutation(yte), yp) for _ in range(10000)])
    p_one = float((null >= ma_mcc).mean())
    p_two = float((np.abs(null) >= abs(ma_mcc)).mean())
    z = (ma_mcc - null.mean()) / (null.std() + 1e-12)
    print(f"  10K perm: z={z:.2f}  p_one={p_one:.4f}  p_two={p_two:.4f}", flush=True)

    return {
        "audit_temporal": temporal_rows,
        "audit_random_per_seed": random_rows,
        "audit_summary": summary,
        "ma_specialist": {
            "mcc": ma_mcc, "balacc": ma_bacc,
            "perm_test": {"z": float(z), "p_one": p_one, "p_two": p_two,
                          "null_mean": float(null.mean()), "null_std": float(null.std()),
                          "n_perm": 10000},
            "n_test": int(len(te_t)),
            "n_train": int(len(tr_t)),
        },
    }


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("FNSPID third-corpus M&A replication", flush=True)
    print("=" * 60, flush=True)

    sample = step1_download_ma_sample()
    print(f"\nFNSPID M&A sample shape: {sample.shape}", flush=True)
    print(f"  date range: {sample['pub_time'].min()} -> {sample['pub_time'].max()}",
          flush=True)

    labeled = step2_fetch_returns(sample)
    if len(labeled) < 200:
        print(f"WARNING: only {len(labeled)} labeled rows; audit may be noisy", flush=True)

    results = step3_audit(labeled)

    out = {
        "meta": {"timestamp": pd.Timestamp.now().isoformat(),
                 "elapsed_s": float(time.time() - t0),
                 "n_sample": int(len(sample)),
                 "n_labeled": int(len(labeled)),
                 "temporal_train_end": TEMPORAL_TRAIN_END,
                 "max_ma_samples": MAX_MA_SAMPLES,
                 "dataset": "FNSPID (Dong et al NeurIPS 2024)"},
        **results,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
