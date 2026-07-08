"""
Label the FNSPID M&A subset with t+1 close-to-close return using the FNSPID
bundled price archive (NOT yfinance). Much faster and handles delisted
tickers.

Inputs:
    data/external/fnspid_ma_filtered_fromstream.parquet
    data/external/hf_cache/.../full_history.zip

Outputs:
    paper/gpu_package_v8/data/fnspid_ma_filtered.parquet
    paper/gpu_package_v8/data/fnspid_ma_labelled.parquet
"""
import os, sys, time, io, zipfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package")
SRC_FILTERED = BASE / "data" / "external" / "fnspid_ma_filtered_fromstream.parquet"
PRICE_ZIP = BASE / "data" / "external" / "hf_cache" / \
    "datasets--Zihan1004--FNSPID" / "snapshots" / \
    "bf9189c41527198897d1af3e17b1a0095279fc45" / "Stock_price" / "full_history.zip"

V8_DATA = Path(r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\paper\gpu_package_v8\data")
V8_DATA.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    df = pd.read_parquet(SRC_FILTERED)
    print(f"Loaded {len(df):,} M&A rows", flush=True)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["pub_date"] = pd.to_datetime(df["date_str"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["pub_date"]).copy()
    df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$")].copy()
    print(f"  after ticker+date clean: {len(df):,}", flush=True)
    # Save the cleaned filtered set for v8 data/
    df_save = df[["date_str", "title", "ticker"]].copy()
    out_filt = V8_DATA / "fnspid_ma_filtered.parquet"
    df_save.to_parquet(out_filt, index=False)
    print(f"  filtered -> {out_filt} ({os.path.getsize(out_filt)/1e6:.1f} MB)", flush=True)

    # Load needed ticker CSVs from zip
    tickers_needed = sorted(df["ticker"].unique())
    print(f"\nUnique tickers needed: {len(tickers_needed)}", flush=True)
    zf = zipfile.ZipFile(PRICE_ZIP)
    available = set(zf.namelist())
    price_map = {}
    n_load_ok, n_load_miss, n_load_fail = 0, 0, 0
    for i, t in enumerate(tickers_needed):
        name = f"full_history/{t}.csv"
        if name not in available:
            n_load_miss += 1; continue
        try:
            with zf.open(name) as f:
                px = pd.read_csv(f, usecols=["date", "close"])
            px["date"] = pd.to_datetime(px["date"], errors="coerce")
            px = px.dropna(subset=["date", "close"])
            px["date"] = px["date"].dt.tz_localize(None).dt.normalize()
            px = px.sort_values("date").drop_duplicates("date").reset_index(drop=True)
            if len(px) >= 5:
                price_map[t] = px
                n_load_ok += 1
            else:
                n_load_fail += 1
        except Exception:
            n_load_fail += 1
        if i % 500 == 0:
            print(f"  ...loaded {i}/{len(tickers_needed)}  ok={n_load_ok}  miss={n_load_miss}  fail={n_load_fail}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nPrice load done: ok={n_load_ok}  miss={n_load_miss}  fail={n_load_fail}  ({time.time()-t0:.0f}s)", flush=True)

    # Label each article
    rows = []
    n_no_px, n_no_t0, n_no_t1, n_flat = 0, 0, 0, 0
    df_arr = df.to_dict(orient="records")
    for r in df_arr:
        t = r["ticker"]
        if t not in price_map:
            n_no_px += 1; continue
        s = price_map[t]
        d = r["pub_date"]
        idx = s["date"].searchsorted(d, side="right")  # first trading day strictly after d
        if idx >= len(s) - 1:
            n_no_t1 += 1; continue
        if idx == 0:
            n_no_t0 += 1; continue
        p_t = float(s["close"].iloc[idx])
        p_t1 = float(s["close"].iloc[idx + 1])
        if p_t <= 0 or not np.isfinite(p_t) or not np.isfinite(p_t1):
            n_flat += 1; continue
        ret = (p_t1 - p_t) / p_t
        if abs(ret) < 1e-6 or not np.isfinite(ret):
            n_flat += 1; continue
        y = int(ret > 0)
        rows.append((r["date_str"], r["title"], t, ret, y))
    print(f"\nLabel done: kept={len(rows):,}  no_px={n_no_px}  no_t0={n_no_t0}  "
          f"no_t1={n_no_t1}  flat={n_flat}", flush=True)

    labelled = pd.DataFrame(rows, columns=["date_str", "title", "ticker", "ret_t1", "y"])
    labelled["pub_date"] = pd.to_datetime(labelled["date_str"]).dt.tz_localize(None)
    labelled = labelled.sort_values("pub_date").reset_index(drop=True)
    labelled = labelled.drop(columns=["pub_date"])

    out_lab = V8_DATA / "fnspid_ma_labelled.parquet"
    labelled.to_parquet(out_lab, index=False)
    print(f"\nlabelled -> {out_lab} ({os.path.getsize(out_lab)/1e6:.1f} MB)", flush=True)
    print(f"  UP rate overall: {labelled['y'].mean():.3f}", flush=True)
    print(f"\nBy year:", flush=True)
    labelled["year"] = pd.to_datetime(labelled["date_str"]).dt.year
    print(labelled.groupby("year").agg(n=("y","size"), up_rate=("y","mean")).to_string(), flush=True)
    print(f"\nDONE  elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
