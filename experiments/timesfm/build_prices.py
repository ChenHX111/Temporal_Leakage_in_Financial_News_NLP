"""
build_prices.py — fetch trailing daily close history for each M&A locked-test ticker (via yfinance) and cache it.
Produces data/prices_cache.parquet: (yf_ticker, date, close). Skips tickers that fail; reports coverage.
Run on a box WITH internet (the server, or locally then upload the cache). ~a few minutes for ~349 tickers.
"""
import os, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "data", "ma_locked_test_meta.parquet")
OUT = os.path.join(HERE, "data", "prices_cache.parquet")
LOOKBACK_CALENDAR_DAYS = 200   # ~128 trading days of context before the earliest article


def main():
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Install first:  pip install yfinance")
    meta = pd.read_parquet(META)
    meta["published_date"] = pd.to_datetime(meta["published_date"])
    tickers = sorted(meta["yf_ticker"].dropna().astype(str).unique())
    start = (meta["published_date"].min() - pd.Timedelta(days=LOOKBACK_CALENDAR_DAYS)).date()
    end = (meta["published_date"].max() + pd.Timedelta(days=3)).date()
    print(f"Fetching {len(tickers)} tickers  {start} -> {end}", flush=True)
    frames, ok, fail = [], 0, []
    for i, t in enumerate(tickers):
        try:
            h = yf.Ticker(t).history(start=str(start), end=str(end), interval="1d", auto_adjust=True)
            if h is None or len(h) < 20:
                fail.append(t); continue
            d = h.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
            d["date"] = d["date"].dt.tz_localize(None)   # drop tz, keep exchange-local wall-clock date (per-ticker single tz)
            d["yf_ticker"] = t; frames.append(d); ok += 1
        except Exception:
            fail.append(t)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tickers)}  ok={ok} fail={len(fail)}", flush=True)
        time.sleep(0.05)
    if not frames:
        sys.exit("No price data fetched — check internet / ticker formats.")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}: {len(out)} rows, {ok}/{len(tickers)} tickers covered "
          f"({100*ok/len(tickers):.0f}%). Missing {len(fail)}.", flush=True)


if __name__ == "__main__":
    main()
