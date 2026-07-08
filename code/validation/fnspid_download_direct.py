"""
FNSPID direct-download approach (replaces the slow streaming pre-download).

Downloads:
  Stock_news/nasdaq_exteral_data.csv  (smaller, US-focused; ~1-3 GB)
  Stock_news/All_external.csv         (full, ~22M rows; ~5-10 GB)
  Stock_price/full_history.zip        (pre-bundled OHLCV for all tickers!)

Then filters to M&A keywords and saves the slim parquet.
"""
import os, sys, time, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE = Path(r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package\data\external")
BASE.mkdir(parents=True, exist_ok=True)

MA_REGEX = re.compile(
    r"\b(?:merger|merging|merge|acquisition|acquir(?:e|es|ed|ing)|"
    r"to be acquired|takeover|tender offer|buyout)\b",
    flags=re.IGNORECASE,
)


def download_file(filename: str) -> Path:
    print(f"\nDownloading {filename}...", flush=True)
    t0 = time.time()
    path = hf_hub_download(
        repo_id="Zihan1004/FNSPID",
        filename=filename,
        repo_type="dataset",
        cache_dir=str(BASE / "hf_cache"),
    )
    print(f"  done {time.time()-t0:.0f}s -> {path}  "
          f"({os.path.getsize(path)/1e6:.0f} MB)", flush=True)
    return Path(path)


def filter_ma_chunked(csv_path: Path, label: str) -> pd.DataFrame:
    """Stream the CSV in 200K-row chunks, keep only M&A keyword matches."""
    print(f"\nFiltering M&A from {label}: {csv_path}", flush=True)
    t0 = time.time()
    kept = []
    n_total = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(
            csv_path, chunksize=200_000, low_memory=False, on_bad_lines='skip')):
        n_total += len(chunk)
        # Determine column names dynamically
        title_col = next((c for c in ["Article_title", "Title", "article_title", "title"]
                          if c in chunk.columns), None)
        date_col = next((c for c in ["Date", "date", "Publication date", "Publish date"]
                         if c in chunk.columns), None)
        tic_col = next((c for c in ["Stock_symbol", "Symbol", "stock_symbol", "symbol", "Ticker", "ticker"]
                        if c in chunk.columns), None)
        if title_col is None or date_col is None or tic_col is None:
            if chunk_idx == 0:
                print(f"  WARN columns: {list(chunk.columns)[:10]}", flush=True)
            continue
        c = chunk[[date_col, title_col, tic_col]].rename(columns={
            date_col: "date_str", title_col: "title", tic_col: "ticker"})
        c["title"] = c["title"].fillna("").astype(str)
        mask = c["title"].str.contains(MA_REGEX)
        if mask.any():
            kept.append(c[mask].copy())
        if chunk_idx % 20 == 0:
            running = sum(len(k) for k in kept)
            print(f"  ...chunk {chunk_idx}  scanned={n_total:>10}  "
                  f"kept={running:>10}  ({time.time()-t0:.0f}s)", flush=True)
    out = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(
        columns=["date_str", "title", "ticker"])
    print(f"\n  {label}: kept {len(out):,} / scanned {n_total:,} = "
          f"{100*len(out)/max(1,n_total):.2f}%   ({time.time()-t0:.0f}s)",
          flush=True)
    return out


def main():
    t0 = time.time()

    nas_csv = download_file("Stock_news/nasdaq_exteral_data.csv")
    nas = filter_ma_chunked(nas_csv, "nasdaq")

    all_csv = download_file("Stock_news/All_external.csv")
    allx = filter_ma_chunked(all_csv, "All_external")

    combined = pd.concat([nas, allx], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date_str", "title", "ticker"]).reset_index(drop=True)
    print(f"\nCombined: {len(combined):,} M&A rows after dedup", flush=True)

    # Normalize date
    combined["date_str"] = combined["date_str"].astype(str)
    combined["ticker"] = combined["ticker"].astype(str).str.upper().str.strip()
    combined = combined[combined["ticker"].str.len() > 0]

    out_path = BASE / "fnspid_ma_filtered.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\n-> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)

    # Per-year summary
    combined["year"] = pd.to_datetime(
        combined["date_str"], errors="coerce").dt.year
    print("\nM&A by year:", flush=True)
    print(combined.groupby("year").size().to_string(), flush=True)
    print(f"\nDONE  elapsed={time.time()-t0:.0f}s", flush=True)

    # Also download the bundled prices for completeness
    print("\nDownloading bundled prices (Stock_price/full_history.zip)...",
          flush=True)
    price_path = download_file("Stock_price/full_history.zip")
    print(f"  prices at {price_path}", flush=True)


if __name__ == "__main__":
    main()
