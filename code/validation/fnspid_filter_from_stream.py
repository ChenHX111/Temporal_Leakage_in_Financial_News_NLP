"""Quick M&A filter on the streamed parts (fallback)."""
import os, sys, time, re, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package\data\external")

MA_REGEX = re.compile(
    r"\b(?:merger|merging|merge|acquisition|acquir(?:e|es|ed|ing)|"
    r"to be acquired|takeover|tender offer|buyout)\b",
    flags=re.IGNORECASE,
)


def main():
    t0 = time.time()
    parts = sorted(BASE.glob("fnspid_slim_full_part*.parquet"))
    print(f"Found {len(parts)} streamed parts", flush=True)
    dfs = [pd.read_parquet(p) for p in parts]
    big = pd.concat(dfs, ignore_index=True)
    print(f"Total streamed rows: {len(big):,}", flush=True)
    print(f"Date range: {big['date_str'].min()} .. {big['date_str'].max()}", flush=True)
    # Per-year
    big["pd"] = pd.to_datetime(big["date_str"], errors="coerce")
    big["year"] = big["pd"].dt.year
    print("\nAll rows by year:", flush=True)
    print(big.groupby("year").size().to_string(), flush=True)
    # Filter
    mask = big["title"].fillna("").astype(str).str.contains(MA_REGEX)
    ma = big[mask].copy()
    print(f"\nM&A rows: {len(ma):,}  ({100*len(ma)/len(big):.2f}%)", flush=True)
    print("\nM&A by year:", flush=True)
    print(ma.groupby("year").size().to_string(), flush=True)
    out = BASE / "fnspid_ma_filtered_fromstream.parquet"
    ma[["date_str", "title", "ticker"]].to_parquet(out, index=False)
    print(f"\n-> {out}  ({os.path.getsize(out)/1e6:.1f} MB)  ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
