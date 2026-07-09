"""
Pre-download a slim copy of FNSPID (Article_title, Stock_symbol, Date) to local
parquet so the GPU pkg #B can be self-contained and re-run anywhere.

Output: data/external/fnspid_slim_full.parquet (expected ~500-800 MB)
"""
import os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd

BASE = r"."
OUT = os.path.join(BASE, "data", "external", "fnspid_slim_full.parquet")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print(f"Streaming Zihan1004/FNSPID -> {OUT}", flush=True)
from datasets import load_dataset
ds = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)

rows = []
t0 = time.time()
n_dropped_no_date = 0
n_dropped_no_ticker = 0
n_dropped_no_title = 0

BATCH_FLUSH = 500_000
batch_idx = 0

for i, item in enumerate(ds):
    if i % 1_000_000 == 0 and i > 0:
        print(f"  scanned {i:>10}  kept {len(rows):>10}  ({time.time()-t0:.0f}s)",
              flush=True)
    title = item.get("Article_title", "") or ""
    if not title or not isinstance(title, str) or len(title) < 5:
        n_dropped_no_title += 1; continue
    ticker = item.get("Stock_symbol", "") or ""
    if not ticker or not isinstance(ticker, str):
        n_dropped_no_ticker += 1; continue
    date = item.get("Date", "") or ""
    if not date:
        n_dropped_no_date += 1; continue
    rows.append((str(date)[:19], title, ticker.upper()))

    # Flush every 500K to keep memory bounded
    if len(rows) >= BATCH_FLUSH:
        df = pd.DataFrame(rows, columns=["date_str", "title", "ticker"])
        part_path = OUT.replace(".parquet", f"_part{batch_idx:04d}.parquet")
        df.to_parquet(part_path, index=False)
        print(f"  flushed batch {batch_idx}: {len(rows)} rows -> {part_path}",
              flush=True)
        rows = []
        batch_idx += 1

# Final flush
if rows:
    df = pd.DataFrame(rows, columns=["date_str", "title", "ticker"])
    part_path = OUT.replace(".parquet", f"_part{batch_idx:04d}.parquet")
    df.to_parquet(part_path, index=False)
    print(f"  flushed final batch {batch_idx}: {len(rows)} rows -> {part_path}",
          flush=True)
    batch_idx += 1

# Coalesce parts
print(f"\nCoalescing {batch_idx} parts -> {OUT}", flush=True)
parts = []
for j in range(batch_idx):
    pp = OUT.replace(".parquet", f"_part{j:04d}.parquet")
    parts.append(pd.read_parquet(pp))
big = pd.concat(parts, ignore_index=True)
big.to_parquet(OUT, index=False)
print(f"  total: {len(big):,} rows  ({os.path.getsize(OUT)/1e6:.1f} MB)",
      flush=True)
print(f"  date range: {big['date_str'].min()} .. {big['date_str'].max()}",
      flush=True)

# Delete part files
for j in range(batch_idx):
    pp = OUT.replace(".parquet", f"_part{j:04d}.parquet")
    if os.path.exists(pp): os.remove(pp)
print(f"\nDONE   elapsed: {time.time()-t0:.0f}s   dropped: "
      f"title={n_dropped_no_title}, ticker={n_dropped_no_ticker}, date={n_dropped_no_date}",
      flush=True)
