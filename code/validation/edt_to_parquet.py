"""Convert EDT evaluate_news.json (1.6 GB) to a slim parquet for fast access."""
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r".")
SRC = ROOT / "data" / "external" / "evaluate_news.json"
OUT = ROOT / "data" / "external" / "edt_evaluate_slim.parquet"

if OUT.exists():
    print(f"Already exists: {OUT}")
    df = pd.read_parquet(OUT)
    print(df.shape, df.columns.tolist())
    print(df.head(3))
    raise SystemExit(0)

print(f"Loading {SRC} ({SRC.stat().st_size / 1e9:.2f} GB) ...")
t0 = time.time()
with open(SRC, "rb") as f:
    raw = f.read()
print(f"Read {time.time()-t0:.1f}s")
t0 = time.time()
data = json.loads(raw)
print(f"JSON parse {time.time()-t0:.1f}s, {len(data)} records")

rows = []
for d in data:
    lbl = d.get("labels") or {}
    rows.append({
        "title": d.get("title") or "",
        "pub_time": d.get("pub_time") or "",
        "ticker": lbl.get("ticker"),
        "start_price_close": lbl.get("start_price_close"),
        "end_price_1day": lbl.get("end_price_1day"),
    })
df = pd.DataFrame(rows)
print(f"DataFrame {df.shape}, columns={df.columns.tolist()}")
df["pub_time"] = pd.to_datetime(df["pub_time"], errors="coerce", utc=True).dt.tz_convert(None)
df = df.dropna(subset=["pub_time", "title", "start_price_close", "end_price_1day"])
df["return_1d"] = (df["end_price_1day"] - df["start_price_close"]) / df["start_price_close"]
df["y"] = (df["return_1d"] > 0).astype(int)
print(f"Valid: {len(df)}, UP rate {df['y'].mean():.3f}")

df.to_parquet(OUT, index=False)
print(f"Saved {OUT}")
