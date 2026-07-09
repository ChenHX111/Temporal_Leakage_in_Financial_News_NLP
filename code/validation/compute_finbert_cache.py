"""Compute FinBERT [CLS] embeddings for all binary articles and cache to disk.
Runs once on CPU (slow, ~1-3 hrs). Output saved to data/embeddings_cache/finbert_title.npy.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time

ROOT = Path(r".")
DATA = ROOT / "data" / "classifier_training_v2.parquet"
OUT = ROOT / "data" / "embeddings_cache" / "finbert_title.npy"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("Loading data ...")
    df = pd.read_parquet(DATA)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["title_en"] = df["title_en"].fillna("").astype(str)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df = df.sort_values("published_date").reset_index(drop=True)
    titles = df["title_en"].tolist()
    print(f"Total titles: {len(titles)}")

    if OUT.exists():
        cached = np.load(OUT)
        if cached.shape[0] == len(titles):
            print(f"Cache exists with matching size {cached.shape}. Done.")
            return
        print(f"Cache size mismatch ({cached.shape[0]} vs {len(titles)}); recomputing.")

    print("Loading FinBERT ...")
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")

    BATCH = 32
    out = np.zeros((len(titles), 768), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(titles), BATCH), desc="FinBERT"):
            batch = titles[i:i + BATCH]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            o = model(**enc)
            # [CLS] = first token of last_hidden_state
            cls = o.last_hidden_state[:, 0, :].cpu().numpy()
            out[i:i + len(batch)] = cls

    np.save(OUT, out)
    print(f"Saved {OUT}, shape={out.shape}, elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
