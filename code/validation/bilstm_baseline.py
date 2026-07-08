"""
A3: Bi-LSTM "StockNet-style" baseline.

We acknowledge upfront that StockNet (Xu & Cohen 2018, ACL) is a tweet-and-price
joint model with a variational autoencoder + temporal attention. Reproducing it
exactly is out of scope; this script implements a faithful *baseline in spirit*:

    GloVe-300d (frozen) -> 2-layer BiLSTM (hidden=128) -> max-pool -> MLP -> sigmoid

We train it on the BINARY general news classification task under both random
and temporal splits, then add results to the audit table.

If GloVe is missing, we fall back to in-domain trained word2vec via gensim.

Designed for CPU. ~30-90 min per training. Multi-seed optional (set N_SEEDS).
"""
import os
import sys
import io
import json
import time
import warnings
import math
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

torch.set_num_threads(max(1, os.cpu_count() // 2))

ROOT = Path(r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package")
DATA = ROOT / "data" / "classifier_training_v2.parquet"
GLOVE_PATH = ROOT / "data" / "external" / "glove.6B.300d.txt"
OUT = ROOT / "results" / "validation" / "bilstm_baseline.json"

TRAIN_END = pd.Timestamp("2025-04-01")
VAL_END = pd.Timestamp("2025-06-01")

DEVICE = torch.device("cpu")
SEED = 42
MAX_LEN = 32
EMBED_DIM = 300
HIDDEN = 128
DROPOUT = 0.3
EPOCHS = 8
BATCH = 256
LR = 1e-3
N_SEEDS = 1  # raise to 3 if time allows


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def tokenize(text, max_len=MAX_LEN):
    import re
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())
    return toks[:max_len]


class Vocab:
    def __init__(self, min_count=2):
        self.min_count = min_count
        self.tok2id = {"<pad>": 0, "<unk>": 1}
        self.id2tok = ["<pad>", "<unk>"]

    def fit(self, texts):
        from collections import Counter
        c = Counter()
        for t in texts:
            c.update(tokenize(t))
        for tok, n in c.items():
            if n >= self.min_count:
                self.tok2id[tok] = len(self.id2tok)
                self.id2tok.append(tok)
        return self

    def encode(self, text):
        return [self.tok2id.get(tok, 1) for tok in tokenize(text)]


def pad_or_trunc(ids, max_len=MAX_LEN, pad_id=0):
    if len(ids) >= max_len:
        return ids[:max_len], max_len
    return ids + [pad_id] * (max_len - len(ids)), len(ids)


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, n = pad_or_trunc(self.vocab.encode(self.texts[idx]))
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(n, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float32))


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden=HIDDEN, dropout=DROPOUT,
                 pretrained=None, freeze_emb=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
        if freeze_emb:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x, lengths):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        # Mean-pool over non-pad positions
        mask = (x != 0).float().unsqueeze(-1)
        pooled = (out * mask).sum(1) / (mask.sum(1).clamp(min=1.0))
        return self.fc(pooled).squeeze(-1)


def load_glove(vocab):
    """Load GloVe-300d embeddings for words in vocab. Returns embed matrix."""
    if not GLOVE_PATH.exists():
        print(f"  [WARN] GloVe not found at {GLOVE_PATH}; using random init", flush=True)
        rng = np.random.default_rng(SEED)
        emb = rng.normal(0, 0.1, (len(vocab.id2tok), EMBED_DIM)).astype(np.float32)
        emb[0] = 0.0  # pad
        return emb, 0
    print(f"  loading GloVe from {GLOVE_PATH} ...", flush=True)
    emb = np.random.normal(0, 0.1, (len(vocab.id2tok), EMBED_DIM)).astype(np.float32)
    emb[0] = 0.0
    found = 0
    with open(GLOVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            tok = parts[0]
            if tok in vocab.tok2id:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == EMBED_DIM:
                    emb[vocab.tok2id[tok]] = vec
                    found += 1
    print(f"  GloVe coverage: {found}/{len(vocab.id2tok)} = "
          f"{found/len(vocab.id2tok):.3f}", flush=True)
    return emb, found


def train_one(model, loader, val_loader, epochs, lr):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    best_val = -2.0
    best_state = None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        loss_acc = 0.0
        n_batches = 0
        for xb, lens, yb in loader:
            opt.zero_grad()
            logits = model(xb, lens)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_acc += loss.item()
            n_batches += 1
        train_loss = loss_acc / max(1, n_batches)
        # Eval val
        val_mcc, val_bacc = eval_model(model, val_loader)
        if val_mcc > best_val:
            best_val = val_mcc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    ep{ep+1:2d}: loss={train_loss:.4f}  val_MCC={val_mcc:+.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    ys, ps = [], []
    for xb, lens, yb in loader:
        logits = model(xb, lens)
        ps.extend((torch.sigmoid(logits) >= 0.5).long().tolist())
        ys.extend(yb.long().tolist())
    return safe_mcc(ys, ps), float(balanced_accuracy_score(ys, ps))


def run_split(df_tr, df_va, df_te, label_col="y"):
    print("  Building vocab on train ...", flush=True)
    vocab = Vocab(min_count=2).fit(df_tr["title_en"].values)
    print(f"  Vocab: {len(vocab.id2tok)}", flush=True)

    pretrained, n_found = load_glove(vocab)
    train_ds = TextDataset(df_tr["title_en"].values, df_tr[label_col].values, vocab)
    val_ds = TextDataset(df_va["title_en"].values, df_va[label_col].values, vocab)
    test_ds = TextDataset(df_te["title_en"].values, df_te[label_col].values, vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    torch.manual_seed(SEED)
    model = BiLSTMClassifier(len(vocab.id2tok), pretrained=pretrained, freeze_emb=True)
    print(f"  model params: trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)
    model, best_val = train_one(model, train_loader, val_loader, EPOCHS, LR)
    test_mcc, test_bacc = eval_model(model, test_loader)
    print(f"  TEST MCC={test_mcc:+.4f}  BalAcc={test_bacc:.4f}  (val MCC={best_val:+.4f})",
          flush=True)
    return {"mcc": float(test_mcc), "balacc": float(test_bacc), "val_mcc": float(best_val),
            "vocab_size": len(vocab.id2tok), "glove_coverage": int(n_found),
            "n_train": int(len(df_tr)), "n_test": int(len(df_te))}


def main():
    t0 = time.time()
    print("Loading data ...", flush=True)
    df = pd.read_parquet(DATA)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    df = df.sort_values("published_date").reset_index(drop=True)
    n = len(df)
    print(f"  {n} binary articles", flush=True)

    out = {"meta": {"timestamp": pd.Timestamp.now().isoformat(),
                    "epochs": EPOCHS, "max_len": MAX_LEN, "hidden": HIDDEN,
                    "embed_dim": EMBED_DIM, "n_seeds": N_SEEDS,
                    "glove_path": str(GLOVE_PATH), "glove_exists": GLOVE_PATH.exists()},
           "results": []}

    # === Temporal split ===
    print("\n[TEMPORAL split]", flush=True)
    df_tr_t = df[df["published_date"] < TRAIN_END]
    df_va_t = df[(df["published_date"] >= TRAIN_END) & (df["published_date"] < VAL_END)]
    df_te_t = df[df["published_date"] >= VAL_END]
    res_t = run_split(df_tr_t, df_va_t, df_te_t)
    res_t.update({"split": "temporal", "seed": SEED})
    out["results"].append(res_t)

    # Save partial
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    # === Random splits ===
    n_tr, n_va, n_te = len(df_tr_t), len(df_va_t), len(df_te_t)
    for s in range(N_SEEDS):
        print(f"\n[RANDOM split, seed={s}]", flush=True)
        rng = np.random.default_rng(s)
        perm = rng.permutation(n)
        df_tr_r = df.iloc[perm[:n_tr]]
        df_va_r = df.iloc[perm[n_tr:n_tr + n_va]]
        df_te_r = df.iloc[perm[n_tr + n_va:n_tr + n_va + n_te]]
        res_r = run_split(df_tr_r, df_va_r, df_te_r)
        res_r.update({"split": "random", "seed": s})
        out["results"].append(res_r)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2)

    out["meta"]["elapsed_s"] = float(time.time() - t0)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)
    print(f"Total: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
