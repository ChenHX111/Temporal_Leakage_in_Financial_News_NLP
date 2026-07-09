"""
Title-vs-content systematic ablation — addresses W10.

Audit weakness W10: paper asserts "title-only is best" but does not systematically
test (title, content, title+content) across multiple architectures.

This script runs a 3 x 5 grid:
    text inputs:  {title-only, content-only, title+content}
    architectures: {TF-IDF+LR, TF-IDF+RF, TF-IDF+GB, FinBERT-frozen+LR, MiniLM-frozen+LR}

Reports temporal-split MCC + BalAcc for each cell. Goal: establish whether
title-only superiority is consistent across architectures or restricted to
certain ones.

Output: results/validation/title_vs_content_ablation.json
"""
import os
import sys
import io
import json
import time
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

BASE = r"."
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
FINBERT_CACHE = os.path.join(BASE, "data", "finbert_cls_cache.npz")
MINILM_CACHE = os.path.join(BASE, "data", "minilm_emb_cache.npz")
OUT = os.path.join(BASE, "results", "validation", "title_vs_content_ablation.json")


def safe_mcc(y, p):
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2: return 0.0
    return float(matthews_corrcoef(y, p))


def fit_eval_model(model_kind, X_tr, y_tr, X_te, y_te):
    if model_kind == "lr":
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, n_jobs=-1)
    elif model_kind == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                     n_jobs=-1, min_samples_leaf=10)
    elif model_kind == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    else:
        raise ValueError(model_kind)
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    return safe_mcc(y_te, yp), float(balanced_accuracy_score(y_te, yp))


def build_text(df, mode):
    if mode == "title":
        return df["title_en"].fillna("").astype(str).tolist()
    if mode == "content":
        return df["content_en"].fillna("").astype(str).tolist()
    if mode == "title_content":
        t = df["title_en"].fillna("").astype(str)
        c = df["content_en"].fillna("").astype(str)
        return (t + " " + c).tolist()
    raise ValueError(mode)


def main():
    t0 = time.time()
    print("Loading data ...", flush=True)
    df = pd.read_parquet(DATA)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    df["content_en"] = df["content_en"].fillna("").astype(str)
    df = df.sort_values("published_date").reset_index(drop=True)

    TRAIN_END = pd.Timestamp("2025-04-01")
    VAL_END = pd.Timestamp("2025-06-01")
    tr = df[df["published_date"] < VAL_END].copy()
    te = df[df["published_date"] >= VAL_END].copy()
    print(f"Train+val: {len(tr)}  Test: {len(te)}", flush=True)

    cells = []
    text_modes = ["title", "content", "title_content"]
    tfidf_archs = ["lr", "rf", "gb"]

    for mode in text_modes:
        print(f"\n[TF-IDF on {mode}]", flush=True)
        tf = TfidfVectorizer(max_features=2000, stop_words="english", min_df=2)
        tr_txt = build_text(tr, mode); te_txt = build_text(te, mode)
        Xtr = tf.fit_transform(tr_txt); Xte = tf.transform(te_txt)
        for arch in tfidf_archs:
            mcc, bacc = fit_eval_model(arch, Xtr, tr["y"].values, Xte, te["y"].values)
            print(f"  TF-IDF {arch:>2}  MCC={mcc:+.4f}  BalAcc={bacc:.4f}", flush=True)
            cells.append({"text": mode, "feat": "tfidf",
                          "model": arch, "mcc": mcc, "balacc": bacc})

    # FinBERT frozen [CLS]
    if os.path.exists(FINBERT_CACHE):
        print(f"\n[FinBERT frozen [CLS]]  (loading cache)", flush=True)
        cache = np.load(FINBERT_CACHE, allow_pickle=True)
        emb_title = cache["emb_title"]
        ids_title = cache["news_ids_title"]
        emb_content = cache["emb_content"] if "emb_content" in cache.files else None
        ids_content = cache["news_ids_content"] if "news_ids_content" in cache.files else None
        # title only
        for mode_name, emb_array, ids_array in [
            ("title", emb_title, ids_title),
            ("content", emb_content, ids_content)
        ]:
            if emb_array is None or ids_array is None:
                print(f"  FinBERT cache for '{mode_name}' missing; skip", flush=True)
                continue
            id_to_pos = {str(nid): i for i, nid in enumerate(ids_array)}
            tr_pos = np.array([id_to_pos.get(str(nid), -1) for nid in tr["news_id"]])
            te_pos = np.array([id_to_pos.get(str(nid), -1) for nid in te["news_id"]])
            tr_keep = tr_pos >= 0; te_keep = te_pos >= 0
            Xtr = emb_array[tr_pos[tr_keep]]; Xte = emb_array[te_pos[te_keep]]
            ytr_k = tr["y"].values[tr_keep]; yte_k = te["y"].values[te_keep]
            print(f"  FinBERT-{mode_name}: tr={Xtr.shape}, te={Xte.shape}", flush=True)
            mcc, bacc = fit_eval_model("lr", Xtr, ytr_k, Xte, yte_k)
            print(f"  FinBERT-{mode_name}+LR  MCC={mcc:+.4f}  BalAcc={bacc:.4f}", flush=True)
            cells.append({"text": mode_name, "feat": "finbert_cls",
                          "model": "lr", "mcc": mcc, "balacc": bacc})
        # title+content: avg of two if both available
        if emb_content is not None and ids_content is not None:
            id_to_t = {str(nid): i for i, nid in enumerate(ids_title)}
            id_to_c = {str(nid): i for i, nid in enumerate(ids_content)}
            common = list(set(id_to_t.keys()) & set(id_to_c.keys()))
            id_to_concat = {}
            arr_concat = []
            for nid in common:
                arr_concat.append(np.concatenate(
                    [emb_title[id_to_t[nid]], emb_content[id_to_c[nid]]]))
                id_to_concat[nid] = len(arr_concat) - 1
            arr_concat = np.array(arr_concat)
            tr_pos = np.array([id_to_concat.get(str(nid), -1) for nid in tr["news_id"]])
            te_pos = np.array([id_to_concat.get(str(nid), -1) for nid in te["news_id"]])
            tr_keep = tr_pos >= 0; te_keep = te_pos >= 0
            Xtr = arr_concat[tr_pos[tr_keep]]; Xte = arr_concat[te_pos[te_keep]]
            ytr_k = tr["y"].values[tr_keep]; yte_k = te["y"].values[te_keep]
            mcc, bacc = fit_eval_model("lr", Xtr, ytr_k, Xte, yte_k)
            print(f"  FinBERT-concat+LR  MCC={mcc:+.4f}  BalAcc={bacc:.4f}", flush=True)
            cells.append({"text": "title_content", "feat": "finbert_cls_concat",
                          "model": "lr", "mcc": mcc, "balacc": bacc})
    else:
        print(f"\n[FinBERT cache not found at {FINBERT_CACHE}; skipping FinBERT rows]",
              flush=True)

    if os.path.exists(MINILM_CACHE):
        print(f"\n[MiniLM frozen emb]  (loading cache)", flush=True)
        cache = np.load(MINILM_CACHE, allow_pickle=True)
        emb = cache["emb"] if "emb" in cache.files else None
        ids = cache["news_ids"] if "news_ids" in cache.files else None
        if emb is not None and ids is not None:
            id_to_pos = {str(nid): i for i, nid in enumerate(ids)}
            tr_pos = np.array([id_to_pos.get(str(nid), -1) for nid in tr["news_id"]])
            te_pos = np.array([id_to_pos.get(str(nid), -1) for nid in te["news_id"]])
            tr_keep = tr_pos >= 0; te_keep = te_pos >= 0
            Xtr = emb[tr_pos[tr_keep]]; Xte = emb[te_pos[te_keep]]
            ytr_k = tr["y"].values[tr_keep]; yte_k = te["y"].values[te_keep]
            mcc, bacc = fit_eval_model("lr", Xtr, ytr_k, Xte, yte_k)
            print(f"  MiniLM-title+LR  MCC={mcc:+.4f}  BalAcc={bacc:.4f}", flush=True)
            cells.append({"text": "title", "feat": "minilm_emb",
                          "model": "lr", "mcc": mcc, "balacc": bacc})
    else:
        print(f"\n[MiniLM cache not found at {MINILM_CACHE}; skipping MiniLM row]",
              flush=True)

    # Summary table
    print("\n[SUMMARY]", flush=True)
    print(f"{'text':<15} {'feat':<20} {'model':<5} {'MCC':>9} {'BalAcc':>9}",
          flush=True)
    for c in cells:
        print(f"{c['text']:<15} {c['feat']:<20} {c['model']:<5} "
              f"{c['mcc']:+.4f}    {c['balacc']:.4f}", flush=True)

    # Per-architecture: which text mode wins?
    by_arch = {}
    for c in cells:
        k = (c["feat"], c["model"])
        if k not in by_arch:
            by_arch[k] = {}
        by_arch[k][c["text"]] = c["mcc"]
    print("\n[Per-architecture winner]", flush=True)
    for k, v in by_arch.items():
        winner = max(v.items(), key=lambda x: x[1])
        print(f"  {k}: {v}  -> WINNER: {winner[0]} ({winner[1]:+.4f})", flush=True)

    out = {
        "meta": {"timestamp": pd.Timestamp.now().isoformat(),
                 "elapsed_s": float(time.time() - t0),
                 "n_train_val": int(len(tr)), "n_test": int(len(te))},
        "cells": cells,
        "by_architecture": {f"{k[0]}__{k[1]}": v for k, v in by_arch.items()},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
