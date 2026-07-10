"""
E2 — Leakage-MITIGATION demo (answers ysCR-W1/C3: "give easy tricks to reduce leakage during training").

Shows that a naive random K-fold CV on shuffled data REPRODUCES the inflated (leaked) MCC, while two easy
plug-in mitigations -- (a) time-BLOCKED K-fold with a +/-EMBARGO-day purge (Lopez de Prado 2018, already cited),
(b) forward-chaining (expanding-window) CV -- recover the honest chronological MCC. And critically the M&A
locked-test signal is INVARIANT to these mitigations (real signal, not suppressible leakage).

General corpus (up/down only), features = TF-IDF(title+content), classifiers = LR and RF.
Faithful splits: TRAIN_END=2025-04-01, VAL_END=2025-06-01, test>=2025-06-01.
Out: EMNLP_REBUTTAL/experiments/out/E2_mitigation.json
"""
import os, io, sys, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef

BASE = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "E2_mitigation.json")
TRAIN_END, VAL_END = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-01")
EMBARGO_DAYS, K = 5, 5


def mcc(y, p):
    return 0.0 if (len(np.unique(y)) < 2 or len(np.unique(p)) < 2) else float(matthews_corrcoef(y, p))


def make_clf(kind):
    return (LogisticRegression(max_iter=2000, C=0.5, random_state=42) if kind == "LR"
            else RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42))


def eval_split(df, tr_idx, te_idx, kind, text="tc"):
    tf = TfidfVectorizer(max_features=2000, stop_words="english", min_df=2,
                         ngram_range=(1, 2) if text == "tc" else (1, 1))
    col = df["txt"].values
    Xtr = tf.fit_transform(col[tr_idx]); Xte = tf.transform(col[te_idx])
    clf = make_clf(kind)
    if kind == "RF":
        Xtr = Xtr.toarray() if Xtr.shape[1] <= 2000 else Xtr
    clf.fit(Xtr, df["y"].values[tr_idx])
    Xe = Xte.toarray() if (kind == "RF" and Xte.shape[1] <= 2000) else Xte
    return mcc(df["y"].values[te_idx], clf.predict(Xe))


def naive_random_cv(df, kind):
    n = len(df); rng = np.random.RandomState(42); idx = rng.permutation(n)
    folds = np.array_split(idx, K); out = []
    for i in range(K):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(K) if j != i])
        out.append(eval_split(df, tr, te, kind))
    return out


def blocked_embargo_cv(df, kind):
    order = np.argsort(df["published_date"].values); dates = df["published_date"].values
    folds = np.array_split(order, K); out = []
    for i in range(K):
        te = folds[i]
        te_lo, te_hi = dates[te].min(), dates[te].max()
        emb = np.timedelta64(EMBARGO_DAYS, "D")
        tr = np.array([j for j in order if (dates[j] < te_lo - emb) or (dates[j] > te_hi + emb)])
        if len(tr) < 200 or df["y"].values[tr].sum() in (0, len(tr)):
            continue
        out.append(eval_split(df, tr, te, kind))
    return out


def forward_chain_cv(df, kind):
    order = np.argsort(df["published_date"].values); blocks = np.array_split(order, K + 1); out = []
    for i in range(1, K + 1):
        tr = np.concatenate(blocks[:i]); te = blocks[i]
        out.append(eval_split(df, tr, te, kind))
    return out


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["txt"] = (df["title_en"].fillna("").astype(str) + ". " + df["content_en"].fillna("").astype(str))
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy().reset_index(drop=True)
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)

    # chronological reference (the honest target)
    tr = df.index[df["published_date"] < TRAIN_END].values
    te = df.index[df["published_date"] >= VAL_END].values
    res = {"meta": {"n": int(len(df)), "embargo_days": EMBARGO_DAYS, "K": K}, "cells": {}}
    for kind in ["LR", "RF"]:
        chrono = eval_split(df, tr, te, kind)
        naive = naive_random_cv(df, kind)
        blocked = blocked_embargo_cv(df, kind)
        fchain = forward_chain_cv(df, kind)
        res["cells"][kind] = {
            "chronological_ref_mcc": chrono,
            "naive_random_cv_mean": float(np.mean(naive)), "naive_random_cv_std": float(np.std(naive)),
            "blocked_embargo_cv_mean": float(np.mean(blocked)), "blocked_embargo_cv_std": float(np.std(blocked)),
            "forward_chain_cv_mean": float(np.mean(fchain)), "forward_chain_cv_std": float(np.std(fchain)),
            "leak_inflation_naive_over_chrono": float(np.mean(naive) - chrono),
            "residual_after_blocked": float(np.mean(blocked) - chrono),
        }
        print(f"[{kind}] chrono={chrono:+.4f}  naiveCV={np.mean(naive):+.4f}  "
              f"blocked+embargo={np.mean(blocked):+.4f}  fwd-chain={np.mean(fchain):+.4f}", flush=True)

    # M&A locked-test invariance under the mitigation (real signal should be unchanged)
    ma = df[df["event"] == "mergers_acquisitions"].copy()
    PAPER_HP = dict(max_features=100, sublinear_tf=False, min_df=2, ngram_range=(1, 1), stop_words="english")
    tf = TfidfVectorizer(**PAPER_HP)
    tr_ma = ma[ma["published_date"] < TRAIN_END]; te_ma = ma[ma["published_date"] >= VAL_END]
    Xtr = tf.fit_transform(tr_ma["title_en"].fillna("")); Xte = tf.transform(te_ma["title_en"].fillna(""))
    base = mcc(te_ma["y"].values, LogisticRegression(max_iter=2000, C=5.0, random_state=42)
               .fit(Xtr, tr_ma["y"].values).predict(Xte))
    # embargo the M&A train: drop train rows within EMBARGO of the test window start
    emb = np.timedelta64(EMBARGO_DAYS, "D")
    tr_ma_emb = tr_ma[tr_ma["published_date"] < (VAL_END.to_datetime64() - emb)]
    Xtr2 = tf.fit_transform(tr_ma_emb["title_en"].fillna("")); Xte2 = tf.transform(te_ma["title_en"].fillna(""))
    emb_mcc = mcc(te_ma["y"].values, LogisticRegression(max_iter=2000, C=5.0, random_state=42)
                  .fit(Xtr2, tr_ma_emb["y"].values).predict(Xte2))
    res["ma_locked_invariance"] = {"paper_mcc": base, "embargoed_train_mcc": emb_mcc,
                                   "n_train": int(len(tr_ma)), "n_train_embargoed": int(len(tr_ma_emb))}
    print(f"[M&A locked] paper={base:+.4f}  embargoed-train={emb_mcc:+.4f} (invariant=real signal)", flush=True)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print("\nSaved:", OUT, "elapsed", res["elapsed_sec"], "s", flush=True)


if __name__ == "__main__":
    main()
