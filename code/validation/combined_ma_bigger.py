"""
A4: Combined-corpus M&A bigger test.

Train a single TF-IDF logistic regression M&A specialist on:
  (our train M&A) + (EDT 2020-Q3 narrow M&A first 70% chronological)

Evaluate on TWO held-out test sets:
  Test 1 - our test M&A (n=786, 2025 Jun-Aug)
  Test 2 - EDT 2021 narrow M&A held-out 30% chronological

A4.5: Earnings event negative control - replicate the same protocol on 'earnings'
event subset; we predict it should NOT show signal because the paper claims M&A
is the only event type with signal.

Output: results/validation/combined_ma_bigger.json
"""
import os
import sys
import io
import json
import re
import time
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

BASE = r"."
OUR = os.path.join(BASE, "data", "classifier_training_v2.parquet")
EDT = os.path.join(BASE, "data", "external", "edt_evaluate_slim.parquet")
OUT = os.path.join(BASE, "results", "validation", "combined_ma_bigger.json")

NARROW_MA = re.compile(r"\b(merger|acquisition|acquir|takeover|tender offer)\b", re.IGNORECASE)


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def load_our():
    df = pd.read_parquet(OUR)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    return df


def load_edt():
    df = pd.read_parquet(EDT)
    df["pub_time"] = pd.to_datetime(df["pub_time"]).dt.tz_localize(None)
    df["title"] = df["title"].fillna("").astype(str)
    return df


def main():
    t0 = time.time()
    rng = np.random.default_rng(42)
    print("Loading our data ...", flush=True)
    ours = load_our()
    our_ma = ours[ours["event"] == "mergers_acquisitions"].copy()
    print(f"  our M&A: {len(our_ma)}", flush=True)

    OUR_TRAIN_END = pd.Timestamp("2025-04-01")
    OUR_VAL_END = pd.Timestamp("2025-06-01")
    our_ma_tr = our_ma[our_ma["published_date"] < OUR_VAL_END].copy()
    our_ma_te = our_ma[our_ma["published_date"] >= OUR_VAL_END].copy()

    print("Loading EDT ...", flush=True)
    edt = load_edt()
    edt_ma = edt[edt["title"].apply(lambda s: bool(NARROW_MA.search(s)))].copy()
    edt_ma = edt_ma.sort_values("pub_time").reset_index(drop=True)
    print(f"  EDT narrow M&A: {len(edt_ma)}  date range "
          f"{edt_ma['pub_time'].min()} to {edt_ma['pub_time'].max()}", flush=True)

    # EDT chronological split: first 70% train, last 30% test
    cut = int(len(edt_ma) * 0.7)
    edt_ma_tr = edt_ma.iloc[:cut].copy()
    edt_ma_te = edt_ma.iloc[cut:].copy()
    cut_date = edt_ma_tr["pub_time"].max()
    print(f"  EDT chrono cut at {cut_date}: train={len(edt_ma_tr)}  test={len(edt_ma_te)}",
          flush=True)

    out = {"meta": {"timestamp": pd.Timestamp.now().isoformat(),
                    "edt_chrono_cut_date": str(cut_date)},
           "n": {"our_train": len(our_ma_tr), "our_test": len(our_ma_te),
                 "edt_train": len(edt_ma_tr), "edt_test": len(edt_ma_te)},
           "experiments": {}}

    # === Experiment 1: OUR-only baseline (replicate paper) ===
    print("\n[E1] OUR-only baseline (paper replication) ...", flush=True)
    tf = TfidfVectorizer(max_features=300, stop_words="english", min_df=2, sublinear_tf=True)
    Xtr = tf.fit_transform(our_ma_tr["title_en"])
    Xte = tf.transform(our_ma_te["title_en"])
    ytr = our_ma_tr["y"].values
    yte = our_ma_te["y"].values
    cl = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    cl.fit(Xtr, ytr)
    yp = cl.predict(Xte)
    e1_mcc = safe_mcc(yte, yp)
    print(f"  E1 our-only: train={len(ytr)} test={len(yte)} MCC={e1_mcc:+.4f}", flush=True)
    out["experiments"]["E1_our_only"] = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
                                          "mcc": float(e1_mcc),
                                          "balacc": float(balanced_accuracy_score(yte, yp))}

    # === Experiment 2: EDT-only baseline (replicate Section 6.4 EDT narrow) ===
    print("\n[E2] EDT-only baseline ...", flush=True)
    tf2 = TfidfVectorizer(max_features=300, stop_words="english", min_df=2, sublinear_tf=True)
    Xtr_e = tf2.fit_transform(edt_ma_tr["title"])
    Xte_e = tf2.transform(edt_ma_te["title"])
    ytr_e = edt_ma_tr["y"].values
    yte_e = edt_ma_te["y"].values
    cl2 = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    cl2.fit(Xtr_e, ytr_e)
    yp_e = cl2.predict(Xte_e)
    e2_mcc = safe_mcc(yte_e, yp_e)
    print(f"  E2 edt-only: train={len(ytr_e)} test={len(yte_e)} MCC={e2_mcc:+.4f}", flush=True)
    out["experiments"]["E2_edt_only"] = {"n_train": int(len(ytr_e)), "n_test": int(len(yte_e)),
                                          "mcc": float(e2_mcc),
                                          "balacc": float(balanced_accuracy_score(yte_e, yp_e))}

    # === Experiment 3: COMBINED training, two test sets ===
    print("\n[E3] COMBINED training, two test sets ...", flush=True)
    # Concatenate text and labels (titles only - our paper's headline result is title-based)
    combined_titles = pd.concat([our_ma_tr["title_en"], edt_ma_tr["title"]],
                                ignore_index=True).values
    combined_y = np.concatenate([our_ma_tr["y"].values, edt_ma_tr["y"].values])
    print(f"  combined train: {len(combined_titles)}", flush=True)

    tf3 = TfidfVectorizer(max_features=300, stop_words="english", min_df=2, sublinear_tf=True)
    Xtr_c = tf3.fit_transform(combined_titles)
    Xte_our = tf3.transform(our_ma_te["title_en"])
    Xte_edt = tf3.transform(edt_ma_te["title"])
    cl3 = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    cl3.fit(Xtr_c, combined_y)
    yp_our = cl3.predict(Xte_our)
    yp_edt = cl3.predict(Xte_edt)
    e3_our_mcc = safe_mcc(yte, yp_our)
    e3_edt_mcc = safe_mcc(yte_e, yp_edt)
    print(f"  E3 -> our test : MCC={e3_our_mcc:+.4f} (n={len(yte)})", flush=True)
    print(f"  E3 -> edt test : MCC={e3_edt_mcc:+.4f} (n={len(yte_e)})", flush=True)
    out["experiments"]["E3_combined"] = {
        "n_train": int(len(combined_y)),
        "test_our": {"n": int(len(yte)), "mcc": float(e3_our_mcc),
                     "balacc": float(balanced_accuracy_score(yte, yp_our))},
        "test_edt": {"n": int(len(yte_e)), "mcc": float(e3_edt_mcc),
                     "balacc": float(balanced_accuracy_score(yte_e, yp_edt))},
    }

    # === Experiment 4: Cross-corpus transfer ===
    # Train on EDT only -> evaluate on our test
    print("\n[E4] Cross-corpus transfer ...", flush=True)
    yp_x1 = cl2.predict(tf2.transform(our_ma_te["title_en"]))
    e4_edt_to_our = safe_mcc(yte, yp_x1)
    yp_x2 = cl.predict(tf.transform(edt_ma_te["title"]))
    e4_our_to_edt = safe_mcc(yte_e, yp_x2)
    print(f"  EDT->ours MCC: {e4_edt_to_our:+.4f}", flush=True)
    print(f"  ours->EDT MCC: {e4_our_to_edt:+.4f}", flush=True)
    out["experiments"]["E4_transfer"] = {
        "edt_to_our": {"n": int(len(yte)), "mcc": float(e4_edt_to_our)},
        "our_to_edt": {"n": int(len(yte_e)), "mcc": float(e4_our_to_edt)},
    }

    # === Permutation tests on combined-trained results ===
    n_perm = 5000
    print(f"\n[Perm] {n_perm} perms on E3 outputs ...", flush=True)
    null_our = np.zeros(n_perm); null_edt = np.zeros(n_perm)
    for k in range(n_perm):
        null_our[k] = safe_mcc(rng.permutation(yte), yp_our)
        null_edt[k] = safe_mcc(rng.permutation(yte_e), yp_edt)
    p_our_one = float((null_our >= e3_our_mcc).mean())
    p_edt_one = float((null_edt >= e3_edt_mcc).mean())
    print(f"  E3 our: p_one={p_our_one:.4f}", flush=True)
    print(f"  E3 edt: p_one={p_edt_one:.4f}", flush=True)
    out["experiments"]["E3_combined"]["test_our"]["p_one_sided"] = p_our_one
    out["experiments"]["E3_combined"]["test_edt"]["p_one_sided"] = p_edt_one

    # === Multi-period EDT split (B2 lite): 2020-Q1, Q2, Q3, Q4, 2021-Q1, Q2 ===
    print("\n[E5] EDT multi-period decomposition ...", flush=True)
    edt_ma_sorted = edt_ma.sort_values("pub_time").reset_index(drop=True)
    edt_ma_sorted["quarter"] = edt_ma_sorted["pub_time"].dt.to_period("Q").astype(str)
    quarters = sorted(edt_ma_sorted["quarter"].unique())
    period_results = []
    for q in quarters:
        sub = edt_ma_sorted[edt_ma_sorted["quarter"] == q]
        if len(sub) < 100:
            continue
        # 70/30 chronological within quarter
        cut_q = int(len(sub) * 0.7)
        tr_q = sub.iloc[:cut_q]; te_q = sub.iloc[cut_q:]
        if len(tr_q) < 50 or len(te_q) < 30:
            continue
        try:
            tfq = TfidfVectorizer(max_features=200, stop_words="english", min_df=2,
                                  sublinear_tf=True)
            Xtq = tfq.fit_transform(tr_q["title"])
            Xeq = tfq.transform(te_q["title"])
            clq = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
            clq.fit(Xtq, tr_q["y"].values)
            ypq = clq.predict(Xeq)
            mccq = safe_mcc(te_q["y"].values, ypq)
            period_results.append({
                "quarter": q,
                "n_train": int(len(tr_q)), "n_test": int(len(te_q)),
                "mcc": float(mccq),
                "true_up": float(te_q["y"].mean()),
                "pred_up": float(ypq.mean()),
            })
            print(f"  {q}: train={len(tr_q)} test={len(te_q)} MCC={mccq:+.4f}", flush=True)
        except Exception as exc:
            print(f"  {q}: ERR {exc}", flush=True)
    out["experiments"]["E5_edt_per_quarter"] = period_results

    # === A4.5: Earnings event negative control ===
    print("\n[E6] Earnings event NEGATIVE CONTROL ...", flush=True)
    # Identify earnings events in our data; the event taxonomy may have several earnings labels
    event_counts = ours["event"].value_counts()
    print("  available events sample:", flush=True)
    for ev in event_counts.head(20).index.tolist():
        print(f"    {ev}: {event_counts[ev]}", flush=True)
    earnings_evs = [e for e in event_counts.index if "earning" in str(e).lower() or
                    "result" in str(e).lower() or "profit" in str(e).lower()]
    print(f"  earnings-like events found: {earnings_evs[:10]}", flush=True)
    earn = ours[ours["event"].isin(earnings_evs)].copy() if earnings_evs else ours.iloc[:0]
    print(f"  earnings articles: {len(earn)}", flush=True)
    earn_tr = earn[earn["published_date"] < OUR_VAL_END]
    earn_te = earn[earn["published_date"] >= OUR_VAL_END]
    if len(earn_tr) > 100 and len(earn_te) > 50:
        tfe = TfidfVectorizer(max_features=300, stop_words="english", min_df=2,
                              sublinear_tf=True)
        Xtre = tfe.fit_transform(earn_tr["title_en"])
        Xtee = tfe.transform(earn_te["title_en"])
        cle = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        cle.fit(Xtre, earn_tr["y"].values)
        ype = cle.predict(Xtee)
        e6_mcc = safe_mcc(earn_te["y"].values, ype)
        print(f"  earnings MCC: train={len(earn_tr)} test={len(earn_te)} MCC={e6_mcc:+.4f}",
              flush=True)
        out["experiments"]["E6_earnings_negcontrol"] = {
            "events_used": earnings_evs,
            "n_train": int(len(earn_tr)), "n_test": int(len(earn_te)),
            "mcc": float(e6_mcc),
            "true_up": float(earn_te["y"].mean()),
            "pred_up": float(ype.mean()),
        }
    else:
        print(f"  insufficient earnings data: tr={len(earn_tr)} te={len(earn_te)}", flush=True)
        out["experiments"]["E6_earnings_negcontrol"] = {
            "events_used": earnings_evs, "n_train": int(len(earn_tr)),
            "n_test": int(len(earn_te)), "mcc": None,
            "note": "insufficient",
        }

    # === Save ===
    out["meta"]["elapsed_s"] = float(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)
    print(f"Total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
