"""
E1 — Neutrals-included / ex-ante-observable M&A robustness  (answers tkJL-W3 survivorship / ex-ante-unobservability).

Faithful to the PAPER-AUTHORITATIVE protocol (cost_aware_backtest_paperHP.py):
  M&A = event=='mergers_acquisitions'; text = title_en; label y = (actual_side=='up');
  splits TRAIN_END=2025-04-01, VAL_END=2025-06-01 (train_only headline = 0.138);
  TF-IDF(max_features=100, sublinear_tf=False, min_df=2, ngram=(1,1), stop_words='english') + LR(C=5.0, seed=42).

Outputs (JSON) to EMNLP_REBUTTAL/experiments/out/E1_neutrals.json. NO change to the repo.

Reported cells:
  S  sanity          : reproduce the up/down-only headline (expect ~0.138 train_only; ~0.068 train+val).
  A  tau0_exante     : include ALL M&A rows (incl. the 303 'neutral'); label = sign(price_change_percentage) at tau=0
                       -> inclusion is outcome-independent (fully ex-ante observable). MCC + 10k-perm p.
  B  three_class     : 3-class UP/NEUTRAL/DOWN via actual_side; multiclass MCC + macro-F1 (train_only, test).
  C  neutral_band    : neutral defined by |price_change_percentage|<tau for tau in {0.3,0.5,1.0}; binary MCC on extremes.
  D  audit_invariance: the random-vs-chronological audit gap on M&A under (i) paper up/down-only vs (ii) tau0 all-rows;
                       shows neutral-filtering does NOT manufacture the leakage gap (same filter on both arms).
"""
import os, io, sys, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, f1_score
from joblib import Parallel, delayed

BASE = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "E1_neutrals.json")
PAPER_HP = dict(max_features=100, sublinear_tf=False, min_df=2, ngram_range=(1, 1), stop_words="english")
PAPER_C, SEED = 5.0, 42
TRAIN_END, VAL_END = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-01")


def mcc(y, p):
    return 0.0 if (len(np.unique(y)) < 2 or len(np.unique(p)) < 2) else float(matthews_corrcoef(y, p))


def fit_eval(tr_txt, tr_y, te_txt, te_y, C=PAPER_C):
    tf = TfidfVectorizer(**PAPER_HP)
    Xtr = tf.fit_transform(tr_txt); Xte = tf.transform(te_txt)
    clf = LogisticRegression(max_iter=2000, C=C, random_state=SEED).fit(Xtr, tr_y)
    return clf.predict(Xte)


def perm_p(y_true, y_pred, n=10000):
    obs = mcc(y_true, y_pred)
    def one(s):
        return matthews_corrcoef(np.random.RandomState(s).permutation(y_true), y_pred)
    ms = np.array(Parallel(n_jobs=-1, batch_size=200)(delayed(one)(s) for s in range(n)))
    return dict(observed_mcc=obs, p_one=float(np.mean(ms >= obs)),
                p_two=float(np.mean(np.abs(ms) >= abs(obs))),
                z=float((obs - ms.mean()) / (ms.std() + 1e-12)))


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    df["pcp"] = pd.to_numeric(df["price_change_percentage"], errors="coerce")
    ma_all = df[df["event"] == "mergers_acquisitions"].copy()
    res = {"meta": {"n_ma_total": int(len(ma_all)),
                    "n_ma_neutral": int((ma_all["actual_side"].str.lower() == "neutral").sum())}}

    def split(d):
        return (d[d["published_date"] < TRAIN_END], d[d["published_date"] < VAL_END], d[d["published_date"] >= VAL_END])

    # --- S: sanity (paper up/down-only) ---
    ma = ma_all[ma_all["actual_side"].str.lower().isin(["up", "down"])].copy()
    ma["y"] = (ma["actual_side"].str.lower() == "up").astype(int)
    tr, trv, te = split(ma)
    yp_to = fit_eval(tr["title_en"], tr["y"].values, te["title_en"], te["y"].values)
    yp_tv = fit_eval(trv["title_en"], trv["y"].values, te["title_en"], te["y"].values)
    res["S_sanity"] = {"train_only_mcc": mcc(te["y"].values, yp_to), "expect_0.138": True,
                       "train_val_mcc": mcc(te["y"].values, yp_tv), "expect_0.068": True,
                       "n_train_only": int(len(tr)), "n_test": int(len(te))}
    print("S sanity train_only MCC =", round(res["S_sanity"]["train_only_mcc"], 4),
          " train+val MCC =", round(res["S_sanity"]["train_val_mcc"], 4), flush=True)

    # --- A: tau=0 ex-ante (all rows incl. neutrals; label = sign(pcp)) ---
    maa = ma_all[ma_all["pcp"].notna()].copy()
    maa["y"] = (maa["pcp"] > 0).astype(int)
    tr, trv, te = split(maa)
    yp = fit_eval(tr["title_en"], tr["y"].values, te["title_en"], te["y"].values)
    A = {"n_train": int(len(tr)), "n_test": int(len(te)), "test_up_rate": float(te["y"].mean())}
    A.update(perm_p(te["y"].values, yp))
    res["A_tau0_exante"] = A
    print("A tau0-exante MCC =", round(A["observed_mcc"], 4), " p_two =", A["p_two"],
          " (n_test=%d, incl neutrals)" % len(te), flush=True)

    # --- B: 3-class UP/NEUTRAL/DOWN ---
    m3 = {"up": 2, "neutral": 1, "down": 0}
    mb = ma_all.copy(); mb["y3"] = mb["actual_side"].str.lower().map(m3)
    mb = mb[mb["y3"].notna()].copy(); mb["y3"] = mb["y3"].astype(int)
    tr, trv, te = split(mb)
    yp3 = fit_eval(tr["title_en"], tr["y3"].values, te["title_en"], te["y3"].values)
    res["B_three_class"] = {"multiclass_mcc": mcc(te["y3"].values, yp3),
                            "macro_f1": float(f1_score(te["y3"].values, yp3, average="macro")),
                            "n_train": int(len(tr)), "n_test": int(len(te)),
                            "test_dist": {int(k): int(v) for k, v in pd.Series(te["y3"]).value_counts().items()}}
    print("B 3-class MCC =", round(res["B_three_class"]["multiclass_mcc"], 4),
          " macroF1 =", round(res["B_three_class"]["macro_f1"], 4), flush=True)

    # --- C: neutral-band sensitivity (extremes-only binary at various tau) ---
    Cc = {}
    for tau in [0.3, 0.5, 1.0]:
        d = ma_all[ma_all["pcp"].notna()].copy()
        d = d[np.abs(d["pcp"]) >= tau].copy(); d["y"] = (d["pcp"] > 0).astype(int)
        tr, trv, te = split(d)
        if len(te) < 40 or te["y"].nunique() < 2:
            Cc[f"tau_{tau}"] = {"note": f"n_test={len(te)}"}; continue
        yp = fit_eval(tr["title_en"], tr["y"].values, te["title_en"], te["y"].values)
        Cc[f"tau_{tau}"] = {"mcc": mcc(te["y"].values, yp), "n_train": int(len(tr)), "n_test": int(len(te))}
    res["C_neutral_band"] = Cc
    print("C neutral-band:", {k: (round(v["mcc"], 3) if "mcc" in v else v) for k, v in Cc.items()}, flush=True)

    # --- D: audit-ratio invariance (random vs chronological) under up/down-only vs tau0-all ---
    def audit_gap(d, ycol, K=10):
        tr, trv, te = split(d)
        yp = fit_eval(tr["title_en"], tr[ycol].values, te["title_en"], te[ycol].values)
        temporal = mcc(te[ycol].values, yp)
        pool = d.reset_index(drop=True); n_tr, n_te = len(tr), len(te)
        rand = []
        for k in range(K):
            idx = np.random.RandomState(100 + k).permutation(len(pool))
            rtr, rte = pool.iloc[idx[:n_tr]], pool.iloc[idx[n_tr:n_tr + n_te]]
            if rtr[ycol].nunique() < 2:
                continue
            yp_r = fit_eval(rtr["title_en"], rtr[ycol].values, rte["title_en"], rte[ycol].values)
            rand.append(mcc(rte[ycol].values, yp_r))
        rand = np.array(rand)
        return dict(temporal_mcc=temporal, random_mcc_mean=float(rand.mean()), random_mcc_std=float(rand.std()),
                    delta_mcc=float(rand.mean() - temporal), n_test=int(n_te))
    updo = ma_all[ma_all["actual_side"].str.lower().isin(["up", "down"])].copy()
    updo["y"] = (updo["actual_side"].str.lower() == "up").astype(int)
    tau0 = ma_all[ma_all["pcp"].notna()].copy(); tau0["y"] = (tau0["pcp"] > 0).astype(int)
    res["D_audit_invariance"] = {"updown_only_paper": audit_gap(updo, "y"),
                                 "tau0_all_rows": audit_gap(tau0, "y")}
    print("D audit dMCC: updown_only =", round(res["D_audit_invariance"]["updown_only_paper"]["delta_mcc"], 4),
          " tau0_all =", round(res["D_audit_invariance"]["tau0_all_rows"]["delta_mcc"], 4), flush=True)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("\nSaved:", OUT, "  elapsed", res["elapsed_sec"], "s", flush=True)


if __name__ == "__main__":
    main()
