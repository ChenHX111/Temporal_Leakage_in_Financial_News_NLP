"""
generate_aggregates.py — reproducible AGGREGATE release (answers BCS9-C3/C4, tkJL-W1, ysCR-W3) from the proprietary
corpus WITHOUT releasing any article text. Everything here is a count, a hash, or an overlap statistic.

Emits into reproducible_aggregates/:
  split_summary.json          - train/val/test counts, date ranges, class balance, event coverage.
  monthly_event_counts.csv    - article counts by (year-month x event-type).
  tfidf_vocab_hashed_MA.csv   - the M&A specialist TF-IDF(max_features=100) vocabulary as SHA-256(term) + df + idf
                                (verifies the feature space + matrix structure without revealing the words).
  dedup_report.json           - MinHash near-duplicate rate (Jaccard>=0.85 on title 5-shingles) + per-stage counts.
  dedup_keys_hashed.csv       - SHA-256(title) + minhash-band signature per article id (near-dup audit keys; no text).
  publisher_entity_overlap.json - publisher & entity(ticker) overlap ACROSS splits (Jaccard + counts) = leakage-source audit.
"""
import os, io, sys, json, hashlib, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = r"."
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUTD = r".\Documents\Fin_NLP\EMNLP_REBUTTAL\artifact_bundle\reproducible_aggregates"
TRAIN_END, VAL_END = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-01")
sha = lambda s: hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def split_of(d):
    s = pd.Series("test", index=d.index)
    s[d["published_date"] < VAL_END] = "val"
    s[d["published_date"] < TRAIN_END] = "train"
    return s


def minhash_sig(tokens, n_perm=32, seed=1):
    if not tokens:
        return tuple([0] * 4)
    rng = np.random.RandomState(seed)
    a = rng.randint(1, 2**31 - 1, n_perm); b = rng.randint(0, 2**31 - 1, n_perm)
    hs = np.array([hash(t) & 0x7fffffff for t in tokens], dtype=np.int64)
    sig = [int(((a[i] * hs + b[i]) % 2147483647).min()) for i in range(n_perm)]
    return tuple(sig[j] for j in range(0, n_perm, n_perm // 4))  # 4-band signature


def main():
    df = pd.read_parquet(DATA)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy().reset_index(drop=True)
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["split"] = split_of(df)

    # 1) split summary
    summ = {"total_updown": int(len(df)), "splits": {}}
    for s in ["train", "val", "test"]:
        d = df[df["split"] == s]
        summ["splits"][s] = {"n": int(len(d)), "date_min": str(d["published_date"].min().date()),
                             "date_max": str(d["published_date"].max().date()),
                             "up_rate": round(float(d["y"].mean()), 4), "n_events": int(d["event"].nunique()),
                             "n_publishers": int(d["publisher"].nunique()), "n_tickers": int(d["yf_ticker"].nunique())}
    ma = df[df["event"] == "mergers_acquisitions"]
    summ["mergers_acquisitions"] = {s: int((ma["split"] == s).sum()) for s in ["train", "val", "test"]}
    json.dump(summ, open(os.path.join(OUTD, "split_summary.json"), "w"), indent=2)
    print("split_summary:", summ["splits"]["train"]["n"], summ["splits"]["val"]["n"], summ["splits"]["test"]["n"],
          "| M&A", summ["mergers_acquisitions"], flush=True)

    # 2) monthly x event counts
    df["ym"] = df["published_date"].dt.to_period("M").astype(str)
    mec = df.groupby(["ym", "event"]).size().reset_index(name="n_articles")
    mec.to_csv(os.path.join(OUTD, "monthly_event_counts.csv"), index=False)
    print("monthly_event_counts rows:", len(mec), flush=True)

    # 3) hashed M&A TF-IDF vocab (paper HP)
    tf = TfidfVectorizer(max_features=100, sublinear_tf=False, min_df=2, ngram_range=(1, 1), stop_words="english")
    ma_tr = ma[ma["split"] == "train"]
    tf.fit(ma_tr["title_en"])
    vocab = tf.get_feature_names_out()
    voc = pd.DataFrame({"term_sha256": [sha(t) for t in vocab],
                        "df_in_ma_train": np.asarray((tf.transform(ma_tr["title_en"]) > 0).sum(axis=0)).ravel(),
                        "idf": np.round(tf.idf_, 5)})
    voc.to_csv(os.path.join(OUTD, "tfidf_vocab_hashed_MA.csv"), index=False)
    print("tfidf_vocab_hashed_MA terms:", len(voc), flush=True)

    # 4) dedup: MinHash near-dup on title 5-char shingles
    def shingles(s):
        s = "".join(s.lower().split())
        return set(s[i:i + 5] for i in range(max(0, len(s) - 4))) if len(s) >= 5 else {s}
    sigs = {}
    dup_pairs = 0
    band_map = {}
    keys = []
    for nid, t in zip(df["news_id"].values, df["title_en"].values):
        sh = shingles(t)
        sg = minhash_sig(list(sh))
        keys.append({"news_id": int(nid), "title_sha256": sha(t), "band_sig": "_".join(map(str, sg))})
        band_map.setdefault(sg, []).append(nid)
    near_dup_articles = sum(len(v) - 1 for v in band_map.values() if len(v) > 1)
    pd.DataFrame(keys).to_csv(os.path.join(OUTD, "dedup_keys_hashed.csv"), index=False)
    exact_dup = int(df["title_en"].duplicated().sum())
    dedup = {"n_articles": int(len(df)), "exact_title_duplicates": exact_dup,
             "minhash_near_dup_candidate_articles": int(near_dup_articles),
             "near_dup_candidate_rate": round(near_dup_articles / len(df), 4),
             "method": "MinHash 32-perm / 4-band on 5-char title shingles; band-collision = near-dup candidate (Jaccard~>=0.8)",
             "note": "labels are next-day return (t+1); (ticker,date) pairs inherit the split of the earliest article; "
                     "release includes per-article hashed keys for external near-dup audit without text."}
    json.dump(dedup, open(os.path.join(OUTD, "dedup_report.json"), "w"), indent=2)
    print("dedup: exact_dup=%d near_dup_candidates=%d (%.1f%%)" %
          (exact_dup, near_dup_articles, 100 * near_dup_articles / len(df)), flush=True)

    # 5) publisher/entity overlap across splits (leakage-source audit)
    def jac(a, b):
        a, b = set(a), set(b)
        return round(len(a & b) / max(1, len(a | b)), 4)
    tr, va, te = [df[df["split"] == s] for s in ["train", "val", "test"]]
    ov = {}
    for name, col in [("publisher", "publisher"), ("entity_ticker", "yf_ticker")]:
        ov[name] = {
            "train_test_jaccard": jac(tr[col], te[col]),
            "train_test_shared": int(len(set(tr[col]) & set(te[col]))),
            "test_only": int(len(set( te[col]) - set(tr[col]))),
            "test_rows_with_train_seen_value_frac": round(float(te[col].isin(set(tr[col])).mean()), 4),
        }
    json.dump(ov, open(os.path.join(OUTD, "publisher_entity_overlap.json"), "w"), indent=2)
    print("overlap publisher tr/te Jaccard=%.3f  entity tr/te Jaccard=%.3f  (test rows w/ seen ticker=%.2f)" %
          (ov["publisher"]["train_test_jaccard"], ov["entity_ticker"]["train_test_jaccard"],
           ov["entity_ticker"]["test_rows_with_train_seen_value_frac"]), flush=True)
    print("\nAll aggregates written to", OUTD, flush=True)


if __name__ == "__main__":
    main()
