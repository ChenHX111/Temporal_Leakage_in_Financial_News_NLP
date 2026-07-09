"""
NER + dependency-parse based role attribution for M&A — addresses W3.

Audit weakness W3: regex-based ACQUIRER/TARGET classification is brittle. Tier B
role_stability showed MCC range [-0.187, +0.331] across 4 regex variants — large
variance suggesting the headline asymmetry could be a regex artifact.

This script uses linguistic structure instead of keyword matching:
    1. spaCy NER finds ORG entities in each M&A title.
    2. Dependency parser finds acquire-verbs (acquire, buy, merge, takeover,
       purchase, sell, divest) and their nsubj/dobj/nsubjpass/agent slots.
    3. The structural role of the FIRST ORG in the title is taken as the
       focal-company role:
           - first ORG is nsubj of acquire-verb -> ACQUIRER
           - first ORG is dobj of acquire-verb -> TARGET
           - first ORG is nsubjpass of acquired-verb -> TARGET
           - first ORG is agent ("by X") of acquired-verb -> ACQUIRER
           - else -> UNCLEAR

Then we replicate the ACQUIRER and TARGET specialists on the NER-defined
subsets and compare to the regex-defined headline (MCC ACQUIRER = +0.160,
TARGET = 0.000 degenerate).

Output: results/validation/ner_role_attribution.json
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
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

BASE = r"."
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(BASE, "results", "validation", "ner_role_attribution.json")

ACQ_LEMMA = {"acquire", "buy", "purchase", "take", "merge", "absorb"}
SELL_LEMMA = {"sell", "divest"}
ALL_DEAL_LEMMA = ACQ_LEMMA | SELL_LEMMA

ACQ_NOUNS = {"acquisition", "merger", "takeover", "buyout"}


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def first_org_span(doc):
    """Return (text, start, end) for the first ORG entity, or None."""
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return (ent.text, ent.start, ent.end)
    return None


def token_dominant_dep_relative_to_verb(token, verb):
    """Walk up from token; if any ancestor is the verb, return the dep label
    on the path step into that verb."""
    cur = token
    visited = 0
    while cur.head is not cur and visited < 8:
        if cur.head is verb:
            return cur.dep_
        cur = cur.head
        visited += 1
    return None


def classify_role_ner(doc, title_text):
    """Return one of: ACQUIRER, TARGET, BOTH, NEITHER, UNCLEAR_NO_ORG, UNCLEAR_NO_VERB."""
    first_org = first_org_span(doc)
    if first_org is None:
        return "UNCLEAR_NO_ORG", None
    org_text, org_start, org_end = first_org

    # Find a relevant acquire-verb in the doc
    verbs = []
    for t in doc:
        if t.pos_ in ("VERB", "AUX") and t.lemma_.lower() in ALL_DEAL_LEMMA:
            verbs.append(t)
        elif t.pos_ == "NOUN" and t.lemma_.lower() in ACQ_NOUNS:
            # Promote nouns like "acquisition of X" to a pseudo-verb test
            verbs.append(t)
    if not verbs:
        return "UNCLEAR_NO_VERB", None

    # Determine role of each token in the first ORG span relative to each verb
    role_votes = []
    for verb in verbs:
        verb_lem = verb.lemma_.lower()
        for tok_idx in range(org_start, org_end):
            tok = doc[tok_idx]
            dep_to_verb = token_dominant_dep_relative_to_verb(tok, verb)
            if dep_to_verb is None:
                continue
            # ACQ verbs (active): subj = acquirer; dobj = target
            if verb_lem in ACQ_LEMMA:
                if dep_to_verb in ("nsubj", "compound"):
                    role_votes.append(("ACQUIRER", verb.text))
                elif dep_to_verb == "dobj":
                    role_votes.append(("TARGET", verb.text))
                elif dep_to_verb == "nsubjpass":
                    role_votes.append(("TARGET", verb.text))
                elif dep_to_verb in ("pobj", "agent"):
                    # "X to be acquired by Y": Y is agent => Y=acquirer; X=subjpass=target
                    if tok.head.text.lower() == "by" or tok.head.dep_ == "agent":
                        role_votes.append(("ACQUIRER", verb.text))
                    elif tok.head.text.lower() == "of":
                        role_votes.append(("TARGET", verb.text))
            elif verb_lem in SELL_LEMMA:
                if dep_to_verb in ("nsubj", "compound"):
                    role_votes.append(("TARGET", verb.text))  # seller (target side)
                elif dep_to_verb == "dobj":
                    role_votes.append(("TARGET", verb.text))  # "sells X" - X is divestiture target
                elif dep_to_verb in ("pobj", "agent"):
                    role_votes.append(("ACQUIRER", verb.text))  # buyer
            else:
                # noun head like "acquisition of X by Y"
                if dep_to_verb in ("compound", "nmod", "poss"):
                    # X's acquisition - X=acquirer typically
                    role_votes.append(("ACQUIRER", verb.text))
                elif dep_to_verb == "pobj":
                    if tok.head.text.lower() == "of":
                        role_votes.append(("TARGET", verb.text))
                    elif tok.head.text.lower() == "by":
                        role_votes.append(("ACQUIRER", verb.text))

    if not role_votes:
        return "UNCLEAR_NO_VERB", None
    acq_n = sum(1 for r, _ in role_votes if r == "ACQUIRER")
    tgt_n = sum(1 for r, _ in role_votes if r == "TARGET")
    if acq_n > 0 and tgt_n == 0:
        return "ACQUIRER", role_votes
    if tgt_n > 0 and acq_n == 0:
        return "TARGET", role_votes
    if acq_n > 0 and tgt_n > 0:
        return "BOTH", role_votes
    return "NEITHER", role_votes


def main():
    t0 = time.time()
    print("Loading data ...", flush=True)
    df = pd.read_parquet(DATA)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["title_en"] = df["title_en"].fillna("").astype(str)

    ma = df[df["event"] == "mergers_acquisitions"].copy()
    TRAIN_END = pd.Timestamp("2025-04-01")
    VAL_END = pd.Timestamp("2025-06-01")
    print(f"M&A total: {len(ma)}", flush=True)

    print("Loading spaCy en_core_web_sm ...", flush=True)
    nlp = spacy.load("en_core_web_sm")
    print("Tagging roles via NER + dep-parse (may take ~2 min) ...", flush=True)

    roles = []
    rationales = []
    titles = ma["title_en"].tolist()
    t_tag0 = time.time()
    for k, doc in enumerate(nlp.pipe(titles, batch_size=128)):
        role, votes = classify_role_ner(doc, titles[k])
        roles.append(role)
        rationales.append(votes)
        if (k + 1) % 500 == 0:
            print(f"  tagged {k+1}/{len(titles)}  ({time.time()-t_tag0:.0f}s)", flush=True)
    ma["role_ner"] = roles
    print(f"  NER tagging done in {time.time()-t_tag0:.0f}s", flush=True)

    role_counts = ma["role_ner"].value_counts().to_dict()
    print("\n[NER role distribution (all M&A)]", flush=True)
    for k, v in role_counts.items():
        print(f"  {k:>20}: {v}", flush=True)

    # Sample examples for each role for sanity
    examples = {}
    for r in ["ACQUIRER", "TARGET", "BOTH", "NEITHER", "UNCLEAR_NO_ORG",
              "UNCLEAR_NO_VERB"]:
        sub = ma[ma["role_ner"] == r].head(5)["title_en"].tolist()
        examples[r] = sub

    # === Compare to regex roles for cross-tab ===
    import re as _re
    ACQ_PAT = _re.compile(
        r"\b(acqui|acquir|takeover|buyer|buy[- ]?out|purchas|to acquire|will acquire|"
        r"completes acquisition|launches offer|tender offer|bid for|offers? to buy)\b",
        _re.IGNORECASE)
    TGT_PAT = _re.compile(
        r"\b(target|to be acquir|being acquir|to be sold|sold to|sale of|divest|"
        r"subject of (an? )?bid|merger with|to merge with|received offer|"
        r"agree(s|d)? to be acquir)\b",
        _re.IGNORECASE)

    def regex_role(t):
        a = bool(ACQ_PAT.search(t)); g = bool(TGT_PAT.search(t))
        if a and g: return "BOTH"
        if a: return "ACQUIRER"
        if g: return "TARGET"
        return "NEITHER"

    ma["role_regex"] = ma["title_en"].apply(regex_role)
    cross = pd.crosstab(ma["role_regex"], ma["role_ner"])
    print("\n[Cross-tab: regex (rows) vs NER (cols)]", flush=True)
    print(cross.to_string(), flush=True)

    # === Specialist models on NER-defined subsets ===
    train_full = ma[ma["published_date"] < VAL_END].copy()
    test_full = ma[ma["published_date"] >= VAL_END].copy()

    def specialist_eval(role_label, tag_col="role_ner"):
        tr = train_full[train_full[tag_col] == role_label]
        te = test_full[test_full[tag_col] == role_label]
        if len(tr) < 20 or len(te) < 20:
            return {"n_train": len(tr), "n_test": len(te), "mcc": None,
                    "balacc": None, "proba_std": None, "pred_up_frac": None}
        if len(np.unique(te["y"])) < 2:
            return {"n_train": len(tr), "n_test": len(te), "mcc": None,
                    "balacc": None, "proba_std": None,
                    "pred_up_frac": float(te["y"].mean()),
                    "note": "test single-class"}
        tf = TfidfVectorizer(max_features=300, stop_words="english",
                             min_df=2, sublinear_tf=True)
        Xtr = tf.fit_transform(tr["title_en"]); Xte = tf.transform(te["title_en"])
        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        clf.fit(Xtr, tr["y"].values)
        yp = clf.predict(Xte); yp_proba = clf.predict_proba(Xte)[:, 1]
        return {"n_train": len(tr), "n_test": len(te),
                "mcc": safe_mcc(te["y"].values, yp),
                "balacc": float(balanced_accuracy_score(te["y"].values, yp)),
                "proba_std": float(yp_proba.std()),
                "pred_up_frac": float(yp.mean())}

    print("\n[Specialist eval on NER-defined subsets]", flush=True)
    results_ner = {}
    for role in ["ACQUIRER", "TARGET"]:
        r = specialist_eval(role, "role_ner")
        results_ner[role] = r
        print(f"  {role}: n_train={r['n_train']}, n_test={r['n_test']}, "
              f"MCC={r['mcc']}, balacc={r['balacc']}, "
              f"proba_std={r['proba_std']}, pred_up_frac={r['pred_up_frac']}",
              flush=True)

    print("\n[Specialist eval on REGEX-defined subsets (sanity)]", flush=True)
    results_regex = {}
    for role in ["ACQUIRER", "TARGET"]:
        r = specialist_eval(role, "role_regex")
        results_regex[role] = r
        print(f"  {role}: n_train={r['n_train']}, n_test={r['n_test']}, "
              f"MCC={r['mcc']}, balacc={r['balacc']}, "
              f"proba_std={r['proba_std']}, pred_up_frac={r['pred_up_frac']}",
              flush=True)

    # === Permutation test on NER-ACQUIRER (if has signal) ===
    if (results_ner["ACQUIRER"]["mcc"] is not None and
            results_ner["ACQUIRER"]["n_test"] >= 30):
        print("\n[10K permutation test on NER-ACQUIRER test]", flush=True)
        sub_te = test_full[test_full["role_ner"] == "ACQUIRER"]
        sub_tr = train_full[train_full["role_ner"] == "ACQUIRER"]
        tf = TfidfVectorizer(max_features=300, stop_words="english", min_df=2, sublinear_tf=True)
        Xtr = tf.fit_transform(sub_tr["title_en"]); Xte = tf.transform(sub_te["title_en"])
        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        clf.fit(Xtr, sub_tr["y"].values)
        yp = clf.predict(Xte)
        obs = results_ner["ACQUIRER"]["mcc"]
        rng = np.random.default_rng(42)
        null = np.array([safe_mcc(rng.permutation(sub_te["y"].values), yp)
                         for _ in range(10000)])
        p_one = float((null >= obs).mean())
        p_two = float((np.abs(null) >= abs(obs)).mean())
        z = (obs - null.mean()) / (null.std() + 1e-12)
        results_ner["ACQUIRER"]["perm_test"] = {
            "observed": obs, "null_mean": float(null.mean()),
            "null_std": float(null.std()), "z": float(z),
            "p_one_sided": p_one, "p_two_sided": p_two, "n_perm": 10000}
        print(f"  observed={obs:+.4f}, null_mean={null.mean():+.4f}, "
              f"z={z:.2f}, p_one={p_one:.4f}, p_two={p_two:.4f}", flush=True)

    out = {
        "meta": {"timestamp": pd.Timestamp.now().isoformat(),
                 "elapsed_s": float(time.time() - t0),
                 "n_ma_total": int(len(ma))},
        "role_counts_ner": role_counts,
        "examples_per_role": examples,
        "cross_regex_ner": cross.to_dict(),
        "specialist_eval_ner_roles": results_ner,
        "specialist_eval_regex_roles": results_regex,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
