"""
NER + dep-parse role attribution v2.

v1 failed (0 of either role) because:
    - 27% of M&A titles have no ORG entity (use tickers or generic words)
    - 72% have ORG entities but my dep-rule filter was too narrow
      (most press releases use NOUN forms: "Completes Acquisition of X",
       "Announces Closing of Acquisition of Y")

v2 approach: walk from ROOT verb and look for deal-noun/verb anywhere in the
parse tree. Use:
    - Active deal verbs (acquires, buys, purchases, merges, absorbs)
       -> subj of verb = ACQUIRER, dobj = TARGET
    - Deal nouns (acquisition, merger, takeover, buyout, purchase, offer)
       used with "of X"/"for X" -> X = TARGET; subject of governing verb = ACQUIRER
    - "be acquired" / "to be acquired" passive -> nsubjpass = TARGET, agent = ACQUIRER
    - Sell verbs -> subj is SELLER (TARGET-side in focal-co terms)
    - "purchase offer for X" / "tender offer for X" -> X = TARGET

Plus a positional fallback if dep-parse fails: find FIRST deal keyword;
ORG before = ACQUIRER, ORG after = TARGET.

Output: results/validation/ner_role_attribution_v2.json
"""
import os
import sys
import io
import json
import time
import warnings
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

BASE = r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package"
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(BASE, "results", "validation", "ner_role_attribution_v2.json")

ACQ_VERBS = {"acquire", "buy", "purchase", "merge", "absorb", "consume",
             "takeover", "absorb"}
SELL_VERBS = {"sell", "divest", "spin"}
DEAL_NOUNS = {"acquisition", "merger", "takeover", "buyout", "purchase",
              "offer", "bid", "tender", "deal", "buyup"}
# Active-verb fragment regex for positional fallback
DEAL_KEYWORD_RE = re.compile(
    r"\b(acquir\w*|buy\w*|purchas\w*|merger?|takeover|buyout|tender offer|"
    r"divest\w*|sell\w*|sold|sale of|to be acquired|being acquired|"
    r"acquisition of|offer for)\b",
    re.IGNORECASE)


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def first_org_token_idx(doc):
    """Index of the FIRST token of the first ORG entity, or None."""
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return (ent.start, ent.end, ent.text)
    return None


def find_deal_anchor(doc):
    """Return (kind, token_idx) where kind in {'acq_verb', 'sell_verb', 'deal_noun', None}.
    The anchor is the first relevant token (deepest meaningful one)."""
    anchors = []
    for t in doc:
        if t.pos_ in ("VERB", "AUX") and t.lemma_.lower() in ACQ_VERBS:
            anchors.append(("acq_verb", t))
        elif t.pos_ in ("VERB", "AUX") and t.lemma_.lower() in SELL_VERBS:
            anchors.append(("sell_verb", t))
        elif t.pos_ == "NOUN" and t.lemma_.lower() in DEAL_NOUNS:
            anchors.append(("deal_noun", t))
    if not anchors: return None
    # Prefer acq_verb > deal_noun > sell_verb (most informative role signal)
    for kind in ("acq_verb", "deal_noun", "sell_verb"):
        for k, t in anchors:
            if k == kind: return (k, t)
    return None


def get_subject(verb):
    """Walk to find nsubj or nsubjpass of a verb."""
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child, child.dep_
    # If verb is xcomp/ccomp, climb to head and use its subject
    if verb.head is not verb and verb.dep_ in ("xcomp", "ccomp", "advcl"):
        return get_subject(verb.head)
    return None, None


def is_org_or_compound_of_org(token, doc, org_span):
    """True if token is within first ORG entity span or compound-modifies it."""
    if org_span is None: return False
    s, e, _ = org_span
    if s <= token.i < e: return True
    # check if token is a head of any token in org_span (e.g., compound)
    for i in range(s, e):
        if doc[i].head is token:
            return True
    return False


def classify_role_v2(doc, title_text, debug=False):
    """Return tag in {ACQUIRER, TARGET, BOTH, NEITHER, UNCLEAR_NO_ORG,
    UNCLEAR_NO_ANCHOR, AMBIGUOUS}, and explanation."""
    org_span = first_org_token_idx(doc)
    has_org = org_span is not None

    anchor = find_deal_anchor(doc)
    if anchor is None:
        return ("UNCLEAR_NO_ANCHOR" if has_org else "UNCLEAR_NO_ORG", "no deal anchor")

    kind, atok = anchor

    # --- Strategy: detect focal company role via dependency parse ---
    role_dep = None
    expl = ""

    if kind == "acq_verb":
        subj, subj_dep = get_subject(atok)
        if subj is not None:
            if subj_dep == "nsubj":
                # active: subject is ACQUIRER
                focal_is_subj = (org_span is not None and
                                 is_org_or_compound_of_org(subj, doc, org_span))
                if focal_is_subj:
                    role_dep = "ACQUIRER"; expl = f"focal=subj of active '{atok.text}'"
                else:
                    # subj is something else -> first ORG might be the dobj (target)
                    # Find dobj
                    for child in atok.children:
                        if child.dep_ == "dobj" and is_org_or_compound_of_org(
                                child, doc, org_span):
                            role_dep = "TARGET"; expl = f"focal=dobj of '{atok.text}'"
            elif subj_dep == "nsubjpass":
                # "X is acquired" -> subject is TARGET
                focal_is_subj = (org_span is not None and
                                 is_org_or_compound_of_org(subj, doc, org_span))
                if focal_is_subj:
                    role_dep = "TARGET"; expl = f"focal=subjpass of '{atok.text}'"
                # check agent for ACQUIRER
                for child in atok.children:
                    if child.dep_ == "agent":
                        for gc in child.children:
                            if (gc.dep_ == "pobj" and
                                    is_org_or_compound_of_org(gc, doc, org_span)):
                                if role_dep is None:
                                    role_dep = "ACQUIRER"
                                    expl = f"focal=agent (by) of '{atok.text}'"

    elif kind == "deal_noun":
        # "X completes/announces acquisition of Y" -> X is ACQUIRER, Y is TARGET
        # The deal_noun's governing verb's subject = ACQUIRER side
        # "of Y" pobj = TARGET
        # First find "of <orgY>" pattern
        target_org = None
        acquirer_via_subj = None

        # Pattern A: deal_noun governs "of X"
        for child in atok.children:
            if child.dep_ == "prep" and child.lemma_.lower() in ("of", "for"):
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        target_org = gc
        # Pattern B: deal_noun's head is a verb, find subj
        if atok.head is not atok:
            verb = atok.head
            subj, sd = get_subject(verb)
            if subj is not None and sd == "nsubj":
                acquirer_via_subj = subj

        # Resolve focal-company role
        if (target_org is not None and
                is_org_or_compound_of_org(target_org, doc, org_span)):
            role_dep = "TARGET"
            expl = f"focal=pobj of {kind} '{atok.text}'"
        elif (acquirer_via_subj is not None and
                is_org_or_compound_of_org(acquirer_via_subj, doc, org_span)):
            role_dep = "ACQUIRER"
            expl = f"focal=subj of verb governing '{atok.text}'"

    elif kind == "sell_verb":
        subj, sd = get_subject(atok)
        if subj is not None and sd == "nsubj":
            # "X sells/divests/agrees to sell" -> X is on the TARGET side (selling)
            focal_is_subj = is_org_or_compound_of_org(subj, doc, org_span)
            if focal_is_subj:
                role_dep = "TARGET"; expl = f"focal=subj of sell-verb '{atok.text}'"
            else:
                # Look for "to Y" pattern (buyer)
                for child in atok.children:
                    if child.dep_ == "prep" and child.lemma_.lower() == "to":
                        for gc in child.children:
                            if (gc.dep_ == "pobj" and
                                    is_org_or_compound_of_org(gc, doc, org_span)):
                                role_dep = "ACQUIRER"
                                expl = f"focal=buyer (to X) of '{atok.text}'"

    # --- Positional fallback if dep-parse didn't yield a role ---
    if role_dep is None:
        # If first ORG is before the deal keyword -> ACQUIRER
        # If after -> TARGET
        if has_org:
            org_idx_start = org_span[0]
            kw_match = DEAL_KEYWORD_RE.search(title_text)
            if kw_match is not None:
                # Map char pos to token pos
                kw_char_start = kw_match.start()
                kw_token = None
                for t in doc:
                    if t.idx <= kw_char_start < t.idx + len(t.text):
                        kw_token = t; break
                if kw_token is None:
                    for t in doc:
                        if t.idx > kw_char_start:
                            kw_token = t; break
                if kw_token is not None:
                    kw = kw_match.group(0).lower()
                    if any(k in kw for k in ("to be acquired", "being acquired",
                                              "sold", "sale of", "divest",
                                              "offer for", "tender offer")):
                        # Target-side keyword: first ORG is likely TARGET if before; ACQUIRER if after
                        if org_idx_start < kw_token.i:
                            role_dep = "TARGET" if "offer for" not in kw else "TARGET"
                            expl = f"positional: focal before target-side keyword '{kw}'"
                        else:
                            role_dep = "ACQUIRER"
                            expl = f"positional: focal after target-side keyword '{kw}'"
                    else:
                        # Acquirer-side keyword
                        if org_idx_start < kw_token.i:
                            role_dep = "ACQUIRER"
                            expl = f"positional: focal before acq-keyword '{kw}'"
                        else:
                            role_dep = "TARGET"
                            expl = f"positional: focal after acq-keyword '{kw}'"

    if role_dep is None:
        return ("AMBIGUOUS", f"anchor={kind}/{atok.text} but no role")
    return (role_dep, expl)


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
    print("Tagging roles via v2 (NER + dep + positional fallback) ...", flush=True)

    roles, expls = [], []
    titles = ma["title_en"].tolist()
    t_tag0 = time.time()
    for k, doc in enumerate(nlp.pipe(titles, batch_size=128)):
        role, expl = classify_role_v2(doc, titles[k])
        roles.append(role); expls.append(expl)
        if (k + 1) % 500 == 0:
            print(f"  tagged {k+1}/{len(titles)}  ({time.time()-t_tag0:.0f}s)", flush=True)
    ma["role_ner"] = roles
    print(f"  done in {time.time()-t_tag0:.0f}s", flush=True)

    role_counts = ma["role_ner"].value_counts().to_dict()
    print("\n[NER v2 role distribution (all M&A)]", flush=True)
    for k, v in role_counts.items():
        print(f"  {k:>20}: {v}", flush=True)

    # Sample examples per role
    examples = {}
    for r in ["ACQUIRER", "TARGET", "BOTH", "AMBIGUOUS", "UNCLEAR_NO_ANCHOR",
              "UNCLEAR_NO_ORG"]:
        sub_ex = []
        sub = ma[ma["role_ner"] == r]
        for i in range(min(8, len(sub))):
            idx = sub.index[i]; pos = ma.index.get_loc(idx)
            sub_ex.append({"title": sub.iloc[i]["title_en"],
                           "explanation": expls[pos]})
        examples[r] = sub_ex

    print("\n[Examples per role]")
    for r, ex_list in examples.items():
        print(f"\n--- {r} ---")
        for e in ex_list[:3]:
            print(f"  {e['title'][:130]}")
            print(f"     -> {e['explanation']}")

    # Compare regex
    ACQ_PAT = re.compile(
        r"\b(acqui|acquir|takeover|buyer|buy[- ]?out|purchas|to acquire|will acquire|"
        r"completes acquisition|launches offer|tender offer|bid for|offers? to buy)\b",
        re.IGNORECASE)
    TGT_PAT = re.compile(
        r"\b(target|to be acquir|being acquir|to be sold|sold to|sale of|divest|"
        r"subject of (an? )?bid|merger with|to merge with|received offer|"
        r"agree(s|d)? to be acquir)\b",
        re.IGNORECASE)

    def regex_role(t):
        a = bool(ACQ_PAT.search(t)); g = bool(TGT_PAT.search(t))
        if a and g: return "BOTH"
        if a: return "ACQUIRER"
        if g: return "TARGET"
        return "NEITHER"

    ma["role_regex"] = ma["title_en"].apply(regex_role)
    cross = pd.crosstab(ma["role_regex"], ma["role_ner"])
    print("\n[Cross-tab: regex (rows) vs NER v2 (cols)]")
    print(cross.to_string())

    # Specialist eval
    train_full = ma[ma["published_date"] < VAL_END].copy()
    test_full = ma[ma["published_date"] >= VAL_END].copy()
    print(f"\nTrain+val M&A: {len(train_full)}, Test M&A: {len(test_full)}", flush=True)

    def specialist_eval(role_label, tag_col="role_ner"):
        tr = train_full[train_full[tag_col] == role_label]
        te = test_full[test_full[tag_col] == role_label]
        if len(tr) < 20 or len(te) < 20:
            return {"n_train": len(tr), "n_test": len(te), "mcc": None,
                    "balacc": None, "proba_std": None, "pred_up_frac": None,
                    "note": "too small"}
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
        return {"n_train": int(len(tr)), "n_test": int(len(te)),
                "mcc": safe_mcc(te["y"].values, yp),
                "balacc": float(balanced_accuracy_score(te["y"].values, yp)),
                "proba_std": float(yp_proba.std()),
                "pred_up_frac": float(yp.mean()),
                "true_up_frac": float(te["y"].mean())}

    print("\n[Specialist eval on NER v2-defined subsets]", flush=True)
    results_ner = {}
    for role in ["ACQUIRER", "TARGET"]:
        r = specialist_eval(role, "role_ner")
        results_ner[role] = r
        print(f"  {role}: {r}", flush=True)

    print("\n[Specialist eval on REGEX-defined subsets]", flush=True)
    results_regex = {}
    for role in ["ACQUIRER", "TARGET"]:
        r = specialist_eval(role, "role_regex")
        results_regex[role] = r
        print(f"  {role}: {r}", flush=True)

    # Permutation test on NER-ACQUIRER if signal exists
    if (results_ner["ACQUIRER"]["mcc"] is not None and
            results_ner["ACQUIRER"]["n_test"] >= 30):
        print("\n[10K perm test on NER v2 ACQUIRER]", flush=True)
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
                 "n_ma_total": int(len(ma)),
                 "method": "v2 NER + dep-parse + positional fallback"},
        "role_counts_ner_v2": role_counts,
        "examples_per_role": examples,
        "cross_regex_ner_v2": cross.to_dict(),
        "specialist_eval_ner_v2_roles": results_ner,
        "specialist_eval_regex_roles": results_regex,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
