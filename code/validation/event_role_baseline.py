"""
Event-Role Extraction Baseline for M&A Articles
================================================
Tests whether explicit rule-based role extraction matches or beats TF-IDF.

This addresses the rubber-duck critique: "Is the M&A signal just role identification?"

Experiments:
1. Rule-based role extraction → predict direction
2. Role-only features vs TF-IDF
3. Role + TF-IDF combined
4. Simple finance heuristic: seller/divestor → UP, acquirer → DOWN

Output: results/validation/event_role_baselines.json
"""

import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from split_config import get_split

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'classifier_training_v2.parquet')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'validation')

# ── Role extraction rules ───────────────────────────────────────────────────

ROLE_PATTERNS = {
    'seller': [
        r'\bsells?\b', r'\bsold\b', r'\bdivest(?:s|ed|iture|ing)\b',
        r'\bdispos(?:es?|ed|al|ing)\b', r'\bspin[- ]?off\b',
        r'\boffload(?:s|ed|ing)?\b', r'\bexit(?:s|ed|ing)?\b',
    ],
    'acquirer': [
        r'\bacquir(?:es?|ed|ing)\b', r'\bbuy(?:s|ing)?\b', r'\bbought\b',
        r'\bpurchas(?:es?|ed|ing)\b', r'\btake(?:s)? over\b', r'\btakeover\b',
        r'\bbid(?:s|ding)?\b',
    ],
    'target': [
        r'\btarget(?:ed|s)?\b', r'\bto be acquired\b', r'\bbeing acquired\b',
        r'\breceiv(?:es?|ed|ing) (?:a |an )?(?:bid|offer)\b',
    ],
    'merger': [
        r'\bmerg(?:es?|ed|er|ing)\b', r'\bcombine(?:s|d)?\b',
        r'\bjoint venture\b', r'\bpartner(?:ship|s|ed|ing)?\b',
    ],
    'completion': [
        r'\bcomplete(?:s|d)?\b', r'\bfinaliz(?:es?|ed|ing)\b',
        r'\bclose(?:s|d)? (?:the )?(?:deal|acquisition|transaction)\b',
        r'\bapproved\b', r'\bcleared\b',
    ],
    'rumor': [
        r'\brumor(?:s|ed)?\b', r'\breport(?:s|ed|edly)?\b',
        r'\ballegedly\b', r'\bexplor(?:es?|ed|ing)\b',
        r'\bconsidering\b', r'\bin talks?\b', r'\bnegotiat(?:es?|ed|ing|ion)\b',
    ],
}

def extract_roles(text: str) -> dict:
    """Extract M&A role features from text."""
    text_lower = str(text).lower()
    features = {}
    for role, patterns in ROLE_PATTERNS.items():
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, text_lower))
        features[f'role_{role}'] = min(count, 3)  # cap at 3
        features[f'has_{role}'] = int(count > 0)
    return features


def extract_role_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract role features for all rows."""
    records = []
    for _, row in df.iterrows():
        title = str(row.get('title_en', ''))
        content = str(row.get('content_en', ''))[:500]
        combined = title + ' ' + content
        feats = extract_roles(combined)
        records.append(feats)
    return pd.DataFrame(records, index=df.index)


# ── Finance heuristic baseline ─────────────────────────────────────────────

def finance_heuristic(text: str) -> str:
    """Simple finance rule: seller/divestor → UP, acquirer → DOWN."""
    text_lower = str(text).lower()
    seller_score = 0
    acquirer_score = 0
    
    for pat in ROLE_PATTERNS['seller']:
        seller_score += len(re.findall(pat, text_lower))
    for pat in ROLE_PATTERNS['acquirer']:
        acquirer_score += len(re.findall(pat, text_lower))
    
    if seller_score > acquirer_score:
        return 'up'
    elif acquirer_score > seller_score:
        return 'down'
    else:
        # Default to majority class
        return 'up'


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    train, val, test = get_split(df)
    
    # M&A subsets
    ma_train = train[train['event'].str.lower().str.contains('m&a|merger|acquisition|takeover', na=False)]
    ma_val = val[val['event'].str.lower().str.contains('m&a|merger|acquisition|takeover', na=False)]
    ma_test = test[test['event'].str.lower().str.contains('m&a|merger|acquisition|takeover', na=False)]
    
    print(f"M&A samples: train={len(ma_train)}, val={len(ma_val)}, test={len(ma_test)}")
    
    results = {}
    
    # ── Exp 1: Finance heuristic (no training needed) ──
    print("\n=== Experiment 1: Finance Heuristic ===")
    for split_name, split_df in [('val', ma_val), ('test', ma_test)]:
        preds = [finance_heuristic(str(row.get('title_en', '')) + ' ' + str(row.get('content_en', ''))[:500]) 
                 for _, row in split_df.iterrows()]
        y_true = split_df['actual_side'].tolist()
        mcc = matthews_corrcoef(y_true, preds)
        acc = accuracy_score(y_true, preds)
        up_rate = sum(1 for p in preds if p == 'up') / len(preds)
        print(f"  {split_name}: MCC={mcc:.4f}, Acc={acc:.4f}, UP_rate={up_rate:.3f}")
        results[f'heuristic_{split_name}'] = {
            'mcc': round(mcc, 4), 'accuracy': round(acc, 4), 'up_rate': round(up_rate, 4),
            'n': len(split_df), 'method': 'finance_heuristic'
        }
    
    # ── Exp 2: Role features only (LogReg) ──
    print("\n=== Experiment 2: Role Features Only ===")
    role_train = extract_role_features(ma_train)
    role_val = extract_role_features(ma_val)
    role_test = extract_role_features(ma_test)
    
    y_train = (ma_train['actual_side'] == 'up').astype(int)
    y_val = (ma_val['actual_side'] == 'up').astype(int)
    y_test = (ma_test['actual_side'] == 'up').astype(int)
    
    clf_role = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf_role.fit(role_train, y_train)
    
    for split_name, X_split, y_split in [('val', role_val, y_val), ('test', role_test, y_test)]:
        pred = clf_role.predict(X_split)
        pred_labels = ['up' if p == 1 else 'down' for p in pred]
        true_labels = ['up' if y == 1 else 'down' for y in y_split]
        mcc = matthews_corrcoef(true_labels, pred_labels)
        acc = accuracy_score(true_labels, pred_labels)
        print(f"  {split_name}: MCC={mcc:.4f}, Acc={acc:.4f}")
        results[f'role_only_{split_name}'] = {
            'mcc': round(mcc, 4), 'accuracy': round(acc, 4),
            'n': len(X_split), 'method': 'role_features_logreg'
        }
    
    # Feature importance for role features
    coefs = dict(zip(role_train.columns, clf_role.coef_[0]))
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    print("  Top role feature coefficients:")
    for feat, coef in sorted_coefs[:10]:
        print(f"    {feat}: {coef:.4f}")
    results['role_feature_importance'] = {k: round(v, 4) for k, v in sorted_coefs}
    
    # ── Exp 3: TF-IDF only (for comparison) ──
    print("\n=== Experiment 3: TF-IDF Only (M&A) ===")
    tfidf = TfidfVectorizer(max_features=3000, min_df=2, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(ma_train['title_en'].astype(str))
    X_val_tfidf = tfidf.transform(ma_val['title_en'].astype(str))
    X_test_tfidf = tfidf.transform(ma_test['title_en'].astype(str))
    
    clf_tfidf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf_tfidf.fit(X_train_tfidf, y_train)
    
    for split_name, X_split, y_split in [('val', X_val_tfidf, y_val), ('test', X_test_tfidf, y_test)]:
        pred = clf_tfidf.predict(X_split)
        pred_labels = ['up' if p == 1 else 'down' for p in pred]
        true_labels = ['up' if y == 1 else 'down' for y in y_split]
        mcc = matthews_corrcoef(true_labels, pred_labels)
        acc = accuracy_score(true_labels, pred_labels)
        print(f"  {split_name}: MCC={mcc:.4f}, Acc={acc:.4f}")
        results[f'tfidf_only_{split_name}'] = {
            'mcc': round(mcc, 4), 'accuracy': round(acc, 4),
            'n': X_split.shape[0], 'method': 'tfidf_logreg'
        }
    
    # ── Exp 4: Role + TF-IDF combined ──
    print("\n=== Experiment 4: Role + TF-IDF Combined ===")
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_tfidf, role_train.values])
    X_val_combined = hstack([X_val_tfidf, role_val.values])
    X_test_combined = hstack([X_test_tfidf, role_test.values])
    
    clf_combined = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf_combined.fit(X_train_combined, y_train)
    
    for split_name, X_split, y_split in [('val', X_val_combined, y_val), ('test', X_test_combined, y_test)]:
        pred = clf_combined.predict(X_split)
        pred_labels = ['up' if p == 1 else 'down' for p in pred]
        true_labels = ['up' if y == 1 else 'down' for y in y_split]
        mcc = matthews_corrcoef(true_labels, pred_labels)
        acc = accuracy_score(true_labels, pred_labels)
        print(f"  {split_name}: MCC={mcc:.4f}, Acc={acc:.4f}")
        results[f'role_tfidf_{split_name}'] = {
            'mcc': round(mcc, 4), 'accuracy': round(acc, 4),
            'n': X_split.shape[0], 'method': 'role+tfidf_logreg'
        }
    
    # ── Exp 5: Role-only heuristic with finer categories ──
    print("\n=== Experiment 5: Detailed Role Distribution ===")
    for split_name, split_df in [('val', ma_val), ('test', ma_test)]:
        role_feats = extract_role_features(split_df)
        for role in ['seller', 'acquirer', 'target', 'merger', 'completion', 'rumor']:
            has_col = f'has_{role}'
            subset = split_df[role_feats[has_col] == 1]
            if len(subset) > 10:
                up_rate = (subset['actual_side'] == 'up').mean()
                print(f"  {split_name} {role}: n={len(subset)}, UP_rate={up_rate:.3f}")
    
    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY: Event-Role vs TF-IDF on M&A")
    print("="*60)
    print(f"{'Method':<25} {'Val MCC':<10} {'Test MCC':<10}")
    print("-"*45)
    for prefix in ['heuristic', 'role_only', 'tfidf_only', 'role_tfidf']:
        val_mcc = results.get(f'{prefix}_val', {}).get('mcc', 'N/A')
        test_mcc = results.get(f'{prefix}_test', {}).get('mcc', 'N/A')
        print(f"{prefix:<25} {val_mcc:<10} {test_mcc:<10}")
    
    print("\nComparison point: M&A TF-IDF (prior result): val MCC=0.093, test MCC=0.071")
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'event_role_baselines.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {os.path.join(RESULTS_DIR, 'event_role_baselines.json')}")


if __name__ == '__main__':
    main()
