"""
FinBERT Baseline — Domain-Specific Financial Text Encoder.
Uses ProsusAI/finbert for financial sentiment and embeddings.
Compares against generic sentence-transformers.
"""
import sys, io, os, json, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

BASE_DIR = r'.'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'classifier_training_v2.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'validation')

df = pd.read_parquet(DATA_PATH)
df['published_date'] = pd.to_datetime(df['published_date'])
df = df[df['actual_side'].str.lower().isin(['up', 'down'])].copy()
df['label'] = (df['actual_side'].str.lower() == 'up').astype(int)

train = df[df['published_date'] < '2025-04-01'].copy()
val = df[(df['published_date'] >= '2025-04-01') & (df['published_date'] < '2025-06-01')].copy()
print(f"Train: {len(train)}, Val: {len(val)}")

# Try FinBERT sentiment first (lighter weight)
print("\n--- FinBERT Sentiment Analysis ---")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = "ProsusAI/finbert"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    def get_finbert_sentiment(texts, batch_size=64):
        sentiments = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).numpy()
            sentiments.append(probs)
            if (i // batch_size) % 20 == 0:
                print(f"  Batch {i//batch_size}/{len(texts)//batch_size}...")
        return np.vstack(sentiments)

    t0 = time.time()
    train_texts = train['title_en'].fillna('').astype(str).tolist()
    val_texts = val['title_en'].fillna('').astype(str).tolist()

    print("Processing train...")
    train_sent = get_finbert_sentiment(train_texts)
    print("Processing val...")
    val_sent = get_finbert_sentiment(val_texts)
    elapsed = time.time() - t0
    print(f"  FinBERT sentiment done in {elapsed:.0f}s")

    # Use raw sentiment probabilities as features
    lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr.fit(train_sent, train['label'])
    preds = lr.predict(val_sent)
    mcc = matthews_corrcoef(val['label'], preds)
    bacc = balanced_accuracy_score(val['label'], preds)
    print(f"  FinBERT sentiment LogReg: MCC={mcc:.4f}, BalAcc={bacc:.4f}")

    # Also try FinBERT on M&A subset
    ma_tr = train[train['event'] == 'mergers_acquisitions']
    ma_vl = val[val['event'] == 'mergers_acquisitions']
    if len(ma_tr) >= 30 and len(ma_vl) >= 15:
        ma_tr_idx = train.index.isin(ma_tr.index)
        ma_vl_idx = val.index.isin(ma_vl.index)
        # Re-index to align
        tr_positions = [list(train.index).index(i) for i in ma_tr.index]
        vl_positions = [list(val.index).index(i) for i in ma_vl.index]
        lr_ma = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lr_ma.fit(train_sent[tr_positions], ma_tr['label'])
        preds_ma = lr_ma.predict(val_sent[vl_positions])
        mcc_ma = matthews_corrcoef(ma_vl['label'], preds_ma)
        bacc_ma = balanced_accuracy_score(ma_vl['label'], preds_ma)
        print(f"  FinBERT sentiment M&A: MCC={mcc_ma:.4f}, BalAcc={bacc_ma:.4f}")

    # Direct sentiment majority vote
    # FinBERT: 0=positive, 1=negative, 2=neutral
    train_pred_labels = np.argmax(train_sent, axis=1)
    val_pred_labels = np.argmax(val_sent, axis=1)
    # Positive sentiment → UP
    direct_preds = (val_pred_labels == 0).astype(int)
    mcc_direct = matthews_corrcoef(val['label'], direct_preds)
    bacc_direct = balanced_accuracy_score(val['label'], direct_preds)
    print(f"  FinBERT direct (pos→UP): MCC={mcc_direct:.4f}, BalAcc={bacc_direct:.4f}")

    # Distribution
    unique, counts = np.unique(val_pred_labels, return_counts=True)
    sent_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    for u, c in zip(unique, counts):
        print(f"    {sent_map.get(u, u)}: {c} ({c/len(val)*100:.1f}%)")

    results = {
        "finbert_sentiment_logreg": {"mcc": round(mcc, 4), "bacc": round(bacc, 4)},
        "finbert_direct": {"mcc": round(mcc_direct, 4), "bacc": round(bacc_direct, 4)},
        "elapsed_seconds": round(elapsed, 0)
    }

except Exception as e:
    print(f"  FinBERT failed: {e}")
    print("  Trying to install transformers...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers', 'torch'], 
                   capture_output=True)
    results = {"error": str(e)}

# Try financial domain sentence embeddings
print("\n--- Financial Domain Embeddings ---")
try:
    from sentence_transformers import SentenceTransformer

    for model_name in ["sentence-transformers/all-mpnet-base-v2"]:
        print(f"\nLoading {model_name}...")
        st_model = SentenceTransformer(model_name)

        t0 = time.time()
        train_emb = st_model.encode(train_texts[:5000], batch_size=64, show_progress_bar=False)
        val_emb = st_model.encode(val_texts, batch_size=64, show_progress_bar=False)
        elapsed = time.time() - t0
        print(f"  Encoding done in {elapsed:.0f}s (train subset=5000)")

        lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lr.fit(train_emb, train['label'].values[:5000])
        preds = lr.predict(val_emb)
        mcc = matthews_corrcoef(val['label'], preds)
        bacc = balanced_accuracy_score(val['label'], preds)
        print(f"  {model_name}: MCC={mcc:.4f}, BalAcc={bacc:.4f}")

        # M&A subset
        ma_vl_texts = ma_vl['title_en'].fillna('').astype(str).tolist()
        ma_tr_texts = ma_tr['title_en'].fillna('').astype(str).tolist()
        ma_tr_emb = st_model.encode(ma_tr_texts, batch_size=64, show_progress_bar=False)
        ma_vl_emb = st_model.encode(ma_vl_texts, batch_size=64, show_progress_bar=False)
        lr_ma = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lr_ma.fit(ma_tr_emb, ma_tr['label'])
        preds_ma = lr_ma.predict(ma_vl_emb)
        mcc_ma = matthews_corrcoef(ma_vl['label'], preds_ma)
        bacc_ma = balanced_accuracy_score(ma_vl['label'], preds_ma)
        print(f"  {model_name} M&A: MCC={mcc_ma:.4f}, BalAcc={bacc_ma:.4f}")

        results[f"mpnet_full"] = {"mcc": round(mcc, 4), "bacc": round(bacc, 4)}
        results[f"mpnet_ma"] = {"mcc": round(mcc_ma, 4), "bacc": round(bacc_ma, 4)}

except Exception as e:
    print(f"  Sentence transformer failed: {e}")

out_path = os.path.join(RESULTS_DIR, 'finbert_baselines.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")
