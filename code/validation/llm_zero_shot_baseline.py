"""
LLM Zero-Shot Baseline for Financial News Stock Prediction
===========================================================
Uses codex-proxy (http://127.0.0.1:12041) to call GPT-5.5 and other models.

Experiments:
1. Zero-shot: title only → UP/DOWN
2. Zero-shot: title + event type → UP/DOWN
3. Zero-shot: title + content → UP/DOWN (truncated)
4. Zero-shot with chain-of-thought reasoning
5. M&A subset: all prompts above

Temporal protocol: LOCKED TEST SET only.
Sampling: stratified random sample for cost control, full M&A subset.

Output: results/validation/llm_zero_shot_baselines.json
"""

import sys, os, json, time, random, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, confusion_matrix
from split_config import get_split, SPLIT_CONFIG
import httpx

# ── Config ──────────────────────────────────────────────────────────────────
PROXY_URL = "http://127.0.0.1:12041/v1/chat/completions"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'validation')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'classifier_training_v2.parquet')

# Sample sizes
GLOBAL_SAMPLE_N = 500      # stratified sample from full test set
MA_FULL = True              # use ALL M&A test samples (~786)
MAX_CONCURRENT = 3          # conservative to avoid rate limits
MAX_CONTENT_CHARS = 800     # truncate content to this length

MODELS = ["claude-sonnet-4-5"]  # Clean outputs, supports temperature=0
# GPT-5.5 is a reasoning model with verbose outputs; claude gives cleaner UP/DOWN

SEED = 42

# ── Prompt Templates ────────────────────────────────────────────────────────

PROMPT_TITLE_ONLY = """You are a financial analyst. Based on the following financial news headline, predict whether the stock price will go UP or DOWN on the same day the news is published.

Headline: {title}

Respond with exactly one word: UP or DOWN"""

PROMPT_TITLE_EVENT = """You are a financial analyst. Based on the following financial news headline and event type, predict whether the stock price will go UP or DOWN on the same day the news is published.

Headline: {title}
Event type: {event}

Respond with exactly one word: UP or DOWN"""

PROMPT_TITLE_CONTENT = """You are a financial analyst. Based on the following financial news article, predict whether the stock price will go UP or DOWN on the same day the news is published.

Headline: {title}

Article excerpt:
{content}

Respond with exactly one word: UP or DOWN"""

PROMPT_COT = """You are a financial analyst. Based on the following financial news headline, predict whether the stock price will go UP or DOWN on the same day the news is published.

Headline: {title}

First, briefly explain your reasoning (2-3 sentences), then on a new line write your final prediction as exactly: PREDICTION: UP or PREDICTION: DOWN"""

PROMPT_MA_ROLE = """You are a financial analyst specializing in mergers and acquisitions. Based on the following M&A news headline, predict whether the mentioned company's stock price will go UP or DOWN on the same day.

Consider: In M&A deals, target companies' stocks typically rise, acquirers' stocks often decline or stay flat, and divestitures can unlock value for sellers.

Headline: {title}

Respond with exactly one word: UP or DOWN"""


# ── LLM Caller ──────────────────────────────────────────────────────────────

async def call_llm(client: httpx.AsyncClient, prompt: str, model: str = "gpt-5.5",
                   max_tokens: int = 100, semaphore: asyncio.Semaphore = None) -> str:
    """Call LLM via codex-proxy with retry."""
    if semaphore:
        async with semaphore:
            return await _call_llm_inner(client, prompt, model, max_tokens)
    return await _call_llm_inner(client, prompt, model, max_tokens)

async def _call_llm_inner(client, prompt, model, max_tokens, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Build request body
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens,
            }
            # Only reasoning models (gpt-5.x) reject temperature
            if not model.startswith("gpt-5"):
                body["temperature"] = 0.0
            resp = await client.post(PROXY_URL, json=body, timeout=120.0)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("retry-after-ms", 2000)) / 1000
                print(f"  Rate limited, waiting {retry_after:.1f}s...")
                await asyncio.sleep(retry_after)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            print(f"  ERROR after {max_retries} retries: {e}")
            return "ERROR"
    return "ERROR"


def parse_prediction(response: str) -> str:
    """Extract UP/DOWN from LLM response. Handles various formats."""
    if not response or response == "ERROR":
        return "unknown"
    resp = response.strip()
    resp_upper = resp.upper()
    
    # Direct single-word answer
    if resp_upper in ("UP", "DOWN"):
        return resp_upper.lower()
    
    # "UP." or "DOWN." with punctuation
    if resp_upper.rstrip(".,!") in ("UP", "DOWN"):
        return resp_upper.rstrip(".,!").lower()
    
    # CoT format: look for "PREDICTION: UP/DOWN"
    if "PREDICTION:" in resp_upper:
        after = resp_upper.split("PREDICTION:")[-1].strip()
        if after.startswith("UP"):
            return "up"
        elif after.startswith("DOWN"):
            return "down"
    
    # "**UP**" or "**DOWN**" markdown bold
    if "**UP**" in resp_upper:
        return "up"
    if "**DOWN**" in resp_upper:
        return "down"
    
    # Look for "UP" or "DOWN" as standalone word in first 200 chars
    import re
    first_part = resp_upper[:200]
    # Check last line first (often the answer)
    lines = resp.strip().split('\n')
    last_line = lines[-1].strip().upper()
    if last_line in ("UP", "DOWN", "UP.", "DOWN."):
        return last_line.rstrip(".").lower()
    
    # Find standalone UP/DOWN
    up_match = re.search(r'\bUP\b', first_part)
    down_match = re.search(r'\bDOWN\b', first_part)
    
    if up_match and not down_match:
        return "up"
    if down_match and not up_match:
        return "down"
    
    # Both found - take the last one (usually the conclusion)
    if up_match and down_match:
        full_upper = resp_upper
        last_up = full_upper.rfind("UP")
        last_down = full_upper.rfind("DOWN")
        if last_up > last_down:
            return "up"
        else:
            return "down"
    
    return "unknown"


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label=""):
    """Compute metrics, filtering out unparseable predictions."""
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p in ("up", "down")]
    if len(valid) < 10:
        return {"label": label, "n_valid": len(valid), "error": "too few valid predictions"}
    
    yt = [v[0] for v in valid]
    yp = [v[1] for v in valid]
    
    mcc = matthews_corrcoef(yt, yp)
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, pos_label="up", average="binary")
    cm = confusion_matrix(yt, yp, labels=["up", "down"]).tolist()
    up_rate = sum(1 for p in yp if p == "up") / len(yp)
    
    return {
        "label": label,
        "n_total": len(y_true),
        "n_valid": len(valid),
        "n_unparseable": len(y_true) - len(valid),
        "mcc": round(mcc, 4),
        "accuracy": round(acc, 4),
        "f1_up": round(f1, 4),
        "up_rate_pred": round(up_rate, 4),
        "up_rate_true": round(sum(1 for t in yt if t == "up") / len(yt), 4),
        "confusion_matrix": cm,
    }


# ── Main Pipeline ───────────────────────────────────────────────────────────

async def run_experiment(df_test, prompt_template, prompt_name, model, semaphore,
                         subset_label="full"):
    """Run a single prompt variant on a dataset."""
    print(f"\n{'='*60}")
    print(f"Experiment: {prompt_name} | Model: {model} | Subset: {subset_label}")
    print(f"Samples: {len(df_test)}")
    print(f"{'='*60}")
    
    predictions = []
    raw_responses = []
    
    async with httpx.AsyncClient() as client:
        tasks = []
        for _, row in df_test.iterrows():
            title = str(row.get("title_en", ""))[:500]
            event = str(row.get("event", "unknown"))
            content = str(row.get("content_en", ""))[:MAX_CONTENT_CHARS]
            
            prompt = prompt_template.format(
                title=title,
                event=event,
                content=content,
            )
            tasks.append(call_llm(client, prompt, model=model, 
                                  max_tokens=200 if "COT" in prompt_name.upper() else 50,
                                  semaphore=semaphore))
        
        # Process in batches for progress reporting
        batch_size = 25
        all_responses = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            all_responses.extend(batch_results)
            n_done = min(i + batch_size, len(tasks))
            print(f"  Progress: {n_done}/{len(tasks)} ({n_done/len(tasks)*100:.0f}%)")
            # Small delay between batches to respect rate limits
            if i + batch_size < len(tasks):
                await asyncio.sleep(1.0)
    
    y_true = df_test["actual_side"].tolist()
    y_pred = [parse_prediction(r) for r in all_responses]
    
    # Count parse failures
    n_unknown = sum(1 for p in y_pred if p == "unknown")
    n_error = sum(1 for r in all_responses if r == "ERROR")
    print(f"  Parsed: {len(y_pred) - n_unknown - n_error} valid, {n_unknown} unknown, {n_error} errors")
    
    metrics = evaluate(y_true, y_pred, label=f"{prompt_name}_{subset_label}")
    metrics["model"] = model
    metrics["prompt_name"] = prompt_name
    metrics["subset"] = subset_label
    metrics["n_errors"] = n_error
    
    # Save a few example responses for inspection
    examples = []
    for i in range(min(5, len(all_responses))):
        examples.append({
            "title": str(df_test.iloc[i].get("title_en", ""))[:200],
            "true": y_true[i],
            "predicted": y_pred[i],
            "raw_response": all_responses[i][:300],
        })
    metrics["examples"] = examples
    
    print(f"  MCC={metrics.get('mcc', 'N/A')}, Acc={metrics.get('accuracy', 'N/A')}, "
          f"F1={metrics.get('f1_up', 'N/A')}, UP_rate={metrics.get('up_rate_pred', 'N/A')}")
    
    return metrics


async def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    _, _, test = get_split(df)
    print(f"Test set: {len(test)} rows")
    
    # M&A subset
    ma_mask = test["event"].str.lower().str.contains("m&a|merger|acquisition|takeover", na=False)
    test_ma = test[ma_mask]
    print(f"M&A test subset: {len(test_ma)} rows")
    
    # Stratified sample for global experiments
    random.seed(SEED)
    np.random.seed(SEED)
    if len(test) > GLOBAL_SAMPLE_N:
        # Stratified by actual_side
        test_up = test[test["actual_side"] == "up"]
        test_down = test[test["actual_side"] == "down"]
        n_up = int(GLOBAL_SAMPLE_N * len(test_up) / len(test))
        n_down = GLOBAL_SAMPLE_N - n_up
        sample = pd.concat([
            test_up.sample(n=n_up, random_state=SEED),
            test_down.sample(n=n_down, random_state=SEED),
        ])
        test_sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    else:
        test_sample = test.reset_index(drop=True)
    
    print(f"Global sample: {len(test_sample)} rows "
          f"(UP={sum(test_sample['actual_side']=='up')}, DOWN={sum(test_sample['actual_side']=='down')})")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    all_results = []
    
    for model in MODELS:
        # ── Experiment 1: Title only (global sample) ──
        r = await run_experiment(test_sample, PROMPT_TITLE_ONLY, "title_only", model, semaphore, "global_sample")
        all_results.append(r)
        
        # ── Experiment 2: Title + Event (global sample) ──
        r = await run_experiment(test_sample, PROMPT_TITLE_EVENT, "title_event", model, semaphore, "global_sample")
        all_results.append(r)
        
        # ── Experiment 3: Title + Content (global sample) ──
        r = await run_experiment(test_sample, PROMPT_TITLE_CONTENT, "title_content", model, semaphore, "global_sample")
        all_results.append(r)
        
        # ── Experiment 4: Chain-of-thought (global sample) ──
        r = await run_experiment(test_sample, PROMPT_COT, "cot", model, semaphore, "global_sample")
        all_results.append(r)
        
        # ── M&A experiments ──
        if len(test_ma) > 0:
            # Title only on M&A
            r = await run_experiment(test_ma.reset_index(drop=True), PROMPT_TITLE_ONLY, "title_only", model, semaphore, "ma")
            all_results.append(r)
            
            # M&A-specific prompt with role awareness
            r = await run_experiment(test_ma.reset_index(drop=True), PROMPT_MA_ROLE, "ma_role_prompt", model, semaphore, "ma")
            all_results.append(r)
            
            # Title + event on M&A
            r = await run_experiment(test_ma.reset_index(drop=True), PROMPT_TITLE_EVENT, "title_event", model, semaphore, "ma")
            all_results.append(r)
    
    # ── Summary ──
    print("\n" + "="*80)
    print("SUMMARY OF ALL LLM ZERO-SHOT EXPERIMENTS")
    print("="*80)
    print(f"{'Experiment':<30} {'Subset':<15} {'MCC':<8} {'Acc':<8} {'F1':<8} {'UP%':<8} {'N':<6}")
    print("-"*80)
    for r in all_results:
        if "error" not in r:
            print(f"{r['prompt_name']:<30} {r['subset']:<15} {r['mcc']:<8} {r['accuracy']:<8} "
                  f"{r['f1_up']:<8} {r['up_rate_pred']:<8} {r['n_valid']:<6}")
        else:
            print(f"{r['prompt_name']:<30} {r['subset']:<15} ERROR: {r.get('error', 'unknown')}")
    
    # Compare to our best models
    print("\n── Comparison to our models (test set) ──")
    print("  Global text TF-IDF LogReg:   MCC=0.022")
    print("  M&A text TF-IDF LogReg:      MCC=0.071")
    print("  FinBERT LogReg:              MCC=0.025")
    print("  Majority baseline:           MCC=0.000")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "llm_zero_shot_baselines.json")
    with open(output_path, "w") as f:
        json.dump({
            "experiments": all_results,
            "config": {
                "global_sample_n": GLOBAL_SAMPLE_N,
                "models": MODELS,
                "seed": SEED,
                "max_content_chars": MAX_CONTENT_CHARS,
                "test_set_size": len(test),
                "ma_test_size": len(test_ma),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
