"""
LLM Multi-Model Consensus Baseline for Financial News Stock Prediction
======================================================================
Implements the MULTI_LLM_TEXT_CONSENSUS methodology:
- 3 diverse LLMs each predict UP/DOWN independently
- Majority vote (>=2/3 agreement) determines final prediction
- If all 3 disagree or tie, mark as "abstain"

Models: gpt-54-reasoning, claude-opus-4-7, claude-sonnet-4-6
(As specified in MULTI_LLM_TEXT_CONSENSUS.md)

This replaces the single-LLM baseline (claude-sonnet-4-5 only).

Output: results/validation/llm_consensus_baselines.json
"""

import sys, os, json, time, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, confusion_matrix
from split_config import get_split
import httpx

# ── Config ──────────────────────────────────────────────────────────────────
PROXY_URL = "http://127.0.0.1:12041/v1/chat/completions"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'validation')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'classifier_training_v2.parquet')

# Multi-LLM Consensus: 3 diverse models
CONSENSUS_MODELS = ["gpt-54-reasoning", "claude-opus-4-7", "claude-sonnet-4-6"]
INCLUSION_THRESHOLD = 2  # need >=2 out of 3 to agree

# Sample sizes (same as original for direct comparison)
GLOBAL_SAMPLE_N = 500
MA_FULL = True
MAX_CONCURRENT = 10
MAX_CONTENT_CHARS = 800
SEED = 42

# ── Prompt Templates (same as original) ─────────────────────────────────────

PROMPT_TITLE_ONLY = """You are a financial analyst. Based on the following financial news headline, predict whether the stock price will go UP or DOWN on the same day the news is published.

Headline: {title}

Respond with exactly one word: UP or DOWN"""

PROMPT_TITLE_EVENT = """You are a financial analyst. Based on the following financial news headline and event type, predict whether the stock price will go UP or DOWN on the same day the news is published.

Headline: {title}
Event type: {event}

Respond with exactly one word: UP or DOWN"""

PROMPT_MA_ROLE = """You are a financial analyst specializing in mergers and acquisitions. Based on the following M&A news headline, predict whether the mentioned company's stock price will go UP or DOWN on the same day.

Consider: In M&A deals, target companies' stocks typically rise, acquirers' stocks often decline or stay flat, and divestitures can unlock value for sellers.

Headline: {title}

Respond with exactly one word: UP or DOWN"""

PROMPT_COT = """You are a financial analyst. Based on the following financial news headline, predict whether the stock price will go UP or DOWN on the same day the news is published.

Headline: {title}

First, briefly explain your reasoning (2-3 sentences), then on a new line write your final prediction as exactly: PREDICTION: UP or PREDICTION: DOWN"""


# ── LLM Caller ──────────────────────────────────────────────────────────────

async def call_llm(client: httpx.AsyncClient, prompt: str, model: str,
                   max_tokens: int = 100, semaphore: asyncio.Semaphore = None) -> str:
    """Call LLM via codex-proxy with retry."""
    if semaphore:
        async with semaphore:
            return await _call_inner(client, prompt, model, max_tokens)
    return await _call_inner(client, prompt, model, max_tokens)


async def _call_inner(client, prompt, model, max_tokens, max_retries=3):
    for attempt in range(max_retries):
        try:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens,
            }
            # Only non-reasoning models support temperature
            if not model.startswith("gpt-5") and not model.startswith("claude-opus"):
                body["temperature"] = 0.0

            resp = await client.post(PROXY_URL, json=body, timeout=120.0)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("retry-after-ms", 3000)) / 1000
                await asyncio.sleep(retry_after)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return "ERROR"
    return "ERROR"


def parse_prediction(response: str) -> str:
    """Extract UP/DOWN from LLM response."""
    if not response or response == "ERROR":
        return "unknown"
    resp = response.strip().upper()

    # Direct answer
    if resp.rstrip(".,!") in ("UP", "DOWN"):
        return resp.rstrip(".,!").lower()

    # CoT format
    if "PREDICTION:" in resp:
        after = resp.split("PREDICTION:")[-1].strip()
        if after.startswith("UP"):
            return "up"
        elif after.startswith("DOWN"):
            return "down"

    # Markdown bold
    if "**UP**" in resp:
        return "up"
    if "**DOWN**" in resp:
        return "down"

    # Last line
    lines = response.strip().split('\n')
    last_line = lines[-1].strip().upper().rstrip(".,!")
    if last_line in ("UP", "DOWN"):
        return last_line.lower()

    # Standalone word search
    import re
    up_match = re.search(r'\bUP\b', resp[:300])
    down_match = re.search(r'\bDOWN\b', resp[:300])
    if up_match and not down_match:
        return "up"
    if down_match and not up_match:
        return "down"
    if up_match and down_match:
        # Take last occurrence as final answer
        last_up = resp.rfind("UP")
        last_down = resp.rfind("DOWN")
        return "up" if last_up > last_down else "down"

    return "unknown"


def majority_vote(predictions: list) -> str:
    """Take majority vote from 3 model predictions. Returns 'up', 'down', or 'abstain'."""
    valid = [p for p in predictions if p in ("up", "down")]
    if len(valid) < 2:
        return "abstain"
    up_count = sum(1 for p in valid if p == "up")
    down_count = sum(1 for p in valid if p == "down")
    if up_count >= INCLUSION_THRESHOLD:
        return "up"
    elif down_count >= INCLUSION_THRESHOLD:
        return "down"
    else:
        return "abstain"


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label=""):
    """Compute metrics, filtering out abstains/unknowns."""
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
    n_abstain = sum(1 for p in y_pred if p == "abstain")

    return {
        "label": label,
        "n_total": len(y_true),
        "n_valid": len(valid),
        "n_abstain": n_abstain,
        "n_unparseable": len(y_true) - len(valid) - n_abstain,
        "mcc": round(mcc, 4),
        "accuracy": round(acc, 4),
        "f1_up": round(f1, 4),
        "up_rate_pred": round(up_rate, 4),
        "up_rate_true": round(sum(1 for t in yt if t == "up") / len(yt), 4),
        "confusion_matrix": cm,
    }


# ── Main Pipeline ───────────────────────────────────────────────────────────

async def run_consensus_experiment(df_test, prompt_template, prompt_name, subset_label="full"):
    """Run multi-LLM consensus for one prompt variant."""
    print(f"\n{'='*70}")
    print(f"CONSENSUS Experiment: {prompt_name} | Subset: {subset_label}")
    print(f"Models: {CONSENSUS_MODELS}")
    print(f"Samples: {len(df_test)} | Threshold: {INCLUSION_THRESHOLD}/3")
    print(f"{'='*70}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    per_model_predictions = {m: [None]*len(df_test) for m in CONSENSUS_MODELS}

    # Build all prompts first
    prompts = []
    for _, row in df_test.iterrows():
        title = str(row.get("title_en", ""))[:500]
        event = str(row.get("event", "unknown"))
        content = str(row.get("content_en", ""))[:MAX_CONTENT_CHARS]
        prompt = prompt_template.format(title=title, event=event, content=content)
        prompts.append(prompt)

    max_tok = 200 if "COT" in prompt_name.upper() else 50

    async with httpx.AsyncClient() as client:
        # Run all models in parallel across all samples
        for model_idx, model in enumerate(CONSENSUS_MODELS):
            print(f"  Running model: {model}...")
            # Process in batches of 20
            batch_size = 20
            results_for_model = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                tasks = [call_llm(client, p, model, max_tokens=max_tok, semaphore=semaphore)
                         for p in batch]
                batch_results = await asyncio.gather(*tasks)
                results_for_model.extend(batch_results)
                if (i + batch_size) % 100 == 0 or i + batch_size >= len(prompts):
                    print(f"    {model}: {min(i+batch_size, len(prompts))}/{len(prompts)}")
                await asyncio.sleep(0.3)

            # Parse predictions
            for i, resp in enumerate(results_for_model):
                per_model_predictions[model][i] = parse_prediction(resp)

    # Majority vote
    all_predictions = []
    for i in range(len(df_test)):
        preds_i = [per_model_predictions[m][i] for m in CONSENSUS_MODELS]
        consensus = majority_vote(preds_i)
        all_predictions.append(consensus)

    # True labels
    y_true = df_test['actual_side'].astype(str).str.lower().tolist()

    # Consensus evaluation
    result = evaluate(y_true, all_predictions, label=f"consensus_{prompt_name}_{subset_label}")
    result["method"] = "multi_llm_consensus"
    result["models"] = CONSENSUS_MODELS
    result["threshold"] = f"{INCLUSION_THRESHOLD}/3"

    # Per-model results (for comparison with consensus)
    result["per_model"] = {}
    for model in CONSENSUS_MODELS:
        model_result = evaluate(y_true, per_model_predictions[model],
                                label=f"{model}_{prompt_name}")
        result["per_model"][model] = model_result

    # Agreement stats
    n_unanimous = 0
    n_majority = 0
    n_disagree = 0
    for i in range(len(df_test)):
        preds_i = [per_model_predictions[m][i] for m in CONSENSUS_MODELS]
        valid_preds = [p for p in preds_i if p in ("up", "down")]
        if len(set(valid_preds)) == 1 and len(valid_preds) >= 2:
            n_unanimous += 1
        elif len(valid_preds) >= 2:
            from collections import Counter
            counts = Counter(valid_preds)
            if counts.most_common(1)[0][1] >= 2:
                n_majority += 1
            else:
                n_disagree += 1
        else:
            n_disagree += 1

    result["agreement_stats"] = {
        "n_unanimous": n_unanimous,
        "n_majority_only": n_majority,
        "n_disagree": n_disagree,
        "unanimity_rate": round(n_unanimous / len(df_test), 4),
    }

    print(f"\n  CONSENSUS MCC: {result.get('mcc', 'N/A')}")
    print(f"  Per-model MCCs: ", end="")
    for m in CONSENSUS_MODELS:
        print(f"{m.split('-')[0]}={result['per_model'][m].get('mcc', 'N/A')} ", end="")
    print(f"\n  Agreement: {n_unanimous} unanimous, {n_majority} majority, {n_disagree} disagree")

    return result


async def main():
    print("=" * 70)
    print("MULTI-LLM CONSENSUS BASELINE")
    print("Methodology: MULTI_LLM_TEXT_CONSENSUS.md (majority vote, 3 models)")
    print("=" * 70)

    # Load data
    df = pd.read_parquet(DATA_PATH)
    df['actual_side'] = df['actual_side'].astype(str).str.lower().str.strip()
    df['published_date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None)
    binary = df[df['actual_side'].isin(['up', 'down'])].copy()

    # Get test set
    test = binary[binary['published_date'] >= pd.Timestamp('2025-06-01')].reset_index(drop=True)
    print(f"\nFull test set: {len(test)} samples")

    # Global sample (same seed for reproducibility)
    np.random.seed(SEED)
    global_sample = test.sample(n=GLOBAL_SAMPLE_N, random_state=SEED).reset_index(drop=True)
    print(f"Global sample: {len(global_sample)} samples")

    # M&A subset
    ma_test = test[test['event'].astype(str).str.lower().str.contains('mergers', na=False)].reset_index(drop=True)
    print(f"M&A test: {len(ma_test)} samples")

    experiments = []

    # ── Experiment 1: Title-only, Global sample ──
    result = await run_consensus_experiment(
        global_sample, PROMPT_TITLE_ONLY, "title_only", "global_sample")
    experiments.append(result)

    # ── Experiment 2: Title+Event, Global sample ──
    result = await run_consensus_experiment(
        global_sample, PROMPT_TITLE_EVENT, "title_event", "global_sample")
    experiments.append(result)

    # ── Experiment 3: CoT, Global sample ──
    result = await run_consensus_experiment(
        global_sample, PROMPT_COT, "cot", "global_sample")
    experiments.append(result)

    # ── Experiment 4: Title-only, M&A full ──
    result = await run_consensus_experiment(
        ma_test, PROMPT_TITLE_ONLY, "title_only", "ma")
    experiments.append(result)

    # ── Experiment 5: M&A role prompt, M&A full ──
    result = await run_consensus_experiment(
        ma_test, PROMPT_MA_ROLE, "ma_role_prompt", "ma")
    experiments.append(result)

    # ── Experiment 6: Title+Event, M&A full ──
    result = await run_consensus_experiment(
        ma_test, PROMPT_TITLE_EVENT, "title_event", "ma")
    experiments.append(result)

    # Save results
    output = {
        "experiments": experiments,
        "config": {
            "models": CONSENSUS_MODELS,
            "consensus_method": "majority_vote",
            "inclusion_threshold": f"{INCLUSION_THRESHOLD}/3",
            "global_sample_n": GLOBAL_SAMPLE_N,
            "seed": SEED,
            "max_content_chars": MAX_CONTENT_CHARS,
            "test_set_size": len(test),
            "ma_test_size": len(ma_test),
        },
        "methodology": "MULTI_LLM_TEXT_CONSENSUS.md - Method 5 (Hybrid) with majority vote for binary decisions",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "llm_consensus_baselines.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Consensus vs Individual Models")
    print(f"{'='*70}")
    print(f"{'Experiment':<30} {'Consensus MCC':<15} {'Model MCCs'}")
    print("-" * 70)
    for exp in experiments:
        model_mccs = " | ".join(
            f"{m.split('-')[0][:8]}={exp['per_model'][m].get('mcc', 'N/A')}"
            for m in CONSENSUS_MODELS
        )
        print(f"{exp['label']:<30} {exp.get('mcc', 'N/A'):<15} {model_mccs}")


if __name__ == "__main__":
    asyncio.run(main())
