---
license: other
task_categories:
- text-classification
language:
- en
tags:
- finance
- financial-news
- temporal-leakage
- chronological-evaluation
- mergers-and-acquisitions
- stock-movement-prediction
pretty_name: Financial-News Temporal-Leakage Corpus
size_categories:
- 10K<n<100K
---

# Financial-News Temporal-Leakage Corpus

56,409 financial-news articles (2020–2025, 81% from 2025; 64 exchanges; 203 event types) with next-day return-direction
labels and 30+ engineered features. Built to study **temporal leakage** in financial-news NLP: how random vs.
chronological train/test splitting inflates apparent performance, and whether event-conditioned signal (notably
**M&A**) survives strict chronological evaluation.

Companion to the paper *"Temporal Leakage in Financial News NLP: A Multi-Architecture Audit with a Regime-Specific M&A
Signal"* (EMNLP 2026). Code + reproducible aggregates + public cross-corpus (EDT, FNSPID) replications are in the
artifact repository.

**Provenance & why it is useful.** The underlying news is public, but the corpus's value is its comprehensive historical
retention: commercial wire feeds typically keep only a few weeks of history per ticker, so a multi-year, multi-exchange
tagged corpus like this is hard to reconstruct independently. It is released with the data provider's approval.

## Load
```python
from datasets import load_dataset
ds = load_dataset("<org>/financial-news-temporal-leakage")  # ⟦final path⟧
```

## Task & splits
- **Label:** `actual_side` ∈ {up, down, neutral}; binary target `y = 1[actual_side=='up']` (drop neutral for binary).
- **Chronological splits (pre-registered):** train `published_date < 2025-04-01` (21,654) · val `2025-04/05` (10,866) ·
  test `≥ 2025-06-01` (17,279). M&A subset: 731 / 369 / 786.
- **Primary metric:** Matthews correlation (MCC); balanced accuracy as complement.

## Key finding (reproducible)
Random splits inflate MCC 1.1×–6.5× over chronological. Under chronological evaluation general prediction is
near-random (best MCC 0.060); **M&A** retains a locked-test signal (TF-IDF+LR MCC **0.138**, permutation p<1e-3),
scoped to the 2024–25 European-tilted regime (definition-matched on public EDT; null on public FNSPID 2009–2020).

## Fields
See `DATA_DICTIONARY.md`. Highlights: `title_en`, `content_en`, `event`, `publisher`, `industry`, `exchange`,
`yf_ticker`, `published_date`, `price_change_percentage`, `index_price_change_percentage`, `actual_side`,
`nextday_side`, plus keyword/timing/sentiment/length features.

## License
Dataset license is **provisional** — to be finalized with the data provider at publication (CC-BY-4.0 intended for the
derived, de-identified corpus, pending provider confirmation). The underlying article text remains subject to the
originating publishers' copyright. Public EDT/FNSPID slices retain their original licenses. Code: MIT.

## Ethics & intended use
Public-company financial news; no private-individual personal data. Intended for **evaluation-methodology / leakage
research**, not deployable trading (the paper's backtest is an explicit upper bound). Signal is regime-scoped; do not
treat as a universal predictor.

## Citation
```bibtex
@inproceedings{temporal-leakage-finnlp-2026, title={Temporal Leakage in Financial News NLP: A Multi-Architecture Audit with a Regime-Specific M&A Signal}, year={2026} }
```
Public corpora: Zhou et al. (2021, EDT); Dong et al. (2024, FNSPID).
