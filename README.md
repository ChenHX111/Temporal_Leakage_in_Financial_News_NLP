# Temporal Leakage in Financial News NLP — reproducibility bundle

This repository is the reproducibility artifact for the paper *"Temporal Leakage in Financial News NLP: A
Multi-Architecture Audit with a Regime-Specific M&A Signal"* (ACL-ARR / EMNLP 2026, submission 16286).

> **Status:** Anonymized reproducibility artifact accompanying submission 16286 (double-blind review). The underlying
> news is public and the data provider has **approved public release**; the corpus's value is its comprehensive
> historical retention (commercial wire feeds typically keep only a few weeks of history per ticker), which makes
> independent reconstruction difficult. Final public host (**HuggingFace Datasets** recommended — built-in viewer,
> streaming loader, DOI, standard dataset card) will be fixed for the camera-ready. This mirror contains the full
> labelled corpus, reproduction code, exact split rules, prompts, and the aggregate-verification layer, and reproduces
> the headline (MCC 0.138 train→val→test; 0.068 train+val refit) directly.

## What's here
```
data/classifier_training_v2.parquet     # the primary corpus: 56,409 financial-news articles x 51 columns (2020–2025)
public_corpora/                          # public cross-corpus replications (no proprietary data)
    edt_evaluate_slim.parquet            #   EDT (Zhou et al. 2021) M&A slice — definition-matched reproduction
    fnspid_ma_filtered_fromstream.parquet#   FNSPID (Dong et al. 2024) 2009–2020 US M&A — cross-corpus null
code/validation/                         # the paper's headline reproductions (audit, M&A locked test, permutation,
                                         #   cross-event, cross-corpus, backtest, role attribution)
code/baseline/                           # baseline classifiers (temporal reproduction)
reproducible_aggregates/                 # count/hash/overlap-only aggregates (verifiable without any text)
experiments/                             # rebuttal robustness additions: E1 neutrals/ex-ante, E2 leakage-mitigation CV,
                                         #   E3 val-calibrated backtest, E4 firm-disjoint; timesfm/ = E5 TimesFM/Chronos
                                         #   price-only probe, self-contained (probe + bundled price cache + result JSON)
CHECKLIST.md                             # reusable temporal-leakage audit protocol (each step -> a runnable script)
DATASHEET.md    REPRODUCE.md    HF_DATASET_CARD.md    DATA_DICTIONARY.md
```

## Headline numbers you can reproduce (see REPRODUCE.md for exact commands + expected values)
- Multi-architecture leakage audit: random-split MCC inflates 1.1×–6.5× over chronological (ΔMCC up to +0.145).
- M&A locked-test specialist (TF-IDF+LR, paper HP): **MCC = 0.138** (train→val→test), **0.068** (train+val refit),
  10k-permutation p<1e-3, weekly-bootstrap CI [+0.066, +0.205].
- Definition-matched EDT reproduction (public); FNSPID 2009–2020 US M&A **null** (public) → regime-scoping.
- Cost-aware backtest = explicit **upper bound**: +2.62 ex-post is reported for continuity; the validation-calibrated top-quartile Sharpe is **+2.33** (all-trade **−0.60**), and remains an upper bound only because slippage/market-impact are unmodelled.

## Reproducibility aggregates (count/hash-only, no article text)
`reproducible_aggregates/` holds only counts, SHA-256 hashes, and overlap statistics — enough to externally verify the
split design, feature space, near-duplicate handling, and leakage sources **without** any article text:
`split_summary.json`, `monthly_event_counts.csv`, `tfidf_vocab_hashed_MA.csv`, `dedup_report.json`,
`dedup_robustness_MA.json`, `dedup_keys_hashed.csv`, `publisher_entity_overlap.json`.

## License
Code: MIT (see below). Data license is **provisional** — to be finalized with the provider at publication. Article text
is redistributed only under the provider-approved release terms; the originating publishers retain copyright, and we make
no CC-BY (or other open-license) claim over third-party article text. Public EDT/FNSPID slices retain their original licenses
(cite Zhou et al. 2021; Dong et al. 2024).
