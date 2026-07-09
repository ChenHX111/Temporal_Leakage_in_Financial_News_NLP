# Temporal Leakage in Financial News NLP — reproducibility bundle

This repository is the reproducibility artifact for the paper *"Temporal Leakage in Financial News NLP: A
Multi-Architecture Audit with a Regime-Specific M&A Signal"* (ACL-ARR / EMNLP 2026, submission 16286).

> **Status:** Anonymized reproducibility artifact accompanying submission 16286 (double-blind review). The dataset
> provider has approved public release; the final public host (**HuggingFace Datasets** recommended — built-in viewer,
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
DATASHEET.md    REPRODUCE.md    HF_DATASET_CARD.md    DATA_DICTIONARY.md
```

## Headline numbers you can reproduce (see REPRODUCE.md for exact commands + expected values)
- Multi-architecture leakage audit: random-split MCC inflates 1.1×–6.5× over chronological (ΔMCC up to +0.125).
- M&A locked-test specialist (TF-IDF+LR, paper HP): **MCC = 0.138** (train→val→test), **0.068** (train+val refit),
  10k-permutation p<1e-3, weekly-bootstrap CI [+0.066, +0.205].
- Definition-matched EDT reproduction (public); FNSPID 2009–2020 US M&A **null** (public) → regime-scoping.
- Cost-aware backtest = explicit **upper bound** (ex-post quantile + no slippage/impact).

## Reproducibility aggregates (releasable even under any residual data restriction)
`reproducible_aggregates/` holds only counts, SHA-256 hashes, and overlap statistics — enough to externally verify the
split design, feature space, near-duplicate handling, and leakage sources **without** any article text:
`split_summary.json`, `monthly_event_counts.csv`, `tfidf_vocab_hashed_MA.csv`, `dedup_report.json`,
`dedup_robustness_MA.json`, `dedup_keys_hashed.csv`, `publisher_entity_overlap.json`.

## License
Code: MIT (see below). Data: released under the provider-approved terms; final license fixed at publication
(CC-BY-4.0 intended for the derived, de-identified corpus). Public EDT/FNSPID slices retain their original licenses
(cite Zhou et al. 2021; Dong et al. 2024).
