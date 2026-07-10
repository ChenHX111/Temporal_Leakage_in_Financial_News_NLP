# Datasheet — Financial-News Temporal-Leakage Corpus (classifier_training_v2)

Following Gebru et al. (2021), "Datasheets for Datasets." ⟦…⟧ = finalize at publication.

## Motivation
- **Purpose.** To audit *temporal leakage* in financial-news direction prediction: how much random vs. chronological
  train/test splitting inflates apparent NLP performance, and whether any event-conditioned signal survives strict
  chronological evaluation. Built for the paper's multi-architecture leakage audit + the M&A locked-test analysis.
- **Creators / funding.** The paper's author team; data sourced from a commercial financial-news provider. ⟦funding⟧.

## Composition
- **Instances.** 56,409 financial-news articles (49,799 after removing neutral / near-zero-return items for the binary
  task), each an (article, market-outcome) pair. **51 columns** (see `DATA_DICTIONARY.md`): text (`title_en`,
  `content_en`), metadata (`event` [203 types], `publisher`, `industry`, `exchange`, `yf_ticker`, `published_date`),
  outcomes (`price_change_percentage`, `index_price_change_percentage`, `actual_side`, `nextday_side`), and 30+
  engineered features (keyword counts, timing flags, sentiment, length stats).
- **Coverage.** 2020-05 → 2025-08; **81% from 2025**; 64 stock exchanges worldwide; English text.
- **Labels.** Binary next-day return direction: `actual_side` ∈ {up, down, neutral} (neutral = near-zero move);
  `y = 1[actual_side=='up']`; a market-adjusted variant uses abnormal return vs. the exchange benchmark.
- **Splits (chronological, pre-registered).** train `<2025-04-01` (21,654) · val `2025-04/05` (10,866) ·
  test `≥2025-06-01` (17,279). M&A subset (`event=='mergers_acquisitions'`): 731 / 369 / 786.
- **Sensitive data.** Public-company financial news only; no personal data about private individuals. Firm/person names
  appear as public market entities.

## Collection
- **Source.** A commercial news provider's tagged feed (203-type event taxonomy, per-article instrument + timestamps).
  The underlying news is public; the provider's value is comprehensive **historical retention** — commercial wire feeds
  typically keep only a few weeks of history per ticker — which makes independent reconstruction difficult and is the
  main reason releasing this curated corpus is useful.
  Returns/benchmarks computed from market data (close-to-close, or close-to-next-open for after-hours; exchange
  timezone/holiday handling per the paper's App. Reproducibility).
- **Sampling.** All articles in the provider window meeting the tagging criteria; no subsampling beyond neutral-removal
  for the binary task.

## Preprocessing / cleaning
- Neutral / near-zero-return articles removed for the binary task (55–57% UP among the retained). Near-duplicate audit:
  exact-title duplicates and MinHash near-dup candidates are reported in `reproducible_aggregates/dedup_report.json`;
  crucially, across the 2-month train/test gap only ~0.1% of M&A test titles near-duplicate a train title and the
  locked-test MCC is unchanged after removing them (`dedup_robustness_MA.json`) — temporal splitting controls near-dup
  leakage.

## Uses
- **Intended.** Leakage auditing / chronological-evaluation research for financial NLP; event-conditioned signal
  analysis; a benchmark for "leakage-audit-as-required-disclosure."
- **Out of scope.** Deployable trading (the backtest is an explicit upper bound); out-of-regime generalization (signal
  is scoped to the 2024–25 European-tilted M&A regime; null on FNSPID 2009–2020 US).

## Distribution & maintenance
- **Release.** Provider-approved public release; recommended host **HuggingFace Datasets** (viewer + loader + DOI).
  Size: 23.6 MB (parquet). Public EDT/FNSPID slices carry their original licenses.
- **License.** Provisional — to be finalized with the provider at publication (CC-BY-4.0 intended for the derived,
  de-identified corpus, pending provider confirmation); the underlying article text remains subject to the originating
  publishers' copyright. ⟦confirm final terms at publication⟧.
- **Maintenance / contact / versioning.** ⟦…⟧.
