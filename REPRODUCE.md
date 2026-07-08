# REPRODUCE — expected numbers + commands

All scripts are CPU-only unless noted; the full audit is minutes. Point scripts at `data/classifier_training_v2.parquet`
(paths in the scripts assume the original layout; edit `BASE` to this repo root or set it via env). Environment:
`pip install -r requirements.txt` (pandas, numpy, scikit-learn, pyarrow, joblib).

## Core headline reproductions
| Result | Script (code/) | Expected |
|---|---|---|
| M&A locked-test specialist + backtest (paper HP) | `validation/cost_aware_backtest_paperHP.py` | train-only locked-test **MCC = 0.138**; train+val **0.068**; all-trade Sharpe@10bps ≈ −0.60; top-quartile Sharpe@10bps ≈ +2.62 (ex-post upper bound) |
| 10k-permutation test (M&A) | `validation/permutation_test_ma_10k.py` | z ≈ 3.8, p_two < 1e-3 |
| Multi-architecture leakage audit | `validation/leakage_audit_definitive.py`, `leakage_audit_full_features.py`, `leakage_audit_multiseed.py` | random/temporal ratio 1.1×–6.5×; ΔMCC up to +0.125 |
| Cross-event replication (M&A vs CLN/LGL/ERN) | `validation/cross_event_audit.py`, `cross_event_full_pack.py` | 2×2 taxonomy; M&A unique signal cell |
| Cross-corpus: EDT def-matched | `validation/edt_broadma_reconcile.py`, `edt_robustness.py` (uses `public_corpora/edt_evaluate_slim.parquet`) | EDT M&A MCC ≈ 0.097 |
| Cross-corpus: FNSPID null | `validation/fnspid_replication.py` (uses `public_corpora/fnspid_ma_filtered_fromstream.parquet`) | within-FNSPID MCC ≈ −0.011 (null, n≈4,235) |
| Extended 6-month window | `validation/w3_extended_window.py` | MCC ≈ +0.133, p<1e-4 |
| Cutoff perturbation ±7/±14d | `validation/tierB_cutoff_perturbation.py` | MCC ∈ [+0.115, +0.156] |
| Negative-control events | `validation/tierB_event_neg_control.py` | only M&A substantially positive |
| Role attribution (acquirer/target) | `validation/tierB_role_stability.py` | acquirer-side qualitative (power-limited) |

## Rebuttal robustness additions (this bundle, `../experiments/` + `reproducible_aggregates/`)
| Result | Script | Expected |
|---|---|---|
| Neutrals-included / τ=0 ex-ante M&A | `experiments/E1_neutrals.py` | τ=0 MCC +0.078 (p=0.019); signal survives ex-ante inclusion |
| Leakage-mitigation demo (purged/embargo CV) | `experiments/E2_mitigation.py` | naive-CV inflates ~2.2×; blocked+embargo & forward-chain recover chronological; M&A locked-test invariant |
| Val-calibrated backtest (upper bound) | `experiments/E3_valcal_backtest.py` | val-threshold≈test-threshold (|Δ|<0.001); top-quartile Sharpe +2.33@10bps, but all-trade −0.60@10bps (break-even ≈5bps) — still an upper bound (slippage/impact unmodelled) |
| Firm-disjoint (entity-holdout) M&A | `experiments/E4_firm_disjoint.py` | temporal + firm-disjoint unseen-firm MCC +0.098 (n=551) — the clean, time-purged estimate |
| Dedup robustness (cross-boundary) | `artifact_bundle/dedup_robustness.py` | ~0.1% test near-dup in train; MCC 0.138→0.139 after removal |

## Aggregate verification (no text needed)
Run `artifact_bundle/generate_aggregates.py` → regenerates `reproducible_aggregates/` (split_summary, monthly event
counts, hashed TF-IDF vocab, dedup report + keys, publisher/entity overlap). These let an external party verify the
split design, feature space, and leakage-source handling from hashes/counts alone.
