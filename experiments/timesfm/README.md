# E5 — zero-shot time-series foundation-model probe

Tests whether a zero-shot numeric time-series foundation model can recover the M&A signal that the text specialist
finds, using only the price path (no text). It cannot: no zero-shot FM beats the text specialist.

## Run
```bash
pip install timesfm chronos-forecasting   # GPU recommended
python run_timesfm_probe.py               # ~2 min on a single GPU
```
Inputs are bundled here under `data/` (`ma_locked_test_meta.parquet`, `prices_cache.parquet`); the probe writes
`results/timesfm_probe.json`. To rebuild the price cache from scratch, run `python build_prices.py` first.

## Expected (see `results/timesfm_probe.json`)
| Model | MCC | n |
|---|---|---|
| TimesFM-2.5 (zero-shot) | −0.054 | ≈693 |
| Chronos (zero-shot) | −0.028 | ≈693 |
| momentum (last-return) | −0.044 | 618 |
| AR(20) mean | −0.065 | 687 |

None of the numeric foundation models or momentum/AR baselines recovers the text specialist's M&A signal
(locked-test MCC 0.138), i.e. the effect is text-borne, not a repackaged price-momentum pattern.
