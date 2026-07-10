#!/usr/bin/env bash
# E5 TimesFM price-only probe — end-to-end runner.
set -e
cd "$(dirname "$0")"
echo "=== E5 TimesFM probe | $(date -u +%FT%TZ) ==="
python -c "import pandas, numpy, sklearn" 2>/dev/null || pip install pandas numpy scikit-learn pyarrow
if [ ! -f data/prices_cache.parquet ]; then
  echo "--- fetching price history (needs internet) ---"
  pip install yfinance >/dev/null 2>&1 || true
  python build_prices.py
fi
echo "--- running TimesFM probe (+ numeric baselines) ---"
pip install timesfm >/dev/null 2>&1 || echo "(timesfm not installed; numeric baselines will still run)"
python run_timesfm_probe.py
echo "--- ship back results/timesfm_probe.json ---"
