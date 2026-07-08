"""Statistical significance tests for the paper."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from scipy.stats import binom
import numpy as np

# Sign test: 10/11 months positive for M&A
n_pos = 10
n_total = 11
p_val = binom.sf(n_pos - 1, n_total, 0.5)
print(f"Sign test: {n_pos}/{n_total} positive months")
print(f"  p-value (one-sided) = {p_val:.6f}")
print(f"  Significant at 0.05? {p_val < 0.05}")
print(f"  Significant at 0.01? {p_val < 0.01}")

# Multiple comparison: per-event sign tests (8 rolling months)
print("\nMultiple comparison (Benjamini-Hochberg FDR=10%):")
events = [
    ("mergers_acquisitions", 0.0811, 7),
    ("shares_issue", 0.0471, 6),
    ("financial_results", 0.0252, 6),
    ("exchange_announcement", 0.0160, 4),
    ("corporate_action", 0.0102, 5),
    ("management_changes", 0.0046, 4),
    ("clinical_study", 0.0000, 0),
    ("press_releases", 0.0000, 0),
    ("interim_information", -0.0129, 2),
    ("share_capital_increase", -0.0211, 3),
    ("annual_general_meeting", -0.0334, 1),
]

pvals = []
for name, mean_mcc, n_pos in events:
    if n_pos == 0:
        pvals.append((name, mean_mcc, n_pos, 1.0))
    else:
        p = binom.sf(n_pos - 1, 8, 0.5)
        pvals.append((name, mean_mcc, n_pos, p))

# Sort by p-value for BH
pvals.sort(key=lambda x: x[3])

print(f"  {'Event':<40} {'mean_MCC':>8} {'pos/8':>5} {'p-val':>8} {'BH sig':>8}")
print("  " + "-" * 75)
n = len(pvals)
for rank, (name, mean_mcc, n_pos, p) in enumerate(pvals, 1):
    bh_threshold = (rank / n) * 0.10
    sig = "YES" if p <= bh_threshold else "no"
    print(f"  {name:<40} {mean_mcc:>8.4f} {n_pos:>3}/8 {p:>8.4f} {sig:>8}")
