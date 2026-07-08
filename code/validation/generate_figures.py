"""
Generate paper-ready figure data and summary tables.
Creates CSV files for plotting and LaTeX table snippets.
"""
import sys, io, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

BASE_DIR = r'C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package'
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'validation')
FIGURES_DIR = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Figure 1: Rolling MCC over time (M&A vs Global vs Majority)
# ============================================================
rolling = json.load(open(os.path.join(RESULTS_DIR, 'rolling_validation_results.json')))
rolling_data = []
for r in rolling['rolling_validation']:
    rolling_data.append({
        'month': r['val_month'],
        'global_mcc': r['mcc_full'],
        'high_signal_mcc': r['mcc_high_signal'] if r['mcc_high_signal'] is not None else np.nan,
        'event_only_mcc': r['mcc_event_only'],
        'n_val': r['val_n']
    })

# Add test months for M&A
test_data = json.load(open(os.path.join(RESULTS_DIR, 'purged_and_final_test.json')))
for r in test_data['final_test']['ma_rolling_test']:
    rolling_data.append({
        'month': r['month'],
        'ma_mcc': r['mcc'],
        'ma_n': r['n'],
        'split': 'test'
    })

rolling_df = pd.DataFrame(rolling_data)
rolling_df.to_csv(os.path.join(FIGURES_DIR, 'fig1_rolling_mcc.csv'), index=False)
print("Figure 1: Rolling MCC data saved")

# Also add M&A-specific rolling from event_specific_classifiers
ma_rolling = json.load(open(os.path.join(RESULTS_DIR, 'event_specific_classifiers.json')))
ma_months = []
for r in ma_rolling['ma_rolling']:
    ma_months.append({
        'month': r['month'],
        'mcc': r['mcc'],
        'bacc': r['bacc'],
        'n': r['val_n'],
        'split': 'val' if r['month'] < '2025-06' else 'test'
    })
# Append test months
for r in test_data['final_test']['ma_rolling_test']:
    ma_months.append({
        'month': r['month'],
        'mcc': r['mcc'],
        'n': r['n'],
        'split': 'test'
    })
pd.DataFrame(ma_months).to_csv(os.path.join(FIGURES_DIR, 'fig1b_ma_rolling_mcc.csv'), index=False)
print("Figure 1b: M&A rolling MCC data saved")

# ============================================================
# Figure 2: Per-event mean MCC with error bars
# ============================================================
event_data = []
for event, windows in rolling['event_stability'].items():
    valid_mccs = [r['mcc'] for r in windows if r['mcc'] is not None]
    if valid_mccs:
        event_data.append({
            'event': event,
            'mean_mcc': np.mean(valid_mccs),
            'std_mcc': np.std(valid_mccs),
            'n_windows': len(valid_mccs),
            'n_positive': sum(1 for m in valid_mccs if m > 0)
        })
event_df = pd.DataFrame(event_data).sort_values('mean_mcc', ascending=False)
event_df.to_csv(os.path.join(FIGURES_DIR, 'fig2_event_stability.csv'), index=False)
print("Figure 2: Event stability data saved")

# ============================================================
# Figure 3: Non-text control comparison
# ============================================================
controls = json.load(open(os.path.join(RESULTS_DIR, 'nontext_controls.json')))
ctrl_data = []
for name, vals in controls['full_dataset'].items():
    ctrl_data.append({
        'baseline': name,
        'subset': 'full',
        'mcc': vals['mcc'],
        'bacc': vals.get('bacc', np.nan)
    })
for name, vals in controls['ma_subset'].items():
    ctrl_data.append({
        'baseline': name,
        'subset': 'ma',
        'mcc': vals['mcc'],
        'bacc': vals.get('bacc', np.nan)
    })
pd.DataFrame(ctrl_data).to_csv(os.path.join(FIGURES_DIR, 'fig3_nontext_controls.csv'), index=False)
print("Figure 3: Non-text controls data saved")

# ============================================================
# Figure 4: Entity-role UP% and model ablation
# ============================================================
role_data = json.load(open(os.path.join(RESULTS_DIR, 'entity_role_analysis.json')))
role_df = pd.DataFrame([
    {'role': role, 'n': vals['n'], 'pct': vals['pct'], 'up_pct': role_data['role_up_pct'][role]}
    for role, vals in role_data['role_distribution'].items()
])
role_df.to_csv(os.path.join(FIGURES_DIR, 'fig4_entity_roles.csv'), index=False)
print("Figure 4: Entity role data saved")

# ============================================================
# Table 1: Main results table (LaTeX-ready)
# ============================================================
print("\n" + "=" * 80)
print("LATEX TABLE: Main Results")
print("=" * 80)
latex = r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Split} & \textbf{Subset} & \textbf{MCC} & \textbf{Bal.~Acc.} \\
\midrule
\multicolumn{5}{l}{\textit{Aggregate models (full coverage)}} \\
Random split RF & random & All & 0.205 & 0.568 \\
GradBoost TF-IDF+num & temporal & All & 0.063 & 0.545 \\
All non-text metadata & temporal & All & 0.057 & 0.527 \\
FinBERT direct & temporal & All & 0.047 & 0.517 \\
TF-IDF title LogReg & temporal & All & 0.029 & 0.512 \\
FinBERT LogReg & temporal & All & 0.025 & 0.504 \\
Global text (test) & \textbf{test} & All & 0.022 & 0.507 \\
\midrule
\multicolumn{5}{l}{\textit{Event-conditioned models (M\&A subset)}} \\
M\&A text (val) & temporal & M\&A & 0.093 & 0.523 \\
M\&A text (rolling avg) & rolling & M\&A & 0.085 & --- \\
M\&A text (\textbf{test}) & \textbf{test} & \textbf{M\&A} & \textbf{0.071} & \textbf{0.520} \\
M\&A text no-role-words & test & M\&A & $-$0.008 & 0.499 \\
M\&A role-only & temporal & M\&A & 0.024 & 0.512 \\
M\&A majority & test & M\&A & 0.000 & 0.500 \\
\bottomrule
\end{tabular}
\caption{Stock movement prediction results under temporal validation. Random splits inflate MCC by 0.18. M\&A articles show genuine text-driven signal (MCC=0.071 on locked test, p=0.006 by sign test over 11 months).}
\label{tab:main_results}
\end{table}
"""
print(latex)

# Save LaTeX table
with open(os.path.join(FIGURES_DIR, 'table1_main_results.tex'), 'w') as f:
    f.write(latex)

# ============================================================
# Table 2: Event-type stability
# ============================================================
print("\n" + "=" * 80)
print("LATEX TABLE: Event-Type Stability")
print("=" * 80)
latex2 = r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{lcccl}
\toprule
\textbf{Event Type} & \textbf{Mean MCC} & \textbf{Std} & \textbf{Pos./8} & \textbf{Signal?} \\
\midrule
mergers\_acquisitions & 0.081 & 0.085 & 7/8 & \checkmark \\
shares\_issue & 0.047 & 0.105 & 6/8 & weak \\
financial\_results & 0.025 & 0.050 & 6/8 & weak \\
exchange\_announcement & 0.016 & 0.187 & 4/8 & unstable \\
corporate\_action & 0.010 & 0.115 & 5/8 & noise \\
management\_changes & 0.005 & 0.067 & 4/8 & $\times$ \\
annual\_general\_meeting & $-$0.033 & 0.036 & 1/8 & $\times$ \\
\bottomrule
\end{tabular}
\caption{Per-event-type temporal stability of TF-IDF text classifiers over 8 rolling monthly validation windows. Only M\&A shows consistent positive signal.}
\label{tab:event_stability}
\end{table}
"""
print(latex2)
with open(os.path.join(FIGURES_DIR, 'table2_event_stability.tex'), 'w') as f:
    f.write(latex2)

print(f"\nAll figure data saved to: {FIGURES_DIR}")
print("Files created:")
for f in os.listdir(FIGURES_DIR):
    print(f"  {f}")
