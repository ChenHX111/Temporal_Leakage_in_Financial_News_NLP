"""
Economic Significance Analysis
==============================
Simple long-short backtest and market-adjusted analysis for M&A signal.

Experiments:
1. Market-adjusted labels: re-label UP/DOWN after removing market return
2. Simple long-short backtest: buy predicted-UP, sell predicted-DOWN
3. Transaction cost analysis
4. Sharpe ratio and drawdown

Output: results/validation/economic_significance.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, accuracy_score
from split_config import get_split

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'classifier_training_v2.parquet')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'validation')

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    train, val, test = get_split(df)
    
    # M&A subsets
    ma_train = train[train['event'].str.lower().str.contains('m&a|merger|acquisition|takeover', na=False)].copy()
    ma_val = val[val['event'].str.lower().str.contains('m&a|merger|acquisition|takeover', na=False)].copy()
    ma_test = test[test['event'].str.lower().str.contains('m&a|merger|acquisition|takeover', na=False)].copy()
    
    print(f"M&A: train={len(ma_train)}, val={len(ma_val)}, test={len(ma_test)}")
    
    results = {}
    
    # ── Check available columns for returns ──
    print("\nColumns related to price changes:")
    price_cols = [c for c in df.columns if 'price' in c.lower() or 'change' in c.lower() or 'return' in c.lower()]
    for c in price_cols:
        print(f"  {c}: {df[c].dtype}, null_rate={df[c].isnull().mean():.3f}")
    
    # ── Experiment 1: Market-Adjusted Labels ──
    print("\n=== Experiment 1: Market-Adjusted Labels ===")
    # Use price_change_percentage and index_price_change_percentage
    # Market-adjusted return = stock return - index return
    # IMPORTANT: These columns are NOT leakage when used as LABELS (not features)
    # We're re-labeling the target to see if signal survives market adjustment
    
    for split_name, split_df in [('val', ma_val), ('test', ma_test)]:
        has_both = split_df['price_change_percentage'].notna() & split_df['index_price_change_percentage'].notna()
        subset = split_df[has_both].copy()
        if len(subset) < 20:
            print(f"  {split_name}: insufficient data with both return columns")
            continue
        
        stock_ret = pd.to_numeric(subset['price_change_percentage'], errors='coerce')
        market_ret = pd.to_numeric(subset['index_price_change_percentage'], errors='coerce')
        abnormal_ret = stock_ret - market_ret
        
        # Market-adjusted labels
        mkt_adj_labels = (abnormal_ret > 0).map({True: 'up', False: 'down'})
        
        # Original labels
        orig_labels = subset['actual_side']
        
        # Agreement rate
        agreement = (mkt_adj_labels == orig_labels).mean()
        print(f"  {split_name}: orig vs mkt-adjusted label agreement: {agreement:.3f}")
        print(f"  {split_name}: mkt-adj UP rate: {(mkt_adj_labels == 'up').mean():.3f}")
        print(f"  {split_name}: orig UP rate: {(orig_labels == 'up').mean():.3f}")
        print(f"  {split_name}: mean abnormal return: {abnormal_ret.mean():.4f}%")
        print(f"  {split_name}: std abnormal return: {abnormal_ret.std():.4f}%")
        
        results[f'mkt_adj_{split_name}'] = {
            'n': len(subset),
            'agreement': round(agreement, 4),
            'mkt_adj_up_rate': round((mkt_adj_labels == 'up').mean(), 4),
            'orig_up_rate': round((orig_labels == 'up').mean(), 4),
            'mean_abnormal_return': round(abnormal_ret.mean(), 4),
            'std_abnormal_return': round(abnormal_ret.std(), 4),
        }
    
    # ── Experiment 2: Train on raw, evaluate on market-adjusted ──
    print("\n=== Experiment 2: Model on Market-Adjusted Labels ===")
    
    # Train TF-IDF model on original labels (as before)
    tfidf = TfidfVectorizer(max_features=3000, min_df=2, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(ma_train['title_en'].astype(str))
    y_train = (ma_train['actual_side'] == 'up').astype(int)
    
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train, y_train)
    
    for split_name, split_df in [('val', ma_val), ('test', ma_test)]:
        has_both = split_df['price_change_percentage'].notna() & split_df['index_price_change_percentage'].notna()
        subset = split_df[has_both].copy()
        if len(subset) < 20:
            continue
        
        X_split = tfidf.transform(subset['title_en'].astype(str))
        pred_prob = clf.predict_proba(X_split)[:, 1]
        pred = clf.predict(X_split)
        
        stock_ret = pd.to_numeric(subset['price_change_percentage'], errors='coerce')
        market_ret = pd.to_numeric(subset['index_price_change_percentage'], errors='coerce')
        abnormal_ret = stock_ret - market_ret
        mkt_adj_label = (abnormal_ret > 0).astype(int).values
        
        # MCC on original labels
        orig_label = (subset['actual_side'] == 'up').astype(int).values
        mcc_orig = matthews_corrcoef(orig_label, pred)
        
        # MCC on market-adjusted labels
        mcc_mkt = matthews_corrcoef(mkt_adj_label, pred)
        
        print(f"  {split_name}: MCC (orig labels) = {mcc_orig:.4f}")
        print(f"  {split_name}: MCC (mkt-adj labels) = {mcc_mkt:.4f}")
        
        results[f'model_mktadj_{split_name}'] = {
            'mcc_orig': round(mcc_orig, 4),
            'mcc_mkt_adjusted': round(mcc_mkt, 4),
            'n': len(subset),
        }
    
    # ── Experiment 3: Simple Long-Short Backtest ──
    print("\n=== Experiment 3: Long-Short Backtest ===")
    
    for split_name, split_df in [('test', ma_test)]:
        has_ret = split_df['price_change_percentage'].notna()
        subset = split_df[has_ret].copy()
        if len(subset) < 20:
            continue
        
        X_split = tfidf.transform(subset['title_en'].astype(str))
        pred = clf.predict(X_split)
        pred_prob = clf.predict_proba(X_split)[:, 1]
        
        stock_ret = pd.to_numeric(subset['price_change_percentage'], errors='coerce').values / 100
        
        # Strategy: go long when predicted UP, short when predicted DOWN
        positions = np.where(pred == 1, 1, -1)
        strategy_returns = positions * stock_ret
        
        # Daily aggregation
        subset_copy = subset.copy()
        subset_copy['strategy_return'] = strategy_returns
        subset_copy['stock_return'] = stock_ret
        subset_copy['date'] = pd.to_datetime(subset_copy['published_date']).dt.date
        
        daily = subset_copy.groupby('date').agg(
            strategy_return=('strategy_return', 'mean'),
            stock_return=('stock_return', 'mean'),
            n_trades=('strategy_return', 'count'),
        ).reset_index()
        
        # Statistics
        total_return = (1 + daily['strategy_return']).prod() - 1
        buy_hold = (1 + daily['stock_return']).prod() - 1
        
        ann_factor = np.sqrt(252)  # annualization
        sharpe = daily['strategy_return'].mean() / max(daily['strategy_return'].std(), 1e-8) * ann_factor
        
        win_rate = (daily['strategy_return'] > 0).mean()
        max_dd = 0
        cumulative = (1 + daily['strategy_return']).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative / peak - 1)
        max_dd = drawdown.min()
        
        # Transaction cost scenarios
        for tc_bps in [0, 10, 30, 50]:
            tc = tc_bps / 10000  # basis points to fraction
            adj_returns = strategy_returns - tc  # one-way cost per trade
            adj_total = (1 + pd.Series(adj_returns)).prod() - 1
            print(f"  {split_name} (tc={tc_bps}bps): total_return={adj_total*100:.2f}%")
        
        print(f"  {split_name}: Sharpe={sharpe:.3f}, Win_rate={win_rate:.3f}, MaxDD={max_dd*100:.2f}%")
        print(f"  {split_name}: Buy&Hold={buy_hold*100:.2f}%, Strategy={total_return*100:.2f}%")
        print(f"  {split_name}: Trading days={len(daily)}, Avg trades/day={daily['n_trades'].mean():.1f}")
        
        results[f'backtest_{split_name}'] = {
            'total_return_pct': round(total_return * 100, 4),
            'buy_hold_pct': round(buy_hold * 100, 4),
            'sharpe': round(sharpe, 4),
            'win_rate': round(win_rate, 4),
            'max_drawdown_pct': round(max_dd * 100, 4),
            'trading_days': len(daily),
            'avg_trades_per_day': round(daily['n_trades'].mean(), 1),
            'tc_scenarios': {
                f'{tc}bps': round(((1 + pd.Series(strategy_returns - tc/10000)).prod() - 1) * 100, 4)
                for tc in [0, 10, 30, 50]
            },
        }
    
    # ── Experiment 4: Confidence-filtered backtest ──
    print("\n=== Experiment 4: Confidence-Filtered Backtest ===")
    for split_name, split_df in [('test', ma_test)]:
        has_ret = split_df['price_change_percentage'].notna()
        subset = split_df[has_ret].copy()
        if len(subset) < 20:
            continue
        
        X_split = tfidf.transform(subset['title_en'].astype(str))
        pred_prob = clf.predict_proba(X_split)[:, 1]
        stock_ret = pd.to_numeric(subset['price_change_percentage'], errors='coerce').values / 100
        
        for threshold in [0.55, 0.60, 0.65]:
            # Only trade when confidence > threshold
            high_conf = (pred_prob > threshold) | (pred_prob < (1 - threshold))
            if high_conf.sum() < 10:
                print(f"  threshold={threshold}: too few trades ({high_conf.sum()})")
                continue
            
            positions_filtered = np.where(pred_prob[high_conf] > 0.5, 1, -1)
            strat_ret = positions_filtered * stock_ret[high_conf]
            total_ret = (1 + pd.Series(strat_ret)).prod() - 1
            win_rate = (strat_ret > 0).mean()
            
            print(f"  threshold={threshold}: n_trades={high_conf.sum()}, "
                  f"total_return={total_ret*100:.2f}%, win_rate={win_rate:.3f}")
            
            results[f'filtered_backtest_{threshold}_{split_name}'] = {
                'threshold': threshold,
                'n_trades': int(high_conf.sum()),
                'coverage': round(high_conf.mean(), 4),
                'total_return_pct': round(total_ret * 100, 4),
                'win_rate': round(win_rate, 4),
            }
    
    # ── Summary ──
    print("\n" + "="*60)
    print("ECONOMIC SIGNIFICANCE SUMMARY")
    print("="*60)
    print("Key question: Is the M&A signal economically exploitable?")
    
    bt = results.get('backtest_test', {})
    if bt:
        print(f"\n  Strategy return (0 tc): {bt.get('total_return_pct', 'N/A')}%")
        print(f"  Buy & Hold:             {bt.get('buy_hold_pct', 'N/A')}%")
        print(f"  Sharpe ratio:           {bt.get('sharpe', 'N/A')}")
        print(f"  Win rate:               {bt.get('win_rate', 'N/A')}")
        print(f"  Max drawdown:           {bt.get('max_drawdown_pct', 'N/A')}%")
        tc_30 = bt.get('tc_scenarios', {}).get('30bps', 'N/A')
        print(f"  After 30bps tc:         {tc_30}%")
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'economic_significance.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {os.path.join(RESULTS_DIR, 'economic_significance.json')}")


if __name__ == '__main__':
    main()
