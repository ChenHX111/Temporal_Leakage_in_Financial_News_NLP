"""
Cost-aware backtest re-run with PAPER-AUTHORITATIVE HP.

Closes Opus reviewer W1: the existing backtest (cost_aware_backtest.py) uses
HP max_features=300, C=0.1, sublinear_tf=True (a calibrated specialist).
This re-run uses the paper-authoritative locked-test HP
(max_features=100, C=5.0, sublinear_tf=False, min_df=2, ngram=(1,1)) so the
economic-significance claim reflects the same specialist used for the
headline MCC=0.138.

Output: results/validation/cost_aware_backtest_paperHP.json
"""
import os
import sys
import io
import json
import time
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

BASE = r"C:\Users\a-chenhaoxue\Documents\Fin_NLP\autoresearch_package"
DATA = os.path.join(BASE, "data", "classifier_training_v2.parquet")
OUT = os.path.join(BASE, "results", "validation", "cost_aware_backtest_paperHP.json")


def safe_mcc(y, p):
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return 0.0
    return float(matthews_corrcoef(y, p))


def daily_metrics(daily_strategy_ret):
    s = pd.Series(daily_strategy_ret).dropna()
    if len(s) < 2:
        return None
    mu = float(s.mean()); sd = float(s.std())
    sharpe = 0.0 if sd < 1e-12 else mu / sd * np.sqrt(252)
    total_ret = float((1 + s).prod() - 1)
    win_rate = float((s > 0).mean())
    cum = (1 + s).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak - 1).min()
    return {
        "n_trading_days": int(len(s)),
        "daily_mean_pct": round(mu * 100, 6),
        "daily_std_pct": round(sd * 100, 6),
        "sharpe_annualized": round(sharpe, 4),
        "total_return_pct": round(total_ret * 100, 4),
        "win_rate": round(win_rate, 4),
        "max_drawdown_pct": round(dd * 100, 4),
    }


def apply_cost(article_ret, tc_bps_per_side):
    rt = 2 * tc_bps_per_side / 10000.0
    return article_ret - rt


def aggregate_daily(subset_df, signed_ret):
    df = subset_df.copy()
    df["strategy_ret"] = signed_ret
    df["date"] = pd.to_datetime(df["published_date"]).dt.date
    return df.groupby("date")["strategy_ret"].mean().sort_index()


def break_even(cost_grid_bps, sharpes):
    pairs = sorted(zip(cost_grid_bps, sharpes))
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    for i in range(1, len(ys)):
        if ys[i] <= 0 and ys[i - 1] > 0:
            slope = (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1])
            if abs(slope) < 1e-12:
                return float(xs[i]), "exact"
            be = xs[i - 1] - ys[i - 1] / slope
            return float(be), "interp"
    if ys[-1] > 0 and len(ys) >= 2 and ys[-1] < ys[-2]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(xs[-1] - ys[-1] / slope), "extrapolated"
    return None, "stays_positive_over_grid"


def main():
    t0 = time.time()
    print("Loading data ...", flush=True)
    df = pd.read_parquet(DATA)
    df["published_date"] = pd.to_datetime(df["published_date"]).dt.tz_localize(None)
    df = df[df["actual_side"].str.lower().isin(["up", "down"])].copy()
    df["y"] = (df["actual_side"].str.lower() == "up").astype(int)
    df["title_en"] = df["title_en"].fillna("").astype(str)
    ma = df[df["event"] == "mergers_acquisitions"].copy()
    ma["price_change_percentage"] = pd.to_numeric(ma["price_change_percentage"],
                                                  errors="coerce")

    TRAIN_END = pd.Timestamp("2025-04-01")
    VAL_END = pd.Timestamp("2025-06-01")
    train_only = ma[ma["published_date"] < TRAIN_END].copy().reset_index(drop=True)
    train_val = ma[ma["published_date"] < VAL_END].copy().reset_index(drop=True)
    test_full = ma[ma["published_date"] >= VAL_END].copy().reset_index(drop=True)
    test_bt = test_full[test_full["price_change_percentage"].notna()].copy().reset_index(drop=True)
    print(f"Train: {len(train_only)}  Train+Val: {len(train_val)}  "
          f"Test (w/ returns): {len(test_bt)}", flush=True)

    # PAPER-AUTHORITATIVE HP (locked-test)
    PAPER_HP = dict(max_features=100, sublinear_tf=False, min_df=2,
                    ngram_range=(1, 1), stop_words="english")
    PAPER_LR_C = 5.0

    cost_grid_bps = [0, 1, 2, 5, 10, 15, 20, 30, 50]

    def run_protocol(train_df, label):
        print(f"\n{'=' * 60}", flush=True)
        print(f"PROTOCOL: {label} (n_train={len(train_df)})", flush=True)
        print(f"  HP: TF-IDF mf=100 sublinear=F min_df=2 ng=(1,1); LR C=5.0", flush=True)
        print(f"{'=' * 60}", flush=True)

        tfidf = TfidfVectorizer(**PAPER_HP)
        Xtr = tfidf.fit_transform(train_df["title_en"])
        Xte = tfidf.transform(test_bt["title_en"])
        ytr = train_df["y"].values
        yte = test_bt["y"].values

        clf = LogisticRegression(max_iter=2000, C=PAPER_LR_C, random_state=42)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        yp_prob = clf.predict_proba(Xte)[:, 1]
        mcc_test = safe_mcc(yte, yp)
        print(f"  Locked-test MCC = {mcc_test:+.4f}  (paper headline: +0.138)", flush=True)

        # Probability spread diagnostics (helps explain why C=0.1 was used originally)
        spread_q = float(np.percentile(np.abs(yp_prob - 0.5), 75))
        n_high_conf = int((np.abs(yp_prob - 0.5) >= 0.2).sum())
        print(f"  Probability spread: 75pct |p-0.5| = {spread_q:.4f}; "
              f"n_high_conf (|p-0.5|>=0.20) = {n_high_conf}/{len(yp_prob)}", flush=True)

        stock_ret = test_bt["price_change_percentage"].values / 100.0
        positions = np.where(yp == 1, 1, -1)
        article_strat_ret = positions * stock_ret

        # All-trade cost grid
        print("\n[All-trade backtest across cost grid]", flush=True)
        all_results = {}
        sharpes_all = []
        for tc in cost_grid_bps:
            netr = apply_cost(article_strat_ret, tc)
            daily = aggregate_daily(test_bt, netr)
            m = daily_metrics(daily.values)
            all_results[f"{tc}bps_per_side"] = m
            sharpes_all.append(m["sharpe_annualized"])
            print(f"  tc={tc:>2d}bps/side: Sharpe={m['sharpe_annualized']:+.3f}  "
                  f"total_ret={m['total_return_pct']:+.2f}%  "
                  f"win_rate={m['win_rate']:.3f}  "
                  f"maxDD={m['max_drawdown_pct']:.2f}%", flush=True)
        be_cost, be_method = break_even(cost_grid_bps, sharpes_all)
        print(f"  Break-even per-side: {be_cost} bps ({be_method})", flush=True)

        # Confidence-filtered (note: at C=5.0 we expect few high-confidence trades)
        print("\n[Confidence-filtered backtest]", flush=True)
        filt_results = {}
        abs_conf = np.abs(yp_prob - 0.5)
        for q_pct in [50, 75, 90]:
            thresh = np.percentile(abs_conf, q_pct)
            mask = abs_conf >= thresh
            n_kept = int(mask.sum())
            if n_kept < 20:
                filt_results[f"top_{100 - q_pct}pct_conf"] = {"note": f"n={n_kept}"}
                continue
            sub_strat = article_strat_ret[mask]
            sub_df = test_bt.iloc[mask].copy()
            by_cost = {}
            for tc in cost_grid_bps:
                netr = apply_cost(sub_strat, tc)
                daily = aggregate_daily(sub_df, netr)
                by_cost[f"{tc}bps_per_side"] = daily_metrics(daily.values)
            filt_results[f"top_{100 - q_pct}pct_conf"] = {
                "n_articles_kept": n_kept,
                "threshold_abs_p_minus_half": float(thresh),
                "cost_grid": by_cost}
            print(f"  top-{100 - q_pct}% (n={n_kept}, thresh={thresh:.3f}): "
                  f"Sharpe@0={by_cost['0bps_per_side']['sharpe_annualized']:+.3f} "
                  f"@10={by_cost['10bps_per_side']['sharpe_annualized']:+.3f} "
                  f"@20={by_cost['20bps_per_side']['sharpe_annualized']:+.3f}", flush=True)

        # Long-only / short-only
        long_mask = (yp == 1); short_mask = (yp == 0)
        sub_results = {}
        for name, sub_strat, sub_df, n in [
                ("long_only", stock_ret[long_mask], test_bt[long_mask].copy(),
                 long_mask.sum()),
                ("short_only", -stock_ret[short_mask], test_bt[short_mask].copy(),
                 short_mask.sum())]:
            if n < 20:
                sub_results[name] = {"note": f"n={n}"}
                continue
            by_cost = {}
            for tc in [0, 10, 20]:
                netr = apply_cost(sub_strat, tc)
                daily = aggregate_daily(sub_df, netr)
                by_cost[f"{tc}bps_per_side"] = daily_metrics(daily.values)
            sub_results[name] = {"n_articles": int(n), "cost_grid": by_cost}

        return {
            "protocol_label": label,
            "n_train": int(len(train_df)),
            "n_test_with_returns": int(len(test_bt)),
            "locked_test_mcc": float(mcc_test),
            "prob_spread_75pct_abs_p_half": spread_q,
            "n_high_conf_p_diff_ge_0_2": n_high_conf,
            "all_trades_cost_grid": all_results,
            "break_even": {
                "per_side_bps": be_cost,
                "round_trip_bps": (2 * be_cost) if be_cost is not None else None,
                "method": be_method},
            "confidence_filtered": filt_results,
            "directional_sub_strategies": sub_results,
        }

    prot_train_only = run_protocol(train_only, "train_only (matches paper headline MCC=0.138)")
    prot_train_val = run_protocol(train_val, "train+val merge (matches paper merge MCC=0.068)")

    summary = {
        "meta": {
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_test_with_returns": int(len(test_bt)),
            "test_window": "2025-06-01 to end (3 months: June, July, August 2025)",
            "model_hp": ("PAPER-AUTHORITATIVE: TF-IDF max_features=100 sublinear_tf=False "
                        "min_df=2 ngram=(1,1) stop_words=english; LR C=5.0 seed=42"),
            "rationale": ("Re-run of cost_aware_backtest.py with the locked-test specialist "
                          "HP (Sec.~5.2 of paper), to address reviewer concern that the "
                          "original backtest used a calibrated specialist (mf=300, C=0.1) "
                          "different from the headline MCC=0.138 specialist."),
        },
        "protocols": {
            "train_only": prot_train_only,
            "train_val_merged": prot_train_val,
        },
        "elapsed_seconds": float(time.time() - t0),
        "headline_summary": {
            "train_only": {
                "locked_test_mcc": prot_train_only["locked_test_mcc"],
                "sharpe_0bps": prot_train_only["all_trades_cost_grid"]["0bps_per_side"]["sharpe_annualized"],
                "sharpe_10bps": prot_train_only["all_trades_cost_grid"]["10bps_per_side"]["sharpe_annualized"],
                "sharpe_20bps": prot_train_only["all_trades_cost_grid"]["20bps_per_side"]["sharpe_annualized"],
                "break_even_per_side_bps": prot_train_only["break_even"]["per_side_bps"],
                "total_return_0bps_pct": prot_train_only["all_trades_cost_grid"]["0bps_per_side"]["total_return_pct"],
            },
            "train_val_merged": {
                "locked_test_mcc": prot_train_val["locked_test_mcc"],
                "sharpe_0bps": prot_train_val["all_trades_cost_grid"]["0bps_per_side"]["sharpe_annualized"],
                "sharpe_10bps": prot_train_val["all_trades_cost_grid"]["10bps_per_side"]["sharpe_annualized"],
                "sharpe_20bps": prot_train_val["all_trades_cost_grid"]["20bps_per_side"]["sharpe_annualized"],
                "break_even_per_side_bps": prot_train_val["break_even"]["per_side_bps"],
                "total_return_0bps_pct": prot_train_val["all_trades_cost_grid"]["0bps_per_side"]["total_return_pct"],
            },
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
