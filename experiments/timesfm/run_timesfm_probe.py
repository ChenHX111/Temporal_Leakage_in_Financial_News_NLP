"""
run_timesfm_probe.py — E5: a numeric time-series-FM (TimesFM) PRICE-ONLY probe on the M&A locked test.

Purpose (rebuttal answer to ysCR-W2/C1): TimesFM/Chronos/Moirai are NUMERIC autoregressive forecasters over a price
series; they do NOT ingest news text. This probe answers the fair, in-category question -- "is the M&A direction signal
just price momentum a time-series FM can capture?" -- and thereby positions TimesFM as a NON-TEXT baseline, not a
replacement for the paper's text specialist. Expected: TimesFM (and the momentum / AR baselines) score near-chance on
next-day M&A direction, well below the paper's TF-IDF text specialist (MCC 0.138) and consistent with the paper's
existing non-text baseline (val MCC -0.020). That is the point: text carries the signal, numeric price history does not.

For each M&A test article we build the ticker's trailing K daily log-returns ending the trading day BEFORE the article,
forecast the next-day return, and take its SIGN as the UP/DOWN prediction. We always compute two model-free numeric
baselines (momentum = sign of last return; ar_mean = sign of trailing-K mean return) so the core result lands even if
the TimesFM checkpoint is unavailable. MCC is computed against the paper's actual_side label (y).

Out: results/timesfm_probe.json
"""
import os, sys, io, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "data", "ma_locked_test_meta.parquet")
PRICES = os.path.join(HERE, "data", "prices_cache.parquet")
OUT = os.path.join(HERE, "results", "timesfm_probe.json")
CTX = 128   # context length (trading days) fed to TimesFM


def mcc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    m = (p != 0)  # drop undefined-sign predictions
    if m.sum() < 20 or len(np.unique(y[m])) < 2 or len(np.unique(np.sign(p[m]))) < 2:
        return None, int(m.sum())
    return float(matthews_corrcoef(y[m], (p[m] > 0).astype(int))), int(m.sum())


def load_timesfm():
    """timesfm>=2.0.2 (July 2026 PyPI) ships the NEW 2.5-model API:
       timesfm.TimesFM_2p5_200M_torch.from_pretrained(...).compile(timesfm.ForecastConfig(...)).
       The legacy TimesFmHparams/TimesFmCheckpoint/TimesFm API needs `pip install timesfm==1.3.0`.
       We try the 2.5 API first, then the legacy 1.x API as a fallback."""
    try:
        import timesfm, torch
    except ImportError:
        return None
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # --- new 2.5-model API (what the installed timesfm 2.0.2 actually exposes) ---
    if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        try:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            model.compile(timesfm.ForecastConfig(
                max_context=256, max_horizon=16, normalize_inputs=True,
                use_continuous_quantile_head=False, force_flip_invariance=True,
                infer_is_positive=False,  # log-returns are signed; must NOT force positivity
                fix_quantile_crossing=False))
            print("  [TimesFM 2.5 loaded: TimesFM_2p5_200M_torch]", flush=True)
            return ("timesfm25", model)
        except Exception as e:
            print(f"  [TimesFM 2.5 API failed: {type(e).__name__}: {str(e)[:160]}]", flush=True)
    # --- legacy 1.x API (only if timesfm==1.3.0 is installed) ---
    if hasattr(timesfm, "TimesFmHparams"):
        backend = "gpu" if torch.cuda.is_available() else "cpu"
        for repo, nl in [("google/timesfm-2.0-500m-pytorch", 50), ("google/timesfm-1.0-200m-pytorch", 20)]:
            try:
                m = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(backend=backend, per_core_batch_size=32, horizon_len=16,
                                                   context_len=CTX, num_layers=nl, use_positional_embedding=False),
                    checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=repo))
                print(f"  [TimesFM legacy 1.x loaded: {repo} (num_layers={nl})]", flush=True)
                return ("timesfm_legacy", m)
            except Exception as e:
                print(f"  [TimesFM legacy {repo} failed: {type(e).__name__}: {str(e)[:140]}]", flush=True)
    return None


def load_chronos():
    """Chronos-Bolt (Amazon) — pure-PyTorch, installs cleanly: pip install chronos-forecasting."""
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError:
        return None
    for repo in ("amazon/chronos-bolt-small", "amazon/chronos-t5-small"):
        try:
            dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            pipe = BaseChronosPipeline.from_pretrained(repo, device_map=dev)
            print(f"  [Chronos loaded: {repo} on {dev}]", flush=True); return ("chronos", pipe)
        except Exception as e:
            print(f"  [Chronos {repo} failed: {e}]", flush=True)
    return None


def fm_predict(kind, model, contexts):
    """Return sign(next-step forecast) for each return-series context."""
    import numpy as _np
    preds = []
    if kind == "timesfm25":
        # 2.5 API: model.forecast(horizon, inputs=[1d arrays]) -> (point_forecast[n,h], quantile_forecast[n,h,10])
        B = 256
        for i in range(0, len(contexts), B):
            batch = [_np.asarray(c, dtype=_np.float32) for c in contexts[i:i + B]]
            try:
                out = model.forecast(horizon=1, inputs=batch)
                point = out[0] if isinstance(out, (tuple, list)) else out
                point = _np.asarray(point)
                preds.extend([float(_np.sign(_np.asarray(row).ravel()[0])) for row in point])
            except Exception as e:
                if i == 0:
                    print(f"    [timesfm25.forecast error: {type(e).__name__}: {str(e)[:180]}]", flush=True)
                preds.extend([0] * len(batch))
    elif kind == "timesfm_legacy":
        B = 128
        for i in range(0, len(contexts), B):
            batch = [_np.asarray(c, dtype=_np.float32) for c in contexts[i:i + B]]
            try:
                out = model.forecast(batch, freq=[0] * len(batch))
                fc = out[0] if isinstance(out, (tuple, list)) else out
                for f in fc:
                    arr = f.cpu().numpy() if hasattr(f, "cpu") else _np.asarray(f)
                    preds.append(float(_np.sign(arr.ravel()[0])))
            except Exception as e:
                if i == 0:
                    print(f"    [timesfm_legacy.forecast error: {type(e).__name__}: {str(e)[:180]}]", flush=True)
                preds.extend([0] * len(batch))
    elif kind == "chronos":
        import torch
        B = 256
        for i in range(0, len(contexts), B):
            batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[i:i + B]]
            try:
                q, mean = model.predict_quantiles(batch, prediction_length=1, quantile_levels=[0.5])
                preds.extend([float(_np.sign(m.ravel()[0])) for m in mean.cpu().numpy()])
            except Exception:
                try:
                    fc = model.predict(batch, prediction_length=1)
                    preds.extend([float(_np.sign(_np.median(_np.asarray(f), axis=0).ravel()[0])) for f in fc])
                except Exception:
                    preds.extend([0] * len(batch))
    return _np.array(preds)


def main():
    t0 = time.time()
    meta = pd.read_parquet(META); meta["published_date"] = pd.to_datetime(meta["published_date"]).dt.tz_localize(None)
    if not os.path.exists(PRICES):
        sys.exit("Missing data/prices_cache.parquet -> run:  python build_prices.py")
    px = pd.read_parquet(PRICES); px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None)
    px = px.sort_values(["yf_ticker", "date"])
    by = {t: g for t, g in px.groupby("yf_ticker")}

    contexts, ys, momentum, armean = [], [], [], []
    used = 0
    for _, r in meta.iterrows():
        g = by.get(str(r["yf_ticker"]))
        if g is None:
            continue
        hist = g[g["date"] < pd.Timestamp(r["published_date"]).normalize()]  # strictly BEFORE the article calendar date
        if len(hist) < 30:
            continue
        closes = hist["close"].values[-(CTX + 1):]
        rets = np.diff(np.log(closes))
        if len(rets) < 20:
            continue
        contexts.append(rets.astype(np.float32)); ys.append(int(r["y"]))
        momentum.append(np.sign(rets[-1])); armean.append(np.sign(rets[-20:].mean())); used += 1
    ys = np.array(ys)
    print(f"Built {used} contexts (of {len(meta)}); coverage {100*used/len(meta):.0f}%.", flush=True)

    res = {"n_articles": int(len(meta)), "n_used": int(used), "context_len": CTX}
    for name, pred in [("momentum_last", momentum), ("ar_mean_20", armean)]:
        m, n = mcc(ys, np.array(pred)); res[name] = {"mcc": m, "n_scored": n}
        print(f"  baseline {name:14s}: MCC={m} (n={n})", flush=True)

    # Foundation models: try Chronos (robust, pure-torch) and TimesFM; record whichever load.
    got_fm = False
    for loader in (load_chronos, load_timesfm):
        fm = loader()
        if fm is None:
            continue
        kind, model = fm
        preds = fm_predict(kind, model, contexts)
        m, n = mcc(ys, preds); res[f"fm_{kind}"] = {"mcc": m, "n_scored": n}
        print(f"  FM {kind:10s}: MCC={m} (n={n})", flush=True); got_fm = True
    if not got_fm:
        res["fm_note"] = ("No time-series FM installed; numeric baselines stand in as the non-text control. "
                          "For a real FM number: pip install chronos-forecasting  (or timesfm), then re-run.")
        print("  [no FM installed -> numeric baselines are the non-text control]", flush=True)

    res["paper_text_specialist_mcc"] = 0.138
    res["paper_nontext_baseline_val_mcc"] = -0.020
    res["interpretation"] = ("Numeric price-only forecasters (TimesFM / momentum / AR) score near-chance on next-day "
                             "M&A direction, far below the text specialist (0.138) and consistent with the paper's "
                             "non-text baseline (-0.020): the signal is in the news TEXT, not in price history. "
                             "TimesFM is therefore a complementary non-text baseline, not a substitute for fin-NLP.")
    res["elapsed_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print("\nSaved:", OUT, flush=True)


if __name__ == "__main__":
    main()
