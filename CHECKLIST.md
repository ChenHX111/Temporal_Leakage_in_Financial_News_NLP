# Temporal Leakage Audit — reusable checklist

A short, reusable protocol for auditing temporal leakage in financial-news (and similar time-ordered) text
classification. Each step names the released script that implements it, so the audit is repeatable on a new corpus.
The *contribution here is the packaged protocol and evidence*, not a new learning algorithm.

1. **Pre-specify the temporal cutoffs and the target metric before touching the locked test.**
   Fix `TRAIN_END` / `VAL_END` and the headline metric (MCC) up front; evaluate the locked test exactly once.
   → `code/validation/cost_aware_backtest_paperHP.py` (train-only 0.138; train+val refit 0.068).

2. **Report ΔMCC, not the random/chronological ratio.**
   ΔMCC = (random-split MCC − chronological MCC) is stable near zero; the *ratio* explodes when the chronological
   denominator is small. Lead with ΔMCC (here up to +0.145).
   → `code/validation/leakage_audit_definitive.py`, `leakage_audit_multiseed.py`.

3. **Cross-check with purged/embargoed K-fold and forward-chaining CV.**
   Naive random CV inflates MCC (~2.2× here); López de Prado purged+embargoed K-fold and forward-chaining CV
   recover the honest chronological estimate.
   → `experiments/E2_mitigation.py`.

4. **Disclose the naive-random-CV inflation factor explicitly.**
   Report the inflation as a number so readers can see the size of the leakage, not just its sign.
   → `experiments/E2_mitigation.py` (naive-CV vs chronological vs mitigated grid).

5. **Use a single one-shot chronological locked test — no refitting or threshold-tuning on test.**
   Any confidence threshold must be calibrated on validation, then applied to test unchanged.
   → `experiments/E3_valcal_backtest.py` (|val−test threshold| = 0.0004; deployable top-quartile Sharpe +2.33@10bps,
   all-trade −0.60@10bps — an upper bound, slippage/impact unmodelled).

6. **Stress the signal under ex-ante inclusion and entity-disjoint splits.**
   Confirm the effect survives (a) ex-ante-observable / neutrals-included sample inclusion and (b) firm-holdout
   (unseen-entity) test firms.
   → `experiments/E1_neutrals.py` (τ=0 ex-ante MCC +0.078, p=0.019); `experiments/E4_firm_disjoint.py`
   (firm-disjoint MCC +0.098, n=551).

All numbers above are reproducible from this bundle; see `REPRODUCE.md` for exact commands and expected outputs.
