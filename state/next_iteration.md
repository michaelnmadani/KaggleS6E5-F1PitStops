# Next Iteration: 002_f1_features

## What to change
Add all four F1-domain feature blocks to the pipeline:
- `f1_tyre_features`: tyre_life_sq, tyre_life_log1p, tyre_life_pct_lap
- `f1_gap_features`: gap_abs, gap_log1p for all gap columns
- `f1_stint_features`: stint_sq, past_half_stint, stint_frac
- `f1_position_features`: position_inv, lap_pct

Feature list in config: `[fill_na_median, f1_tyre_features, f1_gap_features, f1_stint_features, f1_position_features, label_encode]`

Keep all other params identical to 001_baseline (same LGBM hparams, 5-fold stratified CV, seed 42).

## Why
The baseline only uses 15 raw features with no domain engineering. Tyre age is the primary strategic signal for pit decisions — polynomial/log transforms expose the degradation cliff. Gap to next car is the undercut/overcut signal. Stint fraction encodes urgency. All four blocks are pre-built and tested; this is zero new code risk.

## Expected delta in CV AUC
+0.004 to +0.008 (from 0.9486 → ~0.952–0.957).
Reasoning: tyre features alone typically give the biggest lift in pit-stop prediction tasks; the other three add incremental signal.

## Risk
Low. All blocks are additive, use column-name heuristics, and are no-ops if the expected column is missing. The same LGBM hparams are retained so any regression is attributable purely to the features.
