# Review — Iteration 008: Rolling Lap-Time Features

## CV Score
| Metric | Value | vs 005 (best) | vs 007 LGBM |
|---|---|---|---|
| CV AUC | **0.9479** | -0.0020 | -0.0001 |
| Balanced accuracy | 0.878 | +0.038 | -0.005 |
| n_features | 30 | +3 | +3 |
| Elapsed (s) | 348 | — | — |

## Analysis

Rolling features (rolling_lt3, laptime_d1, lt_vs_dr_mean) added 3 columns but **produced no AUC gain**. Comparing to iter 007 LGBM component (same class_weights, same base features, 0.9480): rolling features gave 0.9479 — statistically identical.

**Root cause: random CV folds break rolling feature quality.**
The 5-fold stratified CV randomly assigns rows, so within each training fold, consecutive laps from the same race/driver may be non-consecutive or absent. The rolling window (shift(1).rolling(3)) on a randomly-sampled subset of a race's laps produces noisy, non-representative values. For the val/test set, we fall back to per-driver-race training aggregates — essentially the same information as target encoding already provides.

Rolling features computed on race-ordered data would require **time-based folds** (split by race/season), which changes the CV setup. That's a larger change.

**class_weights: balanced** still present — costs -0.0017 AUC as established in iter 007.

## 3 Concrete Next Ideas

1. **Feature interaction crosses (iter 009)**: Multiply key feature pairs: `tyre_life × GapToLeader`, `stint_frac × position`, `tyre_life_sq × GapToLeader`. These capture compound effects trees can't express in 2 splits. Global block, no leakage. Expected: +0.001–0.004 AUC.
2. **Race-based CV folds**: Switch from stratified random folds to leave-one-race-out or group-by-race CV. This validates that rolling features actually generalize to unseen races and may reveal true signal from rolling features. Requires `data.py` change.
3. **LGBM with deeper trees + Optuna**: num_leaves=255, max_depth=12, Optuna for 50-trial HPO on learning_rate / subsample / colsample. Direct squeeze on the current feature set. Expected: +0.001–0.002 AUC.
