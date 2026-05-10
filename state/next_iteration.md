# Next Iteration: 008_rolling_features

## What to change
Add a new `f1_race_rolling` feature block computing per-driver rolling statistics within each race group:
- `rolling_laptime_3`: rolling mean of last 3 lap times per driver per race
- `laptime_vs_rolling`: current LapTime minus rolling mean (pace degradation signal)
- `stint_pace_degrade`: lap-by-lap slope of lap time within current stint
- `laps_remaining_est`: estimated laps remaining in race based on total lap count distribution

These features capture within-race dynamics that static stint counters miss. They require groupby within fold (no leakage) → implemented as a per-fold block.

New block name: `f1_race_rolling` in features.py.

## Why
Iter 007 (LGBM+CatBoost blend) should confirm whether algorithm diversity gives real AUC gains. If it does, the next lever is new features. If it doesn't, feature engineering is even more critical.

The current 27 features (after target encoding) are all static or aggregate. Rolling lap time trends are the most actionable F1 domain signal for pit timing: teams pit when pace degrades beyond a threshold. The model currently has no direct signal for pace trend.

## Expected delta in CV AUC
+0.003–0.008 AUC (from ~0.950 → ~0.953–0.958). Higher variance than prior iters because this requires new feature engineering code.

## Risk
Medium-high. The rolling groupby must be per-fold (train only) to avoid leakage. Implementation in features.py is new code (additive block). Kaggle runtime ~60–90 min with LGBM.
