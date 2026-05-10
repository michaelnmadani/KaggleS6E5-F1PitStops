# Next Iteration: 009_rolling_no_weights

## What to change
LGBM with rolling features (from iter 008) but **no class_weights**. Iter 007 proved that `class_weights: balanced` costs ~0.0017 AUC. This iteration isolates the rolling feature signal cleanly.

Uses: fill_na_median + f1_tyre_features + f1_gap_features + f1_stint_features + f1_position_features + f1_race_rolling + target_encode_binary. No class_weights. LGBM with num_leaves=127 (best hyperparams from iter 004/005).

**Wait for iter 008 results first.** If iter 008 rolling features improve AUC over 005's 0.9499 baseline, iter 009 is the clean combination. If iter 008 does not improve, iter 009 pivots to feature interaction crosses.

## Why
- Iter 007 identified class_weights=balanced as a -0.0017 AUC tax — must remove
- Rolling features (laptime trend, lap delta, driver-race deviation) are the most physics-grounded new signals available in the dataset
- Combining both learnings (rolling features + no class_weights) should recover at minimum iter 005's 0.9499 and potentially gain +0.003–0.008 from the new features

## Expected delta in CV AUC
+0.003–0.008 AUC (from 0.9499 → ~0.953–0.958) if iter 008 rolling features help.
+0.000–+0.002 if rolling features have minimal impact (iter 008 will tell us).

## Risk
Low-medium. LGBM without class_weights is well-understood. Rolling feature block (f1_race_rolling) was just written and untested on Kaggle — iter 008 is the test. Kaggle runtime ~60–90 min.
