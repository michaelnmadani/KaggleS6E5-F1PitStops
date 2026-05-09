# Next Iteration: 006_catboost

## What to change
Switch model to CatBoost with native categorical handling. Remove target_encode_binary and label_encode — CatBoost receives raw string columns (Driver, Compound, Race) directly. Add `class_weights: balanced` to address the persistent 6pp balanced-accuracy gap.

Code change: `_catboost_fit` in models.py now auto-detects object-dtype columns and passes them as `cat_features` via Pool (additive fix, already committed).

## Why
Five iterations of LGBM/XGB gave cumulative +0.0013 AUC. Both GBM families find similar boundaries on the same features. CatBoost uses:
- **Ordered boosting** — unbiased gradient estimates, different residuals
- **Native categoricals** — symmetric tree splits on raw Driver/Compound/Race without target-encoding leakage
- **Balanced class weights** — directly increases recall on minority pit-stop events

Expected to break the current ceiling by exploiting categorical interaction patterns that smoothed mean encoding obscures.

## Expected delta in CV AUC
+0.002 to +0.005 (from 0.9499 → ~0.952–0.955).

## Risk
Medium. CatBoost is slower than LGBM (may hit 9h Kaggle timeout with 3000 iterations if queue is slow — reduce to 2000 if needed). The models.py fix is small and isolated. If CatBoost underperforms, iter 007 falls back to LGBM with rolling features.
