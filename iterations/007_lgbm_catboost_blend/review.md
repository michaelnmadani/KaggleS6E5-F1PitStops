# Review — Iteration 007: LGBM + CatBoost Blend

## CV Score
| Metric | Value | vs 005 (best) | vs 006 |
|---|---|---|---|
| CV AUC (blend) | **0.9494** | -0.0005 | -0.0002 |
| CV AUC (lgbm) | 0.9480 | -0.0017 | — |
| CV AUC (catboost) | 0.9489 | — | -0.0007 |
| Balanced accuracy | 0.883 | +0.043 | -0.003 |
| Elapsed (s) | 1541 | — | — |

## Analysis

Blend of LGBM + CatBoost **did not outperform** the best single models. Two confounds:

**1. `class_weights: balanced` costs ~0.0017 AUC.**
Iter 005 LGBM without class weights: 0.9497. Iter 007 LGBM with class weights: 0.9480. The delta is -0.0017 — balanced weights push the model to optimize recall at the expense of AUC ranking. For a pure AUC competition metric, this is counterproductive.

**2. CatBoost without native categoricals loses its main advantage.**
CatBoost in iter 007 (with target-encoded features): 0.9489. CatBoost in iter 006 (native cats): 0.9496. Using shared target-encoded features forces CatBoost to operate without ordered categorical boosting, reducing its contribution.

**Blend math:** With both models at similarly low AUC (0.9480 / 0.9489), their 50/50 blend produces 0.9494 — no diversity premium over constituent models.

## Key Insight

Remove `class_weights: balanced` for AUC-metric competitions. It was useful for diagnosing the recall gap but actively hurts the leaderboard metric. Revert to unweighted training.

## 3 Concrete Next Ideas

1. **LGBM no class_weights + rolling features (iter 009)**: Combine the rolling lap-time features from iter 008 (if it scores well) with standard unweighted LGBM. Isolating the feature signal from the class_weight confound. Expected: +0.003–0.008 AUC.
2. **Feature interaction crosses**: Multiply key pairs: `tyre_life × rolling_lt3`, `gap × stint_frac`, `position × tyre_life`. These interaction terms capture compound signals the tree can't express in 2 splits. Expected: +0.001–0.003 AUC.
3. **TabNet or MLP blend**: Neural architectures handle tabular data differently from GBMs — especially interaction patterns. A 3-layer MLP with batch norm as a blend component adds diversity without needing new features. Expected: +0.001–0.003 AUC.
