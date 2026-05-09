# Review — Iteration 004: Target Encoding Fix + LGBM Tuning

## CV Score
| Metric | Value | vs 002 | vs 001 |
|---|---|---|---|
| CV AUC (mean) | **0.9497** | +0.0006 | +0.0011 |
| CV AUC (std) | 0.0009 | +0.0001 | +0.0002 |
| Plain accuracy | 0.8997 | +0.0004 | +0.0007 |
| Balanced accuracy | 0.8388 | +0.0008 | +0.0035 |
| n_features | 27 | +5 | +12 |
| Elapsed (s) | 395 | +156 | +163 |

(Note: iter 003 crashed on object dtype bug — this is the first successful target-encoded run.)

## CV vs LB Gap
LB not yet checked. **Strongly recommend submitting iter 004 via `submit.yml` to calibrate.**

## Analysis

Target encoding added 5 `_te` columns (Driver_te, Compound_te, Race_te + 2 others) and LGBM tuning (num_leaves=127, LR=0.03) together gave +0.0006 AUC over 002. Total gain from baseline: +0.0011.

The incremental improvement rate (~0.0005–0.0006 per experiment) suggests we may be approaching a feature-representation ceiling with single-model LGBM. Key observations:

1. **15 raw features** — the dataset is relatively sparse. All gains come from transforming existing columns, not new data.
2. **Balanced accuracy improving (+0.0035 vs baseline)** — class imbalance handling is getting better but the model still struggles on minority-class recall.
3. **27 features, ~0.9497 AUC** — ensemble diversity is the most likely next lever.

## 3 Concrete Next Ideas

1. **LGBM + XGBoost blend**: The `model: [lgbm, xgb]` config is already supported. Two GBM implementations with orthogonal residual patterns. Expected: +0.001–0.003 AUC.
2. **Class-weighted training**: Add `class_weights: balanced` to config. Directly addresses the balanced-accuracy gap; may trade plain accuracy for recall on pit events. Expected: +0.0005–0.002 AUC on balanced_acc, less certain on AUC.
3. **Submit 004 to LB**: Get actual public LB score to calibrate CV-vs-LB gap and understand true distance to top 5%.
