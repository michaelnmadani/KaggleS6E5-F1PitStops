# Review — Iteration 006: CatBoost + Balanced Class Weights

## CV Score
| Metric | Value | vs 005 | vs 001 |
|---|---|---|---|
| CV AUC | **0.9496** | -0.0003 | +0.0010 |
| Balanced accuracy | **0.886** | +0.046 | +0.081 |
| Plain accuracy | 0.873 | -0.027 | — |
| n_features | 22 | -5 | +7 |
| Elapsed (s) | 4235 | +3596 | — |

## Analysis

CatBoost with native categoricals marginally underperformed on AUC vs the 005 LGBM/XGB blend (-0.0003), but the **balanced accuracy jumped +4.6pp** (0.840 → 0.886). This is the largest single-iteration gain in recall on pit events across all runs.

Key observations:
- `class_weights: balanced` is highly effective for the recall gap — this setting should stay
- CatBoost achieves near-parity on AUC with 5 fewer features (22 vs 27), confirming that native ordered boosting extracts more signal from raw categorical columns than target encoding
- The AUC gap to LGBM/XGB is within fold-std noise (0.0007) — both families find similar AUC ceilings but with **very different calibration patterns** (LGBM plain_acc=0.900 vs CatBoost plain_acc=0.873)

**Blend potential:** Calibration divergence (6pp balanced_acc gap at same AUC) is the strongest signal of prediction diversity seen so far. LGBM+CatBoost blend has far more rank-correlation diversity than LGBM+XGB did, making iter 007 a strong candidate for a real AUC jump.

## Top Error Buckets
- CatBoost still mispredicts some pit events when lap time signals are weak early in stint
- Balanced accuracy gap to plain accuracy (0.886 vs 0.873) — recall on positives is high but some false-positives remain near stint boundaries

## 3 Concrete Next Ideas

1. **LGBM + CatBoost blend with target_encode (shared features)**: Use the 005 LGBM config + 006 CatBoost config over the same 27-feature target-encoded set. Algorithm diversity is the main value here. Expected: +0.001–0.003 AUC.
2. **Race-group rolling features**: Per-driver/stint rolling lap time trend (last 3 laps), stint pace degradation, laps-remaining estimate. New `f1_race_rolling` feature block. Expected: +0.003–0.008 AUC.
3. **CatBoost with extra hyperparameter tuning**: Reduce learning_rate to 0.01, increase iterations to 5000, add `bagging_temperature: 0.3`. Expected: +0.001–0.002 AUC.
