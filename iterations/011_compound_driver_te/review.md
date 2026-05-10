# Review — Iteration 011: Compound×Driver Interaction Target Encoding

## CV Score
| Metric | Value | vs 005 (best) | vs 010 |
|---|---|---|---|
| CV AUC | **0.9492** | -0.0007 | -0.0003 |
| Balanced accuracy | 0.838 | ~same | ~same |
| n_features | 31 | +4 | +2 |
| Elapsed (s) | 190 | — | — |

Fold scores: 0.9501, 0.9479, 0.9493, 0.9487, 0.9498 (std 0.0008)

## Analysis

`compound_x_driver` and `compound_x_race` were created (+2 features), target-encoded, and dropped as object columns — net +2 numeric features (the `_te` columns). Result is worse than iter 010 by -0.0003.

**Root causes:**
1. **High-cardinality sparse combinations**: ~5 compounds × ~20 drivers = ~100 `compound_x_driver` categories; ~5 × ~25 races = ~125 `compound_x_race` categories. With `smoothing=10`, rare categories are heavily shrunk toward the global mean — adding a noisy version of the global pit rate.
2. **Redundancy with existing encodings**: `Driver_te` and `Compound_te` already capture most of this signal independently. The interaction term's marginal value above noise is low.
3. **Overfitting within folds**: High-cardinality target encoding is susceptible to within-fold variance even with smoothing; fold std increased (0.0008 vs 0.0007 in iter 010).

**Pattern:** Eleven iterations all cluster between 0.9479–0.9499. Feature engineering improvements are saturating — we're adding correlated or redundant representations of the same underlying signal (tyre age, position, gap).

## 3 Concrete Next Ideas

1. **LGBM + XGB blend with full working feature set (iter 012)**: Iter 005 blended LGBM+XGB without working stint features (0.9499). Run the same blend architecture with iter 010's corrected feature set (stint_frac, all 4 crosses). Blend diversity was +0.0002 in iter 005; may compound with better features. Expected: 0.9500–0.9503.
2. **Optuna HPO (iter 013)**: 30-trial search over `num_leaves` [63–511], `learning_rate` [0.01–0.1], `subsample` [0.6–1.0], `min_child_samples` [10–100], `reg_lambda` [0.1–5.0]. The current hparams may be a local optimum. Expected: +0.001–0.003 AUC.
3. **New raw features — pit-window indicators (iter 014)**: Add `typical_pit_lap_per_compound` (training mean of pit lap per compound), `laps_since_safe_window` = max(0, TyreLife - compound_min_pit_window). These capture whether a driver is in their expected pit window — a direct signal for PitNextLap.
