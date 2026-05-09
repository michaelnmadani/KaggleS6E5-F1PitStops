# Review — Iteration 005: LGBM + XGBoost Blend

## CV Score
| Metric | Value | vs 004 | vs 001 |
|---|---|---|---|
| CV AUC (blend) | **0.9499** | +0.0002 | +0.0013 |
| CV AUC (lgbm) | 0.9497 | — | — |
| CV AUC (xgb) | 0.9492 | — | — |
| Blend weights | 50/50 | — | — |
| Balanced accuracy | 0.8400 | +0.0012 | +0.0047 |
| Elapsed (s) | 639 | +244 | +407 |

## Analysis

The blend gave only +0.0002 AUC because LGBM and XGB are highly correlated:
- Blend settled at exactly 50/50 (score-proportional)
- XGB per-fold scores track LGBM almost perfectly (0.0005 gap per fold)
- Both are gradient boosting over the same 27-feature set — correlation is structural

**Pattern diagnosis:** 5 iterations in, total CV gain is only +0.0013 (0.9486 → 0.9499). We're at a ceiling for the current feature set + GBM family. The incremental improvements are below the fold-std noise floor (0.0009). To make a real jump we need either:
1. A fundamentally different algorithm (CatBoost with native categoricals handles interactions GBM misses)
2. New features that don't exist in the current 15-column raw set (race-group rolling stats)

## Top Error Buckets
- Balanced accuracy now 0.8400, still 6pp below plain accuracy — recall gap on pit events persists
- Both GBM implementations find similar decision boundaries — blend adds no diversity

## 3 Concrete Next Ideas

1. **CatBoost with balanced class weights**: CatBoost handles raw string categoricals natively (no target encoding needed), uses symmetric trees + ordered boosting that captures different patterns. Combined with `class_weights: balanced` to address recall gap. Expected: +0.002–0.005 AUC.
2. **Race-group rolling features**: New feature block computing per-driver rolling lap time trend (last 3 laps), stint pace degradation rate, and laps-remaining estimate. Requires groupby within the fold — needs a new `f1_race_rolling` feature block. Expected: +0.003–0.008 AUC.
3. **Multi-seed LGBM (seeds 42, 123, 456)**: Use `model: [lgbm, lgbm_rf]` with different seeds via per-model params. Lower risk than new algorithms. Expected: +0.0005–0.001 AUC (pure variance reduction).
