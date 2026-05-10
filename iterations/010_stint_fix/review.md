# Review — Iteration 010: Stint Fix + Deeper LGBM

## CV Score
| Metric | Value | vs 005 (best) | vs 009 |
|---|---|---|---|
| CV AUC | **0.9495** | -0.0004 | +0.0003 |
| Balanced accuracy | 0.839 | ~same | ~same |
| n_features | 29 | +2 | same |
| Elapsed (s) | 227 | — | — |

Fold scores: 0.9502, 0.9483, 0.9498, 0.9491, 0.9502 (std 0.0007)

## Analysis

**Stint features now active for the first time** (iters 001–009 produced zero stint features due to TyreLife detection bug). With the fix, `stint_frac`, `past_half_stint`, and `stint_sq` were generated and all 4 feature crosses (`tyre_x_gap`, `tyre_x_stintfrac`, `stintfrac_x_pos`, `tyresq_x_gap`) were created.

Result: +0.0003 vs iter 009 — marginal improvement. Still -0.0004 below best (iter 005).

**Root causes for limited gain:**
1. `stint_frac = TyreLife / median_TyreLife_per_compound` is highly correlated with raw `TyreLife`. LGBM already captures the non-linear tyre degradation signal via `tyre_life_sq` and `tyre_life_log1p`. Relative stint fraction adds little incremental information.
2. `past_half_stint` (binary flag) is a threshold on TyreLife — LGBM tree splits approximate this already.
3. Deeper trees (`num_leaves=255`) with `learning_rate=0.03` / `n_estimators=3000` didn't help over `num_leaves=127` / `learning_rate=0.05` / `n_estimators=2000`. The feature set may be the bottleneck, not tree capacity.

**Key observation:** All iterations cluster at 0.9479–0.9499. The feature set encodes tyre age, gap, position, and per-category averages — but not driver×compound interaction patterns (e.g., Hamilton on Softs has a different pit window than Verstappen on Softs).

## 3 Concrete Next Ideas

1. **Compound×Driver interaction target encoding (iter 011)**: Create a combined `Compound_Driver` categorical column and target-encode it. This captures per-driver-per-compound pit strategy patterns that independent column encodings miss. Expected: +0.002–0.005 AUC.
2. **Optuna HPO on current feature set (iter 012)**: 30-trial Optuna search over `num_leaves` [63–511], `learning_rate` [0.01–0.1], `subsample` [0.6–1.0], `min_child_samples` [10–50]. The plateau may be a local optimum in hparam space. Expected: +0.001–0.003 AUC.
3. **LGBM + XGB blend with fixed stint features (iter 013)**: Repeat iter 005's winning blend strategy but now with the corrected stint features and all 4 crosses active. Blend diversity between LGBM and XGB was +0.0002 in iter 005; with better features may compound. Expected: +0.001–0.003 AUC.
