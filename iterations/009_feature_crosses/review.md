# Review — Iteration 009: Feature Interaction Crosses

## CV Score
| Metric | Value | vs 005 (best) | vs 004 (LGBM solo) |
|---|---|---|---|
| CV AUC | **0.9492** | -0.0007 | -0.0005 |
| Balanced accuracy | 0.838 | -0.002 | same |
| n_features | 29 | +2 | +2 |
| Elapsed (s) | 165 | — | — |

## Analysis

Feature crosses added only 2 of the intended 4 (`tyre_x_gap`, `tyresq_x_gap`). Root cause: **`f1_stint_features` generates zero new columns** because `TyreLife` doesn’t match the stint detection patterns (`stint/lapsontyre/tyreage/tyre_age`). Without `stint_frac`, the `tyre_x_stintfrac` and `stintfrac_x_pos` crosses are silently skipped.

The 2 crosses that were created are highly correlated with each other (same TyreLife base, different polynomial degree) and LGBM already approximates tyre×gap interactions through sequential splits on each factor. No new signal was captured.

**Bug confirmed:** `f1_stint_features` has never produced `stint_frac`, `past_half_stint`, or `stint_sq` across all 9 iterations. These features were intended but silently missing. `TyreLife` must be added to the stint detection patterns.

## 3 Concrete Next Ideas

1. **Fix f1_stint_features + rerun (iter 010)**: Add `c.lower() in ("tyrelife", "tyre_life")` to stint_col detection. This unlocks `stint_frac` (fraction of expected compound pit window), `past_half_stint`, and `stint_sq` — features that were designed for this dataset but never created. Expected: +0.002–0.006 AUC.
2. **Deeper LGBM (num_leaves=255)**: Wider trees capture higher-order interactions on the existing feature set. Combine with `learning_rate=0.02` and `n_estimators=4000`. Expected: +0.001–0.002 AUC.
3. **Count encoding + target encoding of compound×driver interactions**: Cross-encode `Compound+Driver` as a combined categorical — captures that Hamilton on Softs behaves differently from Verstappen on Softs. Expected: +0.001–0.003 AUC.
