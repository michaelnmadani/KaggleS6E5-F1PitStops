# Review — Iteration 012: LGBM + XGB Blend v2 (Full Feature Set)

## CV Score
| Metric | Value | vs 005 (best) | vs 010 (best LGBM solo) |
|---|---|---|---|
| CV AUC | **0.9499** | tied | +0.0004 |
| LGBM solo | 0.9496 | -0.0003 | +0.0001 |
| XGB solo | 0.9492 | -0.0007 | -0.0003 |
| Balanced accuracy | 0.840 | ~same | ~same |
| n_features | 29 | +2 | same |
| Elapsed (s) | 567 | — | — |

Fold scores: 0.9504, 0.9481, 0.9495, 0.9490, 0.9500 (std 0.0008)

## Analysis

Tied the all-time best (0.9499 from iter 005). The blend of LGBM (0.9496) + XGB (0.9492) with working stint features gives the same result as the iter 005 blend without stint features. Net conclusion: **the corrected feature set adds ~+0.0003 to LGBM solo, but XGB is weaker on this feature set and the blend still tops out at 0.9499**.

**Twelve iterations, 0.9479–0.9499 range.** The plateau is now definitively confirmed. Feature engineering within the existing column set is saturated:
- Tyre age: polynomial, log, percentage, stint fraction — all captured
- Gap: abs and log transforms — captured
- Interaction crosses: 4 pairwise products — captured
- Target encoding: individual columns + interaction combos tried
- Blending: LGBM+XGB, LGBM+CatBoost — minimal diversity gain

**To break past 0.9499** requires either:
1. New raw columns not yet utilized (safety car status, lap times, gap to car ahead/behind are mentioned in competition brief as available)
2. More training data (external dataset: `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`)
3. Variance reduction via multi-seed ensembling

## 3 Concrete Next Ideas

1. **Multi-seed LGBM ensemble (iter 013)**: Blend 5 LGBM models with seeds 42, 123, 456, 789, 2024 — same best feature set, same hyperparams. Pure variance reduction from averaging uncorrelated random forests of trees. Expected: +0.001–0.002 AUC.
2. **External dataset (iter 014)**: Add `aadigupta1601/f1-strategy-dataset-pit-stop-prediction` via `extra_dataset` in config. More training rows from real F1 data could unlock signal the synthetic generator smoothed out. Expected: +0.003–0.010 AUC.
3. **Safety car / lap time features (iter 015)**: The competition brief lists TrackStatus (SC/VSC flag) and LapTime as available features. If TrackStatus is in the data, it's near-perfect for pit stop prediction. Add explicit feature blocks for these unused columns.
