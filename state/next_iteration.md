# Next Iteration: 005_lgbm_xgb_blend

## What to change
Add XGBoost as a second model alongside LGBM. Same feature pipeline as 004 (F1 blocks + target encoding). The pipeline blends the two models' OOF predictions by optimizing weights on the full OOF set. Config uses per-model params dict.

## Why
Four iterations in, gains are ~+0.0003–0.0006 per experiment. Ensemble diversity is the most reliable next lever: LGBM and XGBoost find different local optima (different gradient computation, regularization, split finding). Blending them typically gives +0.001–0.003 AUC with zero feature-engineering risk.

## Expected delta in CV AUC
+0.001 to +0.003 (from 0.9497 → ~0.951–0.953).

## Risk
Low. Multi-model blending is already implemented and tested in train.py. XGBoost is already in FITTERS. The config schema supports `model: [lgbm, xgb]`. Only risk: XGB training adds ~3–5 min to elapsed time.
