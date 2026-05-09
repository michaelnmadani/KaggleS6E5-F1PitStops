# Ideas Backlog — S6E5 F1 Pit Stops

Maintained by reviewer sessions. Strategist picks one idea per session and moves it to `state/next_iteration.md`.

---

## Seeded on day 1

- [ ] **iter 001** — LightGBM baseline: fill_na_median + label_encode, no feature engineering. Establishes working pipeline.
- [ ] **iter 002** — Add `f1_tyre_features` (tyre_life_sq, tyre_life_log1p, tyre_life_pct_lap). Tyre age is expected to be the strongest predictor; polynomial/log features should lift AUC.
- [ ] **iter 003** — Add `f1_gap_features` + `f1_position_features`. Gap to next car is a key undercut/overcut signal. Position inverse captures pressure to stop differently for leaders vs backmarkers.
- [ ] **iter 004** — Add `f1_stint_features`. Stint fraction and past-half-stint binary flag encode pit-window urgency.
- [ ] **iter 005** — Add `target_encode_binary` per-fold on categorical columns (Driver, Team, Circuit, TyreCompound). Supervised encoding should capture team-specific pit strategy patterns.
- [ ] **iter 006** — Try XGBoost instead of LGBM (multi:softprob → binary:logistic). XGB often matches LGBM on tabular; ensemble diversity.
- [ ] **iter 007** — CatBoost native categorical handling. CatBoost can use raw string categoricals without label encoding; might capture interaction patterns LGBM misses.
- [ ] **iter 008** — Group-aware CV: split by Race (or Driver+Race) instead of random stratified. Prevents same race appearing in both train and val; more honest AUC estimate.
- [ ] **iter 009** — Incorporate external F1 Strategy Dataset (aadigupta1601/f1-strategy-dataset-pit-stop-prediction). May add real race data that the synthetic generator didn't capture perfectly.
- [ ] **iter 010** — LGBM + XGB blend (equal or score-proportional). Diversity from two GBM implementations.
- [ ] **iter 011** — Safety car / virtual safety car feature: binary flag for track status on that lap. Pit decisions cluster heavily around safety car periods.
- [ ] **iter 012** — Lap time delta features: current lap time vs driver's median lap time (on same tyre), plus tyre degradation rate estimate (lap_time_slope over last 3 laps).
- [ ] **iter 013** — Multi-seed LGBM ensemble (seeds 42, 43, 44). Variance reduction without new features.
- [ ] **iter 014** — Deeper hyperparameter search: increase num_leaves to 127, reduce learning_rate to 0.02, increase num_boost_round to 5000.
- [ ] **iter 015** — TabPFN bagged subsample ensemble (tabpfn_bagged). Orthogonal decision boundary; useful for small effective-n subproblems like specific circuit strategies.

## From review of iter 005_lgbm_xgb_blend (CV AUC 0.9499)

- [ ] **idea-r005-a** — CatBoost with native categoricals + balanced class weights. Two changes: different algorithm family (ordered boosting), and class imbalance fix. Expected +0.002–0.005 AUC.
- [ ] **idea-r005-b** — Race-group rolling features: per-driver lap-time trend (last 3 laps), stint degradation rate. New `f1_race_rolling` block with groupby. Expected +0.003–0.008 AUC.
- [ ] **idea-r005-c** — Multi-seed LGBM (seeds 42, 123, 456) using `model: [lgbm, lgbm_rf]` with different seed params. Pure variance reduction. Expected +0.0005–0.001 AUC.

## From review of iter 004_target_encode_fix (CV AUC 0.9497)

- [ ] **idea-r004-a** — LGBM + XGBoost blend. Ensemble diversity from two GBM implementations; config already supports `model: [lgbm, xgb]`. Expected +0.001–0.003 AUC.
- [ ] **idea-r004-b** — Class-weighted training (`class_weights: balanced`). Balanced accuracy still 6.1pp below plain accuracy; oversampling minority class may recover recall. Expected +0.0005–0.002 on AUC.
- [ ] **idea-r004-c** — Submit iter 004 to Kaggle LB to calibrate CV-vs-LB gap and understand real distance to top 5%.

## From review of iter 002_f1_features (CV AUC 0.9491)

- [ ] **idea-r002-a** — Per-fold target encoding of Driver/Team/Circuit/TyreCompound. F1 feature blocks gave only +0.0005; label-encoded categoricals are the main bottleneck. Smoothed P(pit|category) directly models team strategy fingerprints. Expected +0.003–0.006 AUC.
- [ ] **idea-r002-b** — Tune LGBM: num_leaves=127, learning_rate=0.02, num_boost_round=5000. Large n_train (439k) can support deeper trees; model may be underfit. Expected +0.002–0.004 AUC.
- [ ] **idea-r002-c** — Blend LGBM + XGBoost 50/50. Ensemble diversity from two GBM implementations. Expected +0.001–0.003 AUC.

## From review of iter 001_baseline (CV AUC 0.9486)

- [ ] **idea-r001-a** — Add all 4 F1 feature blocks at once (tyre+gap+stint+position). Pre-built, zero risk, expected +0.004–0.008 AUC. Addresses tyre degradation cliff and undercut gap signal.
- [ ] **idea-r001-b** — Per-fold target encoding of Driver/Team/Circuit/TyreCompound. Direct smoothed P(pit|category) encoding captures team strategy fingerprints without CV leakage.
- [ ] **idea-r001-c** — Race-group cross-validation: split folds by RaceId. Current random stratified CV is optimistic (same race in train+val). Grouped CV gives honest generalization estimate.

## Rules

- Append new ideas at the bottom with `- [ ]`.
- Mark completed ideas with `- [x]` after the reviewer writes a review.md.
- Do not delete ideas; keep them for audit.
