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

## Rules

- Append new ideas at the bottom with `- [ ]`.
- Mark completed ideas with `- [x]` after the reviewer writes a review.md.
- Do not delete ideas; keep them for audit.
