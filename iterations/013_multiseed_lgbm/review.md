# Review — Iteration 013: 5-Seed LGBM Ensemble

## CV Score
| Metric | Value | vs 005 (best) | vs 012 |
|---|---|---|---|
| CV AUC | **0.9498** | -0.0001 | -0.0001 |
| Per-seed range | 0.9492–0.9493 | — | — |
| Balanced accuracy | 0.839 | ~same | ~same |
| n_features | 29 | +2 | same |
| Elapsed (s) | 852 | — | — |

Fold scores: 0.9501, 0.9481, 0.9494, 0.9489, 0.9498 (std 0.0007)

## Analysis

All 5 seeds (42, 123, 456, 789, 2024) produce essentially the same model — per-seed CV is 0.9492–0.9493 with near-identical blend weights (exactly 0.200 each). The blend gives 0.9498, **marginally below best (0.9499)**.

**Key finding: LightGBM has near-zero seed variance on this dataset.** The 5 seeds produce virtually identical AUC values (range of 0.0001). This is a strong signal that:
1. LGBM has converged to the same optimal tree structure regardless of random initialization
2. The model is **bias-limited, not variance-limited** — averaging more seeds cannot improve AUC when individual predictions are ~99.9% correlated

**Root cause of the plateau:** The Kaggle log confirms `mounts under /kaggle/input: ['competitions']` — no external dataset is mounted. All 14 raw feature columns from the competition data are already being used (fill_na_median keeps numerics; target_encode_binary picks up object cols and low-cardinality ints). Feature engineering has added derived signals on top, but the model has saturated the information available in 439K synthetic rows.

**To break past 0.9499:** Need more training data or fundamentally different information sources.

## 3 Concrete Next Ideas

1. **External dataset (iter 014)**: Mount `aadigupta1601/f1-strategy-dataset-pit-stop-prediction` via `extra_dataset` in config. The real F1 data that generated the synthetic competition rows may add orders of magnitude more training samples with the same signal structure. Expected: +0.003–0.015 AUC.
2. **Race-based CV splits (iter 015)**: Switch from stratified random 5-fold to group-by-race folds. The current CV is optimistic (same races appear in both train and val). Race-based CV gives an honest generalization estimate and may reveal that the current model already generalizes well — or poorly.
3. **Larger training depth sweep (iter 016)**: num_leaves=511, learning_rate=0.005, n_estimators=10000. On 439K rows with 29 features, extremely deep trees with very low LR may find interactions the current depth misses. Expected: +0.001–0.003 AUC.
