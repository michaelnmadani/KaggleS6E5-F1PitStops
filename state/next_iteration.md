# Next Iteration: 003_target_encode

## What to change
Replace `label_encode` with `target_encode_binary` for categorical columns, keeping all F1 feature blocks. Also tune LGBM hyperparameters to exploit the larger tree capacity:
- num_leaves: 63 → 127
- learning_rate: 0.05 → 0.03
- num_boost_round: 2000 → 3000

Feature list: `[fill_na_median, f1_tyre_features, f1_gap_features, f1_stint_features, f1_position_features, target_encode_binary]`

Note: `label_encode` is removed because `target_encode_binary` handles categoricals via smoothed mean encoding. Numeric columns are still covered by `fill_na_median`.

## Why
Iter 002 showed that F1 geometric features gave only +0.0005 lift — LGBM already handles non-linearity internally. The remaining gap is in categorical representation. Driver/Team/Circuit/TyreCompound encoded as smoothed P(pit|category) captures team-specific pit window timing directly. This is per-fold (applied inside CV) so there is no target leakage.

Combining with mild LGBM tuning (deeper trees, slower LR) to exploit the 439k training rows, which can support num_leaves=127 without overfitting.

## Expected delta in CV AUC
+0.004 to +0.008 (from 0.9491 → ~0.953–0.957).
Target encoding is consistently the highest-impact categorical feature in GBM pit-stop prediction tasks.

## Risk
Low-medium. `target_encode_binary` is pre-built and per-fold safe. LGBM tuning is conservative (3000 rounds with early stopping at 100 still terminates early). Combined change makes attribution harder if score drops — but both changes are well-motivated and the config is easy to revert.
