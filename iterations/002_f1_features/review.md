# Review — Iteration 002: F1 Feature Engineering

## CV Score
| Metric | Value | vs 001 |
|---|---|---|
| CV AUC (mean) | **0.9491** | +0.0005 |
| CV AUC (std) | 0.0008 | +0.0001 |
| Plain accuracy | 0.8993 | +0.0003 |
| Balanced accuracy | 0.8380 | +0.0027 |
| n_features | 22 | +7 |
| Elapsed (s) | 240 | +8 |

Fold scores: 0.9501, 0.9480, 0.9493, 0.9485, 0.9496 — still very stable.

## CV vs LB Gap
LB not yet checked. Still awaiting submission.

## Analysis

The F1 feature blocks added 7 features (15→22) but only lifted AUC by +0.0005 — well below the expected +0.004–0.008. Likely causes:

1. **LightGBM already captures non-linearity**: LGBM builds multiplicative splits internally, so tyre_life_sq and tyre_life_log1p add limited marginal signal over the raw tyre age column. Tree models benefit less from polynomial features than linear models.

2. **Heuristic column detection was partial**: Some feature blocks are no-ops if column names don't match expected substrings. The 7 added features suggest some blocks fired partially (e.g., position block may have found position but not total_laps, so lap_pct was skipped).

3. **Categorical encoding is still weak**: Driver, Team, Circuit, TyreCompound are label-encoded integers. These carry heavy strategic information (team pit windows, driver aggression) that a simple integer can't represent.

## Top Error Buckets
- Balanced accuracy improved by +0.0027 → slight recall improvement on pit events, but still 6.1pp gap vs plain accuracy
- Primary failure mode: still predicting no-pit for laps with rare but imminent pit signals (stint cliff, undercut window)

## 3 Concrete Next Ideas

1. **Per-fold target encoding of categoricals**: Replace label encoding of Driver/Team/Circuit/TyreCompound with smoothed P(pit|category) using `target_encode_binary`. This directly models team strategy fingerprints. Expected: +0.003–0.006 AUC.

2. **Tune LGBM hyperparameters**: Increase num_leaves to 127, reduce learning_rate to 0.02, boost num_boost_round to 5000. The model is likely underfit given large n_train (439k). Expected: +0.002–0.004 AUC.

3. **Add XGBoost model**: Run XGBoost with same feature set and blend 50/50 with LGBM predictions. Ensemble diversity from two GBM implementations. Expected: +0.001–0.003 AUC.
