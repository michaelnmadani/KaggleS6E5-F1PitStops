# Review — Iteration 001: Baseline LGBM

## CV Score
| Metric | Value |
|---|---|
| CV AUC (mean) | **0.9486** |
| CV AUC (std) | 0.0007 |
| Plain accuracy | 0.8990 |
| Balanced accuracy | 0.8353 |
| Elapsed (s) | 232 |
| n_features | 15 |

Fold scores: 0.9495, 0.9479, 0.9488, 0.9477, 0.9492 — extremely stable (std=0.0007).

## CV vs LB Gap
LB not yet checked. Will update after submit.yml dispatch.  
Estimate: expect LB within ±0.003 of CV given low fold std.

## Top Error Buckets (OOF analysis)

1. **Class imbalance recall gap**: Balanced accuracy (0.8353) is 6.4pp below plain accuracy (0.8990). The model undershoots recall on the positive class (pit-next-lap=1). Pit stops are rare (~10-15% of laps), so the model defaults toward predicting no-pit. Feature engineering that makes the pit window timing explicit will help.

2. **No tyre degradation signal**: Only 15 features with label encoding. Tyre age / stint length — the primary strategic input to a pit decision — is present as a raw integer but has no polynomial/log transforms. The model is likely missing the cliff-edge degradation pattern (pace drops sharply at ~60-70% tyre life).

3. **Categorical strategy patterns unmodeled**: Driver, team, and circuit are label-encoded as integers. Team-specific pit timing (e.g., Red Bull vs Mercedes under/overcut tendencies) and circuit-specific stop windows are not captured as smoothed probabilities.

## 3 Concrete Next Ideas

1. **Add all F1 feature blocks**: `f1_tyre_features` + `f1_gap_features` + `f1_stint_features` + `f1_position_features`. These are pre-built, zero-risk additions that encode tyre degradation curves, undercut gaps, and stint urgency. Expected: +0.004–0.008 AUC.

2. **Target-encode categoricals per fold**: Replace label encoding of Driver/Team/Circuit/TyreCompound with smoothed P(pit|category). This directly models team strategy fingerprints without leakage. Expected: +0.003–0.006 AUC.

3. **Race-group CV**: Split folds by RaceId rather than random stratified rows. Same race appears in both train and val currently, making CV optimistic. A race-grouped split gives a more honest generalization estimate and helps diagnose overfitting.
