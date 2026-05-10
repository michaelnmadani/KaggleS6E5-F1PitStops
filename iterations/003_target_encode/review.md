# Review — Iteration 003: Target Encode (CRASHED)

## CV Score
| Metric | Value |
|---|---|
| CV AUC | **CRASHED** |
| n_features | N/A |

## Analysis

Kernel crashed with `ValueError: Cannot convert object dtype to numeric` inside `target_encode_binary`. The block returned object-dtype columns (original categorical columns) alongside the new `_te` columns; LightGBM rejected the object dtypes.

**Fix applied in iter 004**: Drop original object columns after encoding (`X_tr = X_tr.drop(columns=obj_encoded)`). Iter 004 ran cleanly with CV AUC 0.9497.

## Next Ideas (superseded by iter 004)

1. Drop object columns after target encoding — implemented in 004.
2. Cast object columns to string before encoding for robustness.
3. Add fallback label encoding for any remaining non-numeric columns.
