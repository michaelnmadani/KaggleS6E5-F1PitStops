# Review — Iteration 014: External Dataset (CRASHED)

## CV Score
| Metric | Value |
|---|---|
| CV AUC | CRASHED — no metrics |
| Crash type | FileNotFoundError + dataset not mounted |

## Root Causes

**Two independent bugs prevented the kernel from running:**

1. **Missing `slug` field in config**: `scripts/build_notebook.py` only adds the external dataset to `kernel-metadata.json` (which tells Kaggle to mount it) when the config has `slug` in `extra_dataset`. Without `slug`, the dataset was never mounted, so `/kaggle/input` only showed `['competitions']`.

2. **Wrong filename + no fallback in `_find_extra_file`**: The config used `file: pit_stop_data.csv` which was a guess. Even if the dataset had been mounted, the filename wouldn't have been found with no auto-discovery fallback to list what was actually available.

## Error Message
```
FileNotFoundError: extra_dataset file 'pit_stop_data.csv' not found under /kaggle/input;
tried mount_dir=/kaggle/input/f1-strategy-dataset-pit-stop-prediction
mounts under /kaggle/input: ['competitions']
```

## Fixes Applied for Iter 015

1. Added `slug: aadigupta1601/f1-strategy-dataset-pit-stop-prediction` to config so `build_notebook.py` adds dataset to `kernel-metadata.json`.
2. Fixed `pipeline/src/data.py _find_extra_file` to auto-discover any CSV in mount_dir if the given filename is not found, and lists all available CSVs in the error message.

## Next Step

Iter 015: same config as 014 with slug field added + data.py fix → first actual run with external F1 data. Expected +0.003–0.015 AUC over plateau (0.9499).
