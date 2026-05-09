"""Feature-engineering blocks addressable by name from config.yaml.

Each block takes (X_train, X_test) and returns (X_train, X_test) with new/modified
columns. Some blocks also accept y_tr for supervised encoders (target encoding).
Blocks must be pure and deterministic. Register blocks in BLOCKS at bottom.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def label_encode(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col in _categorical_cols(X_tr):
        combined = pd.concat([X_tr[col], X_te[col]], axis=0).astype("category")
        codes = combined.cat.codes
        X_tr[col] = codes.iloc[: len(X_tr)].values
        X_te[col] = codes.iloc[len(X_tr) :].values
    return X_tr, X_te


def fill_na_median(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col in _numeric_cols(X_tr):
        med = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(med)
        X_te[col] = X_te[col].fillna(med)
    return X_tr, X_te


def count_encode_categoricals(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col in _categorical_cols(X_tr):
        counts = pd.concat([X_tr[col], X_te[col]]).value_counts()
        X_tr[f"{col}_count"] = X_tr[col].map(counts).fillna(0).astype(int)
        X_te[f"{col}_count"] = X_te[col].map(counts).fillna(0).astype(int)
    return X_tr, X_te


def target_encode_binary(
    X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr=None, smoothing: int = 10, max_nunique: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smoothed mean target encoding for binary classification.

    For each categorical (and low-cardinality integer) column, adds a
    smoothed P(y=1|category) column. Must run per-fold to avoid leakage."""
    if y_tr is None:
        raise ValueError("target_encode_binary requires y_tr")
    X_tr, X_te = X_tr.copy(), X_te.copy()
    y = pd.Series(y_tr).reset_index(drop=True).astype(float)
    global_mean = float(y.mean())

    cat_like = set(_categorical_cols(X_tr))
    for col in X_tr.columns:
        if col in cat_like:
            continue
        if pd.api.types.is_integer_dtype(X_tr[col].dtype):
            try:
                if X_tr[col].nunique(dropna=True) <= max_nunique:
                    cat_like.add(col)
            except Exception:
                pass

    for col in cat_like:
        src_tr = pd.Series(X_tr[col].values).astype(object)
        src_te = pd.Series(X_te[col].values).astype(object)
        temp = pd.DataFrame({"cat": src_tr, "t": y.values})
        agg = temp.groupby("cat")["t"].agg(["mean", "count"])
        smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
        smooth_dict = smooth.to_dict()
        X_tr[f"{col}_te"] = np.asarray(src_tr.map(smooth_dict).astype(float).fillna(global_mean))
        X_te[f"{col}_te"] = np.asarray(src_te.map(smooth_dict).astype(float).fillna(global_mean))
    # Drop original object columns — LightGBM cannot accept object dtype.
    # Low-cardinality integer columns are kept (already numeric).
    obj_encoded = [c for c in cat_like if X_tr[c].dtype == object]
    X_tr = X_tr.drop(columns=obj_encoded)
    X_te = X_te.drop(columns=obj_encoded)
    return X_tr, X_te


def f1_tyre_features(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F1-specific: tyre age polynomial and log features.

    Adds tyre_life_sq, tyre_life_log1p, and tyre_life_pct_lap.
    Column detection is heuristic — checks for 'tyre'+'life'/'age'/'lap' substrings.
    No-op if the expected columns are absent."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    tyre_col = next(
        (c for c in X_tr.columns if "tyre" in c.lower() and any(k in c.lower() for k in ("life", "age", "lap"))),
        None,
    )
    lap_col = next(
        (c for c in X_tr.columns if c.lower() in ("lapnumber", "lap_number", "lap")),
        None,
    )
    for X in (X_tr, X_te):
        if tyre_col:
            v = X[tyre_col].astype(float).fillna(0.0)
            X["tyre_life_sq"] = v ** 2
            X["tyre_life_log1p"] = np.log1p(v)
        if tyre_col and lap_col:
            lap = X[lap_col].astype(float).fillna(1.0).replace(0, 1)
            X["tyre_life_pct_lap"] = X[tyre_col].astype(float).fillna(0.0) / lap
    return X_tr, X_te


def f1_gap_features(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F1-specific: log-transform all gap columns and add abs versions.

    Identifies columns with 'gap' in the name and adds {col}_log1p and {col}_abs."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    gap_cols = [c for c in X_tr.columns if "gap" in c.lower()]
    for col in gap_cols:
        for X in (X_tr, X_te):
            v = X[col].astype(float).fillna(0.0)
            X[f"{col}_abs"] = v.abs()
            X[f"{col}_log1p"] = np.log1p(v.abs())
    return X_tr, X_te


def f1_stint_features(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F1-specific: stint-based features.

    Detects StintLength / LapsOnTyre / TyreAge-style column and creates:
    - stint_remaining_est: rough estimate based on compound median life
    - past_half_stint: binary flag

    No-op if relevant columns are absent."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    stint_col = next(
        (c for c in X_tr.columns if any(k in c.lower() for k in ("stint", "lapsontyre", "tyreage", "tyre_age"))),
        None,
    )
    compound_col = next((c for c in X_tr.columns if "compound" in c.lower()), None)
    if stint_col is None:
        return X_tr, X_te
    for X in (X_tr, X_te):
        v = X[stint_col].astype(float).fillna(0.0)
        if compound_col:
            medians = X_tr.groupby(compound_col)[stint_col].median()
            expected = X[compound_col].map(medians).fillna(v.median())
            X["past_half_stint"] = (v > expected * 0.5).astype(int)
            X["stint_frac"] = v / expected.replace(0, 1)
        X["stint_sq"] = v ** 2
    return X_tr, X_te


def f1_position_features(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F1-specific: position and lap-progress features.

    Adds position_inv (1/position), lap_pct (lap/total_laps if both present)."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    pos_col = next((c for c in X_tr.columns if c.lower() in ("position", "raceposition", "race_position")), None)
    lap_col = next((c for c in X_tr.columns if c.lower() in ("lapnumber", "lap_number", "lap")), None)
    total_col = next((c for c in X_tr.columns if c.lower() in ("totallaps", "total_laps", "racelaps")), None)
    for X in (X_tr, X_te):
        if pos_col:
            p = X[pos_col].astype(float).fillna(10.0).replace(0, 1)
            X["position_inv"] = 1.0 / p
        if lap_col and total_col:
            lap = X[lap_col].astype(float).fillna(0.0)
            total = X[total_col].astype(float).fillna(1.0).replace(0, 1)
            X["lap_pct"] = lap / total
    return X_tr, X_te


BLOCKS = {
    "label_encode": label_encode,
    "fill_na_median": fill_na_median,
    "count_encode_categoricals": count_encode_categoricals,
    "target_encode_binary": target_encode_binary,
    "f1_tyre_features": f1_tyre_features,
    "f1_gap_features": f1_gap_features,
    "f1_stint_features": f1_stint_features,
    "f1_position_features": f1_position_features,
}


def apply_blocks(X_tr: pd.DataFrame, X_te: pd.DataFrame, names: list[str], y_tr=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name in names:
        if name not in BLOCKS:
            raise KeyError(f"unknown feature block: {name!r}. known: {sorted(BLOCKS)}")
        fn = BLOCKS[name]
        if "y_tr" in inspect.signature(fn).parameters:
            X_tr, X_te = fn(X_tr, X_te, y_tr=y_tr)
        else:
            X_tr, X_te = fn(X_tr, X_te)
    return X_tr, X_te
