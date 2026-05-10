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
    """F1-specific: tyre age polynomial and log features."""
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
    """F1-specific: log-transform all gap columns and add abs versions."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    gap_cols = [c for c in X_tr.columns if "gap" in c.lower()]
    for col in gap_cols:
        for X in (X_tr, X_te):
            v = X[col].astype(float).fillna(0.0)
            X[f"{col}_abs"] = v.abs()
            X[f"{col}_log1p"] = np.log1p(v.abs())
    return X_tr, X_te


def f1_stint_features(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F1-specific: stint fraction and polynomial features."""
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
    """F1-specific: position inverse and lap-progress fraction."""
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


def f1_race_rolling(
    X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold rolling lap-time features within each (Race, Driver) group.

    Training rows: sorted by LapNumber, shift(1).rolling(3) to avoid leakage.
    Val/test rows: per-(Race, Driver) training aggregates as a proxy.
    Adds: rolling_lt3, laptime_d1 (lap-to-lap delta), lt_vs_dr_mean.
    No-op if LapTime / LapNumber / Driver / Race columns are absent."""
    X_tr, X_te = X_tr.copy(), X_te.copy()

    lap_col = next(
        (c for c in X_tr.columns if c.lower() in ("lapnumber", "lap_number")), None
    )
    time_col = next(
        (c for c in X_tr.columns if "laptime" in c.lower() or c.lower() == "lap_time"), None
    )
    driver_col = next((c for c in X_tr.columns if c.lower() == "driver"), None)
    race_col = next((c for c in X_tr.columns if c.lower() == "race"), None)

    if not all([lap_col, time_col, driver_col, race_col]):
        return X_tr, X_te

    grp = [race_col, driver_col]

    # Training: sort by (Race, Driver, LapNumber), compute rolling/diff, restore order.
    tr = X_tr.reset_index(drop=True)
    order = tr.sort_values(grp + [lap_col]).index
    ts = tr.loc[order].copy()
    g = ts.groupby(grp, sort=False)
    ts["rolling_lt3"] = g[time_col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    ts["laptime_d1"] = g[time_col].transform("diff")
    ts_back = ts.iloc[np.argsort(np.argsort(order))]  # inverse permutation -> original order
    X_tr["rolling_lt3"] = ts_back["rolling_lt3"].values
    X_tr["laptime_d1"] = ts_back["laptime_d1"].values

    # Per-group training aggregates (proxy for val/test).
    agg = (
        X_tr.groupby(grp, observed=True)
        .agg(_lt_mean=(time_col, "mean"), _roll_proxy=("rolling_lt3", "mean"))
        .reset_index()
    )

    # LapTime deviation from per-(Race, Driver) mean -- training.
    X_tr = X_tr.merge(agg[grp + ["_lt_mean"]], on=grp, how="left")
    X_tr["lt_vs_dr_mean"] = X_tr[time_col] - X_tr["_lt_mean"]
    X_tr.drop(columns=["_lt_mean"], inplace=True)

    # Val/test: merge proxy rolling mean and compute deviation.
    X_te = X_te.merge(agg, on=grp, how="left")
    global_roll = float(X_tr["rolling_lt3"].median())
    global_lt = float(X_tr[time_col].median())
    X_te["rolling_lt3"] = X_te["_roll_proxy"].fillna(global_roll)
    X_te["laptime_d1"] = 0.0
    X_te["lt_vs_dr_mean"] = X_te[time_col] - X_te["_lt_mean"].fillna(global_lt)
    X_te.drop(columns=["_lt_mean", "_roll_proxy"], inplace=True, errors="ignore")

    # Fill NaN in training (first lap of each stint has no prior -> fill with median).
    for col in ("rolling_lt3", "laptime_d1", "lt_vs_dr_mean"):
        fill = float(X_tr[col].median())
        X_tr[col] = X_tr[col].fillna(fill)
        X_te[col] = X_te[col].fillna(fill)

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
    "f1_race_rolling": f1_race_rolling,
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
