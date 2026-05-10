"""Thin wrappers over gradient-boosting libraries with a uniform fit/predict API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    model: Any
    val_pred: np.ndarray
    test_pred: np.ndarray


def _lgbm_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    import lightgbm as lgb

    base = {
        "objective": "binary" if task == "binary" else ("multiclass" if task == "multiclass" else "regression"),
        "verbosity": -1,
        "seed": 42,
    }
    if task == "multiclass":
        base["num_class"] = int(pd.Series(y_tr).nunique())
    base.update(params)
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
    model = lgb.train(
        base,
        dtr,
        num_boost_round=base.pop("num_boost_round", 2000),
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(base.pop("early_stopping_rounds", 100)), lgb.log_evaluation(0)],
    )
    return FitResult(model=model, val_pred=model.predict(X_val), test_pred=model.predict(X_test))


def _xgb_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    import xgboost as xgb

    base = {
        "objective": "binary:logistic" if task == "binary" else ("multi:softprob" if task == "multiclass" else "reg:squarederror"),
        "verbosity": 0,
        "seed": 42,
    }
    if task == "multiclass":
        base["num_class"] = int(pd.Series(y_tr).nunique())
    base.update(params)
    num_boost_round = base.pop("num_boost_round", 2000)
    early_stop = base.pop("early_stopping_rounds", 100)
    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weight)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    try:
        model = xgb.train(
            base, dtr, num_boost_round=num_boost_round,
            evals=[(dval, "val")], early_stopping_rounds=early_stop, verbose_eval=False,
        )
    except TypeError:
        from xgboost.callback import EarlyStopping
        model = xgb.train(
            base, dtr, num_boost_round=num_boost_round,
            evals=[(dval, "val")], callbacks=[EarlyStopping(rounds=early_stop, save_best=True)],
            verbose_eval=False,
        )
    return FitResult(model=model, val_pred=model.predict(dval), test_pred=model.predict(dtest))


def _catboost_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    base = {"verbose": 0, "random_seed": 42, "iterations": 2000, "early_stopping_rounds": 100}
    base.update(params)
    cat_cols = [i for i, c in enumerate(X_tr.columns) if X_tr[c].dtype == object]
    kw = {"cat_features": cat_cols} if cat_cols else {}
    cls = CatBoostClassifier if task in ("binary", "multiclass") else CatBoostRegressor
    model = cls(**base)
    model.fit(
        Pool(X_tr, y_tr, weight=sample_weight, **kw),
        eval_set=Pool(X_val, y_val, **kw),
    )
    if task == "binary":
        val_pred = model.predict_proba(Pool(X_val, **kw))[:, 1]
        test_pred = model.predict_proba(Pool(X_test, **kw))[:, 1]
    elif task == "multiclass":
        val_pred = model.predict_proba(Pool(X_val, **kw))
        test_pred = model.predict_proba(Pool(X_test, **kw))
    else:
        val_pred = model.predict(Pool(X_val, **kw))
        test_pred = model.predict(Pool(X_test, **kw))
    return FitResult(model=model, val_pred=val_pred, test_pred=test_pred)


def _extra_trees_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    base = {
        "n_estimators": 800,
        "min_samples_leaf": 50,
        "max_features": 0.85,
        "n_jobs": -1,
        "random_state": 42,
    }
    base.update({k: v for k, v in params.items() if k in base or k in {"max_depth", "bootstrap", "criterion"}})
    X_tr_num = X_tr.select_dtypes(include=[np.number]).fillna(0)
    X_val_num = X_val.reindex(columns=X_tr_num.columns, fill_value=0).select_dtypes(include=[np.number]).fillna(0)
    X_test_num = X_test.reindex(columns=X_tr_num.columns, fill_value=0).select_dtypes(include=[np.number]).fillna(0)
    if task in ("binary", "multiclass"):
        model = ExtraTreesClassifier(**base)
        model.fit(X_tr_num, y_tr, sample_weight=sample_weight)
        if task == "binary":
            val_pred = model.predict_proba(X_val_num)[:, 1]
            test_pred = model.predict_proba(X_test_num)[:, 1]
        else:
            val_pred = model.predict_proba(X_val_num)
            test_pred = model.predict_proba(X_test_num)
    else:
        model = ExtraTreesRegressor(**base)
        model.fit(X_tr_num, y_tr, sample_weight=sample_weight)
        val_pred = model.predict(X_val_num)
        test_pred = model.predict(X_test_num)
    return FitResult(model=model, val_pred=val_pred, test_pred=test_pred)


def _logreg_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    from sklearn.linear_model import LogisticRegression

    def _prep(df):
        df = df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=float)
        return df

    base = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
        "n_jobs": -1,
    }
    if task == "multiclass":
        base["multi_class"] = "multinomial"
    base.update({k: v for k, v in params.items() if k in {"C", "penalty", "l1_ratio", "max_iter", "solver", "multi_class", "class_weight", "random_state"}})

    all_cols = sorted(set(_prep(X_tr).columns) | set(_prep(X_val).columns) | set(_prep(X_test).columns))

    def _align(df):
        d = _prep(df)
        for c in all_cols:
            if c not in d.columns:
                d[c] = 0.0
        return d[all_cols]

    X_tr_p = _align(X_tr)
    X_val_p = _align(X_val)
    X_test_p = _align(X_test)

    model = LogisticRegression(**base)
    model.fit(X_tr_p, y_tr, sample_weight=sample_weight)
    if task == "binary":
        val_pred = model.predict_proba(X_val_p)[:, 1]
        test_pred = model.predict_proba(X_test_p)[:, 1]
    elif task == "multiclass":
        val_pred = model.predict_proba(X_val_p)
        test_pred = model.predict_proba(X_test_p)
    else:
        val_pred = model.predict(X_val_p)
        test_pred = model.predict(X_test_p)
    return FitResult(model=model, val_pred=val_pred, test_pred=test_pred)


FITTERS = {
    "lgbm": _lgbm_fit,
    "lgbm_rf": _lgbm_fit,
    "lgbm_b": _lgbm_fit,
    "lgbm_c": _lgbm_fit,
    "lgbm_d": _lgbm_fit,
    "lgbm_e": _lgbm_fit,
    "xgb": _xgb_fit,
    "xgb_a": _xgb_fit,
    "xgb_b": _xgb_fit,
    "xgb_c": _xgb_fit,
    "catboost": _catboost_fit,
    "catboost_a": _catboost_fit,
    "catboost_b": _catboost_fit,
    "extra_trees": _extra_trees_fit,
    "logreg": _logreg_fit,
}


def fit_one_fold(name, X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    if name not in FITTERS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(FITTERS)}")
    return FITTERS[name](X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=sample_weight)


def compute_balanced_sample_weights(y_tr) -> np.ndarray:
    """sklearn-style 'balanced': w_i = N / (n_classes * count(y_i))."""
    y = np.asarray(y_tr)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    total = len(y)
    class_weight = {c: total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([class_weight[v] for v in y], dtype=float)
