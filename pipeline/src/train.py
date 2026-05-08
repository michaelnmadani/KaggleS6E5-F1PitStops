"""Orchestrator: reads an iteration config.yaml, runs CV, writes artifacts.

Designed to be called from a Kaggle notebook cell:

    from pipeline.src.train import run
    run(config_path="iterations/001_baseline/config.yaml",
        input_dir="/kaggle/input/<comp-slug>",
        output_dir="/kaggle/working")

Writes into output_dir:
    metrics.json   — CV score, per-fold scores
    oof.csv        — out-of-fold predictions (id, pred_prob)
    submission.csv — test predictions in competition format (id, PitNextLap)
    logs.txt       — human-readable run log
"""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, log_loss,
    mean_absolute_error, mean_squared_error, roc_auc_score,
)

from . import data as data_mod
from . import features as feat_mod
from . import models as models_mod
from . import postprocess as post_mod


def _hard_labels(p: np.ndarray) -> np.ndarray:
    return (p > 0.5).astype(int) if p.ndim == 1 else p.argmax(1)


METRICS = {
    "auc": lambda y, p: roc_auc_score(y, p if p.ndim == 1 else p[:, 1]),
    "logloss": lambda y, p: log_loss(y, p),
    "accuracy": lambda y, p: accuracy_score(y, _hard_labels(p)),
    "balanced_accuracy": lambda y, p: balanced_accuracy_score(y, _hard_labels(p)),
    "rmse": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
    "mae": lambda y, p: mean_absolute_error(y, p),
}


def _split_feature_blocks(names: list[str]) -> tuple[list[str], list[str]]:
    glob, per = [], []
    for n in names:
        if n not in feat_mod.BLOCKS:
            raise KeyError(f"unknown feature block: {n!r}")
        if "y_tr" in inspect.signature(feat_mod.BLOCKS[n]).parameters:
            per.append(n)
        else:
            glob.append(n)
    return glob, per


def run(config_path: str, input_dir: str, output_dir: str) -> dict:
    t0 = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = [f"config: {config_path}"]

    extra_cfg = cfg.get("extra_dataset")
    if extra_cfg and "slug" in extra_cfg and "mount_dir" not in extra_cfg:
        extra_cfg["mount_dir"] = f"/kaggle/input/{extra_cfg['slug'].split('/')[-1]}"
    X, y, X_test, test_ids, inverse_label_map, is_original = data_mod.load(
        Path(input_dir), cfg["target"], cfg["id_col"], extra_dataset=extra_cfg,
    )
    log_lines.append(f"train={X.shape}  test={X_test.shape}  external_rows={int(is_original.sum())}")

    feat_names = cfg.get("features", [])
    global_blocks, per_fold_blocks = _split_feature_blocks(feat_names)
    X, X_test = feat_mod.apply_blocks(X, X_test, global_blocks)
    log_lines.append(f"global features: {global_blocks} -> {X.shape[1]} cols")
    if per_fold_blocks:
        log_lines.append(f"per-fold features: {per_fold_blocks}")

    task = cfg["task"]
    metric_name = cfg["metric"]
    metric_fn = METRICS[metric_name]

    folds = data_mod.make_folds(y, cfg["cv"]["n_splits"], cfg["cv"]["seed"], cfg["cv"].get("stratified", True))

    n_classes = int(pd.Series(y).nunique())
    model_names = cfg["model"] if isinstance(cfg["model"], list) else [cfg["model"]]
    raw_params = cfg.get("params", {})
    is_per_model_params = (
        isinstance(raw_params, dict)
        and len(raw_params) > 0
        and all(k in models_mod.FITTERS for k in raw_params.keys())
    )

    def _params_for(m: str) -> dict:
        if is_per_model_params:
            return dict(raw_params.get(m, {}))
        return dict(raw_params)

    shape_oof = (len(X),) if task != "multiclass" else (len(X), n_classes)
    shape_test = (len(X_test),) if task != "multiclass" else (len(X_test), n_classes)
    per_model_oof = {m: np.zeros(shape_oof) for m in model_names}
    per_model_test = {m: np.zeros(shape_test) for m in model_names}
    per_model_fold_scores: dict[str, list[float]] = {m: [] for m in model_names}
    n_features_final = X.shape[1]

    class_weights_mode = cfg.get("class_weights")

    for f in range(cfg["cv"]["n_splits"]):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        X_tr, y_tr = X.iloc[tr_idx].reset_index(drop=True), y.iloc[tr_idx].reset_index(drop=True)
        X_va, y_va = X.iloc[va_idx].reset_index(drop=True), y.iloc[va_idx].reset_index(drop=True)
        X_te_fold = X_test.copy()

        for name in per_fold_blocks:
            fn = feat_mod.BLOCKS[name]
            val_test = pd.concat([X_va, X_te_fold], axis=0, ignore_index=True)
            X_tr, val_test = fn(X_tr, val_test, y_tr=y_tr)
            X_va = val_test.iloc[: len(X_va)].reset_index(drop=True)
            X_te_fold = val_test.iloc[len(X_va) :].reset_index(drop=True)

        n_features_final = X_tr.shape[1]
        sw = models_mod.compute_balanced_sample_weights(y_tr) if class_weights_mode == "balanced" else None

        extra_weight = (cfg.get("extra_dataset") or {}).get("weight")
        if extra_weight is not None and is_original.any():
            is_orig_tr = is_original[tr_idx]
            if sw is None:
                sw = np.ones(len(y_tr), dtype=float)
            sw = sw * np.where(is_orig_tr, float(extra_weight), 1.0)

        for m in model_names:
            res = models_mod.fit_one_fold(
                m, X_tr, y_tr, X_va, y_va, X_te_fold,
                _params_for(m), task, sample_weight=sw,
            )
            per_model_oof[m][va_idx] = res.val_pred
            per_model_test[m] += res.test_pred / cfg["cv"]["n_splits"]
            s = metric_fn(y_va, res.val_pred)
            per_model_fold_scores[m].append(float(s))
            log_lines.append(f"fold {f} {m}: {metric_name}={s:.5f}")

    per_model_cv = {m: float(metric_fn(y, per_model_oof[m])) for m in model_names}
    for m in model_names:
        log_lines.append(f"model {m}: CV {metric_name}={per_model_cv[m]:.5f}")

    if len(model_names) == 1:
        oof = per_model_oof[model_names[0]]
        test_preds = per_model_test[model_names[0]]
        blend_weights = {model_names[0]: 1.0}
    else:
        manual_w = cfg.get("blend_weights")
        if isinstance(manual_w, dict) and set(manual_w.keys()) == set(model_names):
            raw_w = np.array([float(manual_w[m]) for m in model_names])
            raw_w = np.maximum(raw_w, 0.0)
            weight_mode = "manual"
        else:
            raw_w = np.array([per_model_cv[m] for m in model_names])
            raw_w = np.maximum(raw_w, 1e-9)
            weight_mode = "score_proportional"
        w = raw_w / raw_w.sum()
        blend_weights = {m: float(wi) for m, wi in zip(model_names, w)}
        oof = sum(wi * per_model_oof[m] for m, wi in zip(model_names, w))
        test_preds = sum(wi * per_model_test[m] for m, wi in zip(model_names, w))
        log_lines.append(f"blend weights ({weight_mode}): {blend_weights}")

    fold_scores = [
        float(np.mean([per_model_fold_scores[m][f] for m in model_names]))
        for f in range(cfg["cv"]["n_splits"])
    ]
    cv_score = float(metric_fn(y, oof))

    pp_raw = cfg.get("postprocess")
    pp_stages = pp_raw if isinstance(pp_raw, list) else ([pp_raw] if pp_raw else [])
    postprocess_info = None

    if "logit_bias" in pp_stages and oof.ndim == 2:
        info = post_mod.tune_bias_nested_cv(oof, np.asarray(y), folds)
        if test_preds.ndim == 2:
            test_preds = post_mod.apply_bias_to_probs(test_preds, info["bias_all_folds"])
        oof_biased = post_mod.apply_bias_to_probs(oof, info["bias_all_folds"])
        post_cv = float(balanced_accuracy_score(np.asarray(y), oof_biased.argmax(axis=1)))
        postprocess_info = {**info, "post_bias_bal_acc_full": post_cv}
        log_lines.append(f"logit-bias: delta_nested={info['delta_nested']:+.5f}")
        oof = oof_biased
        cv_score = float(metric_fn(y, oof))

    if "class_weight_optuna" in pp_stages and oof.ndim == 2:
        cwo = post_mod.class_weight_optuna(oof, test_preds, np.asarray(y), metric_fn,
                                           n_trials=int(cfg.get("optuna_trials", 200)))
        log_lines.append(
            f"class_weight_optuna: delta={cwo['post_score'] - cwo['pre_score']:+.5f}"
        )
        if cwo["post_score"] > cwo["pre_score"]:
            oof = cwo["oof"]
            test_preds = cwo["test"]
            cv_score = float(metric_fn(y, oof))
            log_lines.append(f"class_weight_optuna applied")
        postprocess_info = cwo

    # OOF CSV
    if oof.ndim == 1:
        oof_df = pd.DataFrame({cfg["id_col"]: np.arange(len(oof)), "pred": oof})
    else:
        oof_df = pd.DataFrame(oof, columns=[f"p_class_{i}" for i in range(oof.shape[1])])
        oof_df.insert(0, cfg["id_col"], np.arange(len(oof)))
        oof_df["pred"] = oof.argmax(1)
    oof_df.to_csv(out / "oof.csv", index=False)

    # Submission
    hard_label_metrics = {"accuracy", "balanced_accuracy"}
    if test_preds.ndim == 1:
        test_labels = test_preds
    elif metric_name in hard_label_metrics or task == "multiclass":
        test_labels = test_preds.argmax(1)
    else:
        test_labels = test_preds
    if inverse_label_map is not None and np.ndim(test_labels) == 1:
        test_labels = np.array([inverse_label_map[int(v)] for v in test_labels])
    pd.DataFrame({cfg["id_col"]: test_ids, cfg["target"]: test_labels}).to_csv(out / "submission.csv", index=False)

    plain_acc = float(accuracy_score(y, _hard_labels(oof)))
    bal_acc = float(balanced_accuracy_score(y, _hard_labels(oof)))
    metrics = {
        "cv_score": cv_score,
        "metric": metric_name,
        "plain_accuracy": plain_acc,
        "balanced_accuracy": bal_acc,
        "fold_scores": fold_scores,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "elapsed_sec": round(time.time() - t0, 1),
        "model": cfg["model"],
        "per_model_cv": per_model_cv,
        "blend_weights": blend_weights,
        "postprocess": postprocess_info,
        "n_features": int(n_features_final),
        "n_train": int(len(X)),
        "n_test": int(len(X_test)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log_lines.append(f"CV {metric_name}={cv_score:.5f}  plain_acc={plain_acc:.5f}  balanced_acc={bal_acc:.5f}")
    (out / "logs.txt").write_text("\n".join(log_lines))
    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.config, args.input_dir, args.output_dir), indent=2))
