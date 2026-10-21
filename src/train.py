"""Training pipeline: transparent baseline vs monotone-constrained LightGBM.

1. BASELINE  score = 100 * (1 - severity(max distance feature)); no learning
             (src/scoring.baseline_score).
2. MODEL     LightGBM regressor for health_score with the monotone constraints from
             src/features.py. Hyperparameters are tuned by GroupKFold on run_id over
             the TRAIN split only (random search ranked by CV RMSE). The best
             candidates are then refit on the full train split with early stopping
             on VAL, and the one with the lowest VAL RMSE is kept.
3. STATES    health_state is derived from the predicted score with the bands in
             config.STATE_BAND_EDGES (src/scoring.score_to_state).
4. ARTIFACTS joblib model + JSON (feature names, band edges, window, params, metrics)
             + the full search table as CSV.

The TEST split is never featurized, scored or otherwise used in this module:
`run_training` has no test argument and `main` discards the test frame right after
the run-level split. src/evaluate.py owns the test split.

Run from the project root:
    python -m src.train                  # full run: 20 candidates x 5 folds
    python -m src.train --n-trials 6     # quicker
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

import config as cfg
from src import data, scoring
from src import features as F

# --------------------------------------------------------------------------
# Training settings
# --------------------------------------------------------------------------
N_TRIALS = 20  # candidate hyperparameter sets (trial 0 is DEFAULT_PARAMS)
N_FOLDS = 5  # GroupKFold folds over the 50 train runs
TOP_K = 3  # best-by-CV candidates that are refit and compared on val
MAX_ROUNDS = 3000  # upper bound on boosting rounds (a 2000 cap truncated one slow candidate)
CV_PATIENCE = 50  # early-stopping patience inside the CV (held-out train runs)
FINAL_PATIENCE = 100  # early-stopping patience for the final fits (val split)

# Fixed for every candidate. scikit-learn-style names so that the same dict works
# for lgb.cv (as aliases) and for LGBMRegressor.
FIXED_PARAMS: dict[str, Any] = {
    "objective": "regression",  # L2 on the 0-100 score
    "subsample_freq": 1,  # row bagging every iteration (subsample < 1 turns it on)
    "monotone_constraints_method": "intermediate",
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}

# Trial 0 of the search: a sensible hand-picked configuration.
DEFAULT_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 200,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "reg_lambda": 1.0,
}


def sample_params(rng: np.random.Generator) -> dict[str, Any]:
    """One random draw from the search space. Large min_child_samples values guard
    against memorizing single runs (1,344 strongly autocorrelated rows each)."""
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(0.04), np.log(0.15)))),
        "num_leaves": int(rng.choice([7, 15, 31, 63])),
        "min_child_samples": int(rng.choice([50, 100, 200, 400, 800])),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(30.0)))),
    }


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """(X, score, state, run_id) for a frame of whole runs, via the shared feature code."""
    X = F.build_features_by_run(df)[F.get_feature_names()]
    data.assert_no_forbidden_columns(X)  # no label/meta column may reach the model
    return (
        X,
        df[cfg.HEALTH_SCORE].to_numpy(dtype="float64"),
        df[cfg.HEALTH_STATE].to_numpy(dtype="int64"),
        df[cfg.RUN_ID].to_numpy(),
    )


# --------------------------------------------------------------------------
# Cross-validation and fitting
# --------------------------------------------------------------------------
def make_group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold on run_id. Raises if any run lands on both sides of a fold."""
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if not 2 <= n_splits <= n_groups:
        raise ValueError(f"n_splits must be between 2 and the number of runs ({n_groups}), got {n_splits}")
    folds = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
    for fit_idx, held_idx in folds:
        shared = set(groups[fit_idx]) & set(groups[held_idx])
        if shared:
            raise data.SplitError(f"Run(s) {sorted(int(r) for r in shared)[:10]} are on both sides of a CV fold")
    return folds


def cross_validate(
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, float]:
    """Run-grouped CV for one candidate with lgb.cv. Early stopping picks the single
    iteration that minimizes the mean held-out RMSE across folds."""
    cv_params = {
        **FIXED_PARAMS,
        **params,
        "metric": "rmse",
        "seed": cfg.RANDOM_SEED,
        "monotone_constraints": F.get_monotone_constraints(list(X.columns)),
    }
    res = lgb.cv(
        cv_params,
        lgb.Dataset(X, label=y),
        num_boost_round=max_rounds,
        folds=folds,
        stratified=False,
        callbacks=[lgb.early_stopping(CV_PATIENCE, verbose=False)],
    )
    mean_key = next(k for k in res if k.endswith("rmse-mean"))
    std_key = next(k for k in res if k.endswith("rmse-stdv"))
    best = int(np.argmin(res[mean_key]))
    return {
        "cv_rmse": float(res[mean_key][best]),
        "cv_rmse_std": float(res[std_key][best]),
        "cv_best_iter": best + 1,
    }


def tune(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int = N_TRIALS,
    n_folds: int = N_FOLDS,
    max_rounds: int = MAX_ROUNDS,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Random search ranked by mean CV RMSE. Uses only the rows passed in (train split).

    Returns (candidates, results) where results has one row per trial, sorted best
    first. Trial ids are 1-based (trial 1 = DEFAULT_PARAMS), so the parameters of a
    row are `candidates[row.trial - 1]`.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    folds = make_group_folds(groups, n_folds)
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    candidates = [dict(DEFAULT_PARAMS)] + [sample_params(rng) for _ in range(n_trials - 1)]

    rows = []
    for i, params in enumerate(candidates):
        t0 = time.time()
        res = cross_validate(params, X, y, folds, max_rounds)
        rows.append({"trial": i + 1, **res, **params})
        if verbose:
            print(
                f"  trial {i + 1:2d}/{n_trials}  cv_rmse={res['cv_rmse']:.4f} (+/-{res['cv_rmse_std']:.4f})  "
                f"iters={res['cv_best_iter']:4d}  lr={params['learning_rate']:.3f}  leaves={params['num_leaves']:2d}  "
                f"min_leaf={params['min_child_samples']:3d}  {time.time() - t0:5.1f}s",
                flush=True,
            )
    results = pd.DataFrame(rows).sort_values("cv_rmse", kind="stable").reset_index(drop=True)
    return candidates, results


def fit_model(
    params: dict[str, Any],
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_stop: pd.DataFrame,
    y_stop: np.ndarray,
    max_rounds: int = MAX_ROUNDS,
    patience: int = FINAL_PATIENCE,
) -> lgb.Booster:
    """Fit one monotone-constrained booster; early-stop on (X_stop, y_stop)."""
    train_params = {
        **FIXED_PARAMS,
        **params,
        "metric": "rmse",
        "seed": cfg.RANDOM_SEED,
        "monotone_constraints": F.get_monotone_constraints(list(X_fit.columns)),
    }
    d_fit = lgb.Dataset(X_fit, label=y_fit)
    d_stop = lgb.Dataset(X_stop, label=y_stop, reference=d_fit)
    return lgb.train(
        train_params,
        d_fit,
        num_boost_round=max_rounds,
        valid_sets=[d_stop],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )


def predict_score(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Predicted health score clipped to [SCORE_MIN, SCORE_MAX].

    Columns are taken by name in training order; the best iteration is explicit.
    """
    raw = model.predict(X[F.get_feature_names()], num_iteration=model.best_iteration or None)
    return scoring.clip_score(raw)


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def band_edges_dict(edges: tuple[float, float, float] = cfg.STATE_BAND_EDGES) -> dict[str, float]:
    normal_above, caution_from, warning_from = edges
    return {"normal_above": normal_above, "caution_from": caution_from, "warning_from": warning_from}


# --------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------
def run_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    n_trials: int = N_TRIALS,
    n_folds: int = N_FOLDS,
    top_k: int = TOP_K,
    max_rounds: int = MAX_ROUNDS,
    save: bool = True,
    model_path: Path = cfg.MODEL_PATH,
    meta_path: Path = cfg.MODEL_META_PATH,
    cv_results_path: Path = cfg.CV_RESULTS_PATH,
    verbose: bool = True,
) -> dict[str, Any]:
    """Baseline + tuned model on train/val. There is deliberately no test argument."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    data.verify_no_run_overlap({"train": train_df, "val": val_df})

    X_tr, y_tr, _, groups = prepare(train_df)
    X_va, y_va, state_va, _ = prepare(val_df)

    # ---- 1. baseline (no learning) -------------------------------------------------
    baseline_va = scoring.baseline_score(X_va)

    # ---- 2. tune on train only (GroupKFold by run_id) ------------------------------
    if verbose:
        n_runs = len(np.unique(groups))
        print(f"\n=== Tuning: {n_trials} candidates x {n_folds}-fold GroupKFold over {n_runs} train runs ===")
    candidates, cv_results = tune(X_tr, y_tr, groups, n_trials, n_folds, max_rounds, verbose)

    # ---- 3. refit the top-k on the full train split, early-stop and select on val ---
    selection = []
    best = None
    for rank in range(min(top_k, len(cv_results))):
        row = cv_results.iloc[rank]
        params = candidates[int(row["trial"]) - 1]
        model = fit_model(params, X_tr, y_tr, X_va, y_va, max_rounds)
        val_rmse = float(np.sqrt(np.mean((predict_score(model, X_va) - y_va) ** 2)))
        selection.append(
            {
                "trial": int(row["trial"]),
                "cv_rmse": float(row["cv_rmse"]),
                "val_rmse": val_rmse,
                "best_iteration": int(model.best_iteration),
                "params": params,
            }
        )
        if best is None or val_rmse < best["val_rmse"]:
            best = {"val_rmse": val_rmse, "model": model, "params": params, "trial": int(row["trial"])}
    assert best is not None
    model, best_params = best["model"], best["params"]
    model_va = predict_score(model, X_va)

    # ---- 4. validation metrics (states derived through the bands) ------------------
    metrics = {
        "baseline": scoring.evaluate_scores(y_va, baseline_va, state_va),
        "model": scoring.evaluate_scores(y_va, model_va, state_va),
    }
    ceiling_state_acc = float((scoring.score_to_state(y_va) == state_va).mean())
    inside = scoring.max_distance(X_va) == 0.0

    if verbose:
        sel = pd.DataFrame(selection).drop(columns="params")
        sel["chosen"] = sel["trial"] == best["trial"]
        print("\n=== Top candidates refit on full train, early-stopped and ranked on VAL ===")
        print(sel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n=== Validation metrics ({len(val_df)} rows, {val_df[cfg.RUN_ID].nunique()} runs) ===")
        table = scoring.metrics_table({"baseline (formula)": metrics["baseline"], "LightGBM (monotone)": metrics["model"]})
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"state_acc ceiling (bands applied to the TRUE val score): {ceiling_state_acc:.4f}")
        base_err = np.abs(baseline_va - y_va)
        print(
            f"baseline MAE on rows inside every safe limit: {base_err[inside].mean():.3f} "
            f"({inside.mean():.1%} of rows, it predicts 100 but their true mean is {y_va[inside].mean():.1f}); "
            f"on the other rows: {base_err[~inside].mean():.3f}"
        )
        for name, pred in (("LightGBM", model_va), ("baseline", baseline_va)):
            print(f"\nConfusion matrix, {name} (val; rows = true state):")
            print(scoring.confusion_table(state_va, pred).to_string())

    # ---- 5. artifacts --------------------------------------------------------------
    if save:
        for path in (model_path, meta_path, cv_results_path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        cv_results.to_csv(cv_results_path, index=False)
        meta = {
            "model_file": Path(model_path).name,
            "model_type": "lightgbm.Booster",
            "feature_names": F.get_feature_names(),
            "monotone_decreasing": list(F.MONOTONE_DECREASING),
            "monotone_constraints": dict(zip(F.get_feature_names(), F.get_monotone_constraints())),
            "feature_window": cfg.FEATURE_WINDOW,
            "sample_period_h": cfg.SAMPLE_PERIOD_H,
            "score_range": [cfg.SCORE_MIN, cfg.SCORE_MAX],
            "band_edges": band_edges_dict(),
            "band_rule": "score > normal_above: 0 Normal; caution_from <= score <= normal_above: 1 Caution; "
            "warning_from <= score < caution_from: 2 Warning; score < warning_from: 3 Critical",
            "state_names": cfg.HEALTH_STATE_NAMES,
            "hyperparameters": {**FIXED_PARAMS, **best_params},
            "best_iteration": int(model.best_iteration),
            "random_seed": cfg.RANDOM_SEED,
            "tuning": {
                "n_trials": n_trials,
                "n_folds": n_folds,
                "top_k": top_k,
                "best_cv_rmse": float(cv_results.loc[0, "cv_rmse"]),
                "selection_on_val": selection,
            },
            "fit_data": {
                "train_rows": len(train_df),
                "train_runs": int(train_df[cfg.RUN_ID].nunique()),
                "val_rows": len(val_df),
                "val_runs": int(val_df[cfg.RUN_ID].nunique()),
            },
            "val_metrics": {**metrics, "state_acc_ceiling_true_score": ceiling_state_acc},
            "libraries": {
                "python": platform.python_version(),
                "lightgbm": lgb.__version__,
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        }
        Path(meta_path).write_text(json.dumps(_jsonable(meta), indent=2) + "\n", encoding="utf-8")
        if verbose:
            print(f"\nSaved model    -> {model_path}\nSaved metadata -> {meta_path}\nSaved CV table -> {cv_results_path}")

    return {
        "model": model,
        "best_params": best_params,
        "cv_results": cv_results,
        "selection": selection,
        "metrics": metrics,
        "X_train": X_tr,
        "X_val": X_va,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the THMS health-score model.")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="hyperparameter candidates (default %(default)s)")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="GroupKFold folds (default %(default)s)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="candidates compared on val (default %(default)s)")
    parser.add_argument("--data", type=Path, default=None, help="path to thms_dataset.csv (default: config.DATA_PATH)")
    args = parser.parse_args(argv)

    df = data.load_dataset(args.data)
    train_df, val_df, _held_out = data.split_by_run(df)  # also verifies no run is shared between splits
    del _held_out, df  # the test split stays untouched until the final evaluation (src/evaluate.py)
    data.print_class_balance({"train": train_df, "val": val_df})
    run_training(train_df, val_df, n_trials=args.n_trials, n_folds=args.folds, top_k=args.top_k)


if __name__ == "__main__":
    main()
