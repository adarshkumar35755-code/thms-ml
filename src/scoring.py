"""Score <-> state logic, the transparent baseline, and metrics.

Shared by train.py, evaluate.py and infer.py so the bands, the baseline and the
metric definitions exist exactly once.

Baseline (no learning)
----------------------
    d     = max(d_h2, d_oil_temp, d_load, d_vib, d_abs_resid)   # worst quantity
    score = 100 * (1 - severity(d))                              # piecewise linear

The severity knots live in config.BASELINE_SEVERITY_KNOTS: 100 at d = 0, 75 just
past the Caution threshold, 50 at the average Warning position, 25 at the
Critical threshold and 0 at the clip limit. Every distance feature is exactly 0
at or below its Caution threshold, so the formula returns 100 for every row that
is inside all safe limits and cannot rank healthy rows against each other.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score

import config as cfg
from src.features import FEATURE_NAMES

__all__ = [
    "DISTANCE_FEATURES",
    "clip_score",
    "score_to_state",
    "state_names",
    "max_distance",
    "baseline_score",
    "evaluate_scores",
    "metrics_table",
    "confusion_table",
]

DISTANCE_FEATURES: tuple[str, ...] = tuple(n for n in FEATURE_NAMES if n.startswith("d_"))


def clip_score(score: np.ndarray | pd.Series | Sequence[float] | float) -> np.ndarray:
    """Clip scores to [SCORE_MIN, SCORE_MAX] (the regressor can overshoot slightly)."""
    return np.clip(np.asarray(score, dtype="float64"), cfg.SCORE_MIN, cfg.SCORE_MAX)


def _check_edges(edges: Sequence[float]) -> tuple[float, float, float]:
    e = tuple(float(x) for x in edges)
    if len(e) != 3 or not (e[0] > e[1] > e[2]):
        raise ValueError(f"band edges must be three strictly descending numbers, got {edges!r}")
    return e  # type: ignore[return-value]


def score_to_state(
    score: np.ndarray | pd.Series | Sequence[float] | float,
    edges: Sequence[float] = cfg.STATE_BAND_EDGES,
) -> np.ndarray:
    """Map health scores to states: 0 Normal, 1 Caution, 2 Warning, 3 Critical.

    score > e0 -> 0 | e1 <= score <= e0 -> 1 | e2 <= score < e1 -> 2 | score < e2 -> 3,
    with the default edges (75, 50, 25). NaN raises: a state is never guessed.
    """
    e_normal, e_caution, e_warning = _check_edges(edges)
    s = np.asarray(score, dtype="float64")
    if np.isnan(s).any():
        raise ValueError("score contains NaN; cannot derive a health state")
    state = np.select([s > e_normal, s >= e_caution, s >= e_warning], [0, 1, 2], default=3)
    return np.asarray(state, dtype="int64")


def state_names() -> list[str]:
    """State names ordered by state id."""
    return [cfg.HEALTH_STATE_NAMES[k] for k in sorted(cfg.HEALTH_STATE_NAMES)]


def max_distance(features: pd.DataFrame) -> np.ndarray:
    """Row-wise maximum of the five normalized distance-to-limit features."""
    missing = [c for c in DISTANCE_FEATURES if c not in features.columns]
    if missing:
        raise ValueError(f"Missing distance features: {missing}")
    d = features.loc[:, list(DISTANCE_FEATURES)].to_numpy(dtype="float64").max(axis=1)
    if np.isnan(d).any():
        raise ValueError("distance features contain NaN")
    return d


def baseline_score(features: pd.DataFrame) -> np.ndarray:
    """Formula health score (0-100) from the distance features of `build_features`."""
    knots = np.asarray(cfg.BASELINE_SEVERITY_KNOTS, dtype="float64")
    severity = np.interp(max_distance(features), knots[:, 0], knots[:, 1])  # flat beyond last knot
    return cfg.SCORE_MAX * (1.0 - severity)


def evaluate_scores(
    true_score: np.ndarray | pd.Series,
    pred_score: np.ndarray | pd.Series,
    true_state: np.ndarray | pd.Series,
    edges: Sequence[float] = cfg.STATE_BAND_EDGES,
) -> dict[str, float]:
    """MAE, RMSE, R2 on the score, plus accuracy and macro-F1 of the derived state.

    The predicted state is derived from `pred_score` with the band edges and
    compared with the dataset's `health_state` label.
    """
    y = np.asarray(true_score, dtype="float64")
    p = np.asarray(pred_score, dtype="float64")
    state_true = np.asarray(true_state, dtype="int64")
    state_pred = score_to_state(p, edges)
    labels = sorted(cfg.HEALTH_STATE_NAMES)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "state_acc": float((state_pred == state_true).mean()),
        "macro_F1": float(f1_score(state_true, state_pred, labels=labels, average="macro", zero_division=0)),
    }


def metrics_table(results: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """One row per method, one column per metric."""
    return pd.DataFrame(results).T


def confusion_table(
    true_state: np.ndarray | pd.Series,
    pred_score: np.ndarray | pd.Series,
    edges: Sequence[float] = cfg.STATE_BAND_EDGES,
) -> pd.DataFrame:
    """Confusion matrix (rows = true state, columns = state derived from the predicted score)."""
    labels = sorted(cfg.HEALTH_STATE_NAMES)
    cm = confusion_matrix(np.asarray(true_state, dtype="int64"), score_to_state(pred_score, edges), labels=labels)
    names = state_names()
    return pd.DataFrame(cm, index=[f"true {n}" for n in names], columns=[f"pred {n}" for n in names])
