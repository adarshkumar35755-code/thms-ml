"""Tests for src/scoring.py: bands, baseline formula, metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as cfg
from src import scoring as S


def dist_frame(**overrides) -> pd.DataFrame:
    """Distance-feature frame: all zeros except the given columns (scalars or equal-length arrays)."""
    n = max((len(np.atleast_1d(v)) for v in overrides.values()), default=1)
    cols = {c: np.zeros(n) for c in S.DISTANCE_FEATURES}
    for name, value in overrides.items():
        cols[name] = np.broadcast_to(np.asarray(value, dtype=float), n).copy()
    return pd.DataFrame(cols)


# ------------------------------------------------------------------- bands
@pytest.mark.parametrize(
    "score, state",
    [
        (100.0, 0), (75.0001, 0),
        (75.0, 1), (60.0, 1), (50.0, 1),
        (49.9999, 2), (30.0, 2), (25.0, 2),
        (24.9999, 3), (0.0, 3),
    ],
)  # fmt: skip
def test_score_to_state_bands_and_edges(score, state):
    assert int(S.score_to_state(score)) == state


def test_score_to_state_accepts_arrays_and_series_and_returns_ints():
    scores = [100, 80, 75, 60, 49, 30, 24, 0]
    expected = [0, 0, 1, 1, 2, 2, 3, 3]
    for arr in (scores, np.array(scores, dtype=float), pd.Series(scores, index=list("abcdefgh"))):
        out = S.score_to_state(arr)
        assert out.dtype.kind == "i" and out.tolist() == expected


def test_score_to_state_nan_raises_and_bad_edges_raise():
    with pytest.raises(ValueError, match="NaN"):
        S.score_to_state([80.0, np.nan])
    for bad in [(50, 75, 25), (75, 75, 25), (75, 50), (75, 50, 25, 10)]:
        with pytest.raises(ValueError, match="descending"):
            S.score_to_state(50.0, edges=bad)


def test_custom_edges_are_used():
    assert S.score_to_state([90, 70, 10], edges=(80, 40, 20)).tolist() == [0, 1, 3]


def test_clip_score_and_state_names():
    assert S.clip_score([-5.0, 0.0, 50.0, 100.0, 104.2]).tolist() == [0.0, 0.0, 50.0, 100.0, 100.0]
    assert S.state_names() == ["Normal", "Caution", "Warning", "Critical"]


# ---------------------------------------------------------------- baseline
def test_baseline_config_is_consistent():
    knots = np.asarray(cfg.BASELINE_SEVERITY_KNOTS)
    assert (np.diff(knots[:, 0]) > 0).all()  # d strictly increasing
    assert (np.diff(knots[:, 1]) >= 0).all()  # severity never decreases
    assert knots[0].tolist() == [0.0, 0.0] and knots[-1].tolist() == [cfg.DIST_CLIP_MAX, 1.0]
    assert 0.0 < cfg.BASELINE_ONSET_D < cfg.BASELINE_D_WARNING < 1.0
    # derived from the five thresholds only
    per_feature = [
        (t.warning - t.caution) / (t.critical - t.caution) for t in cfg.THRESHOLDS.values()
    ]
    assert cfg.BASELINE_D_WARNING == pytest.approx(np.mean(per_feature))


def test_distance_features_are_the_five_expected():
    assert S.DISTANCE_FEATURES == ("d_h2", "d_oil_temp", "d_load", "d_vib", "d_abs_resid")


@pytest.mark.parametrize(
    "d, score",
    [
        (0.0, 100.0),
        (cfg.BASELINE_ONSET_D / 2, 87.5),
        (cfg.BASELINE_ONSET_D, 75.0),
        (cfg.BASELINE_D_WARNING, 50.0),
        (1.0, 25.0),
        (cfg.DIST_CLIP_MAX, 0.0),
        (3.0, 0.0),
    ],
)
def test_baseline_values_at_knots(d, score):
    assert S.baseline_score(dist_frame(d_h2=d))[0] == pytest.approx(score)


def test_baseline_is_non_increasing_and_bounded():
    d = np.linspace(0.0, cfg.DIST_CLIP_MAX, 400)
    s = S.baseline_score(dist_frame(d_oil_temp=d))
    assert (np.diff(s) <= 1e-12).all()
    assert s.min() >= 0.0 and s.max() <= 100.0


def test_baseline_is_driven_by_the_worst_distance_feature():
    f = dist_frame(d_h2=0.2, d_oil_temp=0.7, d_load=0.1, d_vib=0.0, d_abs_resid=0.3)
    expected = S.baseline_score(dist_frame(d_oil_temp=0.7))
    assert S.baseline_score(f)[0] == pytest.approx(expected[0])


def test_baseline_crossing_each_threshold_lands_in_the_right_band():
    eps = 1e-3
    d = [0.0, cfg.BASELINE_ONSET_D + eps, cfg.BASELINE_D_WARNING + eps, 1.0 + eps, cfg.DIST_CLIP_MAX]
    states = S.score_to_state(S.baseline_score(dist_frame(d_vib=d)))
    assert states.tolist() == [0, 1, 2, 3, 3]


def test_baseline_ignores_every_other_column_and_validates_input():
    f = dist_frame(d_h2=0.5)
    noisy = f.assign(h2_ppm=1e6, oil_temp_C=-50.0, health_score=3.0)
    assert S.baseline_score(noisy)[0] == S.baseline_score(f)[0]
    with pytest.raises(ValueError, match="Missing distance features"):
        S.baseline_score(f.drop(columns="d_vib"))
    with pytest.raises(ValueError, match="NaN"):
        S.baseline_score(f.assign(d_load=np.nan))


def test_max_distance():
    f = dist_frame(d_h2=[0.0, 0.4, 1.5], d_vib=[0.1, 0.2, 0.0])
    assert S.max_distance(f).tolist() == [0.1, 0.4, 1.5]


# ----------------------------------------------------------------- metrics
def test_evaluate_scores_perfect_prediction():
    y = np.array([95.0, 80.0, 60.0, 40.0, 10.0, 90.0])
    state = S.score_to_state(y)
    m = S.evaluate_scores(y, y, state)
    assert m == {"MAE": 0.0, "RMSE": 0.0, "R2": 1.0, "state_acc": 1.0, "macro_F1": 1.0}


def test_evaluate_scores_known_values():
    y = np.array([90.0, 70.0, 40.0, 10.0])
    p = np.array([80.0, 70.0, 40.0, 20.0])  # errors 10, 0, 0, 10
    m = S.evaluate_scores(y, p, np.array([0, 1, 2, 3]))
    assert m["MAE"] == pytest.approx(5.0)
    assert m["RMSE"] == pytest.approx(np.sqrt(50.0))
    assert m["R2"] == pytest.approx(1 - 200.0 / np.sum((y - y.mean()) ** 2))
    # predicted states: 80->0, 70->1, 40->2, 20->3 -> all four correct
    assert m["state_acc"] == 1.0
    wrong = S.evaluate_scores(y, np.array([70.0, 70.0, 40.0, 10.0]), np.array([0, 1, 2, 3]))
    assert wrong["state_acc"] == 0.75  # 70 -> Caution instead of Normal


def test_confusion_and_metrics_tables():
    y_state = np.array([0, 0, 1, 2, 3, 3])
    pred = np.array([90.0, 70.0, 60.0, 45.0, 10.0, 30.0])  # states 0,1,1,2,3,2
    cm = S.confusion_table(y_state, pred)
    assert cm.shape == (4, 4) and int(cm.to_numpy().sum()) == 6
    assert cm.loc["true Normal", "pred Caution"] == 1
    assert cm.loc["true Critical", "pred Warning"] == 1
    assert list(cm.index) == ["true Normal", "true Caution", "true Warning", "true Critical"]

    t = S.metrics_table({"a": {"MAE": 1.0, "RMSE": 2.0}, "b": {"MAE": 3.0, "RMSE": 4.0}})
    assert list(t.index) == ["a", "b"] and list(t.columns) == ["MAE", "RMSE"]
