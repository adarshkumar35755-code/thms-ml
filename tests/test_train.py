"""Tests for src/train.py: CV folds, search space, a small end-to-end run on real
runs (artifacts, monotonicity, determinism) and the train/val/test guards."""
from __future__ import annotations

import inspect
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

import config as cfg
from src import data, scoring
from src import features as F
from src import train as T

needs_data = pytest.mark.skipif(not cfg.DATA_PATH.exists(), reason="thms_dataset.csv not present")

SMOKE = dict(n_trials=2, n_folds=3, top_k=2, max_rounds=60, verbose=False)


# ------------------------------------------------------------ CV and search
def test_group_folds_never_split_a_run_and_cover_every_row():
    runs = np.repeat(np.arange(12), 30)
    folds = T.make_group_folds(runs, 4)
    assert len(folds) == 4
    held = np.concatenate([h for _, h in folds])
    assert sorted(held.tolist()) == list(range(len(runs)))  # each row held out exactly once
    for fit_idx, held_idx in folds:
        assert not set(runs[fit_idx]) & set(runs[held_idx])


def test_group_folds_validate_split_count():
    runs = np.repeat(np.arange(5), 10)
    for bad in (1, 6):
        with pytest.raises(ValueError, match="n_splits"):
            T.make_group_folds(runs, bad)


def test_sample_params_stay_in_bounds_and_are_reproducible():
    draws = [T.sample_params(np.random.default_rng(7)) for _ in range(2)]
    assert draws[0] == draws[1]
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = T.sample_params(rng)
        assert 0.04 <= p["learning_rate"] <= 0.15
        assert p["num_leaves"] in (7, 15, 31, 63)
        assert p["min_child_samples"] in (50, 100, 200, 400, 800)
        assert 0.6 <= p["colsample_bytree"] <= 1.0 and 0.6 <= p["subsample"] <= 1.0
        assert 1e-3 <= p["reg_lambda"] <= 30.0
    assert set(T.DEFAULT_PARAMS) == set(T.sample_params(rng))  # same keys, so any trial is comparable


def test_there_is_no_test_argument_anywhere_in_the_training_api():
    for fn in (T.run_training, T.tune, T.cross_validate, T.fit_model, T.prepare):
        assert not [p for p in inspect.signature(fn).parameters if "test" in p.lower()], fn.__name__


# ---------------------------------------------------------- end-to-end smoke
@pytest.fixture(scope="module")
def frames():
    df = data.load_dataset()
    train, val, _held_out = data.split_by_run(df)
    train_runs = sorted(train[cfg.RUN_ID].unique())[:12]
    val_runs = sorted(val[cfg.RUN_ID].unique())[:3]
    return (
        train[train[cfg.RUN_ID].isin(train_runs)].reset_index(drop=True),
        val[val[cfg.RUN_ID].isin(val_runs)].reset_index(drop=True),
    )


@pytest.fixture(scope="module")
def smoke(frames, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("artifacts")
    paths = dict(
        model_path=out_dir / "model.joblib", meta_path=out_dir / "meta.json", cv_results_path=out_dir / "cv.csv"
    )
    result = T.run_training(*frames, **SMOKE, **paths)
    return result, paths


@needs_data
def test_smoke_run_returns_a_booster_and_sane_outputs(smoke, frames):
    result, _ = smoke
    model, X_va = result["model"], result["X_val"]
    assert isinstance(model, lgb.Booster) and model.best_iteration >= 1
    assert list(result["X_train"].columns) == F.get_feature_names() == list(X_va.columns)
    data.assert_no_forbidden_columns(result["X_train"])
    pred = T.predict_score(model, X_va)
    assert len(pred) == len(frames[1]) and np.isfinite(pred).all()
    assert pred.min() >= cfg.SCORE_MIN and pred.max() <= cfg.SCORE_MAX
    assert set(result["metrics"]) == {"baseline", "model"}
    assert result["metrics"]["model"]["RMSE"] < result["metrics"]["baseline"]["RMSE"]


@needs_data
def test_search_table_is_ranked_and_selection_covers_top_k(smoke):
    result, _ = smoke
    cv = result["cv_results"]
    assert len(cv) == SMOKE["n_trials"]
    assert cv["cv_rmse"].is_monotonic_increasing
    assert len(result["selection"]) == SMOKE["top_k"]
    assert {s["trial"] for s in result["selection"]} == set(cv["trial"].iloc[: SMOKE["top_k"]])
    best_val = min(s["val_rmse"] for s in result["selection"])
    chosen = [s for s in result["selection"] if s["params"] == result["best_params"]]
    assert chosen and chosen[0]["val_rmse"] == pytest.approx(best_val)


@needs_data
def test_saved_artifacts_round_trip(smoke, frames):
    result, paths = smoke
    loaded = joblib.load(paths["model_path"])
    X_va = result["X_val"]
    np.testing.assert_array_equal(T.predict_score(loaded, X_va), T.predict_score(result["model"], X_va))

    meta = json.loads(paths["meta_path"].read_text(encoding="utf-8"))
    assert meta["feature_names"] == F.get_feature_names()
    assert meta["band_edges"] == {"normal_above": 75.0, "caution_from": 50.0, "warning_from": 25.0}
    assert tuple(meta["band_edges"].values()) == cfg.STATE_BAND_EDGES
    assert meta["feature_window"] == cfg.FEATURE_WINDOW
    assert meta["score_range"] == [cfg.SCORE_MIN, cfg.SCORE_MAX]
    assert list(meta["monotone_constraints"]) == F.get_feature_names()
    assert set(meta["monotone_constraints"].values()) == {-1, 0}
    assert {k for k, v in meta["monotone_constraints"].items() if v == -1} == set(F.MONOTONE_DECREASING)
    assert meta["best_iteration"] == result["model"].best_iteration
    assert meta["fit_data"]["train_rows"] == len(frames[0]) and meta["fit_data"]["val_rows"] == len(frames[1])
    assert "test" not in json.dumps(meta["fit_data"]).lower()
    assert set(meta["val_metrics"]) >= {"baseline", "model", "state_acc_ceiling_true_score"}

    cv = pd.read_csv(paths["cv_results_path"])
    assert len(cv) == SMOKE["n_trials"] and "cv_rmse" in cv.columns


@needs_data
@pytest.mark.parametrize("feature", F.MONOTONE_DECREASING)
def test_predictions_are_monotone_decreasing_in_every_constrained_feature(smoke, feature):
    """Sweep one constrained feature while holding all others fixed: the score must never rise."""
    result, _ = smoke
    model, X_tr, X_va = result["model"], result["X_train"], result["X_val"]
    base = X_va.dropna().sample(40, random_state=0)
    grid = np.linspace(X_tr[feature].min(), X_tr[feature].max(), 15)
    preds = []
    for value in grid:
        probe = base.copy()
        probe[feature] = value
        preds.append(model.predict(probe[F.get_feature_names()], num_iteration=model.best_iteration))
    assert (np.diff(np.vstack(preds), axis=0) <= 1e-9).all(), feature


@needs_data
def test_training_is_deterministic(smoke, frames):
    result, _ = smoke
    again = T.run_training(*frames, **SMOKE, save=False)
    np.testing.assert_allclose(
        T.predict_score(again["model"], again["X_val"]),
        T.predict_score(result["model"], result["X_val"]),
        rtol=0,
        atol=1e-9,
    )


# ------------------------------------------------------------------- guards
@needs_data
def test_overlapping_runs_between_train_and_val_raise(frames):
    train_df, val_df = frames
    leaked = pd.concat([val_df, train_df[train_df[cfg.RUN_ID] == train_df[cfg.RUN_ID].iloc[0]]], ignore_index=True)
    with pytest.raises(data.SplitError, match="train"):
        T.run_training(train_df, leaked, **SMOKE, save=False)


@needs_data
def test_prepare_rejects_leaky_feature_sets(frames, monkeypatch):
    X, y, state, groups = T.prepare(frames[0])
    assert list(X.columns) == F.get_feature_names()
    assert len(X) == len(y) == len(state) == len(groups) == len(frames[0])

    monkeypatch.setattr(F, "get_feature_names", lambda: F.FEATURE_NAMES[:-1] + ("health_score",))
    monkeypatch.setattr(F, "build_features_by_run", lambda df, window=cfg.FEATURE_WINDOW: df)  # label columns present
    with pytest.raises((data.LeakageError, KeyError)):
        T.prepare(frames[0])


def test_band_edges_dict_matches_config():
    assert tuple(T.band_edges_dict().values()) == cfg.STATE_BAND_EDGES
    assert scoring.score_to_state(75.0) == 1  # the JSON band_rule and the code agree on the edge
