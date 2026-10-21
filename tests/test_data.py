"""Tests for src/data.py: validation, leakage guard, run-level splits, class balance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as cfg
from src import data

N_RUNS = 10  # runs 0-5 train, 6-7 val, 8-9 test


def make_frame() -> pd.DataFrame:
    """Small synthetic dataset that satisfies the full schema contract."""
    rng = np.random.default_rng(0)
    n = cfg.SAMPLES_PER_RUN
    total = N_RUNS * n
    run_id = np.repeat(np.arange(N_RUNS), n)
    idx = np.tile(np.arange(n), N_RUNS)
    t_hours = idx * cfg.SAMPLE_PERIOD_H
    split = np.array(["train"] * 6 + ["val"] * 2 + ["test"] * 2)[run_id]
    fault = np.array(cfg.KNOWN_FAULT_TYPES)[run_id % len(cfg.KNOWN_FAULT_TYPES)]
    state = (idx // 50) % 4
    names = np.array([cfg.HEALTH_STATE_NAMES[i] for i in range(4)])[state]

    def u(lo: float, hi: float) -> np.ndarray:
        return rng.uniform(lo, hi, total)

    return pd.DataFrame(
        {
            cfg.RUN_ID: run_id,
            cfg.SPLIT: split,
            cfg.T_HOURS: t_hours,
            cfg.HOUR_OF_DAY: t_hours % 24,
            cfg.INPUT_CURRENT_A: u(1, 10),
            cfg.OUTPUT_CURRENT_A: u(30, 280),
            cfg.OUTPUT_VOLTAGE_V: u(400, 430),
            cfg.AMBIENT_TEMP_C: u(15, 35),
            cfg.HUMIDITY_PCT: u(30, 80),
            cfg.VIBRATION_RMS: u(0.5, 1.5),
            cfg.H2_PPM: u(0, 90),
            cfg.OIL_TEMP_C: u(40, 70),
            cfg.FAULT_TYPE: fault,
            cfg.HEALTH_STATE: state,
            cfg.HEALTH_STATE_NAME: names,
            cfg.HEALTH_SCORE: 100 - 25 * state - rng.uniform(0, 5, total),
        }
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()


# ---------------------------------------------------------------- validation
def test_valid_frame_passes(frame):
    data.validate(frame)


def test_missing_column_raises(frame):
    with pytest.raises(data.DataValidationError, match="h2_ppm"):
        data.validate(frame.drop(columns=cfg.H2_PPM))


def test_extra_column_raises(frame):
    frame["surprise"] = 1.0
    with pytest.raises(data.DataValidationError, match="surprise"):
        data.validate(frame)


def test_nan_raises(frame):
    frame.loc[3, cfg.H2_PPM] = np.nan
    with pytest.raises(data.DataValidationError, match="NaN"):
        data.validate(frame)


def test_out_of_range_raises(frame):
    frame.loc[3, cfg.HUMIDITY_PCT] = 101.0
    with pytest.raises(data.DataValidationError, match="humidity_pct"):
        data.validate(frame)


def test_bad_dtype_raises(frame):
    frame[cfg.INPUT_CURRENT_A] = frame[cfg.INPUT_CURRENT_A].astype(str)
    with pytest.raises(data.DataValidationError, match="input_current_A"):
        data.validate(frame)


def test_unknown_split_label_raises(frame):
    frame.loc[0, cfg.SPLIT] = "validation"
    with pytest.raises(data.DataValidationError, match="split"):
        data.validate(frame)


def test_state_name_mismatch_raises(frame):
    frame.loc[0, cfg.HEALTH_STATE_NAME] = "Critical"  # row 0 has state 0
    with pytest.raises(data.DataValidationError, match="health_state_name"):
        data.validate(frame)


def test_broken_time_grid_raises(frame):
    frame.loc[5, cfg.T_HOURS] += 0.1
    with pytest.raises(data.DataValidationError, match="t_hours"):
        data.validate(frame)


def test_run_spanning_two_splits_raises_in_validate(frame):
    frame.loc[0, cfg.SPLIT] = "val"
    with pytest.raises(data.DataValidationError, match="more than one split"):
        data.validate(frame)


# ------------------------------------------------------------ leakage guard
@pytest.mark.parametrize("col", sorted(cfg.FORBIDDEN_COLUMNS))
def test_every_label_and_meta_column_is_rejected(col):
    with pytest.raises(data.LeakageError, match=col):
        data.assert_no_forbidden_columns(list(cfg.INPUT_FEATURES) + [col])


@pytest.mark.parametrize("col", ["health_score_lag1", "t_hours_sin", "hour_of_day_cos", "fault_type_code", "run_id_mod"])
def test_derived_label_or_meta_names_are_rejected(col):
    with pytest.raises(data.LeakageError):
        data.assert_no_forbidden_columns([cfg.H2_PPM, col])


def test_sensor_and_engineered_names_pass():
    data.assert_no_forbidden_columns(list(cfg.INPUT_FEATURES) + ["load_pu", "h2_ppm_roll_mean_16"])


def test_full_frame_with_labels_is_rejected(frame):
    with pytest.raises(data.LeakageError):
        data.assert_no_forbidden_columns(frame)


def test_ndarray_is_rejected():
    with pytest.raises(TypeError):
        data.assert_no_forbidden_columns(np.zeros((3, 8)))


def test_get_model_inputs_returns_exactly_the_eight_sensors(frame):
    X = data.get_model_inputs(frame)
    assert list(X.columns) == list(cfg.INPUT_FEATURES)
    assert len(X) == len(frame)


# ---------------------------------------------------------------- splitting
def test_split_by_run_is_disjoint_and_complete(frame):
    train, val, test = data.split_by_run(frame)
    assert (len(train), len(val), len(test)) == (6 * cfg.SAMPLES_PER_RUN, 2 * cfg.SAMPLES_PER_RUN, 2 * cfg.SAMPLES_PER_RUN)
    assert len(train) + len(val) + len(test) == len(frame)
    runs = [set(p[cfg.RUN_ID]) for p in (train, val, test)]
    assert not (runs[0] & runs[1]) and not (runs[0] & runs[2]) and not (runs[1] & runs[2])
    for name, part in zip(cfg.SPLIT_NAMES, (train, val, test)):
        assert set(part[cfg.SPLIT]) == {name}


def test_verify_no_run_overlap_detects_overlap(frame):
    train, val, test = data.split_by_run(frame)
    with pytest.raises(data.SplitError, match="train"):
        data.verify_no_run_overlap({"train": train, "val": train.copy(), "test": test})


def test_split_by_run_raises_if_a_run_is_in_two_splits(frame):
    frame.loc[0, cfg.SPLIT] = "val"  # run 0 now appears in train and val
    with pytest.raises(data.SplitError):
        data.split_by_run(frame)


def test_split_by_run_raises_on_empty_split(frame):
    with pytest.raises(data.SplitError, match="test"):
        data.split_by_run(frame[frame[cfg.SPLIT] != "test"])


# ------------------------------------------------------------ class balance
def test_class_balance_tables_and_printout(frame, capsys):
    train, val, test = data.split_by_run(frame)
    splits = {"train": train, "val": val, "test": test}

    for column in (cfg.HEALTH_STATE, cfg.FAULT_TYPE):
        counts, pct = data.class_balance(splits, column)
        assert list(counts.columns) == list(cfg.SPLIT_NAMES)
        for name, part in splits.items():
            assert counts[name].sum() == len(part)
            assert pct[name].sum() == pytest.approx(100.0)

    data.print_class_balance(splits)
    out = capsys.readouterr().out
    for token in ("health_state", "fault_type", "train", "val", "test", "Critical", "thermal_fault"):
        assert token in out


# ------------------------------------------------------------- real dataset
@pytest.mark.skipif(not cfg.DATA_PATH.exists(), reason="thms_dataset.csv not present")
def test_real_dataset_loads_validates_and_splits_cleanly():
    train, val, test = data.load_splits(verbose=False)
    assert len(train) + len(val) + len(test) == 94_080
    runs = [set(p[cfg.RUN_ID]) for p in (train, val, test)]
    assert not (runs[0] & runs[1]) and not (runs[0] & runs[2]) and not (runs[1] & runs[2])
    assert len(runs[0] | runs[1] | runs[2]) == 70
    data.assert_no_forbidden_columns(data.get_model_inputs(train))
