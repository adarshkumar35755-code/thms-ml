"""Tests for src/features.py: causality (no look-ahead), live-buffer equivalence,
formulas, warm-up handling, per-run wrapper, names and monotone constraints."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as cfg
from src import data
from src import features as F

W = cfg.FEATURE_WINDOW
SLOPES = ["h2_rate_ppm_per_day", "oil_temp_slope_C_per_h"]


def make_run(n: int = 200, seed: int = 0, run_id: int = 0, with_meta: bool = True) -> pd.DataFrame:
    """One synthetic run with plausible sensor values (finite features everywhere)."""
    rng = np.random.default_rng(seed)
    out = rng.uniform(0.3, 1.2, n) * cfg.RATED_LV_CURRENT_A
    expected_in = np.sqrt((out / cfg.TURNS_RATIO) ** 2 + cfg.NO_LOAD_CURRENT_A**2)
    df = pd.DataFrame(
        {
            cfg.INPUT_CURRENT_A: expected_in * rng.normal(1.0, 0.02, n),
            cfg.OUTPUT_CURRENT_A: out,
            cfg.OUTPUT_VOLTAGE_V: rng.normal(415, 8, n),
            cfg.AMBIENT_TEMP_C: rng.uniform(15, 35, n),
            cfg.HUMIDITY_PCT: rng.uniform(30, 80, n),
            cfg.VIBRATION_RMS: rng.uniform(0.4, 2.5, n),
            cfg.H2_PPM: np.cumsum(rng.uniform(0, 30, n)),
            cfg.OIL_TEMP_C: 50 + np.cumsum(rng.normal(0, 0.8, n)),
        }
    )
    if with_meta:
        df.insert(0, cfg.RUN_ID, run_id)
        df.insert(1, cfg.T_HOURS, np.arange(n) * cfg.SAMPLE_PERIOD_H)
    return df


def one_row(**overrides: float) -> pd.DataFrame:
    """Single healthy-looking sample (input current exactly as expected); override any sensor."""
    out = overrides.get(cfg.OUTPUT_CURRENT_A, cfg.RATED_LV_CURRENT_A)
    expected_in = np.sqrt((out / cfg.TURNS_RATIO) ** 2 + cfg.NO_LOAD_CURRENT_A**2)
    row = {
        cfg.INPUT_CURRENT_A: expected_in,
        cfg.OUTPUT_CURRENT_A: out,
        cfg.OUTPUT_VOLTAGE_V: 415.0,
        cfg.AMBIENT_TEMP_C: 25.0,
        cfg.HUMIDITY_PCT: 50.0,
        cfg.VIBRATION_RMS: 0.9,
        cfg.H2_PPM: 0.0,
        cfg.OIL_TEMP_C: 60.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def assert_features_equal(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------ no look-ahead
@pytest.mark.parametrize("cut", [1, 5, 23, 24, 25, 100, 199])
def test_truncating_the_future_does_not_change_earlier_rows(cut):
    df = make_run()
    full = F.build_features(df)
    assert_features_equal(F.build_features(df.iloc[:cut]), full.iloc[:cut])


@pytest.mark.parametrize("t", [0, 10, 23, 24, 60, 150])
def test_overwriting_future_samples_does_not_change_past_features(t):
    df = make_run()
    base = F.build_features(df)
    rng = np.random.default_rng(1)
    mutated = df.copy()
    future = np.arange(len(df)) > t
    for col in cfg.INPUT_FEATURES:
        garbage = rng.uniform(-1e3, 1e3, len(df))
        mutated[col] = np.where(future, garbage, mutated[col].to_numpy())
    assert_features_equal(F.build_features(mutated).iloc[: t + 1], base.iloc[: t + 1])


def test_checks_are_not_vacuous_current_and_recent_past_do_matter():
    df = make_run()
    t = 100
    base = F.build_features(df)

    cur = df.copy()
    cur.loc[t, cfg.H2_PPM] += 500.0  # current sample
    assert F.build_features(cur).loc[t, "h2_roll_max"] != base.loc[t, "h2_roll_max"]

    past = df.copy()
    past.loc[t - 5, cfg.H2_PPM] += 500.0  # inside the window
    assert F.build_features(past).loc[t, "h2_rate_ppm_per_day"] != pytest.approx(base.loc[t, "h2_rate_ppm_per_day"])


def test_window_is_exactly_current_plus_window_minus_one_past_samples():
    df = make_run()
    t = 100
    j = t - W  # first sample OUTSIDE row t's window, but inside row t-1's window
    base = F.build_features(df)

    mutated = df.copy()
    mutated.loc[j, cfg.H2_PPM] += 500.0
    mutated.loc[j, cfg.VIBRATION_RMS] += 5.0
    mutated.loc[j, cfg.OIL_TEMP_C] += 20.0
    new = F.build_features(mutated)

    for col in ("h2_rate_ppm_per_day", "h2_roll_max", "vib_roll_mean", "oil_temp_slope_C_per_h"):
        assert new.loc[t, col] == pytest.approx(base.loc[t, col]), col  # outside the window
    assert new.loc[t - 1, "h2_rate_ppm_per_day"] != pytest.approx(base.loc[t - 1, "h2_rate_ppm_per_day"])
    assert new.loc[t - 1, "vib_roll_mean"] != pytest.approx(base.loc[t - 1, "vib_roll_mean"])


@pytest.mark.parametrize("buffer_len", [W, W + 16])
def test_live_rolling_buffer_matches_full_history(buffer_len):
    df = make_run()
    full = F.build_features(df)
    for t in range(buffer_len - 1, len(df), 7):
        live = F.build_features(df.iloc[t - buffer_len + 1 : t + 1]).iloc[-1]
        np.testing.assert_allclose(live.to_numpy(), full.iloc[t].to_numpy(), rtol=1e-9, atol=1e-9)


def test_sensor_only_frame_gives_same_features_as_frame_with_meta():
    with_meta = make_run()
    sensors_only = with_meta[list(cfg.INPUT_FEATURES)]
    assert_features_equal(F.build_features(sensors_only), F.build_features(with_meta))


# ---------------------------------------------------------------- formulas
def test_load_oil_rise_and_residual():
    f = F.build_features(one_row(**{cfg.OUTPUT_CURRENT_A: 139.0})).iloc[0]
    assert f["load_pu"] == pytest.approx(0.5)
    assert f["oil_rise"] == pytest.approx(35.0)
    assert f["in_out_resid"] == pytest.approx(0.0, abs=1e-12)

    expected_in = np.sqrt((139.0 / 26.5) ** 2 + (0.015 * 10.5) ** 2)
    f = F.build_features(one_row(**{cfg.OUTPUT_CURRENT_A: 139.0, cfg.INPUT_CURRENT_A: 1.1 * expected_in})).iloc[0]
    assert f["in_out_resid"] == pytest.approx(0.1)
    f = F.build_features(one_row(**{cfg.OUTPUT_CURRENT_A: 139.0, cfg.INPUT_CURRENT_A: 0.9 * expected_in})).iloc[0]
    assert f["in_out_resid"] == pytest.approx(-0.1)
    assert f["abs_in_out_resid"] == pytest.approx(0.1)


def test_vibration_ratio():
    f = F.build_features(one_row(**{cfg.VIBRATION_RMS: 0.9})).iloc[0]  # V=415, load_pu=1 -> expected 0.9
    assert f["vib_ratio"] == pytest.approx(1.0)
    f = F.build_features(one_row(**{cfg.OUTPUT_VOLTAGE_V: 207.5, cfg.VIBRATION_RMS: 0.4875})).iloc[0]
    assert f["vib_ratio"] == pytest.approx(1.0)  # 0.55 * 0.25 + 0.35 * 1


@pytest.mark.parametrize(
    "feature, overrides, expected",
    [
        ("d_h2", {cfg.H2_PPM: 10.0}, 0.0),
        ("d_h2", {cfg.H2_PPM: 100.0}, 0.0),
        ("d_h2", {cfg.H2_PPM: 950.0}, 0.5),
        ("d_h2", {cfg.H2_PPM: 1800.0}, 1.0),
        ("d_h2", {cfg.H2_PPM: 5000.0}, 1.5),
        ("d_oil_temp", {cfg.OIL_TEMP_C: 60.0}, 0.0),
        ("d_oil_temp", {cfg.OIL_TEMP_C: 92.5}, 0.5),
        ("d_oil_temp", {cfg.OIL_TEMP_C: 105.0}, 1.0),
        ("d_oil_temp", {cfg.OIL_TEMP_C: 200.0}, 1.5),
        ("d_load", {cfg.OUTPUT_CURRENT_A: 1.1 * 278.0}, 0.0),
        ("d_load", {cfg.OUTPUT_CURRENT_A: 1.3 * 278.0}, 0.5),
        ("d_load", {cfg.OUTPUT_CURRENT_A: 1.5 * 278.0}, 1.0),
        ("d_load", {cfg.OUTPUT_CURRENT_A: 3.0 * 278.0}, 1.5),
        ("d_vib", {cfg.VIBRATION_RMS: 2.0}, 0.0),
        ("d_vib", {cfg.VIBRATION_RMS: 3.5}, 0.5),
        ("d_vib", {cfg.VIBRATION_RMS: 5.0}, 1.0),
        ("d_vib", {cfg.VIBRATION_RMS: 9.0}, 1.5),
    ],
)
def test_normalized_distance_to_limit(feature, overrides, expected):
    assert F.build_features(one_row(**overrides)).iloc[0][feature] == pytest.approx(expected)


@pytest.mark.parametrize(
    "scale, expected",
    [(1.0, 0.0), (1.05, 0.0), (0.875, 0.5), (1.125, 0.5), (1.2, 1.0), (0.8, 1.0), (2.0, 1.5)],
)
def test_distance_uses_absolute_residual(scale, expected):
    expected_in = np.sqrt((cfg.RATED_LV_CURRENT_A / cfg.TURNS_RATIO) ** 2 + cfg.NO_LOAD_CURRENT_A**2)
    f = F.build_features(one_row(**{cfg.INPUT_CURRENT_A: scale * expected_in})).iloc[0]
    assert f["d_abs_resid"] == pytest.approx(expected, abs=1e-9)


def test_vib_ratio_is_nan_not_inf_when_expected_vibration_is_zero():
    f = F.build_features(one_row(**{cfg.OUTPUT_VOLTAGE_V: 0.0, cfg.OUTPUT_CURRENT_A: 0.0})).iloc[0]
    assert np.isnan(f["vib_ratio"])


# -------------------------------------------------------------- trend units
def test_slope_units_and_values():
    n = 40
    df = make_run(n=n)
    df[cfg.H2_PPM] = 10.0 * np.arange(n)  # +10 ppm per 15-min sample
    df[cfg.OIL_TEMP_C] = 50.0 + 0.2 * np.arange(n)  # +0.2 C per sample
    f = F.build_features(df)
    assert f["h2_rate_ppm_per_day"].iloc[W - 1 :].to_numpy() == pytest.approx(960.0)  # 10 * 96
    assert f["oil_temp_slope_C_per_h"].iloc[W - 1 :].to_numpy() == pytest.approx(0.8)  # 0.2 * 4


def test_slope_matches_numpy_polyfit():
    df = make_run()
    f = F.build_features(df)
    for t in (W - 1, 60, 199):
        seg = df.iloc[t - W + 1 : t + 1]
        x = np.arange(W)
        h2_slope = np.polyfit(x, seg[cfg.H2_PPM].to_numpy(), 1)[0] * cfg.SAMPLES_PER_DAY
        oil_slope = np.polyfit(x, seg[cfg.OIL_TEMP_C].to_numpy(), 1)[0] / cfg.SAMPLE_PERIOD_H
        assert f.loc[t, "h2_rate_ppm_per_day"] == pytest.approx(h2_slope, rel=1e-9)
        assert f.loc[t, "oil_temp_slope_C_per_h"] == pytest.approx(oil_slope, rel=1e-9)


def test_rolling_mean_and_max_over_window():
    df = make_run()
    f = F.build_features(df)
    t = 120
    seg = df.iloc[t - W + 1 : t + 1]
    assert f.loc[t, "vib_roll_mean"] == pytest.approx(seg[cfg.VIBRATION_RMS].mean())
    assert f.loc[t, "h2_roll_max"] == seg[cfg.H2_PPM].max()


# ------------------------------------------------------ missing-history rule
def test_warmup_handling_is_explicit():
    df = make_run()
    f = F.build_features(df)
    for col in SLOPES:
        assert f[col].iloc[: W - 1].isna().all()
        assert f[col].iloc[W - 1 :].notna().all()
    others = [c for c in F.FEATURE_NAMES if c not in SLOPES]
    assert f[others].notna().all().all()  # everything else is defined from row 0

    # expanding mean/max while history is short
    assert f["h2_roll_max"].iloc[0] == df[cfg.H2_PPM].iloc[0]
    assert f["vib_roll_mean"].iloc[0] == pytest.approx(df[cfg.VIBRATION_RMS].iloc[0])
    k = 10
    assert f["h2_roll_max"].iloc[k] == df[cfg.H2_PPM].iloc[: k + 1].max()
    assert f["vib_roll_mean"].iloc[k] == pytest.approx(df[cfg.VIBRATION_RMS].iloc[: k + 1].mean())

    mask = F.warmup_mask(df)
    assert int(mask.sum()) == W - 1
    assert mask.iloc[: W - 1].all() and not mask.iloc[W - 1 :].any()
    assert (mask == f[SLOPES].isna().any(axis=1)).all()


def test_nan_inside_window_gives_nan_slope_only_while_it_is_in_the_window():
    df = make_run()
    df.loc[50, cfg.H2_PPM] = np.nan
    f = F.build_features(df)
    assert f.loc[50 : 50 + W - 1, "h2_rate_ppm_per_day"].isna().all()
    assert f.loc[50 + W, "h2_rate_ppm_per_day"] == f.loc[50 + W, "h2_rate_ppm_per_day"]  # finite again
    assert f.loc[49, "h2_rate_ppm_per_day"] == f.loc[49, "h2_rate_ppm_per_day"]


# ------------------------------------------------------------------ wrapper
def test_wrapper_matches_per_run_and_never_crosses_runs():
    a, b = make_run(seed=1, run_id=0), make_run(seed=2, run_id=1)
    both = pd.concat([a, b], ignore_index=True)
    got = F.build_features_by_run(both)
    expected = pd.concat([F.build_features(a), F.build_features(b)], ignore_index=True)
    assert_features_equal(got, expected)
    # first rows of run 1 are warm-up: nothing leaked from the tail of run 0
    assert got.iloc[len(a) : len(a) + W - 1][SLOPES].isna().all().all()


def test_wrapper_preserves_index_and_row_order():
    a, b = make_run(seed=1, run_id=0), make_run(seed=2, run_id=1)
    both = pd.concat([a, b], ignore_index=True)
    both.index = both.index + 1000
    got = F.build_features_by_run(both)
    assert got.index.equals(both.index)
    assert list(got.columns) == F.get_feature_names()
    assert_features_equal(got.reset_index(drop=True), F.build_features_by_run(both.reset_index(drop=True)))


def test_wrapper_requires_run_id():
    with pytest.raises(ValueError, match="run_id"):
        F.build_features_by_run(make_run(with_meta=False))


# ------------------------------------------------------------------- guards
def test_build_features_does_not_mutate_its_input():
    df = make_run()
    before = df.copy()
    F.build_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_missing_sensor_column_raises():
    with pytest.raises(ValueError, match="h2_ppm"):
        F.build_features(make_run().drop(columns=cfg.H2_PPM))


def test_multiple_runs_in_one_call_raise():
    both = pd.concat([make_run(run_id=0), make_run(run_id=1)], ignore_index=True)
    with pytest.raises(ValueError, match="ONE run"):
        F.build_features(both)


def test_out_of_order_rows_raise():
    df = make_run().iloc[::-1]
    with pytest.raises(ValueError, match="time order"):
        F.build_features(df)


@pytest.mark.parametrize("bad", [1, 0, -3, 2.5, True])
def test_bad_window_raises(bad):
    with pytest.raises(ValueError, match="window"):
        F.build_features(make_run(), window=bad)


def test_other_window_sizes_stay_causal():
    df = make_run()
    for window in (2, 6, 48):
        full = F.build_features(df, window=window)
        assert_features_equal(F.build_features(df.iloc[:77], window=window), full.iloc[:77])


# --------------------------------------------------- names and constraints
def test_feature_names_and_output_columns():
    names = F.get_feature_names()
    assert len(names) == len(set(names)) == 22
    assert list(F.build_features(make_run()).columns) == names
    assert names[:8] == list(cfg.INPUT_FEATURES)
    data.assert_no_forbidden_columns(names)


def test_monotone_constraints():
    names = F.get_feature_names()
    cons = F.get_monotone_constraints()
    assert len(cons) == len(names) and set(cons) <= {-1, 0}
    constrained = {n for n, c in zip(names, cons) if c == -1}
    assert constrained == set(F.MONOTONE_DECREASING)
    required = {
        "d_h2", "d_oil_temp", "d_load", "d_vib", "d_abs_resid",  # distance features
        "h2_rate_ppm_per_day", "oil_temp_C", "vibration_rms_ms2", "load_pu", "abs_in_out_resid",
    }  # fmt: skip
    assert required <= constrained
    assert "in_out_resid" not in constrained  # signed residual: both directions are bad
    assert F.get_monotone_constraints(["load_pu", "humidity_pct"]) == [-1, 0]
    with pytest.raises(KeyError):
        F.get_monotone_constraints(["not_a_feature"])


# ------------------------------------------------------------- real dataset
@pytest.mark.skipif(not cfg.DATA_PATH.exists(), reason="thms_dataset.csv not present")
def test_real_dataset_features_are_finite_bounded_and_nan_only_in_warmup():
    df = data.load_dataset()
    feats = F.build_features_by_run(df)
    assert list(feats.columns) == F.get_feature_names() and len(feats) == len(df)
    assert not np.isinf(feats.to_numpy()).any()

    warm = F.warmup_mask(df)
    assert feats.loc[~warm].notna().all().all()
    assert feats.loc[warm, SLOPES].isna().all().all()

    d_cols = [c for c in F.FEATURE_NAMES if c.startswith("d_")]
    assert feats[d_cols].to_numpy().min() >= 0.0
    assert feats[d_cols].to_numpy().max() <= cfg.DIST_CLIP_MAX
