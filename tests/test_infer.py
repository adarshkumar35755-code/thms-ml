"""Tests for src/infer.py: streaming == batch, warm-up, degraded mode, clamping, reasons.

The replay tests use runs from the VAL split (the TEST split belongs to evaluate.py).
Run from the project root:  pytest tests/test_infer.py
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

import config as cfg
from src import data, scoring
from src import features as F
from src import infer as I
from src import train as T

pytestmark = pytest.mark.skipif(
    not (cfg.MODEL_PATH.exists() and cfg.MODEL_META_PATH.exists() and cfg.DATA_PATH.exists()),
    reason="trained model artifacts or dataset not found (run `python -m src.train` first)",
)

SENSORS = list(cfg.INPUT_FEATURES)
W = cfg.FEATURE_WINDOW


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def val_df() -> pd.DataFrame:
    _train, val, _test = data.load_splits(verbose=False)
    return val


@pytest.fixture(scope="module")
def runs(val_df):
    """One normal run and one faulty run from the val split."""
    normal = val_df[val_df[cfg.FAULT_TYPE] == "normal"][cfg.RUN_ID].iloc[0]
    faulty_id = val_df[val_df[cfg.FAULT_TYPE] != "normal"][cfg.RUN_ID].iloc[0]
    return {rid: val_df[val_df[cfg.RUN_ID] == rid].reset_index(drop=True) for rid in (normal, faulty_id)}


@pytest.fixture()
def scorer():
    return I.HealthScorer()


@pytest.fixture(scope="module")
def booster_model():
    import joblib

    return joblib.load(cfg.MODEL_PATH)


def good_packet(i: int = 0) -> dict:
    """A plausible, healthy packet."""
    return {
        "input_current_A": 3.0, "output_current_A": 80.0, "output_voltage_V": 415.0,
        "ambient_temp_C": 28.0, "humidity_pct": 60.0, "vibration_rms_ms2": 0.6,
        "h2_ppm": 30.0 + 0.01 * i, "oil_temp_C": 50.0,
    }


def fill(scorer: I.HealthScorer, n: int) -> dict:
    out = {}
    for i in range(n):
        out = scorer.update(good_packet(i))
    return out


# --------------------------------------------------------------------------
# Streaming == batch (the core guarantee)
# --------------------------------------------------------------------------
def test_streaming_matches_batch_predictions(scorer, runs, booster_model):
    for rid, run in runs.items():
        scorer.reset()
        streamed = [scorer.update(rec) for rec in run[SENSORS].to_dict("records")]
        stream_score = np.array([r["health_score"] for r in streamed], dtype=float)

        # independent batch path: training's own build_features_by_run + train.predict_score
        batch = T.predict_score(booster_model, F.build_features_by_run(run))
        np.testing.assert_allclose(stream_score, batch, rtol=0, atol=1e-9, err_msg=f"run {rid}")

        batch_state = scoring.score_to_state(batch)
        assert [r["health_state"] for r in streamed] == batch_state.tolist()
        assert [r["state_name"] for r in streamed] == [cfg.HEALTH_STATE_NAMES[s] for s in batch_state]


def test_warming_up_flag_matches_feature_warmup_mask(scorer, runs):
    run = next(iter(runs.values()))
    flags = [scorer.update(rec)["warming_up"] for rec in run[SENSORS].to_dict("records")]
    expected = F.warmup_mask(run).tolist()
    assert flags == expected
    assert sum(flags) == W - 1 and not flags[W - 1]


def test_provisional_warmup_can_be_withheld(runs):
    sc = I.HealthScorer(provisional_warmup=False)
    run = next(iter(runs.values()))
    outs = [sc.update(rec) for rec in run[SENSORS].to_dict("records")[: W + 2]]
    assert all(o["health_score"] is None and o["state_name"] == "Unknown" for o in outs[: W - 1])
    assert all(o["warming_up"] for o in outs[: W - 1])
    assert outs[W - 1]["health_score"] is not None and not outs[W - 1]["warming_up"]


def test_first_packet_returns_a_result(scorer):
    out = scorer.update(good_packet())
    assert out["warming_up"] is True
    assert out["health_score"] is not None


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------
def test_output_schema_and_ranges(scorer):
    out = fill(scorer, W + 3)
    assert set(out) == {
        "health_score", "health_state", "state_name", "top_reasons", "warming_up", "degraded",
        "degraded_reasons", "imputed_sensors", "status", "samples_seen", "timestamp",
    }
    assert 0.0 <= out["health_score"] <= 100.0
    assert out["health_state"] in (0, 1, 2, 3)
    assert out["state_name"] == cfg.HEALTH_STATE_NAMES[out["health_state"]]
    assert out["warming_up"] is False and out["degraded"] is False and out["status"] == "ok"
    assert out["samples_seen"] == W + 3


def test_timestamp_is_echoed_and_extra_keys_ignored(scorer):
    pkt = good_packet() | {"timestamp": "2026-10-18T12:00:00", "run_id": 99, "junk": "x"}
    assert scorer.update(pkt)["timestamp"] == "2026-10-18T12:00:00"


def test_non_mapping_packet_is_a_programming_error(scorer):
    with pytest.raises(TypeError):
        scorer.update([1, 2, 3])  # type: ignore[arg-type]


def test_buffer_is_a_deque_of_at_least_window_and_reset_clears(scorer):
    assert scorer._buffer.maxlen >= scorer.window == W
    fill(scorer, W + 10)
    assert scorer.n_buffered == W  # trailing window only, no unbounded growth
    scorer.reset()
    assert scorer.n_buffered == 0 and scorer.warming_up
    with pytest.raises(ValueError):
        I.HealthScorer(buffer_size=W - 1)


def test_update_uses_shared_build_features(scorer, monkeypatch):
    calls = []
    real = F.build_features

    def spy(df, window=cfg.FEATURE_WINDOW):
        calls.append((len(df), window))
        return real(df, window)

    monkeypatch.setattr(I.F, "build_features", spy)
    scorer.update(good_packet())
    scorer.update(good_packet(1))
    assert calls == [(1, W), (2, W)]


# --------------------------------------------------------------------------
# Missing / NaN handling and degraded mode
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [np.nan, None, float("inf"), "n/a", True, -5.0])
def test_bad_h2_value_is_imputed_not_fatal(scorer, bad):
    fill(scorer, W + 2)
    ref = I.HealthScorer()
    fill(ref, W + 2)
    pkt = good_packet(W + 2) | {"h2_ppm": bad}
    out = scorer.update(pkt)
    # equals a scorer that was simply fed the last valid H2 value again
    held = ref.update(good_packet(W + 2) | {"h2_ppm": good_packet(W + 1)["h2_ppm"]})
    assert out["degraded"] is True and out["status"] == "degraded"
    assert out["imputed_sensors"] == {"h2_ppm": 1}
    assert any("h2_ppm" in r for r in out["degraded_reasons"])
    assert out["health_score"] == pytest.approx(held["health_score"], abs=1e-9)
    assert 0.0 <= out["health_score"] <= 100.0


def test_missing_key_counts_as_missing(scorer):
    fill(scorer, W + 2)
    pkt = good_packet(W + 2)
    del pkt["vibration_rms_ms2"]
    out = scorer.update(pkt)
    assert out["degraded"] and "vibration_rms_ms2" in out["imputed_sensors"]


def test_out_of_range_value_is_rejected_and_range_check_can_be_disabled():
    sc = I.HealthScorer()
    fill(sc, W + 1)
    out = sc.update(good_packet() | {"humidity_pct": 250.0})
    assert out["degraded"] and "out of range" in " ".join(out["degraded_reasons"])
    lenient = I.HealthScorer(validate_ranges=False)
    fill(lenient, W + 1)
    assert lenient.update(good_packet() | {"humidity_pct": 250.0})["degraded"] is False


def test_degraded_flag_persists_while_imputed_samples_are_in_window(scorer):
    fill(scorer, W + 2)
    scorer.update(good_packet() | {"h2_ppm": np.nan})
    flags = [scorer.update(good_packet(i))["degraded"] for i in range(W)]
    assert all(flags[: W - 1])  # still inside the window
    assert flags[-1] is False  # the imputed sample has left the window


def test_all_sensors_nan_after_history_never_raises(scorer):
    fill(scorer, W + 2)
    out = scorer.update({s: np.nan for s in SENSORS})
    assert out["degraded"] and set(out["imputed_sensors"]) == set(SENSORS)
    assert out["health_score"] is not None  # held values, within max_hold_samples


def test_sensor_never_seen_makes_packet_unscorable_and_unbuffered(scorer):
    pkt = good_packet()
    pkt["h2_ppm"] = np.nan
    out = scorer.update(pkt)
    assert out["health_score"] is None and out["health_state"] is None
    assert out["state_name"] == "Unknown" and out["status"] == "unscorable" and out["degraded"]
    assert scorer.n_buffered == 0
    assert scorer.update(good_packet())["health_score"] is not None  # recovers on the next good packet


def test_sensor_lost_too_long_withholds_the_score():
    sc = I.HealthScorer(max_hold_samples=3)
    fill(sc, W + 2)
    outs = [sc.update(good_packet() | {"h2_ppm": np.nan}) for _ in range(5)]
    assert [o["health_score"] is not None for o in outs] == [True, True, True, False, False]
    assert outs[-1]["state_name"] == "Unknown" and "withheld" in " ".join(outs[-1]["degraded_reasons"])
    assert sc.update(good_packet())["health_score"] is not None  # recovers


# --------------------------------------------------------------------------
# Clamping
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [(137.5, 100.0), (-42.0, 0.0), (55.5, 55.5)])
def test_score_is_clamped_to_0_100(scorer, monkeypatch, raw, expected):
    class Fake:
        def __init__(self, real):
            self.real = real

        def predict(self, X, **kw):
            if kw.get("pred_contrib"):
                return self.real.predict(X, **kw)
            return np.full(len(X), raw)

        def num_feature(self):
            return self.real.num_feature()

    monkeypatch.setattr(scorer, "_booster", Fake(scorer._booster))
    out = scorer.update(good_packet())
    assert out["health_score"] == expected
    assert out["health_state"] == int(scoring.score_to_state(expected))


# --------------------------------------------------------------------------
# Reasons
# --------------------------------------------------------------------------
def test_contributions_are_additive(scorer):
    fill(scorer, W + 2)
    feats = F.build_features(pd.DataFrame(list(scorer._buffer), columns=SENSORS), W).iloc[[-1]]
    contrib = scorer._booster.predict(feats[scorer.feature_names], num_iteration=scorer._num_iteration,
                                      pred_contrib=True)[0]
    raw = scorer._booster.predict(feats[scorer.feature_names], num_iteration=scorer._num_iteration)[0]
    assert contrib.sum() == pytest.approx(raw, abs=1e-6)


def test_top_reasons_structure_and_ordering(scorer, runs):
    faulty = list(runs.values())[1]
    scorer.reset()
    out = None
    for rec in faulty[SENSORS].to_dict("records")[:900]:
        out = scorer.update(rec)
    reasons = out["top_reasons"]
    assert 1 <= len(reasons) <= 3
    impacts = [abs(r["impact_points"]) for r in reasons]
    assert impacts == sorted(impacts, reverse=True)
    for r in reasons:
        assert {"name", "label", "impact_points", "direction", "reading", "unit", "features"} <= set(r)
        assert r["direction"] == ("lowers_score" if r["impact_points"] < 0 else "raises_score")
        assert r["impact_points"] == pytest.approx(sum(r["features"].values()))
        assert r["name"] in I.DRIVERS


def test_per_feature_reasons_and_explain_off():
    per_feature = I.HealthScorer(group_reasons=False, top_k=3)
    out = fill(per_feature, W + 2)
    assert all(r["name"] in F.FEATURE_NAMES for r in out["top_reasons"])
    quiet = I.HealthScorer(explain=False)
    assert fill(quiet, W + 2)["top_reasons"] == []


def test_a_hydrogen_fault_is_blamed_on_hydrogen(scorer):
    fill(scorer, W + 2)
    out = None
    for i in range(30):  # H2 climbing to 1500 ppm
        out = scorer.update(good_packet() | {"h2_ppm": 30 + i * 50})
    assert out["top_reasons"][0]["name"] == "hydrogen"
    assert out["top_reasons"][0]["direction"] == "lowers_score"
    assert out["state_name"] != "Normal"


# --------------------------------------------------------------------------
# Artifact guards
# --------------------------------------------------------------------------
def test_mismatching_metadata_is_rejected(tmp_path):
    meta = json.loads(cfg.MODEL_META_PATH.read_text())
    bad = copy.deepcopy(meta)
    bad["feature_names"] = bad["feature_names"][:-1]
    p = tmp_path / "bad_meta.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(I.ArtifactMismatchError):
        I.HealthScorer(meta_path=p)

    bad = copy.deepcopy(meta)
    bad["sample_period_h"] = 1.0
    p.write_text(json.dumps(bad))
    with pytest.raises(I.ArtifactMismatchError):
        I.HealthScorer(meta_path=p)


def test_missing_artifact_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        I.HealthScorer(model_path=tmp_path / "nope.joblib")


# --------------------------------------------------------------------------
# CSV workflow
# --------------------------------------------------------------------------
def test_score_dataframe_matches_streaming_and_keeps_columns(scorer, runs):
    run = next(iter(runs.values())).iloc[:60]
    scored = I.score_dataframe(run, scorer)
    assert len(scored) == len(run) and {"health_score", "state_name", "warming_up", "degraded"} <= set(scored.columns)
    assert scored.columns.is_unique
    assert "health_score_input" in scored.columns  # the dataset's label column was kept, renamed
    scorer.reset()
    expect = [scorer.update(r)["health_score"] for r in run[SENSORS].to_dict("records")]
    np.testing.assert_allclose(scored["health_score"].to_numpy(dtype=float), expect, atol=1e-12)
    with pytest.raises(ValueError):
        I.score_dataframe(run.drop(columns=["h2_ppm"]), scorer)


def test_cli_scores_a_csv_with_renamed_columns(tmp_path, runs):
    run = next(iter(runs.values())).iloc[:40]
    custom = run[SENSORS].rename(columns={"h2_ppm": "hydrogen", "oil_temp_C": "oil_t"})
    src = tmp_path / "mine.csv"
    custom.to_csv(src, index=False)
    out = tmp_path / "scored.csv"
    assert I.main(["--csv", str(src), "--out", str(out), "--map", "hydrogen=h2_ppm,oil_t=oil_temp_C"]) == 0
    got = pd.read_csv(out)
    assert len(got) == 40 and got["health_score"].between(0, 100).all()
    with pytest.raises(ValueError):
        I.main(["--csv", str(src), "--map", "x=not_a_sensor"])
