"""Live inference for the THMS health score.

    from src.infer import HealthScorer
    scorer = HealthScorer()                       # loads models/thms_lgbm.joblib + JSON
    out = scorer.update({"input_current_A": 3.1, "output_current_A": 82.0, ...})
    out["health_score"], out["state_name"], out["top_reasons"], out["warming_up"]

Score a whole CSV of readings from the command line (see GUIDE in the chat / docstring of main):

    python -m src.infer --csv my_readings.csv --out my_scored.csv

How it works
------------
`update(packet)` appends one sensor packet (one every config.SAMPLE_PERIOD_H = 15 min)
to a rolling buffer, calls THE SAME `features.build_features` that training used on the
buffer, takes the last row, and predicts with the saved LightGBM Booster. Because every
window in build_features is trailing and the buffer holds `window` samples, the last row
equals the row the model saw in training: replaying a run through `update` reproduces
the batch predictions exactly (tests/test_infer.py checks this).

Output of update() (always the same keys)
-----------------------------------------
    health_score      float in [0, 100] (clamped), or None if the sample cannot be scored
    health_state      0 Normal / 1 Caution / 2 Warning / 3 Critical, or None
    state_name        "Normal" ... "Critical", or "Unknown"
    top_reasons       up to top_k dicts, biggest absolute effect on the score first
    warming_up        True until the buffer holds `window` samples (6 h)
    degraded          True if any sensor value was missing/invalid and was replaced
    degraded_reasons  human-readable list
    imputed_sensors   {sensor: consecutive samples it has been missing}
    status            "ok" | "degraded" | "unscorable"
    samples_seen, timestamp (echoed from the packet if present)

Decisions I made (please review)
--------------------------------
- Missing / None / NaN / inf / non-numeric / out-of-range (config.VALUE_RANGES) sensor
  values are replaced by the sensor's LAST VALID value (last-observation-carried-forward)
  and the output is flagged degraded. NaN is never passed to the model: a model trained
  without NaNs reads NaN as 0, and in the evaluation a dead H2 sensor that way caused most
  Warning/Critical samples to be missed.
- A sensor held longer than `max_hold_samples` (default 16 = 4 h) makes the sample
  "unscorable" (score None, state "Unknown") instead of reporting a score built on a stale
  value. A sensor that has never produced a valid value also makes it unscorable, and
  that packet is not added to the buffer.
- The output stays "degraded" while ANY imputed sample is still inside the 24-sample window,
  because trend features (H2 rate, oil slope) are computed over it.
- During warm-up a provisional score is returned (trend features NaN, which the model saw in
  training; its validation MAE on warm-up rows was about the same as elsewhere) with
  warming_up=True. Use provisional_warmup=False to withhold it instead.
- top_reasons are grouped by physical driver (H2, oil temperature, loading, vibration,
  current mismatch, voltage, ambient, humidity): contributions of correlated features such
  as h2_ppm / h2_roll_max / d_h2 are summed so credit is not split arbitrarily. Use
  group_reasons=False for per-feature reasons. Contributions are exact TreeSHAP values from
  LightGBM (`pred_contrib`), in score points; no `shap` package needed.
- Packets must arrive at the training cadence (15 min). The rate/slope features assume it.
  Average faster data into 15-minute packets first (see --resample-15min).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # lets `python src/infer.py` work as well as `python -m src.infer`
    sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402
from src import features as F  # noqa: E402
from src import scoring  # noqa: E402

__all__ = ["HealthScorer", "ArtifactMismatchError", "score_dataframe", "DRIVERS", "main"]

SENSORS: tuple[str, ...] = tuple(cfg.INPUT_FEATURES)
UNKNOWN_STATE_NAME = "Unknown"


class ArtifactMismatchError(ValueError):
    """The saved model/metadata does not match the current features.py / config.py."""


# --------------------------------------------------------------------------
# Reason groups: every model feature belongs to exactly one physical driver
# --------------------------------------------------------------------------
DRIVERS: dict[str, dict[str, Any]] = {
    "hydrogen": {
        "label": "H2 concentration", "reading": "h2_ppm", "unit": "ppm",
        "features": ("h2_ppm", "d_h2", "h2_rate_ppm_per_day", "h2_roll_max"),
    },
    "oil_temperature": {
        "label": "Oil temperature", "reading": "oil_temp_C", "unit": "degC",
        "features": ("oil_temp_C", "oil_rise", "d_oil_temp", "oil_temp_slope_C_per_h"),
    },
    "loading": {
        "label": "Loading", "reading": "load_pu", "unit": "pu",
        "features": ("output_current_A", "load_pu", "d_load"),
    },
    "vibration": {
        "label": "Vibration", "reading": "vibration_rms_ms2", "unit": "m/s2",
        "features": ("vibration_rms_ms2", "vib_ratio", "d_vib", "vib_roll_mean"),
    },
    "current_mismatch": {
        "label": "Input/output current mismatch", "reading": "in_out_resid", "unit": "fraction",
        "features": ("input_current_A", "in_out_resid", "abs_in_out_resid", "d_abs_resid"),
    },
    "supply_voltage": {
        "label": "Output voltage", "reading": "output_voltage_V", "unit": "V",
        "features": ("output_voltage_V",),
    },
    "ambient_temperature": {
        "label": "Ambient temperature", "reading": "ambient_temp_C", "unit": "degC",
        "features": ("ambient_temp_C",),
    },
    "humidity": {
        "label": "Humidity", "reading": "humidity_pct", "unit": "%RH",
        "features": ("humidity_pct",),
    },
}


def _check_drivers() -> None:
    members = [f for d in DRIVERS.values() for f in d["features"]]
    if sorted(members) != sorted(F.FEATURE_NAMES):
        missing = sorted(set(F.FEATURE_NAMES) - set(members))
        extra = sorted(set(members) - set(F.FEATURE_NAMES))
        raise RuntimeError(f"DRIVERS out of sync with features.py (missing {missing}, unknown {extra})")


_check_drivers()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _coerce_reading(value: Any, sensor: str, validate_ranges: bool) -> tuple[float | None, str | None]:
    """Return (value, None) if usable, else (None, reason)."""
    if value is None:
        return None, "missing"
    if isinstance(value, (bool, np.bool_)):
        return None, "non-numeric"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None, "non-numeric"
    if not math.isfinite(x):
        return None, "NaN/inf"
    if validate_ranges and sensor in cfg.VALUE_RANGES:
        lo, hi = cfg.VALUE_RANGES[sensor]
        if not (lo <= x <= hi):
            return None, f"out of range [{lo:g}, {hi:g}]"
    return x, None


def _band_edges_from_meta(meta: Mapping[str, Any]) -> tuple[float, float, float]:
    raw = meta.get("band_edges")
    if isinstance(raw, Mapping):
        try:
            return (float(raw["normal_above"]), float(raw["caution_from"]), float(raw["warning_from"]))
        except (KeyError, TypeError, ValueError):
            pass
    elif isinstance(raw, (list, tuple)) and len(raw) == 3:
        return tuple(float(x) for x in raw)  # type: ignore[return-value]
    return tuple(cfg.STATE_BAND_EDGES)  # type: ignore[return-value]


# --------------------------------------------------------------------------
# HealthScorer
# --------------------------------------------------------------------------
class HealthScorer:
    """Streaming health scorer: one `update(packet)` call per sensor packet."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        meta_path: str | Path | None = None,
        *,
        buffer_size: int | None = None,
        explain: bool = True,
        top_k: int = 3,
        group_reasons: bool = True,
        min_impact_points: float = 0.05,
        provisional_warmup: bool = True,
        max_hold_samples: int = 16,
        validate_ranges: bool = True,
    ) -> None:
        model_path = Path(model_path) if model_path else cfg.MODEL_PATH
        meta_path = Path(meta_path) if meta_path else cfg.MODEL_META_PATH
        for p in (model_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"Artifact not found: {p} (run `python -m src.train` first)")

        meta = json.loads(meta_path.read_text())
        self.meta = meta
        self._model = joblib.load(model_path)
        self._booster = getattr(self._model, "booster_", self._model)  # accept a Booster or an LGBMRegressor
        self._num_iteration = getattr(self._booster, "best_iteration", 0) or None

        self.feature_names: list[str] = list(meta.get("feature_names", []))
        if self.feature_names != F.get_feature_names():
            raise ArtifactMismatchError(
                "Model metadata feature_names differ from features.py; retrain the model or restore features.py."
            )
        if self._booster.num_feature() != len(self.feature_names):
            raise ArtifactMismatchError("Model feature count does not match its metadata.")
        if float(meta.get("sample_period_h", cfg.SAMPLE_PERIOD_H)) != cfg.SAMPLE_PERIOD_H:
            raise ArtifactMismatchError("Model was trained with a different sample period than config.py.")

        self.window = int(meta.get("feature_window", cfg.FEATURE_WINDOW))
        size = self.window if buffer_size is None else int(buffer_size)
        if size < self.window:
            raise ValueError(f"buffer_size ({size}) must be >= the feature window ({self.window})")
        self.buffer_size = size
        self.edges = _band_edges_from_meta(meta)

        self.explain = bool(explain)
        self.top_k = int(top_k)
        self.group_reasons = bool(group_reasons)
        self.min_impact_points = float(min_impact_points)
        self.provisional_warmup = bool(provisional_warmup)
        self.max_hold_samples = int(max_hold_samples)
        self.validate_ranges = bool(validate_ranges)

        self._lock = threading.Lock()
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        """Forget all history (call when a new monitoring session / transformer starts)."""
        self._buffer: deque[dict[str, float]] = deque(maxlen=self.buffer_size)
        self._imputed: deque[frozenset[str]] = deque(maxlen=self.buffer_size)  # sensors imputed per buffered sample
        self._last_valid: dict[str, float] = {}
        self._age: dict[str, int] = {s: 0 for s in SENSORS}  # consecutive invalid samples per sensor
        self._seen = 0

    @property
    def n_buffered(self) -> int:
        return len(self._buffer)

    @property
    def warming_up(self) -> bool:
        return len(self._buffer) < self.window

    # ------------------------------------------------------------- batch API
    def predict_batch(self, feats: pd.DataFrame) -> np.ndarray:
        """Clamped scores for a feature matrix from build_features (same call used by update)."""
        raw = self._booster.predict(feats[self.feature_names], num_iteration=self._num_iteration)
        return scoring.clip_score(raw)

    # ---------------------------------------------------------------- update
    def update(self, packet: Mapping[str, Any]) -> dict[str, Any]:
        """Ingest one sensor packet and return the current health assessment (never raises on bad sensor data)."""
        if not isinstance(packet, Mapping):
            raise TypeError(f"packet must be a dict-like mapping of sensor name -> value, got {type(packet).__name__}")
        with self._lock:
            return self._update(packet)

    def _update(self, packet: Mapping[str, Any]) -> dict[str, Any]:
        self._seen += 1
        reasons_bad: dict[str, str] = {}
        row: dict[str, float] = {}
        unscorable: list[str] = []

        for s in SENSORS:
            x, why = _coerce_reading(packet.get(s), s, self.validate_ranges)
            if x is not None:
                self._last_valid[s] = x
                self._age[s] = 0
                row[s] = x
                continue
            reasons_bad[s] = why or "invalid"
            self._age[s] += 1
            if s in self._last_valid:
                row[s] = self._last_valid[s]  # carry the last valid value forward
            else:
                unscorable.append(f"{s} has never reported a valid value")

        if unscorable:  # nothing to carry forward: this packet cannot enter the buffer
            return self._result(packet, None, None, [], reasons_bad, unscorable)

        self._buffer.append(row)
        self._imputed.append(frozenset(reasons_bad))

        stale = [s for s in reasons_bad if self._age[s] > self.max_hold_samples]
        feats = F.build_features(pd.DataFrame(list(self._buffer), columns=list(SENSORS)), self.window)
        feat_row = feats.iloc[[-1]]

        if stale:
            msg = [f"{s} lost for {self._age[s]} samples (> {self.max_hold_samples}); score withheld" for s in stale]
            return self._result(packet, None, feat_row, [], reasons_bad, msg)
        if self.warming_up and not self.provisional_warmup:
            return self._result(packet, None, feat_row, [], reasons_bad, [])

        score = float(self.predict_batch(feat_row)[0])
        reasons = self._explain(feat_row) if self.explain else []
        return self._result(packet, score, feat_row, reasons, reasons_bad, [])

    # --------------------------------------------------------------- explain
    def _explain(self, feat_row: pd.DataFrame) -> list[dict[str, Any]]:
        contrib = self._booster.predict(
            feat_row[self.feature_names], num_iteration=self._num_iteration, pred_contrib=True
        )[0]
        per_feature = dict(zip(self.feature_names, (float(c) for c in contrib[:-1])))  # last column = expected value
        values = feat_row.iloc[0]

        def reading(name: str) -> float | None:
            v = values[name]
            return None if pd.isna(v) else float(v)

        items: list[dict[str, Any]] = []
        if self.group_reasons:
            for key, spec in DRIVERS.items():
                members = {f: per_feature[f] for f in spec["features"]}
                items.append(
                    {
                        "name": key,
                        "label": spec["label"],
                        "impact_points": sum(members.values()),
                        "reading": reading(spec["reading"]),
                        "unit": spec["unit"],
                        "features": dict(sorted(members.items(), key=lambda kv: -abs(kv[1]))),
                    }
                )
        else:
            for f, c in per_feature.items():
                spec = next(d for d in DRIVERS.values() if f in d["features"])
                items.append({"name": f, "label": f, "impact_points": c, "reading": reading(f),
                              "unit": spec["unit"] if f == spec["reading"] else "", "features": {f: c}})

        items = [i for i in items if abs(i["impact_points"]) >= self.min_impact_points]
        items.sort(key=lambda i: -abs(i["impact_points"]))
        for i in items:
            i["direction"] = "lowers_score" if i["impact_points"] < 0 else "raises_score"
        return items[: self.top_k]

    # ---------------------------------------------------------------- result
    def _result(
        self,
        packet: Mapping[str, Any],
        score: float | None,
        feat_row: pd.DataFrame | None,
        reasons: list[dict[str, Any]],
        bad_now: dict[str, str],
        blockers: list[str],
    ) -> dict[str, Any]:
        degraded_reasons = [
            f"{s}: {why}" + (f" (holding last value, {self._age[s]} sample(s) old)" if s in self._last_valid else "")
            for s, why in bad_now.items()
        ]
        recent: dict[str, int] = {}
        for sensors in self._imputed:
            for s in sensors:
                recent[s] = recent.get(s, 0) + 1
        window_imputed = {s: n for s, n in recent.items() if s not in bad_now}
        if window_imputed:
            degraded_reasons.append(
                "recent imputed samples still inside the feature window: "
                + ", ".join(f"{s} x{n}" for s, n in sorted(window_imputed.items()))
            )
        degraded_reasons.extend(blockers)
        degraded = bool(degraded_reasons)

        if score is None:
            state, name, status = None, UNKNOWN_STATE_NAME, "unscorable"
        else:
            score = float(scoring.clip_score(score))
            state = int(scoring.score_to_state(score, self.edges))
            name = cfg.HEALTH_STATE_NAMES[state]
            status = "degraded" if degraded else "ok"

        return {
            "health_score": score,
            "health_state": state,
            "state_name": name,
            "top_reasons": reasons,
            "warming_up": self.warming_up,
            "degraded": degraded,
            "degraded_reasons": degraded_reasons,
            "imputed_sensors": {s: self._age[s] for s in bad_now},
            "status": status,
            "samples_seen": self._seen,
            "timestamp": packet.get("timestamp"),
        }


# --------------------------------------------------------------------------
# Scoring a whole table of readings (CSV workflow)
# --------------------------------------------------------------------------
def _reasons_text(reasons: list[dict[str, Any]]) -> str:
    return "; ".join(f"{r['label']} {r['impact_points']:+.1f}" for r in reasons)


def score_dataframe(df: pd.DataFrame, scorer: HealthScorer | None = None) -> pd.DataFrame:
    """Stream the rows of `df` (oldest first, one row per 15 min) through a HealthScorer.

    `df` needs the 8 sensor columns (config.INPUT_FEATURES); other columns are kept. A fresh
    buffer is used (scorer.reset()). Input columns that share a name with an output column
    (e.g. a `health_score` label) are kept as `<name>_input`. Returns df plus health_score, health_state, state_name,
    warming_up, degraded, status, top_reasons (text) and degraded_reasons (text).
    """
    scorer = scorer or HealthScorer()
    missing = [c for c in SENSORS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns {missing}; needed: {list(SENSORS)}. Use --map to rename columns.")
    scorer.reset()
    out_rows = []
    for rec in df[list(SENSORS)].to_dict("records"):
        r = scorer.update(rec)
        out_rows.append(
            {
                "health_score": r["health_score"],
                "health_state": r["health_state"],
                "state_name": r["state_name"],
                "warming_up": r["warming_up"],
                "degraded": r["degraded"],
                "status": r["status"],
                "top_reasons": _reasons_text(r["top_reasons"]),
                "degraded_reasons": " | ".join(r["degraded_reasons"]),
            }
        )
    res = pd.DataFrame(out_rows, index=df.index)
    res["health_state"] = res["health_state"].astype("Int64")
    clash = {c: f"{c}_input" for c in df.columns if c in res.columns}  # e.g. the dataset's own label columns
    return pd.concat([df.rename(columns=clash), res], axis=1)


def _parse_map(spec: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for part in filter(None, (spec or "").split(",")):
        if "=" not in part:
            raise ValueError(f"--map expects my_column=model_column pairs, got {part!r}")
        mine, model_col = (x.strip() for x in part.split("=", 1))
        if model_col not in SENSORS:
            raise ValueError(f"{model_col!r} is not a model sensor column; choose from {list(SENSORS)}")
        mapping[mine] = model_col
    return mapping


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score a CSV of transformer sensor readings (one row per 15 min).")
    ap.add_argument("--csv", required=True, help="input CSV, oldest row first")
    ap.add_argument("--out", default=None, help="output CSV (default: <input>_scored.csv)")
    ap.add_argument("--map", default=None, help="rename columns: my_col=model_col,my_col2=model_col2")
    ap.add_argument("--timestamp-col", default=None, help="timestamp column (enables the cadence check)")
    ap.add_argument("--resample-15min", action="store_true",
                    help="average readings into 15-minute bins first (needs --timestamp-col)")
    ap.add_argument("--no-explain", action="store_true", help="skip top_reasons (faster)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--meta", default=None)
    args = ap.parse_args(argv)

    src = Path(args.csv)
    df = pd.read_csv(src).rename(columns=_parse_map(args.map))

    if args.timestamp_col:
        if args.timestamp_col not in df.columns:
            raise ValueError(f"timestamp column {args.timestamp_col!r} not in CSV columns {list(df.columns)}")
        df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col])
        df = df.sort_values(args.timestamp_col).reset_index(drop=True)
        if args.resample_15min:
            have = [c for c in SENSORS if c in df.columns]
            df = (df.set_index(args.timestamp_col)[have].resample("15min").mean().reset_index())
        else:
            step = df[args.timestamp_col].diff().dropna().median()
            if pd.notna(step) and abs(step - pd.Timedelta(minutes=15)) > pd.Timedelta(minutes=1):
                print(f"WARNING: median spacing between rows is {step}, but the model expects 15 minutes. "
                      "H2-rate and oil-slope features will be mis-scaled; re-run with --resample-15min.")
    elif args.resample_15min:
        raise ValueError("--resample-15min needs --timestamp-col")

    scorer = HealthScorer(args.model, args.meta, explain=not args.no_explain)
    scored = score_dataframe(df, scorer)

    out = Path(args.out) if args.out else src.with_name(src.stem + "_scored.csv")
    scored.to_csv(out, index=False)
    n = len(scored)
    print(f"scored {n} rows -> {out}")
    print(scored["state_name"].value_counts().to_string())
    print(f"warming-up rows: {int(scored['warming_up'].sum())} | degraded rows: {int(scored['degraded'].sum())} | "
          f"unscorable rows: {int((scored['status'] == 'unscorable').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
