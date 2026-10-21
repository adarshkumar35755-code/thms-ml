"""Central configuration for the Transformer Health Monitoring System (THMS) ML pipeline.

Everything that other modules need to agree on lives here: column names, the
meta/label/input partition, transformer constants, safe-limit thresholds,
window sizes, validation ranges and the random seed. Nothing in this file
computes features; feature code belongs in src/features.py only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("THMS_DATA_PATH", PROJECT_ROOT / "data" / "thms_dataset.csv"))
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "thms_lgbm.joblib"
MODEL_META_PATH = MODELS_DIR / "thms_model_meta.json"
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.csv"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Column names
# --------------------------------------------------------------------------
# Meta columns: never model inputs.
RUN_ID = "run_id"
SPLIT = "split"
T_HOURS = "t_hours"
HOUR_OF_DAY = "hour_of_day"

# Raw sensor columns: the ONLY columns allowed as model inputs.
INPUT_CURRENT_A = "input_current_A"
OUTPUT_CURRENT_A = "output_current_A"
OUTPUT_VOLTAGE_V = "output_voltage_V"
AMBIENT_TEMP_C = "ambient_temp_C"
HUMIDITY_PCT = "humidity_pct"
VIBRATION_RMS = "vibration_rms_ms2"
H2_PPM = "h2_ppm"
OIL_TEMP_C = "oil_temp_C"

# Label columns: never model inputs.
FAULT_TYPE = "fault_type"
HEALTH_STATE = "health_state"
HEALTH_STATE_NAME = "health_state_name"
HEALTH_SCORE = "health_score"

INPUT_FEATURES: tuple[str, ...] = (
    INPUT_CURRENT_A,
    OUTPUT_CURRENT_A,
    OUTPUT_VOLTAGE_V,
    AMBIENT_TEMP_C,
    HUMIDITY_PCT,
    VIBRATION_RMS,
    H2_PPM,
    OIL_TEMP_C,
)
META_COLUMNS: tuple[str, ...] = (RUN_ID, SPLIT, T_HOURS, HOUR_OF_DAY)
LABEL_COLUMNS: tuple[str, ...] = (FAULT_TYPE, HEALTH_STATE, HEALTH_STATE_NAME, HEALTH_SCORE)
ALL_COLUMNS: tuple[str, ...] = META_COLUMNS + INPUT_FEATURES + LABEL_COLUMNS

# Leakage guard: exact names plus prefixes, so derived names such as
# "health_score_lag1" or "hour_of_day_sin" are caught as well.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(META_COLUMNS + LABEL_COLUMNS)
FORBIDDEN_PREFIXES: tuple[str, ...] = (RUN_ID, SPLIT, T_HOURS, HOUR_OF_DAY, "fault_", "health_")

# Expected pandas kind per column: "int", "num" (int or float) or "str".
COLUMN_KINDS: dict[str, str] = {
    RUN_ID: "int",
    SPLIT: "str",
    T_HOURS: "num",
    HOUR_OF_DAY: "num",
    **{col: "num" for col in INPUT_FEATURES},
    FAULT_TYPE: "str",
    HEALTH_STATE: "int",
    HEALTH_STATE_NAME: "str",
    HEALTH_SCORE: "num",
}

# --------------------------------------------------------------------------
# Categorical vocabularies
# --------------------------------------------------------------------------
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
HEALTH_STATE_NAMES: dict[int, str] = {0: "Normal", 1: "Caution", 2: "Warning", 3: "Critical"}
KNOWN_FAULT_TYPES: tuple[str, ...] = (
    "normal",
    "overload",
    "cooling_degradation",
    "thermal_fault",
    "partial_discharge",
    "winding_fault",
    "core_mechanical",
)

# --------------------------------------------------------------------------
# Transformer constants: 200 kVA, 11 kV / 415 V
# --------------------------------------------------------------------------
RATED_KVA = 200.0
HV_VOLTAGE_V = 11_000.0
LV_VOLTAGE_V = 415.0
RATED_LV_CURRENT_A = 278.0
TURNS_RATIO = 26.5


class Limits(NamedTuple):
    """Safe-limit thresholds for one quantity."""

    caution: float
    warning: float
    critical: float


# Caution / Warning / Critical. The residual is a fraction (0.05 = 5 %).
THRESHOLDS: dict[str, Limits] = {
    "h2_ppm": Limits(100.0, 700.0, 1800.0),  # IEEE C57.104 bands
    "oil_temp_C": Limits(80.0, 95.0, 105.0),
    "load_pu": Limits(1.1, 1.3, 1.5),  # output_current_A / RATED_LV_CURRENT_A
    "vibration_rms_ms2": Limits(2.0, 3.3, 5.0),
    "input_current_residual": Limits(0.05, 0.10, 0.20),
}

# --------------------------------------------------------------------------
# Time base and feature window (used by src/features.py)
# --------------------------------------------------------------------------
SAMPLE_PERIOD_H = 0.25  # 15-minute samples
SAMPLES_PER_DAY = int(round(24 / SAMPLE_PERIOD_H))  # 96
SAMPLES_PER_RUN = 1344  # 14 days x 96 samples per day
# Trailing window in samples (current sample + 23 past ones = 6 h). The live
# dashboard buffer must hold at least this many samples to get trend features.
FEATURE_WINDOW = 24

# --------------------------------------------------------------------------
# Constants used inside feature formulas (src/features.py)
# --------------------------------------------------------------------------
NO_LOAD_CURRENT_FRAC = 0.015  # no-load (magnetising) current, fraction of rated HV current
HV_RATED_CURRENT_A = 10.5  # rated HV-side current, 200 kVA / (sqrt(3) * 11 kV)
NO_LOAD_CURRENT_A = NO_LOAD_CURRENT_FRAC * HV_RATED_CURRENT_A  # 0.1575 A
VIB_COEF_VOLTAGE = 0.55  # expected vibration = 0.55*(V/415)^2 + 0.35*load_pu^2
VIB_COEF_LOAD = 0.35
DIST_CLIP_MAX = 1.5  # normalized distance-to-limit is clipped to [0, DIST_CLIP_MAX]

# --------------------------------------------------------------------------
# Score -> state bands (shared by training, evaluation and live inference)
# --------------------------------------------------------------------------
SCORE_MIN = 0.0  # predicted scores are clipped to [SCORE_MIN, SCORE_MAX]
SCORE_MAX = 100.0
# Edges as (normal_above, caution_from, warning_from):
#   score >  75        -> 0 Normal
#   50 <= score <= 75  -> 1 Caution
#   25 <= score <  50  -> 2 Warning
#   score <  25        -> 3 Critical
STATE_BAND_EDGES: tuple[float, float, float] = (75.0, 50.0, 25.0)

# --------------------------------------------------------------------------
# Transparent baseline score (no learning; see src/scoring.py)
# --------------------------------------------------------------------------
# Where the Warning threshold sits on the normalized-distance axis
# (0 = Caution threshold, 1 = Critical threshold), averaged over the five
# distance features. Derived from THRESHOLDS only, never from data.
BASELINE_D_WARNING: float = sum(
    (THRESHOLDS[k].warning - THRESHOLDS[k].caution) / (THRESHOLDS[k].critical - THRESHOLDS[k].caution)
    for k in THRESHOLDS
) / len(THRESHOLDS)
# The distance features are exactly 0 at or below the Caution threshold, so a
# quantity that has just crossed it must already land on the Normal/Caution edge
# (score 75). BASELINE_ONSET_D is the d at which severity reaches 0.25: a short,
# steep ramp from 100 (d = 0) down to 75 (d = BASELINE_ONSET_D).
BASELINE_ONSET_D = 0.02
# Piecewise-linear severity curve over d = max(distance features), as (d, severity)
# knots, severity in [0, 1]. Score = 100 * (1 - severity):
#   d = 0 -> 100 | onset -> 75 | average Warning position -> 50 |
#   Critical threshold (d = 1) -> 25 | clip limit -> 0.
BASELINE_SEVERITY_KNOTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.00),
    (BASELINE_ONSET_D, 0.25),
    (BASELINE_D_WARNING, 0.50),
    (1.0, 0.75),
    (DIST_CLIP_MAX, 1.00),
)

# --------------------------------------------------------------------------
# Validation ranges (inclusive). These are physical-plausibility bounds, not
# the safe limits above: a faulty transformer must still pass validation.
# --------------------------------------------------------------------------
VALUE_RANGES: dict[str, tuple[float, float]] = {
    RUN_ID: (0, 1_000_000),
    T_HOURS: (0.0, SAMPLES_PER_RUN * SAMPLE_PERIOD_H),
    HOUR_OF_DAY: (0.0, 23.999),
    INPUT_CURRENT_A: (0.0, 50.0),
    OUTPUT_CURRENT_A: (0.0, 600.0),
    OUTPUT_VOLTAGE_V: (300.0, 500.0),
    AMBIENT_TEMP_C: (-20.0, 60.0),
    HUMIDITY_PCT: (0.0, 100.0),
    VIBRATION_RMS: (0.0, 20.0),
    H2_PPM: (0.0, 10_000.0),
    OIL_TEMP_C: (0.0, 200.0),
    HEALTH_STATE: (0, 3),
    HEALTH_SCORE: (0.0, 100.0),
}

