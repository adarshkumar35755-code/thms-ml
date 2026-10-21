"""Data loading, validation, leakage guards and run-level splitting for THMS.

This module contains no feature engineering. Feature code lives in
src/features.py and must call `assert_no_forbidden_columns` on the final model
input matrix, both in training and in live inference.

Run from the project root, e.g. `python -m src.data`.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

import config as cfg

__all__ = [
    "DataValidationError",
    "LeakageError",
    "SplitError",
    "load_raw",
    "load_dataset",
    "load_splits",
    "validate",
    "find_forbidden_columns",
    "assert_no_forbidden_columns",
    "get_model_inputs",
    "split_by_run",
    "verify_no_run_overlap",
    "class_balance",
    "print_class_balance",
]


class DataValidationError(ValueError):
    """The dataset violates the expected schema, dtypes, values or structure."""


class LeakageError(ValueError):
    """A label or meta column (or something derived from one) is in the model inputs."""


class SplitError(ValueError):
    """The train/val/test split is invalid (empty split or run_id overlap)."""


_KIND_CHECKS = {
    "int": lambda s: ptypes.is_integer_dtype(s) and not ptypes.is_bool_dtype(s),
    "num": lambda s: ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s),
    "str": ptypes.is_string_dtype,
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the CSV as-is, with no validation or reordering."""
    path = Path(path) if path is not None else cfg.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path} (set THMS_DATA_PATH to override)")
    return pd.read_csv(path)


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load, order by (run_id, t_hours), and validate the full dataset."""
    df = load_raw(path)
    if {cfg.RUN_ID, cfg.T_HOURS} <= set(df.columns):
        df = df.sort_values([cfg.RUN_ID, cfg.T_HOURS], kind="stable").reset_index(drop=True)
    validate(df)
    return df


def load_splits(
    path: str | Path | None = None, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, validate and split by run. Returns (train, val, test)."""
    df = load_dataset(path)
    train, val, test = split_by_run(df)
    if verbose:
        print_class_balance({"train": train, "val": val, "test": test})
    return train, val, test


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _raise_if(problems: list[str]) -> None:
    if problems:
        raise DataValidationError("Dataset validation failed:\n  - " + "\n  - ".join(problems))


def validate(df: pd.DataFrame) -> None:
    """Check columns, dtypes, NaNs, value ranges, vocabularies and run structure.

    Raises DataValidationError listing every problem found in a stage.
    Row order is NOT changed; the time-grid check assumes rows are already in
    time order within each run (load_dataset sorts before calling this).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"validate expects a DataFrame, got {type(df).__name__}")
    if df.empty:
        raise DataValidationError("Dataset is empty")
    if df.columns.has_duplicates:
        raise DataValidationError(f"Duplicate columns: {df.columns[df.columns.duplicated()].tolist()}")

    missing = [c for c in cfg.ALL_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    problems: list[str] = []
    extra = [c for c in df.columns if c not in cfg.ALL_COLUMNS]
    if extra:
        problems.append(f"Unexpected columns: {extra}")

    # dtypes
    bad_dtype: set[str] = set()
    for col, kind in cfg.COLUMN_KINDS.items():
        if not _KIND_CHECKS[kind](df[col]):
            problems.append(f"{col}: dtype {df[col].dtype} is not '{kind}'")
            bad_dtype.add(col)

    # NaNs
    nan_counts = df.isna().sum()
    for col, n in nan_counts[nan_counts > 0].items():
        problems.append(f"{col}: {int(n)} NaN values")

    # value ranges (inf also fails here)
    for col, (lo, hi) in cfg.VALUE_RANGES.items():
        if col in bad_dtype:
            continue
        s = df[col]
        out = s.notna() & ~s.between(lo, hi)
        if out.any():
            problems.append(
                f"{col}: {int(out.sum())} values outside [{lo}, {hi}] "
                f"(observed min {s.min():.6g}, max {s.max():.6g})"
            )

    # vocabularies
    vocab = {
        cfg.SPLIT: cfg.SPLIT_NAMES,
        cfg.FAULT_TYPE: cfg.KNOWN_FAULT_TYPES,
        cfg.HEALTH_STATE: tuple(cfg.HEALTH_STATE_NAMES),
    }
    for col, allowed in vocab.items():
        bad = df[col].notna() & ~df[col].isin(allowed)
        if bad.any():
            problems.append(f"{col}: unexpected values {sorted(map(str, df.loc[bad, col].unique()))[:5]}")

    # health_state <-> health_state_name consistency
    expected_name = df[cfg.HEALTH_STATE].map(cfg.HEALTH_STATE_NAMES)
    mismatch = expected_name.notna() & (expected_name != df[cfg.HEALTH_STATE_NAME])
    if mismatch.any():
        problems.append(f"{cfg.HEALTH_STATE_NAME}: {int(mismatch.sum())} rows do not match {cfg.HEALTH_STATE}")

    _raise_if(problems)

    # run structure (only meaningful once the columns above are clean)
    by_run = df.groupby(cfg.RUN_ID)
    sizes = by_run.size()
    wrong_len = sizes[sizes != cfg.SAMPLES_PER_RUN]
    if len(wrong_len):
        problems.append(
            f"{len(wrong_len)} run(s) do not have {cfg.SAMPLES_PER_RUN} rows, e.g. "
            f"{wrong_len.head(5).to_dict()}"
        )

    dt = by_run[cfg.T_HOURS].diff().dropna()
    off_grid = ~np.isclose(dt, cfg.SAMPLE_PERIOD_H, atol=1e-6)
    if off_grid.any():
        bad_runs = sorted(df.loc[dt.index[off_grid], cfg.RUN_ID].unique().tolist())[:10]
        problems.append(
            f"t_hours is not a strictly increasing {cfg.SAMPLE_PERIOD_H} h grid within run "
            f"(runs: {bad_runs})"
        )

    split_per_run = by_run[cfg.SPLIT].nunique()
    straddling = split_per_run[split_per_run > 1]
    if len(straddling):
        problems.append(f"{len(straddling)} run(s) span more than one split: {straddling.index.tolist()[:10]}")

    _raise_if(problems)


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------
def find_forbidden_columns(columns: Iterable[str]) -> list[str]:
    """Return the names that are label/meta columns or start with a reserved prefix."""
    found = []
    for name in columns:
        low = str(name).lower()
        if name in cfg.FORBIDDEN_COLUMNS or low.startswith(tuple(p.lower() for p in cfg.FORBIDDEN_PREFIXES)):
            found.append(str(name))
    return found


def assert_no_forbidden_columns(columns_or_frame: pd.DataFrame | pd.Index | Iterable[str]) -> None:
    """Raise LeakageError if any label or meta column ends up in the model input matrix.

    Accepts a DataFrame or an iterable of column names. NumPy arrays carry no
    names, so they are rejected: check the names before converting.
    """
    if isinstance(columns_or_frame, np.ndarray):
        raise TypeError("Pass column names or a DataFrame, not a bare ndarray")
    names = list(columns_or_frame.columns) if isinstance(columns_or_frame, pd.DataFrame) else list(columns_or_frame)
    if not names:
        raise LeakageError("Model input matrix has no columns")
    offenders = find_forbidden_columns(names)
    if offenders:
        raise LeakageError(f"Label/meta columns found in model inputs: {offenders}")


# Fail at import time if someone edits INPUT_FEATURES in config.py to include a label or meta column.
assert_no_forbidden_columns(cfg.INPUT_FEATURES)


def get_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the 8 raw sensor columns, after running the leakage guard."""
    cols = list(cfg.INPUT_FEATURES)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing input columns: {missing}")
    X = df.loc[:, cols].copy()
    assert_no_forbidden_columns(X)
    return X


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def verify_no_run_overlap(splits: Mapping[str, pd.DataFrame]) -> None:
    """Raise SplitError if any run_id appears in more than one split."""
    names = list(splits)
    run_sets = {n: set(splits[n][cfg.RUN_ID].unique()) for n in names}
    problems = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = run_sets[a] & run_sets[b]
            if shared:
                ids = sorted(int(r) for r in shared)[:10]
                problems.append(f"{len(shared)} run_id(s) appear in both '{a}' and '{b}': {ids}")
    if problems:
        raise SplitError("Run leakage between splits:\n  - " + "\n  - ".join(problems))


def split_by_run(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on the provided `split` column (never by random row). Returns (train, val, test)."""
    for col in (cfg.RUN_ID, cfg.SPLIT):
        if col not in df.columns:
            raise SplitError(f"Column '{col}' is required to split")
    unknown = set(df[cfg.SPLIT].unique()) - set(cfg.SPLIT_NAMES)
    if unknown:
        raise SplitError(f"Unknown split labels: {sorted(map(str, unknown))}")

    splits = {name: df.loc[df[cfg.SPLIT] == name].reset_index(drop=True) for name in cfg.SPLIT_NAMES}
    empty = [name for name, part in splits.items() if part.empty]
    if empty:
        raise SplitError(f"Empty split(s): {empty}")

    verify_no_run_overlap(splits)
    if sum(len(p) for p in splits.values()) != len(df):
        raise SplitError("Rows were lost or duplicated while splitting")
    return splits["train"], splits["val"], splits["test"]


# --------------------------------------------------------------------------
# Class balance
# --------------------------------------------------------------------------
def class_balance(splits: Mapping[str, pd.DataFrame], column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts, percentages) with one row per class and one column per split."""
    if column == cfg.HEALTH_STATE:
        classes: list = list(cfg.HEALTH_STATE_NAMES)
    elif column == cfg.FAULT_TYPE:
        classes = list(cfg.KNOWN_FAULT_TYPES)
    else:
        raise ValueError(f"class_balance supports {cfg.HEALTH_STATE!r} and {cfg.FAULT_TYPE!r}, got {column!r}")
    counts = pd.DataFrame({name: part[column].value_counts() for name, part in splits.items()})
    counts = counts.reindex(classes).fillna(0).astype(int)
    pct = counts.div(counts.sum(axis=0), axis=1) * 100
    return counts, pct


def print_class_balance(splits: Mapping[str, pd.DataFrame]) -> None:
    """Print split sizes and the health_state / fault_type balance per split."""
    print("\n=== Split sizes ===")
    sizes = pd.DataFrame(
        {
            "rows": {n: len(p) for n, p in splits.items()},
            "runs": {n: p[cfg.RUN_ID].nunique() for n, p in splits.items()},
        }
    )
    print(sizes.to_string())

    for column in (cfg.HEALTH_STATE, cfg.FAULT_TYPE):
        counts, pct = class_balance(splits, column)
        shown = counts.astype(str) + " (" + pct.round(1).astype(str) + "%)"
        if column == cfg.HEALTH_STATE:
            shown.index = [f"{k} {cfg.HEALTH_STATE_NAMES[k]}" for k in counts.index]
        print(f"\n=== Class balance: {column} ===")
        print(shown.to_string())
        absent = [(str(c), s) for c in counts.index for s in counts.columns if counts.loc[c, s] == 0]
        if absent:
            print(f"WARNING: classes absent from a split: {absent}")


if __name__ == "__main__":
    load_splits()
