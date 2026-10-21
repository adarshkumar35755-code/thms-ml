"""One-shot TEST-split evaluation for the THMS health-score model.

Run from the project root:

    python -m src.evaluate                      # default artifacts in models/
    python -m src.evaluate --model models/health_model.joblib --meta models/health_model.json
    python -m src.evaluate --allow-rerun        # deliberately re-evaluate the same model

What it does (all on the TEST split only, in a single pass)
-----------------------------------------------------------
1. Headline metrics: score MAE / RMSE / R2, 4-state accuracy and macro-F1,
   confusion matrix, and the dangerous error (true Warning/Critical predicted Normal).
2. Breakdown by fault_type and detection lead time.
3. Robustness: (a) sensor noise doubled, (b) h2_ppm replaced by a 20 %-error
   prediction, (c) h2_ppm removed, (d) vibration removed.
4. Every number above is computed for the model AND a no-learning baseline.
5. SHAP summary plots and `top_reasons(...)`, the dashboard helper that returns
   the features pulling one score down the most.

"Once" is enforced: after a successful run a lock file (reports/.test_eval_lock.json)
records the SHA-256 of the model file. Re-running with the SAME model is refused
unless --allow-rerun is given, so the test set cannot be quietly tuned against.
A different model file (different hash) is allowed.

Decisions I made (please review)
--------------------------------
- Headline metrics exclude each run's warm-up rows (first FEATURE_WINDOW-1 samples,
  where trend features are NaN). The live scorer reports `warming_up` instead of a
  score there, so scoring those rows would be unrealistic.
- Score -> state bands: >75 Normal, >50 Caution, >25 Warning, otherwise Critical
  (matches how the labels were built). Overridden by `band_edges` in the model JSON.
- "Noise doubled" means the TOTAL noise std is 2x the dataset's: extra noise with std
  sqrt(2^2 - 1) = sqrt(3) x the dataset's sigma is added (the data already contains
  1x). Sigmas in NOISE_SIGMAS are copied from the dataset generator.
- "h2 replaced by a prediction" = each h2 reading multiplied by (1 + N(0, 0.20)),
  independent per sample (harsh: it also corrupts the h2 rate-of-rise feature).
- "Removed" sensors are set to NaN (not 0) before features are rebuilt. A model trained
  without NaNs treats NaN as an ordinary value, so "removed" shows how the deployed
  model behaves when a sensor dies without retraining; it is not a "trained without H2" result.
- Detection lead time uses the first SUSTAINED drop below the Normal band
  (LEAD_MIN_CONSEC consecutive samples = 1 h) so single noisy samples don't count.
  lead = t(true first Caution) - t(model first drop); positive = model flagged earlier.
  Runs with no true Caution onset (incl. all "normal" runs) are scored as false-alarm checks.
- The baseline is `baseline_score` below (max of the distance-to-limit features,
  mapped to severity). Pass --baseline-from-train to use `src.train.baseline_score` instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # lets `python src/evaluate.py` work as well as `python -m src.evaluate`
    sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402
from src import data  # noqa: E402
from src import features as F  # noqa: E402

__all__ = ["top_reasons", "get_explainer", "baseline_score", "score_to_state", "main"]

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
BAND_EDGES = (75.0, 50.0, 25.0)  # score > 75 Normal, > 50 Caution, > 25 Warning, else Critical
LEAD_MIN_CONSEC = 4  # samples (= 1 h at 15 min) a drop must persist to count as a detection
NOISE_MULT = 2.0  # scenario (a): total noise std = NOISE_MULT x the dataset's
H2_PRED_REL_ERR = 0.20  # scenario (b): relative error std of the H2 "prediction"
SHAP_SAMPLE = 4000  # rows used for the SHAP summary plots
LOCK_NAME = ".test_eval_lock.json"

# (absolute std, relative std) per sensor: from the dataset generator's sensor model.
NOISE_SIGMAS: dict[str, tuple[float, float]] = {
    cfg.INPUT_CURRENT_A: (0.0, 0.015),
    cfg.OUTPUT_CURRENT_A: (0.0, 0.010),
    cfg.OUTPUT_VOLTAGE_V: (0.0, 0.008),
    cfg.AMBIENT_TEMP_C: (0.30, 0.0),
    cfg.HUMIDITY_PCT: (1.2, 0.0),
    cfg.VIBRATION_RMS: (0.03, 0.03),
    cfg.H2_PPM: (2.0, 0.05),
    cfg.OIL_TEMP_C: (0.15, 0.0),
}

STATE_LABELS = [cfg.HEALTH_STATE_NAMES[i] for i in range(4)]
SYSTEMS = ("model", "baseline")

FEATURE_LABELS = {
    "input_current_A": "HV-side current",
    "output_current_A": "LV-side current",
    "output_voltage_V": "output voltage",
    "ambient_temp_C": "ambient temperature",
    "humidity_pct": "humidity",
    "vibration_rms_ms2": "vibration (100 Hz RMS)",
    "h2_ppm": "H2 concentration",
    "oil_temp_C": "oil temperature",
    "load_pu": "loading (per unit)",
    "oil_rise": "oil rise over ambient",
    "in_out_resid": "input/output current mismatch",
    "abs_in_out_resid": "input/output current mismatch",
    "vib_ratio": "vibration vs expected for this load",
    "d_h2": "H2 distance to limit",
    "d_oil_temp": "oil temp distance to limit",
    "d_load": "load distance to limit",
    "d_vib": "vibration distance to limit",
    "d_abs_resid": "current-mismatch distance to limit",
    "h2_rate_ppm_per_day": "H2 rate of rise",
    "oil_temp_slope_C_per_h": "oil temperature trend",
    "vib_roll_mean": "recent mean vibration",
    "h2_roll_max": "recent peak H2",
}


# --------------------------------------------------------------------------
# Score -> state, baseline
# --------------------------------------------------------------------------
def score_to_state(score, edges: tuple[float, float, float] = BAND_EDGES) -> np.ndarray:
    """0 Normal (> edges[0]), 1 Caution (> edges[1]), 2 Warning (> edges[2]), 3 Critical."""
    s = np.asarray(score, dtype="float64")
    state = np.full(s.shape, 3, dtype=int)
    state[s > edges[2]] = 2
    state[s > edges[1]] = 1
    state[s > edges[0]] = 0
    return state


_D_COLS = ("d_h2", "d_oil_temp", "d_load", "d_vib", "d_abs_resid")


def baseline_score(feats: pd.DataFrame) -> np.ndarray:
    """No-learning baseline: 100 * (1 - worst per-sensor severity).

    Severity per distance feature d (0 at Caution limit, 1 at Critical limit):
    0 while d == 0 (below the Caution limit), then linear 0.25 (Caution) -> 0.75
    (Critical) -> 1.0 at d = DIST_CLIP_MAX. A missing (NaN) sensor contributes no
    evidence (severity 0). Note: it jumps from 100 to 75 when a limit is first crossed.
    """
    d = feats[list(_D_COLS)].to_numpy(dtype="float64")
    sev = np.where(d > 0, np.interp(d, [0.0, 1.0, cfg.DIST_CLIP_MAX], [0.25, 0.75, 1.0]), 0.0)
    sev = np.where(np.isnan(d), 0.0, sev)
    return 100.0 * (1.0 - sev.max(axis=1))


def _get_baseline(from_train: bool):
    if from_train:
        from src import train as T  # noqa: PLC0415

        return T.baseline_score, "src.train.baseline_score"
    return baseline_score, "src.evaluate.baseline_score (distance-to-limit formula)"


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_artifacts(model_arg: str | None, meta_arg: str | None) -> tuple[Path, Path | None]:
    if model_arg:
        model_path = Path(model_arg)
    else:
        default = cfg.MODELS_DIR / "health_model.joblib"
        found = sorted(cfg.MODELS_DIR.glob("*.joblib"))
        if default.exists():
            model_path = default
        elif len(found) == 1:
            model_path = found[0]
        else:
            raise FileNotFoundError(
                f"Pass --model: expected {default} or exactly one *.joblib in {cfg.MODELS_DIR} (found {[p.name for p in found]})"
            )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if meta_arg:
        meta_path: Path | None = Path(meta_arg)
    else:
        candidates = [model_path.with_suffix(".json"), cfg.MODELS_DIR / "model_meta.json"]
        meta_path = next((p for p in candidates if p.exists()), None)
    return model_path, meta_path


def _parse_edges(raw) -> tuple[float, float, float]:
    try:
        vals = list(raw.values()) if isinstance(raw, dict) else list(raw)
        vals = sorted((float(v) for v in vals), reverse=True)
        if len(vals) == 3:
            return (vals[0], vals[1], vals[2])
    except (TypeError, ValueError):
        pass
    print(f"WARNING: could not parse band_edges={raw!r}; using {BAND_EDGES}")
    return BAND_EDGES


def _load_artifacts(model_path: Path, meta_path: Path | None):
    model = joblib.load(model_path)
    meta: dict = {}
    if meta_path is not None and meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        print("WARNING: no model metadata JSON found; using features.FEATURE_NAMES and default bands")
    names = list(meta.get("feature_names") or meta.get("features") or F.FEATURE_NAMES)
    unknown = [n for n in names if n not in F.FEATURE_NAMES]
    if unknown:
        raise ValueError(f"Model expects features that features.py does not produce: {unknown}")
    edges = _parse_edges(meta["band_edges"]) if "band_edges" in meta else BAND_EDGES
    return model, names, edges


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model.predict(X), dtype="float64"), 0.0, 100.0)


# --------------------------------------------------------------------------
# Scenarios (perturb the RAW sensor columns, then rebuild every feature)
# --------------------------------------------------------------------------
def _clean(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return df.copy()


def _noise_doubled(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    extra = float(np.sqrt(NOISE_MULT**2 - 1.0))  # data already holds 1x noise
    for col, (abs_s, rel_s) in NOISE_SIGMAS.items():
        x = out[col].to_numpy(dtype="float64")
        sigma = np.sqrt(abs_s**2 + (rel_s * np.abs(x)) ** 2) * extra
        out[col] = np.maximum(x + rng.standard_normal(len(x)) * sigma, 0.0)
    out[cfg.HUMIDITY_PCT] = out[cfg.HUMIDITY_PCT].clip(upper=100.0)
    return out


def _h2_noisy_prediction(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    h2 = out[cfg.H2_PPM].to_numpy(dtype="float64")
    out[cfg.H2_PPM] = np.maximum(h2 * (1.0 + rng.normal(0.0, H2_PRED_REL_ERR, len(h2))), 0.0)
    return out


def _h2_removed(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    out[cfg.H2_PPM] = np.nan
    return out


def _vib_removed(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    out[cfg.VIBRATION_RMS] = np.nan
    return out


SCENARIOS = {
    "clean": _clean,
    "noise_x2": _noise_doubled,
    "h2_predicted_20pct": _h2_noisy_prediction,
    "h2_removed": _h2_removed,
    "vibration_removed": _vib_removed,
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def dangerous_stats(y_state: np.ndarray, p_state: np.ndarray) -> dict:
    """Dangerous error = true Warning/Critical predicted Normal."""
    sev = y_state >= 2
    bad = sev & (p_state == 0)
    return {
        "n_true_warn_crit": int(sev.sum()),
        "n_dangerous": int(bad.sum()),
        "dangerous_rate": float(bad.sum() / sev.sum()) if sev.any() else float("nan"),
        "dangerous_true_warning": int((bad & (y_state == 2)).sum()),
        "dangerous_true_critical": int((bad & (y_state == 3)).sum()),
    }


def metrics_block(y_score, p_score, y_state, edges=BAND_EDGES) -> dict:
    y_score, p_score, y_state = map(np.asarray, (y_score, p_score, y_state))
    p_state = score_to_state(p_score, edges)
    labels = sorted(set(y_state.tolist()) | set(p_state.tolist()))
    out = {
        "n": int(len(y_score)),
        "mae": float(mean_absolute_error(y_score, p_score)),
        "rmse": float(np.sqrt(mean_squared_error(y_score, p_score))),
        "r2": float(r2_score(y_score, p_score)) if len(y_score) > 1 else float("nan"),
        "state_acc": float((y_state == p_state).mean()),
        "macro_f1": float(f1_score(y_state, p_state, labels=labels, average="macro", zero_division=0)),
    }
    out.update(dangerous_stats(y_state, p_state))
    return out


def _first_sustained(mask: np.ndarray, m: int) -> int | None:
    if len(mask) < m:
        return None
    hits = np.flatnonzero(np.convolve(mask.astype(int), np.ones(m, dtype=int), mode="valid") == m)
    return int(hits[0]) if hits.size else None


def lead_time_table(test: pd.DataFrame, pred: np.ndarray, valid: np.ndarray, edges=BAND_EDGES) -> pd.DataFrame:
    """One row per run: when the label first turns Caution vs when the score first drops below the Normal band."""
    rows = []
    for rid, pos in test.groupby(cfg.RUN_ID, sort=False).indices.items():
        pos = pos[valid[pos]]
        sub = test.iloc[pos]
        t = sub[cfg.T_HOURS].to_numpy()
        state = sub[cfg.HEALTH_STATE].to_numpy()
        below = pred[pos] < edges[0]
        i_true = int(np.argmax(state >= 1)) if (state >= 1).any() else None
        i_sus = _first_sustained(below, LEAD_MIN_CONSEC)
        i_raw = int(np.argmax(below)) if below.any() else None
        t_true = float(t[i_true]) if i_true is not None else np.nan
        t_pred = float(t[i_sus]) if i_sus is not None else np.nan
        rows.append(
            {
                "run_id": int(rid),
                "fault_type": sub[cfg.FAULT_TYPE].iloc[0],
                "has_onset": i_true is not None,
                "true_caution_h": t_true,
                "model_drop_h": t_pred,
                "first_any_drop_h": float(t[i_raw]) if i_raw is not None else np.nan,
                "detected": i_sus is not None,
                "lead_h": t_true - t_pred if (i_true is not None and i_sus is not None) else np.nan,
                "frac_samples_below_75": float(below.mean()),
            }
        )
    return pd.DataFrame(rows)


def lead_summary(lt: pd.DataFrame) -> dict:
    onset = lt[lt["has_onset"]]
    none = lt[~lt["has_onset"]]
    lead = onset["lead_h"].dropna()
    return {
        "runs_with_onset": int(len(onset)),
        "detection_rate": float(onset["detected"].mean()) if len(onset) else float("nan"),
        "median_lead_h": float(lead.median()) if len(lead) else float("nan"),
        "mean_lead_h": float(lead.mean()) if len(lead) else float("nan"),
        "pct_flagged_early_or_on_time": float((lead >= 0).mean() * 100) if len(lead) else float("nan"),
        "runs_without_onset": int(len(none)),
        "false_alarm_run_rate": float(none["detected"].mean()) if len(none) else float("nan"),
    }


# --------------------------------------------------------------------------
# SHAP: explanations for the dashboard
# --------------------------------------------------------------------------
_EXPLAINERS: dict[int, object] = {}


def _model_feature_names(model) -> list[str]:
    for attr in ("feature_name_", "feature_names_in_"):
        v = getattr(model, attr, None)
        if v is not None:
            return list(v)
    try:
        return list(model.feature_name())  # lightgbm.Booster
    except Exception:  # noqa: BLE001
        return list(F.FEATURE_NAMES)


def get_explainer(model):
    """Cached shap.TreeExplainer (build it once at dashboard start-up)."""
    key = id(model)
    if key not in _EXPLAINERS:
        import shap  # noqa: PLC0415

        _EXPLAINERS[key] = shap.TreeExplainer(model)
    return _EXPLAINERS[key]


def top_reasons(model, feature_row, k: int = 3, explainer=None) -> list[dict]:
    """The k features pulling ONE score down the most (SHAP, in score points).

    `feature_row`: one row of build_features output (pd.Series, 1-row DataFrame or dict).
    Returns a list (most harmful first) of
        {"feature", "label", "value", "impact_points"}  with impact_points < 0.
    Fewer than k items are returned if fewer features lower the score; an empty
    list means nothing is pulling the score below its baseline.
    """
    names = _model_feature_names(model)
    if isinstance(feature_row, pd.DataFrame):
        if len(feature_row) != 1:
            raise ValueError("feature_row must hold exactly one row")
        row = feature_row.iloc[0]
    elif isinstance(feature_row, dict):
        row = pd.Series(feature_row)
    else:
        row = feature_row
    X = pd.DataFrame([row.reindex(names).astype("float64").to_numpy()], columns=names)

    explainer = explainer or get_explainer(model)
    sv = explainer.shap_values(X)
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(-1)
    order = np.argsort(sv)  # most negative first
    out = []
    for i in order[:k]:
        if sv[i] >= 0:
            break
        out.append(
            {
                "feature": names[i],
                "label": FEATURE_LABELS.get(names[i], names[i]),
                "value": None if np.isnan(X.iloc[0, i]) else float(X.iloc[0, i]),
                "impact_points": float(sv[i]),
            }
        )
    return out


def _shap_plots(model, X: pd.DataFrame, out_dir: Path) -> None:
    import shap  # noqa: PLC0415

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    if len(X) > SHAP_SAMPLE:
        X = X.iloc[np.sort(rng.choice(len(X), SHAP_SAMPLE, replace=False))]
    sv = get_explainer(model).shap_values(X)
    sv = sv[0] if isinstance(sv, list) else sv

    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.title("SHAP: what moves the health score (points)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=130)
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=15)
    plt.title("Mean |SHAP| (score points)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _plot_confusions(cms: dict[str, np.ndarray], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sysname in zip(axes, SYSTEMS):
        cm = cms[sysname]
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.1f}%", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
        for i in (2, 3):  # dangerous cells: true Warning/Critical predicted Normal
            ax.add_patch(Rectangle((-0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=3))
        ax.set_xticks(range(4), STATE_LABELS)
        ax.set_yticks(range(4), STATE_LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{sysname} (red = dangerous: true Warning/Critical -> Normal)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrices.png", dpi=130)
    plt.close(fig)


def _plot_by_fault(by_fault: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (col, title) in zip(axes, [("mae", "Score MAE (points)"), ("state_acc", "State accuracy")]):
        piv = by_fault.pivot(index="fault_type", columns="system", values=col)[list(SYSTEMS)]
        piv.plot.bar(ax=ax, rot=30)
        ax.set_title(title)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_by_fault_type.png", dpi=130)
    plt.close(fig)


def _plot_robustness(rob: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    spec = [("mae", "Score MAE (points)"), ("state_acc", "State accuracy"),
            ("dangerous_rate", "Dangerous-miss rate (true Warn/Crit -> Normal)")]
    for ax, (col, title) in zip(axes, spec):
        piv = rob.pivot(index="scenario", columns="system", values=col).reindex(list(SCENARIOS))[list(SYSTEMS)]
        piv.plot.bar(ax=ax, rot=25)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "robustness.png", dpi=130)
    plt.close(fig)


def _plot_lead(lt_by_sys: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.38
    faults = [f for f in cfg.KNOWN_FAULT_TYPES if f != "normal"]
    for k, sysname in enumerate(SYSTEMS):
        lt = lt_by_sys[sysname]
        vals = [lt[(lt.fault_type == f) & lt.has_onset]["lead_h"].median() for f in faults]
        ax.bar(np.arange(len(faults)) + (k - 0.5) * width, vals, width, label=sysname)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(faults)), faults, rotation=25)
    ax.set_ylabel("median lead time (h)\n+ = flagged before the label turns Caution")
    ax.set_title("Detection lead time (clean test data; NaN bars = never detected)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lead_time.png", dpi=130)
    plt.close(fig)


def _plot_example_runs(test: pd.DataFrame, preds: dict[str, np.ndarray], out_dir: Path) -> None:
    picks = []
    for ft in cfg.KNOWN_FAULT_TYPES:
        ids = test.loc[test[cfg.FAULT_TYPE] == ft, cfg.RUN_ID].unique()
        if len(ids):
            picks.append((ft, ids[0]))
    n = len(picks)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.4 * rows), squeeze=False)
    for ax, (ft, rid) in zip(axes.ravel(), picks):
        pos = np.flatnonzero((test[cfg.RUN_ID] == rid).to_numpy())
        t = test[cfg.T_HOURS].to_numpy()[pos] / 24
        ax.plot(t, test[cfg.HEALTH_SCORE].to_numpy()[pos], "k", lw=1.5, label="true")
        ax.plot(t, preds["model"][pos], lw=0.9, label="model")
        ax.plot(t, preds["baseline"][pos], lw=0.9, alpha=0.8, label="baseline")
        for edge in BAND_EDGES:
            ax.axhline(edge, color="gray", ls=":", lw=0.7)
        ax.set_title(f"{ft} (run {rid})", fontsize=10)
        ax.set_ylim(-2, 102)
        ax.set_xlabel("days")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "example_runs.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _fmt(df: pd.DataFrame, digits: int = 3) -> str:
    return df.round(digits).to_string()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=None, help="CSV path (default: config.DATA_PATH)")
    ap.add_argument("--model", default=None, help="joblib model file (default: models/health_model.joblib)")
    ap.add_argument("--meta", default=None, help="model metadata JSON (feature_names, band_edges)")
    ap.add_argument("--out", default=None, help="reports directory (default: config.REPORTS_DIR)")
    ap.add_argument("--allow-rerun", action="store_true", help="re-evaluate a model already evaluated on TEST")
    ap.add_argument("--baseline-from-train", action="store_true", help="use src.train.baseline_score")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else cfg.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path, meta_path = _resolve_artifacts(args.model, args.meta)
    model_hash = _sha256(model_path)
    lock_path = out_dir / LOCK_NAME
    if lock_path.exists() and not args.allow_rerun:
        lock = json.loads(lock_path.read_text())
        if lock.get("model_sha256") == model_hash:
            print(
                f"REFUSED: this exact model was already evaluated on the TEST split ({lock.get('timestamp')}).\n"
                "Re-running invites tuning against the test set. Use --allow-rerun if you really mean it."
            )
            return 2

    model, names, edges = _load_artifacts(model_path, meta_path)
    baseline_fn, baseline_name = _get_baseline(args.baseline_from_train)
    print(f"model: {model_path.name} | baseline: {baseline_name} | bands: {edges}")

    # ---- TEST split only (train/val are dropped immediately)
    _train, _val, test = data.load_splits(args.data, verbose=False)
    del _train, _val
    if not (test[cfg.SPLIT] == "test").all():
        raise RuntimeError("Non-test rows reached the evaluation set")
    test = test.reset_index(drop=True)
    valid = ~F.warmup_mask(test).to_numpy()
    print(f"TEST: {len(test)} rows, {test[cfg.RUN_ID].nunique()} runs; "
          f"{int((~valid).sum())} warm-up rows excluded from metrics")

    y_score = test[cfg.HEALTH_SCORE].to_numpy()
    y_state = test[cfg.HEALTH_STATE].to_numpy()
    fault = test[cfg.FAULT_TYPE].to_numpy()

    # ---- run every scenario once: perturb raw sensors -> rebuild features -> predict
    preds: dict[str, dict[str, np.ndarray]] = {}
    clean_feats = None
    for k, (name, fn) in enumerate(SCENARIOS.items()):
        rng = np.random.default_rng(cfg.RANDOM_SEED + 1000 + k)
        feats = F.build_features_by_run(fn(test, rng))
        if name == "clean":
            clean_feats = feats
        preds[name] = {
            "model": _predict(model, feats[names]),
            "baseline": np.clip(np.asarray(baseline_fn(feats), dtype="float64"), 0.0, 100.0),
        }

    # ---- 1. headline metrics + confusion matrices (clean)
    overall_rows, cms = [], {}
    for sysname in SYSTEMS:
        p = preds["clean"][sysname]
        r = metrics_block(y_score[valid], p[valid], y_state[valid], edges)
        r["system"] = sysname
        overall_rows.append(r)
        cms[sysname] = confusion_matrix(y_state[valid], score_to_state(p[valid], edges), labels=[0, 1, 2, 3])
    overall = pd.DataFrame(overall_rows).set_index("system")

    # ---- 2. by fault type + lead time
    bf_rows = []
    for ft in cfg.KNOWN_FAULT_TYPES:
        m = (fault == ft) & valid
        if not m.any():
            continue
        for sysname in SYSTEMS:
            r = metrics_block(y_score[m], preds["clean"][sysname][m], y_state[m], edges)
            r.update(fault_type=ft, system=sysname)
            bf_rows.append(r)
    by_fault = pd.DataFrame(bf_rows)

    lt_clean = {s: lead_time_table(test, preds["clean"][s], valid, edges) for s in SYSTEMS}
    lt_all = pd.concat([lt_clean[s].assign(system=s) for s in SYSTEMS], ignore_index=True)

    dangerous_runs = []
    for sysname in SYSTEMS:
        ps = score_to_state(preds["clean"][sysname], edges)
        bad = valid & (y_state >= 2) & (ps == 0)
        if bad.any():
            g = test.loc[bad].groupby([cfg.RUN_ID, cfg.FAULT_TYPE]).size().rename("n_dangerous_rows").reset_index()
            dangerous_runs.append(g.assign(system=sysname))
    dangerous_by_run = pd.concat(dangerous_runs, ignore_index=True) if dangerous_runs else pd.DataFrame()

    # ---- 3. robustness (all scenarios, both systems)
    rob_rows = []
    for name in SCENARIOS:
        for sysname in SYSTEMS:
            p = preds[name][sysname]
            r = metrics_block(y_score[valid], p[valid], y_state[valid], edges)
            r.update(lead_summary(lead_time_table(test, p, valid, edges)))
            r.update(scenario=name, system=sysname)
            rob_rows.append(r)
    rob = pd.DataFrame(rob_rows)
    base = rob[rob.scenario == "clean"].set_index("system")
    for col in ("mae", "state_acc", "macro_f1", "dangerous_rate", "median_lead_h", "false_alarm_run_rate"):
        rob[f"d_{col}"] = rob[col] - rob["system"].map(base[col])
    rob["mae_increase_pct"] = 100.0 * rob["d_mae"] / rob["system"].map(base["mae"])

    # ---- 4. console report
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n=== Headline metrics (TEST, warm-up excluded) ===")
    print(_fmt(overall[["n", "mae", "rmse", "r2", "state_acc", "macro_f1"]]))
    print("\n=== DANGEROUS ERROR: true Warning/Critical predicted Normal ===")
    print(_fmt(overall[["n_true_warn_crit", "n_dangerous", "dangerous_rate",
                        "dangerous_true_warning", "dangerous_true_critical"]]))
    if len(dangerous_by_run):
        print("where it happens:")
        print(dangerous_by_run.to_string(index=False))
    for sysname in SYSTEMS:
        print(f"\nConfusion matrix ({sysname}); rows = true, cols = predicted {STATE_LABELS}")
        print(pd.DataFrame(cms[sysname], index=STATE_LABELS, columns=STATE_LABELS).to_string())
    print("\n=== By fault type ===")
    print(_fmt(by_fault.pivot(index="fault_type", columns="system",
                              values=["mae", "state_acc", "n_dangerous"])))
    print("\n=== Detection lead time, clean data (lead_h > 0: flagged before the label turns Caution) ===")
    print(_fmt(lt_all[["system", "run_id", "fault_type", "has_onset", "true_caution_h",
                       "model_drop_h", "lead_h", "detected", "frac_samples_below_75"]]
               .sort_values(["system", "fault_type", "run_id"]).set_index("system"), 2))
    print("\n=== Robustness (deltas are vs the same system on clean data) ===")
    cols = ["scenario", "system", "mae", "d_mae", "mae_increase_pct", "state_acc", "d_state_acc",
            "dangerous_rate", "d_dangerous_rate", "detection_rate", "median_lead_h", "false_alarm_run_rate"]
    print(_fmt(rob[cols].set_index("scenario")))

    # ---- 5. files
    overall.to_csv(out_dir / "test_metrics_overall.csv")
    by_fault.to_csv(out_dir / "test_metrics_by_fault_type.csv", index=False)
    lt_all.to_csv(out_dir / "test_lead_time_by_run.csv", index=False)
    rob.to_csv(out_dir / "test_robustness.csv", index=False)
    if len(dangerous_by_run):
        dangerous_by_run.to_csv(out_dir / "test_dangerous_errors_by_run.csv", index=False)
    for sysname in SYSTEMS:
        pd.DataFrame(cms[sysname], index=STATE_LABELS, columns=STATE_LABELS).to_csv(
            out_dir / f"test_confusion_{sysname}.csv")
    (out_dir / "test_metrics.json").write_text(json.dumps(
        {"model_file": model_path.name, "model_sha256": model_hash, "baseline": baseline_name,
         "band_edges": list(edges), "overall": overall.reset_index().to_dict("records"),
         "robustness": rob.to_dict("records")}, indent=2, default=float))

    _plot_confusions(cms, out_dir)
    _plot_by_fault(by_fault, out_dir)
    _plot_robustness(rob, out_dir)
    _plot_lead(lt_clean, out_dir)
    _plot_example_runs(test, preds["clean"], out_dir)
    try:
        _shap_plots(model, clean_feats.loc[valid, names], out_dir)
        sample_idx = int(np.flatnonzero(valid & (y_state >= 2))[0]) if (valid & (y_state >= 2)).any() else int(np.flatnonzero(valid)[0])
        print("\n=== Example dashboard explanation (top 3 reasons for one Warning/Critical sample) ===")
        print(f"row {sample_idx}: true score {y_score[sample_idx]:.1f}, "
              f"model score {preds['clean']['model'][sample_idx]:.1f}")
        for r in top_reasons(model, clean_feats.iloc[sample_idx][names]):
            print(f"  {r['impact_points']:+.1f} pts  {r['label']} ({r['feature']} = {r['value']:.4g})")
    except Exception as exc:  # noqa: BLE001 - SHAP problems must not discard the metrics above
        print(f"\nWARNING: SHAP step failed ({type(exc).__name__}: {exc}); all other reports were saved.")

    lock_path.write_text(json.dumps(
        {"model_sha256": model_hash, "model_file": model_path.name,
         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "n_test_rows": int(len(test))}, indent=2))
    print(f"\nReports written to {out_dir}  (test lock: {lock_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
