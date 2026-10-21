"""Replay one dataset run through HealthScorer, exactly as a live dashboard would, and plot it.

    python -m src.replay                                  # a faulty run from the TEST split
    python -m src.replay --fault-type thermal_fault
    python -m src.replay --run-id 61
    python -m src.replay --split val                      # use another split
    python -m src.replay --dropout h2_ppm:150:12          # H2 sensor dead for 12 h starting at hour 150

Each row is sent to `HealthScorer.update` one at a time (no look-ahead). The script
then rebuilds the features for the whole run in one batch and checks that streaming
and batch scores are identical, prints a summary, and saves
reports/replay_run<ID>.png (score vs time, states, key sensors, warm-up and degraded spans).

This is a demonstration of the live path, not a model-selection tool: do not tune the
model on what you see here for the test split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402
from src import data  # noqa: E402
from src import features as F  # noqa: E402
from src.infer import HealthScorer  # noqa: E402

STATE_COLORS = {0: "#2e7d32", 1: "#f9a825", 2: "#ef6c00", 3: "#c62828"}


def pick_run(df: pd.DataFrame, run_id: int | None, fault_type: str | None) -> int:
    if run_id is not None:
        if run_id not in set(df[cfg.RUN_ID]):
            raise SystemExit(f"run_id {run_id} is not in this split; available: {sorted(df[cfg.RUN_ID].unique())}")
        return int(run_id)
    if fault_type:
        ids = df.loc[df[cfg.FAULT_TYPE] == fault_type, cfg.RUN_ID].unique()
        if not len(ids):
            raise SystemExit(f"no run with fault_type {fault_type!r} in this split")
        return int(ids[0])
    faulty = df[df[cfg.FAULT_TYPE] != "normal"]
    # default: the faulty run that reaches the worst health state (most informative plot)
    worst = faulty.groupby(cfg.RUN_ID)[cfg.HEALTH_STATE].max().sort_values(ascending=False)
    return int(worst.index[0] if len(worst) else df[cfg.RUN_ID].iloc[0])


def apply_dropout(run: pd.DataFrame, spec: str | None) -> tuple[pd.DataFrame, np.ndarray]:
    """'sensor:start_h:duration_h' -> sensor readings become NaN in that window."""
    mask = np.zeros(len(run), dtype=bool)
    if not spec:
        return run, mask
    sensor, start, dur = spec.split(":")
    if sensor not in cfg.INPUT_FEATURES:
        raise SystemExit(f"--dropout sensor must be one of {list(cfg.INPUT_FEATURES)}")
    t = run[cfg.T_HOURS].to_numpy()
    mask = (t >= float(start)) & (t < float(start) + float(dur))
    run = run.copy()
    run.loc[mask, sensor] = np.nan
    return run, mask


def stream(scorer: HealthScorer, run: pd.DataFrame) -> pd.DataFrame:
    scorer.reset()
    rows = []
    for rec in run[list(cfg.INPUT_FEATURES)].to_dict("records"):
        r = scorer.update(rec)
        rows.append(
            {
                "score": r["health_score"],
                "state": r["health_state"],
                "warming_up": r["warming_up"],
                "degraded": r["degraded"],
                "status": r["status"],
                "top": r["top_reasons"][0]["label"] if r["top_reasons"] else "",
            }
        )
    return pd.DataFrame(rows, index=run.index)


def make_plot(run: pd.DataFrame, res: pd.DataFrame, title: str, path: Path) -> None:
    t = run[cfg.T_HOURS].to_numpy() / 24.0
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1, 2]})
    # --- score
    for lo, hi, c in [(75, 100, STATE_COLORS[0]), (50, 75, STATE_COLORS[1]),
                      (25, 50, STATE_COLORS[2]), (0, 25, STATE_COLORS[3])]:
        ax1.axhspan(lo, hi, color=c, alpha=0.08)
    ax1.plot(t, run[cfg.HEALTH_SCORE], color="k", lw=1.6, label="true score (label)")
    ax1.plot(t, res["score"], color="tab:blue", lw=1.0, label="streamed HealthScorer score")
    warm = res["warming_up"].to_numpy()
    if warm.any():
        ax1.axvspan(t[warm][0], t[warm][-1], color="gray", alpha=0.25, label="warming up (trend features NaN)")
    deg = res["degraded"].to_numpy()
    if deg.any():
        ax1.scatter(t[deg], np.full(deg.sum(), 2.5), marker="|", color="red", s=60, label="degraded (sensor fallback)")
    ax1.set_ylim(0, 102)
    ax1.set_ylabel("health score")
    ax1.set_title(title)
    ax1.legend(loc="lower left", fontsize=8)

    # --- state strip
    st = res["state"].astype("float").to_numpy()
    ax2.scatter(t, np.where(np.isnan(st), -1, st), c=[STATE_COLORS.get(int(s), "#9e9e9e") if s == s else "#9e9e9e"
                                                       for s in st], s=4)
    ax2.set_yticks([-1, 0, 1, 2, 3], ["Unknown", "Normal", "Caution", "Warning", "Critical"])
    ax2.set_ylabel("state")

    # --- sensors
    ax3.plot(t, run[cfg.H2_PPM], color="tab:purple", lw=0.9, label="H2 (ppm)")
    ax3.set_ylabel("H2 (ppm)", color="tab:purple")
    ax3b = ax3.twinx()
    ax3b.plot(t, run[cfg.OIL_TEMP_C], color="tab:red", lw=0.9, label="oil temp (C)")
    ax3b.plot(t, run[cfg.VIBRATION_RMS] * 20, color="tab:green", lw=0.9, label="vibration x20 (m/s2)")
    ax3b.set_ylabel("oil temp (C) / vibration x20")
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax3.set_xlabel("days")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="test", choices=list(cfg.SPLIT_NAMES))
    ap.add_argument("--run-id", type=int, default=None)
    ap.add_argument("--fault-type", default=None, choices=list(cfg.KNOWN_FAULT_TYPES))
    ap.add_argument("--dropout", default=None, help="sensor:start_h:duration_h, e.g. h2_ppm:150:12")
    ap.add_argument("--data", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out", default=None, help="PNG path (default: reports/replay_run<ID>.png)")
    args = ap.parse_args(argv)

    splits = dict(zip(cfg.SPLIT_NAMES, data.load_splits(args.data, verbose=False)))
    df = splits[args.split]
    rid = pick_run(df, args.run_id, args.fault_type)
    run = df[df[cfg.RUN_ID] == rid].reset_index(drop=True)
    fault = run[cfg.FAULT_TYPE].iloc[0]
    print(f"replaying run {rid} ({fault}) from the {args.split} split: {len(run)} samples")

    run, drop_mask = apply_dropout(run, args.dropout)
    scorer = HealthScorer(args.model, args.meta)
    res = stream(scorer, run)

    # batch reference on the same (possibly dropout-affected) clean data only when nothing was dropped
    if not drop_mask.any():
        batch = scorer.predict_batch(F.build_features_by_run(run))
        diff = np.abs(res["score"].to_numpy(dtype=float) - batch)
        print(f"streaming vs batch: max |diff| = {diff.max():.2e} score points "
              f"({'MATCH' if diff.max() < 1e-9 else 'MISMATCH'})")
    else:
        print(f"dropout simulated on {int(drop_mask.sum())} samples; batch comparison skipped "
              "(batch features would see NaN, streaming holds the last value)")

    scored = res["score"].notna()
    mae = float(np.abs(res.loc[scored, "score"] - run.loc[scored, cfg.HEALTH_SCORE]).mean())
    first_true = run.loc[run[cfg.HEALTH_STATE] >= 1, cfg.T_HOURS]
    first_pred = run.loc[res["state"].fillna(0) >= 1, cfg.T_HOURS]
    print(f"MAE vs label: {mae:.3f} | warming-up samples: {int(res['warming_up'].sum())} | "
          f"degraded: {int(res['degraded'].sum())} | unscorable: {int((res['status'] == 'unscorable').sum())}")
    print("first Caution-or-worse at hour: "
          f"label {first_true.iloc[0] if len(first_true) else 'never'} | streamed "
          f"{first_pred.iloc[0] if len(first_pred) else 'never'}")
    print("final sample:", {k: res.iloc[-1][k] for k in ("score", "state", "top")})

    out = Path(args.out) if args.out else cfg.REPORTS_DIR / f"replay_run{rid}.png"
    make_plot(run, res, f"Replay of run {rid} ({fault}, {args.split} split)"
              + (f" with {args.dropout} dropout" if args.dropout else ""), out)
    print(f"plot saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
