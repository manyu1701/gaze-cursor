"""Vertical-bias diagnosis (M2 step 3, last part). Pure functions; the gate is in PREREGISTRATIONS.md.

Targets are grouped into bands by screen position (split at the midpoints between the v2 grid rows
or columns). Per band: signed bias in degrees (vertical: positive = predicted too LOW), error
mean / median / p90, and bias vs within-fixation spread, with bootstrap CIs over targets.
"""

from __future__ import annotations

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.metrics import cm_to_deg, px_to_cm
from gaze_cursor.eval.stats import TargetStats, bootstrap_ci
from gaze_cursor.eval.targets import calibration_grid_v2

ROW_NAMES = ("top", "upper", "lower", "bottom")
COL_NAMES = ("left", "centre-left", "centre-right", "right")
GATE_MIN_DEG = 0.5   # pre-specified in PREREGISTRATIONS.md (vertical-bias decision gate)


def band_edges(cfg: ScreenConfig, axis: int) -> np.ndarray:
    """The 3 split points between the 4 v2 grid rows (axis 1) or columns (axis 0)."""
    lines = np.unique(np.round(calibration_grid_v2(cfg)[:, axis], 6))
    return (lines[:-1] + lines[1:]) / 2


def band_of(values: np.ndarray, cfg: ScreenConfig, axis: int = 1) -> np.ndarray:
    """Band index 0-3 (top to bottom, or left to right) for screen coordinates on ``axis``."""
    return np.searchsorted(band_edges(cfg, axis), np.asarray(values, dtype=float), side="right")


def signed_deg(px: np.ndarray, cfg: ScreenConfig) -> np.ndarray:
    """Signed visual angle of an on-screen offset in points."""
    px = np.asarray(px, dtype=float)
    return np.sign(px) * np.asarray(cm_to_deg(px_to_cm(np.abs(px), cfg), cfg))


def band_report(t: TargetStats, cfg: ScreenConfig, axis: int = 1, n_boot: int = 10_000,
                seed: int = 0) -> list[dict]:
    """One dict per band: n targets, signed bias along ``axis`` (deg, mean + CI), error stats (deg),
    mean |bias| (2-D) and mean spread (deg), so bias and noise can be told apart."""
    band = band_of(t.xy[:, axis], cfg, axis)
    signed = signed_deg(t.bias_px[:, axis], cfg)
    bias2d = signed_deg(np.linalg.norm(t.bias_px, axis=1), cfg)
    spread = signed_deg(t.spread_px, cfg)
    names = ROW_NAMES if axis == 1 else COL_NAMES
    out = []
    for b, name in enumerate(names):
        m = band == b
        if not m.any():
            out.append({"band": name, "n": 0})
            continue
        lo, hi = bootstrap_ci(signed[m], n_boot, seed)
        err = t.err_deg[m]
        out.append({"band": name, "n": int(m.sum()), "bias_deg": float(signed[m].mean()), "ci": (lo, hi),
                    "err_mean": float(err.mean()), "err_median": float(np.median(err)),
                    "err_p90": float(np.percentile(err, 90)), "abs_bias_deg": float(bias2d[m].mean()),
                    "spread_deg": float(np.nanmean(spread[m]))})
    return out


def gate(rows: list[dict], min_deg: float = GATE_MIN_DEG) -> tuple[bool, dict[str, bool]]:
    """The pre-specified gate on a vertical ``band_report``: is the top band predicted too low?"""
    top, bottom = rows[0], rows[-1]
    if not top.get("n") or not bottom.get("n"):
        raise ValueError("the gate needs targets in both the top and the bottom band")
    crit = {
        f"top band bias >= +{min_deg} deg": top["bias_deg"] >= min_deg,
        "top band CI lower bound > 0": top["ci"][0] > 0,
        f"top exceeds bottom by >= {min_deg} deg": top["bias_deg"] - bottom["bias_deg"] >= min_deg,
    }
    return all(crit.values()), crit
