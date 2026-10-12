"""Per-target aggregation, bootstrap confidence intervals and paired comparisons. Pure functions.

Everything is computed per TARGET: frames within a fixation are highly correlated, so the number
of independent samples is the number of targets, and bootstraps resample targets, not frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.metrics import cm_to_deg, px_to_cm


@dataclass(frozen=True)
class TargetStats:
    """One row per target. Errors are means over the target's frames."""

    session: np.ndarray      # (m,) session index
    target: np.ndarray       # (m,) unique target key
    xy: np.ndarray           # (m, 2) target position (screen points)
    err_px: np.ndarray       # (m,) mean frame error
    err_deg: np.ndarray      # (m,) mean frame error, degrees of visual angle
    err_cm: np.ndarray       # (m,)
    bias_px: np.ndarray      # (m, 2) mean prediction - target
    spread_px: np.ndarray    # (m,) within-fixation std, sqrt(var_x + var_y)
    n_frames: np.ndarray     # (m,)


def per_target(pred: np.ndarray, y: np.ndarray, target: np.ndarray, session: np.ndarray,
               cfg: ScreenConfig) -> TargetStats:
    """Aggregate frame-level predictions to one row per target."""
    keys = np.unique(target)
    rows = {k: np.flatnonzero(target == k) for k in keys}
    err = np.linalg.norm(pred - y, axis=1)
    out = {f: [] for f in ("session", "xy", "err_px", "bias", "spread", "n")}
    for k in keys:
        i = rows[k]
        out["session"].append(session[i[0]]); out["xy"].append(y[i[0]])
        out["err_px"].append(err[i].mean()); out["bias"].append(pred[i].mean(axis=0) - y[i[0]])
        out["spread"].append(np.sqrt(pred[i].var(axis=0, ddof=1).sum()) if len(i) > 1 else np.nan)
        out["n"].append(len(i))
    err_px = np.array(out["err_px"])
    err_cm = np.asarray(px_to_cm(err_px, cfg))
    # Degrees from the per-frame errors (cm_to_deg is not linear), then averaged per target.
    deg_frames = np.asarray(cm_to_deg(px_to_cm(err, cfg), cfg))
    err_deg = np.array([deg_frames[rows[k]].mean() for k in keys])
    return TargetStats(np.array(out["session"]), keys, np.array(out["xy"]), err_px, err_deg,
                       err_cm, np.array(out["bias"]), np.array(out["spread"]), np.array(out["n"]))


def concat(parts: list[TargetStats]) -> TargetStats:
    return TargetStats(*(np.concatenate([getattr(p, f) for p in parts])
                         for f in TargetStats.__dataclass_fields__))


def summarize(t: TargetStats, cfg: ScreenConfig) -> dict[str, float]:
    """Mean / median / p90 over targets, plus jitter (mean within-fixation spread)."""
    jitter_px = float(np.nanmean(t.spread_px)) if np.isfinite(t.spread_px).any() else float("nan")
    return {
        "n_targets": int(len(t.target)), "n_frames": int(t.n_frames.sum()),
        "err_px_mean": float(t.err_px.mean()), "err_px_median": float(np.median(t.err_px)),
        "err_px_p90": float(np.percentile(t.err_px, 90)), "err_cm_mean": float(t.err_cm.mean()),
        "err_deg_mean": float(t.err_deg.mean()), "err_deg_median": float(np.median(t.err_deg)),
        "err_deg_p90": float(np.percentile(t.err_deg, 90)),
        "jitter_px": jitter_px,
        "jitter_deg": float(cm_to_deg(px_to_cm(jitter_px, cfg), cfg)),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of the mean, resampling ``values`` (targets) with replacement."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def bootstrap_ci_clustered(values: np.ndarray, groups: np.ndarray, n_boot: int = 10_000,
                           seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Two-level bootstrap: resample groups (sessions), then targets within each chosen group."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    members = [v[g == k] for k in np.unique(g)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(members), size=len(members))
        means[b] = np.mean(np.concatenate(
            [m[rng.integers(0, len(m), size=len(m))] for m in (members[i] for i in picked)]))
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_comparison(ref: TargetStats, new: TargetStats, n_boot: int = 10_000,
                      seed: int = 0) -> dict[str, float | list]:
    """Paired comparison on identical targets. d = err(ref) - err(new) in degrees, so a positive
    difference means ``new`` is better. Targets must align row for row."""
    if not (np.array_equal(ref.target, new.target) and np.array_equal(ref.session, new.session)):
        raise ValueError("paired comparison needs identical targets in the same order")
    d = ref.err_deg - new.err_deg
    lo, hi = bootstrap_ci(d, n_boot, seed)
    clo, chi = bootstrap_ci_clustered(d, ref.session, n_boot, seed)
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    sessions = np.unique(ref.session)
    per_session = [float(d[ref.session == s].mean()) for s in sessions]
    return {
        "n_targets": int(len(d)), "mean_diff_deg": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
        "cluster_ci_lo": clo, "cluster_ci_hi": chi,
        "effect_size_dz": float(d.mean() / sd) if sd and np.isfinite(sd) else float("nan"),
        "sessions": [int(s) for s in sessions], "per_session_diff_deg": per_session,
        "sessions_new_better": int(sum(x > 0 for x in per_session)),
        "targets_new_better": int((d > 0).sum()),
    }
