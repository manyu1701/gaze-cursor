"""Session quality metrics (M3 step 6b). Pure functions: numbers only, no thresholds.

Every function takes a loaded ``Session`` and returns plain floats (NaN = not computable for this
kind of session, e.g. no test phase in a quick calibration; a NaN metric is not scored). The
thresholds that turn numbers into pass / warn / fail are written in PLAN.md (M3 step 6b) from the
existing sessions before the report is used on a new one.

All errors are within-session (calibration -> test targets, or leave-one-target-out): neither is
cross-session accuracy, which stays the headline number elsewhere.

The unit of data is a target (fixation), never a frame: held-out scores are per-target means and
the held-out fit is leave-one-target-out (or calibration -> test targets), never a random frame split.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.configs import REGISTRY
from gaze_cursor.eval.metrics import cm_to_deg, px_to_cm
from gaze_cursor.eval.session import CALIB, TEST, Session
from gaze_cursor.eval.stats import TargetStats, per_target

FIT_CONFIG = "gaze_a10"  # the model Run / Cursor mode use (quick.QUICK_CONFIG)
OUTLIER_DEG = 8.0        # a target further than this from where it was looked at is an outlier target
MIN_FRAMES = 10         # same as quick.MIN_DOT_FRAMES: fewer valid frames on a target is a bad target
CONDITION_TAGS = ("camera", "glasses", "person_id", "distance_cm", "posture_note")  # CLAUDE.md "Generalisation readiness"


def refusal(s: Session) -> str | None:
    """Why this session must not be scored by the report, or None.

    Round 4 (PREREGISTRATIONS.md) was frozen without the quality report, so the calibrations that
    start Cursor-mode sessions (note ``quick calibration (Cursor)``) are not reported on until the
    H6 test has run. Sessions on a distorted canvas have biased target positions (``is_clean``).
    """
    if "(Cursor)" in str(s.meta.get("note", "")):
        return "Cursor-mode calibration: round-4 data, not reported until H6 has run"
    if not s.is_clean:
        return "targets were not drawn 1:1 on the display (unclean window)"
    return None


def screen_of(s: Session) -> ScreenConfig:
    return ScreenConfig(**s.meta["screen"])


def _px_to_deg(px: float, cfg: ScreenConfig) -> float:
    return float(cm_to_deg(px_to_cm(px, cfg), cfg))


def _target_xy(s: Session, phase: int) -> np.ndarray:
    """One (x, y) per target of ``phase`` (all rows, valid or not, so a lost target still counts)."""
    m = s.phase == phase
    idx, first = np.unique(s.target_idx[m], return_index=True)
    return s.target_xy[m][first] if len(idx) else np.zeros((0, 2))


def coverage(s: Session) -> dict[str, float]:
    """Do the calibration targets span the tested area, including near the edges?"""
    cfg = screen_of(s)
    cal, test = _target_xy(s, CALIB), _target_xy(s, TEST)
    out = {"n_calib": float(len(cal)), "n_test": float(len(test))}
    if len(cal) < 3:
        return out | {"calib_extent_x": float("nan"), "calib_extent_y": float("nan"),
                      "test_outside_hull": float("nan")}
    out["calib_extent_x"] = float(np.ptp(cal[:, 0]) / cfg.width_px)
    out["calib_extent_y"] = float(np.ptp(cal[:, 1]) / cfg.height_px)
    if len(test):
        try:
            outside = Delaunay(cal).find_simplex(test) < 0
            out["test_outside_hull"] = float(outside.mean())
        except Exception:  # collinear calibration targets
            out["test_outside_hull"] = 1.0
    else:
        out["test_outside_hull"] = float("nan")
    return out


def fixation_quality(s: Session) -> dict[str, float]:
    """Valid-frame share, thin targets and dropped frames."""
    valid = s.valid
    out: dict[str, float] = {"valid_fraction": float(valid.mean()) if len(valid) else 0.0}
    keys = np.unique(s.phase.astype(np.int64) * 100_000 + s.target_idx)
    counts = np.array([(valid & (s.phase.astype(np.int64) * 100_000 + s.target_idx == k)).sum() for k in keys])
    out["thin_targets"] = float((counts < MIN_FRAMES).sum())
    out["min_target_frames"] = float(counts.min()) if len(counts) else 0.0
    # Frame gaps within one target's recording only: the pause between targets is not a dropped frame.
    key = s.phase.astype(np.int64) * 100_000 + s.target_idx
    gaps = np.diff(s.t_ms.astype(float))[key[1:] == key[:-1]]
    gaps = gaps[gaps > 0]
    out["dropped_gap_share"] = float((gaps > 2.0 * np.median(gaps)).mean()) if len(gaps) else float("nan")
    return out


def _fit_predict(s: Session, train: np.ndarray, test: np.ndarray, cfg: ScreenConfig) -> np.ndarray:
    x, y, _, _ = s.select_all(REGISTRY[FIT_CONFIG].feature_set)
    model = REGISTRY[FIT_CONFIG].make_model(cfg).fit(x[train], y[train])
    return model.predict(x[test])


def held_out_stats(s: Session) -> tuple[TargetStats | None, TargetStats | None]:
    """(calibration -> test targets, leave-one-target-out over every target), per-target stats.

    Either is None when it cannot be computed (no test phase, or fewer than 3 targets).
    """
    cfg = screen_of(s)
    _, y, tidx, phase = s.select_all(REGISTRY[FIT_CONFIG].feature_set)
    tkey = phase.astype(np.int64) * 100_000 + tidx
    within = None
    cal, tst = np.flatnonzero(phase == CALIB), np.flatnonzero(phase == TEST)
    if len(cal) and len(tst):
        within = per_target(_fit_predict(s, cal, tst, cfg), y[tst], tkey[tst], np.zeros(len(tst), int), cfg)
    loto = None
    keys = np.unique(tkey)
    if len(keys) >= 3:
        pred = np.empty_like(y)
        for k in keys:
            te = tkey == k
            pred[te] = _fit_predict(s, np.flatnonzero(~te), np.flatnonzero(te), cfg)
        loto = per_target(pred, y, tkey, np.zeros(len(y), int), cfg)
    return within, loto


def error_summary(t: TargetStats | None, prefix: str) -> dict[str, float]:
    """Mean / median / p90 of per-target error (deg), worst target, and the share of targets worse
    than ``OUTLIER_DEG`` (a share, so 9-dot and 25-target sessions are comparable)."""
    nan = float("nan")
    if t is None or len(t.err_deg) == 0:
        return {f"{prefix}_{k}": nan for k in ("mean", "median", "p90", "worst", "far_share")}
    e = t.err_deg
    return {f"{prefix}_mean": float(e.mean()), f"{prefix}_median": float(np.median(e)),
            f"{prefix}_p90": float(np.percentile(e, 90)), f"{prefix}_worst": float(e.max()),
            f"{prefix}_far_share": float((e > OUTLIER_DEG).mean())}


def bias_vs_noise(t: TargetStats | None, cfg: ScreenConfig) -> dict[str, float]:
    """Per-target bias (mean prediction - target) against within-fixation spread, and the compression
    slope (1.0 = none, < 1 = predictions pulled toward the screen centre) per axis."""
    nan = float("nan")
    if t is None or len(t.err_deg) < 3:
        return {"bias_deg": nan, "spread_deg": nan, "bias_share": nan, "slope_x": nan, "slope_y": nan}
    bias_px = np.linalg.norm(t.bias_px, axis=1)
    bias_deg = float(np.mean([_px_to_deg(b, cfg) for b in bias_px]))
    spread_deg = float(np.nanmean([_px_to_deg(v, cfg) for v in t.spread_px]))
    centre = np.array([cfg.width_px / 2, cfg.height_px / 2])
    tc = t.xy - centre
    pc = tc + t.bias_px  # mean prediction relative to the centre
    slopes = []
    for a in (0, 1):
        d = float(tc[:, a] @ tc[:, a])
        slopes.append(float(tc[:, a] @ pc[:, a] / d) if d > 0 else nan)
    return {"bias_deg": bias_deg, "spread_deg": spread_deg,
            "bias_share": bias_deg ** 2 / (bias_deg ** 2 + spread_deg ** 2) if spread_deg == spread_deg else nan,
            "slope_x": slopes[0], "slope_y": slopes[1]}


def _rot_angle_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    def unit(r: np.ndarray) -> np.ndarray:
        return r / np.linalg.norm(r, axis=0, keepdims=True)
    c = (np.trace(unit(r1).T @ unit(r2)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def drift_and_head(s: Session, loto: TargetStats | None) -> dict[str, float]:
    """Head movement between the first and last quarter of the recording, and error early vs late."""
    nan = float("nan")
    out = {"head_shift_cm": nan, "head_rot_deg": nan, "late_minus_early_deg": nan}
    if s.transform is not None:
        ok = s.valid & ~np.isnan(s.transform).reshape(len(s.transform), -1).any(axis=1)
    else:
        ok = np.zeros(len(s.valid), bool)
    if s.transform is not None and ok.sum() >= 8:
        tr = s.transform[ok]
        q = max(len(tr) // 4, 1)
        first, last = tr[:q], tr[-q:]
        out["head_shift_cm"] = float(np.linalg.norm(np.median(last[:, :3, 3], axis=0) - np.median(first[:, :3, 3], axis=0)))
        out["head_rot_deg"] = _rot_angle_deg(np.median(first[:, :3, :3], axis=0), np.median(last[:, :3, :3], axis=0))
    if loto is not None and len(loto.err_deg) >= 6:
        valid = s.valid
        tkey = s.phase.astype(np.int64) * 100_000 + s.target_idx
        t_med = np.array([np.median(s.t_ms[valid & (tkey == k)]) for k in _loto_keys(s)])
        if len(t_med) == len(loto.err_deg):
            order = np.argsort(t_med)
            h = len(order) // 2
            out["late_minus_early_deg"] = float(loto.err_deg[order[h:]].mean() - loto.err_deg[order[:h]].mean())
    return out


def _loto_keys(s: Session) -> np.ndarray:
    """Target keys in the order ``held_out_stats`` aggregates them (np.unique order, valid rows only)."""
    _, _, tidx, phase = s.select_all(REGISTRY[FIT_CONFIG].feature_set)
    return np.unique(phase.astype(np.int64) * 100_000 + tidx)


def conditions(s: Session) -> dict[str, float | str]:
    """Lighting gate inputs and which structured condition tags the session carries."""
    raw = s.meta.get("lighting")
    light: dict = raw if isinstance(raw, dict) else {}
    out: dict[str, float | str] = {
        "face_brightness": float(light.get("face", float("nan"))),
        "lighting_ratio": float(light.get("ratio", float("nan"))),
        "lighting_label": str(light.get("label", "")),
        "tags_present": float(sum(1 for k in CONDITION_TAGS if s.meta.get(k) not in (None, ""))),
        "tags_expected": float(len(CONDITION_TAGS)),
    }
    return out


def all_metrics(s: Session) -> dict[str, float | str]:
    """Every number the report needs for one session."""
    cfg = screen_of(s)
    within, loto = held_out_stats(s)
    # Bias, spread (= jitter) and compression slope always come from leave-one-target-out over every
    # target, so full and quick sessions are measured the same way. For a 9-dot quick session an
    # edge dot left out is extrapolated, which makes its slope and error pessimistic.
    return {**coverage(s), **fixation_quality(s), **error_summary(within, "heldout"),
            **error_summary(loto, "loto"), **bias_vs_noise(loto, cfg),
            **drift_and_head(s, loto), **conditions(s)}
