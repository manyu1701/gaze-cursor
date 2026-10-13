"""Pre-registered round 3 (PROGRESS.md): head position. Every threshold here is the written protocol.

H4: the current approach (REF trained on the centre session only) loses accuracy when the head moves.
H5: the frozen candidate, trained on all other calibrations, beats REF trained on the same data on a
    held-out position.

Set A (development) is named explicitly below. Set B (confirmatory) is selected from sessions that
start after the candidate is frozen: slot from the note (a label), confirmed by the measured position
(the evidence), front-lit, >= 80% valid frames, earliest qualifying session per slot.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.configs import Config
from gaze_cursor.eval.prereg2 import lighting_class
from gaze_cursor.eval.correction import apply_correction, choose_dots, fit_correction
from gaze_cursor.eval.session import CALIB, TEST, Session
from gaze_cursor.eval.splits import build_dataset
from gaze_cursor.eval.stats import (
    TargetStats, bootstrap_diff_ci, concat, paired_comparison, per_target)
from gaze_cursor.posture import posture_metrics

DISPLACED = ("N", "R", "L", "Rt", "Lo")
SLOTS_A = ("C", *DISPLACED)
SLOTS_B = ("C", *DISPLACED, "C2")

# Set A, logged in PROGRESS.md (round-3 recording log) before any model was fitted to it.
SET_A = {
    "C": "session_20261013_052100.npz",
    "N": "session_20261013_052253.npz",
    "R": "session_20261013_053547.npz",
    "L": "session_20261013_052751.npz",
    "Rt": "session_20261013_052908.npz",
    "Lo": "session_20261013_053130.npz",
}

# Frozen by the freeze commit (M2 step 3), BEFORE set B is recorded. None = not frozen yet: the
# confirmatory run refuses, and no session can count as set B.
PREREG3_CANDIDATE: str | None = None
PREREG3_FREEZE_AFTER: str | None = None   # local time of the freeze commit, "YYYY-MM-DDTHH:MM:SS"
PREREG3_REFERENCE = "gaze_a10"

NEAR_MIN, FAR_MAX = 1.12, 0.90            # distance relative to C
SHIFT_MIN = 0.07                          # |dx| for L / Rt, dy for Lo (fraction of the frame)
C2_DIST_TOL, C2_SHIFT_TOL = 0.05, 0.03
MIN_VALID = 0.8
H4_MIN_GAP_DEG, H4_MIN_SESSIONS = 1.0, 4
H5_MIN_GAIN_DEG, H5_MIN_SESSIONS = 0.5, 4
N_BOOT, SEED = 10_000, 0


@dataclass(frozen=True)
class Position:
    """Median head position over a session's valid frames (``posture_metrics`` units)."""

    distance: float   # inter-iris distance / frame width (larger = closer)
    x: float          # face centre in the unflipped frame, 0..1 (larger = the user's left)
    y: float          # 0..1, larger = lower

    def relative_to(self, ref: Position) -> tuple[float, float, float]:
        """(distance ratio, dx, dy) relative to ``ref``."""
        return self.distance / ref.distance, self.x - ref.x, self.y - ref.y


def session_position(s: Session) -> Position:
    """Median ``posture_metrics`` over the frames that have landmarks."""
    size = tuple(s.meta.get("image_size", (1280, 720)))
    rows = [posture_metrics(p, t, size) for p, t in zip(s.points, s.transform)
            if not np.isnan(p).any() and not np.isnan(t).any()]
    if not rows:
        raise ValueError(f"{s.path.name}: no frames with landmarks")
    return Position(*(float(np.median([r[k] for r in rows])) for k in ("distance", "x", "y")))


def slot_from_note(note: str | None) -> str | None:
    """The slot a note names: 'pos C', 'C2', 'pos Rt', 'R', ... (case-insensitive), else None."""
    m = re.fullmatch(r"\s*(?:pos(?:ition)?[\s:_-]*)?(c2|c|n|rt|r|lo|l)\b.*", (note or "").lower())
    return {"c": "C", "c2": "C2", "n": "N", "r": "R", "l": "L", "rt": "Rt", "lo": "Lo"}[m.group(1)] if m else None


def position_ok(slot: str, pos: Position, ref: Position) -> bool:
    """Does the measured position satisfy the pre-registered rule for ``slot`` (relative to C)?"""
    ratio, dx, dy = pos.relative_to(ref)
    return {
        "C": True,
        "N": ratio >= NEAR_MIN,
        "R": ratio <= FAR_MAX,
        "L": dx >= SHIFT_MIN,
        "Rt": dx <= -SHIFT_MIN,
        "Lo": dy >= SHIFT_MIN,
        "C2": abs(ratio - 1) <= C2_DIST_TOL and abs(dx) <= C2_SHIFT_TOL and abs(dy) <= C2_SHIFT_TOL,
    }[slot]


def describe(pos: Position, ref: Position | None) -> str:
    if ref is None:
        return f"distance {pos.distance:.3f}, x {pos.x:.3f}, y {pos.y:.3f}"
    ratio, dx, dy = pos.relative_to(ref)
    return f"dist x{ratio:.2f}  dx {dx:+.3f}  dy {dy:+.3f}"


@dataclass
class PositionSet:
    sessions: dict[str, Session] = field(default_factory=dict)        # slot -> session
    positions: dict[str, Position] = field(default_factory=dict)
    excluded: list[tuple[str, str]] = field(default_factory=list)      # (session, reason)
    ignored: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def set_a(clean: Sequence[Session]) -> PositionSet:
    """The named development sessions."""
    by_name = {s.path.name: s for s in clean}
    out = PositionSet()
    for slot in SLOTS_A:
        s = by_name.get(SET_A[slot])
        if s is None:
            out.missing.append(f"set A {slot}: {SET_A[slot]}")
        else:
            out.sessions[slot] = s
            out.positions[slot] = session_position(s)
    return out


def _qualifies(s: Session) -> str | None:
    """Why ``s`` cannot count (lighting, validity, layout), or None if it can."""
    if s.layout != "v2":
        return f"layout {s.layout}"
    if lighting_class(s.meta) != "F":
        li = s.meta.get("lighting")
        return f"lighting {li['label']} (face {li['face']:.0f}) is not front-lit" if li else "no captured lighting"
    if s.valid_fraction < MIN_VALID:
        return f"only {s.valid_fraction:.0%} valid frames"
    return None


def select_set_b(clean: Sequence[Session], freeze_after: str | None = PREREG3_FREEZE_AFTER) -> PositionSet:
    """Set B by the pre-registered rules. Only sessions started after the freeze can count."""
    out = PositionSet()
    if freeze_after is None:
        out.missing.append("the candidate is not frozen yet (set B is recorded after the freeze)")
        return out
    named = set(SET_A.values())
    candidates = [s for s in clean if s.path.name not in named
                  and str(s.meta.get("started", "")) >= freeze_after and slot_from_note(s.meta.get("note"))]
    candidates.sort(key=lambda s: str(s.meta.get("started", "")))
    ref: Position | None = None
    for s in candidates:                                   # C first: it is the reference
        if slot_from_note(s.meta.get("note")) == "C" and _qualifies(s) is None:
            out.sessions["C"], ref = s, session_position(s)
            out.positions["C"] = ref
            break
    for s in candidates:
        slot = slot_from_note(s.meta.get("note"))
        if s is out.sessions.get("C"):
            continue
        why = _qualifies(s)
        if why is None and ref is None:
            why = "no qualifying C session to measure it against"
        if why is None:
            pos = session_position(s)
            if not position_ok(slot, pos, ref):
                why = f"labelled {slot} but measured {describe(pos, ref)} does not satisfy the {slot} rule"
        if why is not None:
            out.excluded.append((s.path.name, why))
        elif slot in out.sessions:
            out.ignored.append(f"{s.path.name} ({slot}: extra qualifying session)")
        else:
            out.sessions[slot], out.positions[slot] = s, pos
    out.missing += [f"set B {slot}" for slot in SLOTS_B if slot not in out.sessions]
    return out


def fit_predict(sessions: Sequence[Session], config: Config, train: Sequence[int], test: int,
                screen: ScreenConfig) -> TargetStats:
    """Fit ``config`` on all targets of sessions ``train`` and score every target of ``test``."""
    ds = build_dataset(sessions, config.feature_set)
    tr = np.flatnonzero(np.isin(ds.session, list(train)))
    te = np.flatnonzero(ds.session == test)
    model = config.make_model(screen).fit(ds.x[tr], ds.y[tr])
    return per_target(model.predict(ds.x[te]), ds.y[te], ds.target[te], ds.session[te], screen)


def leave_one_position_out(sessions: Sequence[Session], test_idx: Sequence[int], config: Config,
                           screen: ScreenConfig) -> list[TargetStats]:
    """For each test session: train on every OTHER session in ``sessions`` (identical pool for every
    config), test on all of its targets."""
    return [fit_predict(sessions, config, [i for i in range(len(sessions)) if i != t], t, screen)
            for t in test_idx]


def run_h4(b: PositionSet, ref: Config, screen: ScreenConfig) -> dict:
    """REF trained on set-B C only; error on the displaced sessions vs on C2."""
    order = ["C", *DISPLACED, "C2"]
    sessions = [b.sessions[k] for k in order]
    stats = [fit_predict(sessions, ref, [0], i, screen) for i in range(1, len(order))]
    displaced, c2 = stats[:-1], stats[-1]
    disp = concat(displaced).err_deg
    lo, hi = bootstrap_diff_ci(disp, c2.err_deg, N_BOOT, SEED)
    means = {k: float(s.err_deg.mean()) for k, s in zip(DISPLACED, displaced)}
    c2_mean = float(c2.err_deg.mean())
    gap = float(disp.mean() - c2_mean)
    worse = sum(m > c2_mean for m in means.values())
    crit = {
        "(a) CI lower bound > 0": lo > 0,
        f"(b) gap >= {H4_MIN_GAP_DEG} deg": gap >= H4_MIN_GAP_DEG,
        f"(c) >= {H4_MIN_SESSIONS} of 5 displaced sessions worse than C2": worse >= H4_MIN_SESSIONS,
    }
    return {"gap": gap, "ci": (lo, hi), "per_position": means, "c2": c2_mean, "n_displaced": int(len(disp)),
            "criteria": crit, "supported": all(crit.values())}


def run_h5(a: PositionSet, b: PositionSet, ref: Config, cand: Config, screen: ScreenConfig) -> dict:
    """Leave-one-position-out on set B, training pool = all of set A + set B except the test session."""
    b_order = ["C", *DISPLACED, "C2"]
    sessions = [b.sessions[k] for k in b_order] + [a.sessions[k] for k in SLOTS_A]
    test_idx = list(range(1, 1 + len(DISPLACED)))
    r = concat(leave_one_position_out(sessions, test_idx, ref, screen))
    c = concat(leave_one_position_out(sessions, test_idx, cand, screen))
    comp = paired_comparison(r, c, N_BOOT, SEED)
    crit = {
        "(a) CI lower bound > 0": comp["ci_lo"] > 0,
        f"(b) mean d >= {H5_MIN_GAIN_DEG} deg": comp["mean_diff_deg"] >= H5_MIN_GAIN_DEG,
        f"(c) d > 0 in >= {H5_MIN_SESSIONS} of 5 test sessions": comp["sessions_new_better"] >= H5_MIN_SESSIONS,
    }
    # Reported, not criteria.
    c_only = concat([fit_predict(sessions, ref, [0], t, screen) for t in test_idx])
    vs_today = paired_comparison(c_only, c, N_BOOT, SEED)
    cand_c2 = leave_one_position_out(sessions, [len(DISPLACED) + 1], cand, screen)[0]
    return {"comparison": comp, "criteria": crit, "supported": all(crit.values()),
            "err_ref": float(r.err_deg.mean()), "err_cand": float(c.err_deg.mean()),
            "per_position": dict(zip(DISPLACED, comp["per_session_diff_deg"])),
            "vs_c_only": vs_today, "cand_c2": float(cand_c2.err_deg.mean())}


def few_dot(sessions: Sequence[Session], config: Config, train: Sequence[int], test: int, screen: ScreenConfig,
            k: int, kind: str = "offset") -> TargetStats:
    """EXPLORATORY: a model trained on ``train`` sessions, corrected with ``k`` dots of session ``test``.

    The dots are k of the test session's calibration targets (centre first, then spread out); the
    error is on its held-out targets only, so the dots are never scored. k = 0: no correction."""
    ds = build_dataset(sessions, config.feature_set)
    tr = np.flatnonzero(np.isin(ds.session, list(train)))
    model = config.make_model(screen).fit(ds.x[tr], ds.y[tr])
    in_s = ds.session == test
    te = np.flatnonzero(in_s & (ds.phase == TEST))
    pred = model.predict(ds.x[te])
    if k:
        cal = np.flatnonzero(in_s & (ds.phase == CALIB))
        keys = np.unique(ds.target[cal])
        xy = np.array([ds.y[cal][ds.target[cal] == t][0] for t in keys])
        dots = keys[choose_dots(xy, k, (screen.width_px / 2, screen.height_px / 2))]
        p_cal = model.predict(ds.x[cal])
        mean_pred = np.array([p_cal[ds.target[cal] == t].mean(axis=0) for t in dots])
        truth = np.array([ds.y[cal][ds.target[cal] == t][0] for t in dots])
        pred = apply_correction(pred, fit_correction(mean_pred, truth, kind))
    return per_target(pred, ds.y[te], ds.target[te], ds.session[te], screen)


def own_calibration(sessions: Sequence[Session], config: Config, test: int, screen: ScreenConfig) -> TargetStats:
    """The session's own 16 calibration targets -> its held-out targets (the ceiling)."""
    ds = build_dataset(sessions, config.feature_set)
    in_s = ds.session == test
    tr, te = np.flatnonzero(in_s & (ds.phase == CALIB)), np.flatnonzero(in_s & (ds.phase == TEST))
    model = config.make_model(screen).fit(ds.x[tr], ds.y[tr])
    return per_target(model.predict(ds.x[te]), ds.y[te], ds.target[te], ds.session[te], screen)


DOT_METHODS = ("none", "offset", "affine", "fresh", "refit")


def dot_strategy(sessions: Sequence[Session], config: Config, train: Sequence[int], test: int,
                 screen: ScreenConfig, k: int, method: str) -> TargetStats:
    """EXPLORATORY: what a user sitting down at a new position gets after looking at ``k`` quick dots.

    The dots are k of the test session's calibration targets (centre first, then spread out). The
    error is on ALL its other targets (25 - k), so every method at the same k is scored on the same
    targets and can be paired. Methods: ``none`` (the pooled model unchanged), ``offset`` / ``affine``
    (a correction fitted on the dots), ``fresh`` (a new model on the dots only, no earlier sessions),
    ``refit`` (the pooled model refitted with the dots added, weighted to half of the data)."""
    if method not in DOT_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {DOT_METHODS}")
    ds = build_dataset(sessions, config.feature_set)
    in_s = ds.session == test
    cal = np.flatnonzero(in_s & (ds.phase == CALIB))
    keys = np.unique(ds.target[cal])
    xy = np.array([ds.y[cal][ds.target[cal] == t][0] for t in keys])
    dots = keys[choose_dots(xy, k, (screen.width_px / 2, screen.height_px / 2))]
    dot_rows = np.flatnonzero(in_s & np.isin(ds.target, dots))
    te = np.flatnonzero(in_s & ~np.isin(ds.target, dots))
    tr = np.flatnonzero(np.isin(ds.session, list(train)))
    if method == "fresh":
        model = config.make_model(screen).fit(ds.x[dot_rows], ds.y[dot_rows])
    elif method == "refit":
        rep = max(1, round(len(tr) / len(dot_rows)))
        rows = np.concatenate([tr, np.tile(dot_rows, rep)])
        model = config.make_model(screen).fit(ds.x[rows], ds.y[rows])
    else:
        model = config.make_model(screen).fit(ds.x[tr], ds.y[tr])
    pred = model.predict(ds.x[te])
    if method in ("offset", "affine"):
        p_dot = model.predict(ds.x[dot_rows])
        mean_pred = np.array([p_dot[ds.target[dot_rows] == t].mean(axis=0) for t in dots])
        truth = np.array([ds.y[dot_rows][ds.target[dot_rows] == t][0] for t in dots])
        pred = apply_correction(pred, fit_correction(mean_pred, truth, method))
    return per_target(pred, ds.y[te], ds.target[te], ds.session[te], screen)
