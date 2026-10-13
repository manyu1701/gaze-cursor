"""Pre-registered round 3 (PROGRESS.md): head position. Every threshold here is the written protocol.

H4: the current approach (REF trained on the centre session only) loses accuracy when the head moves.
H5 (as amended before set B, PROGRESS.md): at a new position, a fresh recalibration on
    PREREG3_DOTS quick dots beats REF trained on all other sessions (no dots), scored on the
    position's other 25 - PREREG3_DOTS targets.

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
PREREG3_CANDIDATE: str | None = "gaze_a10"
PREREG3_FREEZE_AFTER: str | None = "2026-10-13T13:12:00"   # local time of the freeze commit
PREREG3_REFERENCE = "gaze_a10"
# Amendment (before set B): the candidate is a fresh fit on this many dots of the test session, chosen
# by eval.correction.choose_dots from its calibration targets; REF gets no dots.
PREREG3_DOTS = 9
PREREG3_METHOD = "fresh"
# An EARLIER, separate freeze of the same round (claude.ai session, commit dfa75ff, 05:57 IST), made on a
# diverged branch the protocol above was written without: candidate gaze_a100 trained on all other
# calibrations (pooled, no dots) vs REF on the set-B centre session only. Set B (13:37-14:05) postdates
# both freezes. Kept so it can be run once and reported next to the protocol above.
EARLIER_FREEZE_CANDIDATE = "gaze_a100"
EARLIER_FREEZE_AFTER = "2026-10-13T06:00:00"

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


def run_h5(a: PositionSet, b: PositionSet, ref: Config, cand: Config, screen: ScreenConfig,
           dots: int = PREREG3_DOTS, method: str = PREREG3_METHOD) -> dict:
    """Amended H5 on set B. For each displaced session T: REF trained on all of set A + set B except T,
    no dots; CAND = ``method`` with ``dots`` dots of T (``fresh``: fitted on those dots only). Both are
    scored on T's other 25 - ``dots`` targets, paired by target."""
    b_order = ["C", *DISPLACED, "C2"]
    sessions = [b.sessions[k] for k in b_order] + [a.sessions[k] for k in SLOTS_A]
    test_idx = list(range(1, 1 + len(DISPLACED)))
    c2 = len(DISPLACED) + 1
    def pool(t: int) -> list[int]:
        return [i for i in range(len(sessions)) if i != t]
    r = concat([dot_strategy(sessions, ref, pool(t), t, screen, dots, "none") for t in test_idx])
    c = concat([dot_strategy(sessions, cand, pool(t), t, screen, dots, method) for t in test_idx])
    comp = paired_comparison(r, c, N_BOOT, SEED)
    crit = {
        "(a) CI lower bound > 0": comp["ci_lo"] > 0,
        f"(b) mean d >= {H5_MIN_GAIN_DEG} deg": comp["mean_diff_deg"] >= H5_MIN_GAIN_DEG,
        f"(c) d > 0 in >= {H5_MIN_SESSIONS} of 5 test sessions": comp["sessions_new_better"] >= H5_MIN_SESSIONS,
    }
    # Reported, not criteria.
    c_only = concat([dot_strategy(sessions, ref, [0], t, screen, dots, "none") for t in test_idx])
    refit = concat([dot_strategy(sessions, ref, pool(t), t, screen, dots, "refit") for t in test_idx])
    own = concat([own_calibration(sessions, ref, t, screen) for t in test_idx])
    cand_c2 = dot_strategy(sessions, cand, pool(c2), c2, screen, dots, method)
    return {"comparison": comp, "criteria": crit, "supported": all(crit.values()),
            "err_ref": float(r.err_deg.mean()), "err_cand": float(c.err_deg.mean()),
            "per_position": dict(zip(DISPLACED, comp["per_session_diff_deg"])),
            "vs_c_only": paired_comparison(c_only, c, N_BOOT, SEED),
            "vs_refit": paired_comparison(refit, c, N_BOOT, SEED),
            "own_16": float(own.err_deg.mean()), "cand_c2": float(cand_c2.err_deg.mean())}


def run_h5_earlier_freeze(a: PositionSet, b: PositionSet, ref: Config, cand: Config,
                          screen: ScreenConfig) -> dict:
    """H5 as frozen in dfa75ff: CAND trained on all of set A + set B except the test session versus REF
    trained on set-B C only (today's Run), each displaced set-B session held out in turn; paired on its
    targets. Same criteria as ``run_h5``."""
    b_order = ["C", *DISPLACED, "C2"]
    sessions = [b.sessions[k] for k in b_order] + [a.sessions[k] for k in SLOTS_A]
    test_idx = list(range(1, 1 + len(DISPLACED)))
    today = concat([fit_predict(sessions, ref, [0], t, screen) for t in test_idx])
    c = concat(leave_one_position_out(sessions, test_idx, cand, screen))
    comp = paired_comparison(today, c, N_BOOT, SEED)
    crit = {
        "(a) CI lower bound > 0": comp["ci_lo"] > 0,
        f"(b) mean d >= {H5_MIN_GAIN_DEG} deg": comp["mean_diff_deg"] >= H5_MIN_GAIN_DEG,
        f"(c) d > 0 in >= {H5_MIN_SESSIONS} of 5 test sessions": comp["sessions_new_better"] >= H5_MIN_SESSIONS,
    }
    # Reported, not criteria: REF on the same pool, CAND on C2.
    same_pool = concat(leave_one_position_out(sessions, test_idx, ref, screen))
    cand_c2 = leave_one_position_out(sessions, [len(DISPLACED) + 1], cand, screen)[0]
    return {"comparison": comp, "criteria": crit, "supported": all(crit.values()),
            "err_ref": float(today.err_deg.mean()), "err_cand": float(c.err_deg.mean()),
            "per_position": dict(zip(DISPLACED, comp["per_session_diff_deg"])),
            "vs_same_pool": paired_comparison(same_pool, c, N_BOOT, SEED), "cand_c2": float(cand_c2.err_deg.mean())}


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


# --- Live slot guidance for recording set B (calibration checklist) ---------------------------
# Guidance only: whether a session counts is decided afterwards by ``select_set_b`` on the session
# medians. Green needs a margin past the rule, so that a little drift during the recording does not
# drop the median back across it. The "keep steady" limits are heuristics from the set-A lesson
# (sideways shifts also moved the face down), not inclusion rules.
SLOT_AIM = {"N": 1.20, "R": 0.82, "L": 0.10, "Rt": -0.10, "Lo": 0.10}
RATIO_MARGIN, SHIFT_MARGIN, C2_MARGIN = 0.03, 0.015, 0.7
STEADY_SHIFT, STEADY_RATIO = 0.05, 0.12
IPD_CM = 6.3   # only to show shifts in rough centimetres


def _cm(dx: float, dy: float, ref: Position, aspect: float) -> tuple[float, float]:
    """Rough centimetres of a face shift (frame fractions), scaled by the inter-iris distance."""
    per = IPD_CM / ref.distance
    return dx * per, dy * aspect * per


def slot_guide(slot: str, live: Position | None, ref: Position | None, ref_name: str = "pos C",
               aspect: float = 720 / 1280):
    """A checklist step that coaches the user into ``slot`` relative to ``ref`` (set-B C, or set-A C
    when recording C itself). Levels: good (rule met with margin, everything else steady), adjust,
    off. ``aspect`` = frame height / width, for the height shown in cm."""
    from gaze_cursor.posture import CheckItem
    names = {"C": "normal, centred (as in set A)", "N": "closer", "R": "further back", "L": "shifted to your left",
             "Rt": "shifted to your right", "Lo": "lower (slouched)", "C2": "back to your pos C position"}
    title = f"Position {slot}: {names[slot]}"
    if slot == "C":
        need = "No rule for C (it is the reference): match set A's C so set B starts from the same place."
    elif slot == "C2":
        need = "Rule: distance within 5% of pos C and face within 0.03 of it sideways and in height."
    elif slot in ("N", "R"):
        need = (f"Rule: distance {'>=' if slot == 'N' else '<='} x{NEAR_MIN if slot == 'N' else FAR_MAX:.2f} "
                f"of pos C (aim x{SLOT_AIM[slot]:.2f}). Keep your side and height as in pos C.")
    else:
        need = (f"Rule: face moved >= {SHIFT_MIN:.2f} of the frame {'sideways' if slot != 'Lo' else 'down'} "
                f"from pos C (about {abs(_cm(SHIFT_MIN, 0, ref, aspect)[0]) if ref else 4:.0f} cm; aim "
                f"{abs(SLOT_AIM[slot]):.2f}). Keep {'your height' if slot != 'Lo' else 'your side'} and distance.")
    if ref is None:
        return CheckItem("slot", title, need, "off", f"no {ref_name} session yet",
                         f"record {ref_name} first: the other positions are measured against it")
    if live is None:
        return CheckItem("slot", title, need, "off", "no face detected", "look at the camera")
    ratio, dx, dy = live.relative_to(ref)
    xcm, ycm = _cm(dx, dy, ref, aspect)
    detail = (f"vs {ref_name}: distance x{ratio:.2f}   sideways {abs(xcm):.1f} cm {'left' if xcm > 0 else 'right'}"
              f"   height {abs(ycm):.1f} cm {'lower' if ycm > 0 else 'higher'}")
    side = "move to your right" if dx > 0 else "move to your left"
    height = "sit up a little" if dy > 0 else "sit a little lower"
    dist = "move back a little" if ratio > 1 else "move a little closer"
    if slot in ("C", "C2"):
        tol_r, tol_s = C2_DIST_TOL * C2_MARGIN, C2_SHIFT_TOL * C2_MARGIN
        off = [(abs(ratio - 1) / tol_r, dist), (abs(dx) / tol_s, side), (abs(dy) / tol_s, height)]
        worst, hint = max(off)
        level = "good" if worst <= 1 else "adjust" if worst <= 1 / C2_MARGIN else "off"
        return CheckItem("slot", title, need, level, detail, "" if level == "good" else hint)
    if slot in ("N", "R"):
        primary = ratio
        rule_ok = ratio >= NEAR_MIN if slot == "N" else ratio <= FAR_MAX
        margin_ok = ratio >= NEAR_MIN + RATIO_MARGIN if slot == "N" else ratio <= FAR_MAX - RATIO_MARGIN
        go = "move closer" if slot == "N" else "lean / move further back"
        steady = [(abs(dx) / STEADY_SHIFT, side), (abs(dy) / STEADY_SHIFT, height)]
    else:
        primary = {"L": dx, "Rt": -dx, "Lo": dy}[slot]
        rule_ok, margin_ok = primary >= SHIFT_MIN, primary >= SHIFT_MIN + SHIFT_MARGIN
        go = {"L": "shift further to your left", "Rt": "shift further to your right", "Lo": "slouch / sit lower"}[slot]
        steady = [(abs(ratio - 1) / STEADY_RATIO, dist)]
        steady.append((abs(dy) / STEADY_SHIFT, height) if slot != "Lo" else (abs(dx) / STEADY_SHIFT, side))
    s_worst, s_hint = max(steady)
    if not rule_ok:
        return CheckItem("slot", title, need, "off", detail, go)
    if not margin_ok:
        return CheckItem("slot", title, need, "adjust", detail, f"just inside the rule: {go} a bit more")
    if s_worst > 1:
        return CheckItem("slot", title, need, "adjust", detail, f"{s_hint} (keep the rest as in {ref_name})")
    return CheckItem("slot", title, need, "good", detail)


def slot_reference(slot: str, clean: Sequence[Session]) -> tuple[Position | None, str]:
    """The position a slot is guided against: set A's C for C itself, set B's counted C otherwise."""
    if slot == "C":
        s = {x.path.name: x for x in clean}.get(SET_A["C"])
        return (session_position(s) if s else None), "set-A pos C"
    b = select_set_b(clean)
    return b.positions.get("C"), "pos C"


def slot_verdict(session_name: str, clean: Sequence[Session]) -> str:
    """After a recording: does this session count for set B, and if not, why (the frozen rules)."""
    b = select_set_b(clean)
    for slot, s in b.sessions.items():
        if s.path.name == session_name:
            pos = "" if slot == "C" else f" ({describe(b.positions[slot], b.positions['C'])})"
            left = [k for k in SLOTS_B if k not in b.sessions]
            return f"COUNTS as set B {slot}{pos}. " + (f"Still needed: {', '.join(left)}." if left else "Set B is complete.")
    for name, why in b.excluded:
        if name == session_name:
            return f"DOES NOT COUNT: {why}. Re-record this slot (at most 2 retries per slot)."
    for note in b.ignored:
        if session_name in note:
            return "Not used: this slot already has an earlier qualifying session."
    return "Not a set-B session (no slot in the note, or recorded before the freeze)."
