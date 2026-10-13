"""Head-transform check (EXPLORATORY; decision rule in PREREGISTRATIONS.md, written before running).

Does the translation of MediaPipe's facial transformation matrix track the real head moves in set B
(laptop fixed), where the eye-corner estimate ``head_x/y`` did not? Fits screen x, y ~ 1 + iris (4) +
yaw + pitch + head x + head y over all frames and compares the head coefficients with the screen
geometry (x: -1 / cm_per_px, y: +1 / cm_per_px points per cm), once per head estimate.

  uv run scripts/diagnose_head_transform.py
"""

from __future__ import annotations

import sys

import numpy as np

from gaze_cursor.eval.prereg3 import DISPLACED, SLOTS_A, select_set_b, set_a
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.eval.runner import common_screen
from gaze_cursor.eval.session import Session, clean_sessions

RATIO_LO, RATIO_HI = 0.5, 2.0   # pre-registered: within a factor of 2 of the geometry
MIN_AXIS_R = 0.5                # pre-registered: |r| with head_x / head_y below this = does not track


def transform_translation(transform: np.ndarray) -> np.ndarray:
    """(n, 3) head translation from (n, 4, 4) facial transformation matrices (canonical-face units, ~cm)."""
    return np.asarray(transform)[:, :3, 3]


def axis_signs(t_xy: np.ndarray, corner_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signs mapping (tx, ty) to the image convention of ``head_x/y`` and the correlations they come from."""
    r = np.array([np.corrcoef(t_xy[:, k], corner_xy[:, k])[0, 1] for k in range(2)])
    return np.where(r < 0, -1.0, 1.0), r


def head_ratios(features: np.ndarray, head_xy: np.ndarray, screen_xy: np.ndarray,
                cm_per_px: float) -> tuple[np.ndarray, float]:
    """Fitted head coefficients (points per cm) on screen x and y divided by the geometric prediction
    (x: -1 / cm_per_px, y: +1 / cm_per_px), and the rms residual in points. ``features`` holds the 12
    base features (iris in columns 0-3, yaw 4, pitch 5)."""
    design = np.column_stack([np.ones(len(features)), features[:, 0:4], features[:, 4:6], head_xy])
    coef, *_ = np.linalg.lstsq(design, screen_xy, rcond=None)
    k = 1 / cm_per_px
    ratios = np.array([coef[7, 0] / -k, coef[8, 1] / k])
    resid = screen_xy - design @ coef
    return ratios, float(np.sqrt((resid ** 2).sum(axis=1).mean()))


def stack(sessions: dict[str, Session]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid frames of all sessions: base features, transform translation, target positions."""
    f = np.concatenate([s.features[s.valid] for s in sessions.values()])
    t = np.concatenate([transform_translation(s.transform[s.valid]) for s in sessions.values()])
    y = np.concatenate([s.target_xy[s.valid] for s in sessions.values()])
    return f, t, y


def report(name: str, sessions: dict[str, Session], cm_per_px: float) -> dict:
    f, t, y = stack(sessions)
    signs, r = axis_signs(t[:, :2], f[:, 9:11])
    t_img = t[:, :2] * signs
    corner, corner_rms = head_ratios(f, f[:, 9:11], y, cm_per_px)
    trans, trans_rms = head_ratios(f, t_img, y, cm_per_px)
    print(f"\n== {name}: {len(f)} frames ==")
    print(f"axis check: r(tx, head_x) = {r[0]:+.2f}, r(ty, head_y) = {r[1]:+.2f} (signs used {signs[0]:+.0f} / {signs[1]:+.0f})")
    print(f"{'session':8s} {'tx':>7s} {'ty':>7s} {'tz':>7s}   {'corner x':>8s} {'corner y':>8s} {'corner z':>8s}   (median, vs C)")
    c_t = np.median(transform_translation(sessions["C"].transform[sessions["C"].valid]), axis=0)
    c_f = np.median(sessions["C"].features[sessions["C"].valid][:, 9:12], axis=0)
    for k, s in sessions.items():
        mt = np.median(transform_translation(s.transform[s.valid]), axis=0)
        mf = np.median(s.features[s.valid][:, 9:12], axis=0)
        d = (mt - c_t) * np.array([signs[0], signs[1], 1.0])
        print(f"{k:8s} {d[0]:+7.1f} {d[1]:+7.1f} {d[2]:+7.1f}   {mf[0] - c_f[0]:+8.1f} {mf[1] - c_f[1]:+8.1f}"
              f" {mf[2] - c_f[2]:+8.1f}   tz ratio {mt[2] / c_t[2]:.2f}, corner z ratio {mf[2] / c_f[2]:.2f}")
    print(f"head coefficient / geometry:  eye corners x {corner[0]:+.2f}, y {corner[1]:+.2f} (rms {corner_rms:.0f} pt);"
          f"  transform x {trans[0]:+.2f}, y {trans[1]:+.2f} (rms {trans_rms:.0f} pt)")
    return {"r": r, "corner": corner, "transform": trans}


def within(ratios: np.ndarray) -> bool:
    return bool(np.all((ratios >= RATIO_LO) & (ratios <= RATIO_HI)))


def main() -> int:
    clean = clean_sessions()
    a, b = set_a(clean), select_set_b(clean)
    if not (a.complete and b.complete):
        print("Set A or B is incomplete on this machine:\n  - " + "\n  - ".join(a.missing + b.missing), file=sys.stderr)
        return 2
    screen = common_screen([*a.sessions.values(), *b.sessions.values()])
    print(f"Head-transform check (EXPLORATORY). Geometry: {1 / screen.cm_per_px:.1f} points per cm.")
    rb = report("set B (laptop fixed; the rule)", {k: b.sessions[k] for k in ("C", *DISPLACED, "C2")}, screen.cm_per_px)
    report("set A (reported only)", {k: a.sessions[k] for k in SLOTS_A}, screen.cm_per_px)
    tracks_axes = bool(np.all(np.abs(rb["r"]) >= MIN_AXIS_R))
    transform_ok = tracks_axes and within(rb["transform"])
    corner_ok = within(rb["corner"])
    print("\nDecision (set B):")
    print(f"  [{'PASS' if tracks_axes else 'FAIL'}] transform axes correlate with the eye-corner estimate (|r| >= {MIN_AXIS_R})")
    print(f"  [{'PASS' if transform_ok else 'FAIL'}] transform ratios in [{RATIO_LO}, {RATIO_HI}] on x and y")
    print(f"  eye-corner ratios in range: {'yes' if corner_ok else 'no'}")
    if transform_ok and corner_ok:
        verdict = "both track head moves on set B: set A's failure was the laptop; prefer the eye-corner estimate"
    elif transform_ok:
        verdict = "the transform tracks head moves: un-park the geometric gaze-ray route (pre-register on new sessions)"
    elif corner_ok:
        verdict = "only the eye-corner estimate tracks head moves on set B: set A's failure was the laptop"
    else:
        verdict = "neither tracks head moves: the geometric route stays parked; head adjustment via M3 click recalibration"
    print(f"=> {verdict}")
    append_rows([{"time": now(), "mode": "head_transform_check", "confirmatory": 0, "config": "free fit, set B",
                  "n_sessions": 7, "note": f"ratios to geometry: transform x {rb['transform'][0]:+.2f} y {rb['transform'][1]:+.2f}; "
                                           f"eye corners x {rb['corner'][0]:+.2f} y {rb['corner'][1]:+.2f}; {verdict}"}])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
