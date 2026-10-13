"""Diagnose the head-geometry estimate on set A (EXPLORATORY; numbers only, never opens images).

1. Per session: median head position (cm / depth units), head pose, and own-calibration error
   (16 targets -> its 9 held-out) for gaze_a10 and geo_a10. If geo is bad even within a session,
   the model form is wrong; if it is good within but bad across sessions, the head estimate is.
2. A free linear fit over all set-A frames: screen x / y in points ~ iris + yaw + pitch + head x, y.
   The head coefficients are printed in points per cm next to what the screen geometry predicts
   (x: -1 / cm_per_px, y: +1 / cm_per_px), so sign or scale errors in the head estimate show up.

  uv run scripts/diagnose_positions.py
"""

from __future__ import annotations

import sys

import numpy as np

from gaze_cursor.eval.configs import REGISTRY
from gaze_cursor.eval.prereg3 import SLOTS_A, set_a
from gaze_cursor.eval.runner import common_screen
from gaze_cursor.eval.session import CALIB, TEST, Session, clean_sessions
from gaze_cursor.eval.stats import per_target


def own_error(s: Session, name: str, screen) -> float:
    cfg = REGISTRY[name]
    x_tr, y_tr, _ = s.select(CALIB, cfg.feature_set)
    x_te, y_te, t_te = s.select(TEST, cfg.feature_set)
    m = cfg.make_model(screen).fit(x_tr, y_tr)
    return float(per_target(m.predict(x_te), y_te, t_te, np.zeros(len(t_te), int), screen).err_deg.mean())


def diagnose(sessions: dict[str, Session]) -> str:
    screen = common_screen(list(sessions.values()))
    lines = [f"{'slot':4s} {'head_x':>7s} {'head_y':>7s} {'head_z':>7s} {'yaw':>6s} {'pitch':>6s} {'roll':>6s}"
             f" {'own gaze':>9s} {'own geo':>8s}"]
    xs, ys = [], []
    for k, s in sessions.items():
        f = s.features[s.valid]
        med = np.median(f, axis=0)
        lines.append(f"{k:4s} {med[9]:7.2f} {med[10]:7.2f} {med[11]:7.1f} {med[4]:6.1f} {med[5]:6.1f} {med[6]:6.1f}"
                     f" {own_error(s, 'gaze_a10', screen):9.2f} {own_error(s, 'geo_a10', screen):8.2f}")
        xs.append(f); ys.append(s.target_xy[s.valid])
    f, y = np.concatenate(xs), np.concatenate(ys)
    design = np.column_stack([np.ones(len(f)), f[:, 0:4], f[:, 4:6], f[:, 9], f[:, 10]])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    k = 1 / screen.cm_per_px
    lines.append("\nfree fit over all set-A frames, head coefficients in screen points per cm of head movement:")
    lines.append(f"  screen x ~ head_x {coef[7, 0]:+8.1f}  (geometry: {-k:+.1f})   screen x ~ head_y {coef[8, 0]:+8.1f}  (geometry: +0.0)")
    lines.append(f"  screen y ~ head_x {coef[7, 1]:+8.1f}  (geometry: +0.0)   screen y ~ head_y {coef[8, 1]:+8.1f}  (geometry: {k:+.1f})")
    resid = y - design @ coef
    lines.append(f"  (rms residual {np.sqrt((resid ** 2).sum(axis=1).mean()):.0f} points over {len(f)} frames)")
    return "\n".join(lines)


def main() -> int:
    a = set_a(clean_sessions())
    if not a.complete:
        print("Set A is incomplete on this machine:\n  - " + "\n  - ".join(a.missing), file=sys.stderr)
        return 2
    print("Set A head-geometry diagnosis (EXPLORATORY)\n")
    print(diagnose({k: a.sessions[k] for k in SLOTS_A}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
