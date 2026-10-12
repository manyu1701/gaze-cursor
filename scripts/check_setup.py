"""Posture check: are you sitting like you did in the recorded sessions?

Shows your camera view, a gauge per measure (head pitch/yaw/roll, distance, face position), and a
green "IN POSITION" once you have held it for a second. The reference is computed from recorded
clean sessions, so it updates itself as you add sessions. Lighting is shown but never blocks.

  uv run scripts/check_setup.py              # reference = all clean sessions
  uv run scripts/check_setup.py --last 2     # only the 2 most recent clean sessions
  uv run scripts/check_setup.py --sessions data/sessions/session_X.npz

SPACE or ESC exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gaze_cursor.capture import Camera
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import clean_sessions, load_session
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.posture import LABELS, METRICS, Reference
from gaze_cursor.sinks.overlay import FullscreenCanvas
from gaze_cursor.sinks.posture_view import run_posture_check


def reference_from_args(sessions: list[Path] | None, last: int | None) -> Reference:
    chosen = [load_session(p) for p in sessions] if sessions else clean_sessions()
    if last:
        chosen = chosen[-last:]
    if not chosen:
        raise SystemExit("No recorded sessions to build a reference from. Run calibrate.py first.")
    return Reference.from_sessions(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", type=Path)
    ap.add_argument("--last", type=int, help="use only the N most recent clean sessions")
    args = ap.parse_args()
    ref = reference_from_args(args.sessions, args.last)
    print(f"reference from {ref.n_sessions} session(s), {ref.n_frames} frames:")
    for m in METRICS:
        print(f"  {LABELS[m]:22s} {ref.center[m]:8.3f}  (tolerance unit {ref.scale[m]:.3f})")
    with Camera() as cam, FaceTracker() as tracker, FullscreenCanvas(ScreenConfig.from_system()) as canvas:
        res = run_posture_check(cam, tracker, canvas, ref)
    print("in position" if res.in_position else "not in position", "when you exited")


if __name__ == "__main__":
    main()
