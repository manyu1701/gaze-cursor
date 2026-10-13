"""Cursor mode (M3): tap Right Command and the real cursor jumps to where you look; the trackpad finishes.

Starts with the 9-dot quick calibration where you sit (same as Run), then gets out of the way: a
transparent status chip in the top-right corner, every app keeps working, clicks go through. The cursor
only ever moves when you tap the hotkey, and not when the tracker cannot see your eyes.

Hotkeys (work in any app): tap Right Command = jump to gaze; Cmd+Option+G = on / off;
Cmd+Option+R = recalibrate; Cmd+Option+Q (or Ctrl+C here) = quit.

Every frame (gaze, head shift) and every hotkey / click is saved to data/quick/cursor_<time>.npz
(positions only, no images) and summarised in data/results.csv, so self-recalibration from clicks can be
built and judged offline. Needs Input Monitoring permission for Terminal (it explains how).

  uv run scripts/cursor_mode.py [--show-dot] [--warp-key right_command|right_option]
"""

from __future__ import annotations

import argparse
import time


from gaze_cursor.capture import Camera
from gaze_cursor.config import ScreenConfig
from gaze_cursor.cursor_session import CursorSession
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.features import assemble, extract_features
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.livetest import head_shift
from gaze_cursor.posture import Reference, posture_metrics
from gaze_cursor.quick import QUICK_CONFIG
from gaze_cursor.sinks.cursor import (
    DEFAULT_WARP_KEY, PERMISSION_HELP, WARP_KEYS, HotkeyDetector, MacCursor, input_monitoring_granted,
    request_input_monitoring)
from gaze_cursor.sinks.cursor_view import AMBER, GREEN, GREY, draw_gaze_dot, draw_status
from gaze_cursor.sinks.overlay import ClickThroughOverlay, FullscreenCanvas
from gaze_cursor.sinks.quick_view import QUICK_DIR, QuickFit, calibrate_here

NOTE = "quick calibration (Cursor)"
LEVEL_COLORS = {"ok": GREEN, "warn": AMBER, "off": GREY}


def calibrate(cam: Camera, tracker: FaceTracker, cfg: ScreenConfig) -> QuickFit | None:
    """The quick calibration on its own fullscreen canvas (closed again afterwards)."""
    with FullscreenCanvas(cfg) as canvas:
        out = calibrate_here(cam, tracker, canvas, cfg, NOTE)
    if isinstance(out, str):
        print("Cursor mode always uses a fresh quick calibration (L is not available here).")
        return None
    return out


def process_frame(session: CursorSession, frame, result, names: tuple[str, ...], ref: Reference | None) -> None:
    """Features -> session.frame (prediction, smoothing, input range, head shift, log)."""
    x = shift = None
    if frame is not None and result is not None:
        h, w = frame.image.shape[:2]
        if ref is not None:
            shift = head_shift(posture_metrics(result.points, result.transform, (w, h)), ref.center)
        feats = extract_features(result.points, result.transform, (w, h))
        if feats is not None:
            x = assemble(feats, result.blendshapes, names, "gaze")
    t = frame.timestamp_ms if frame is not None else time.monotonic_ns() / 1e6
    session.frame(t, x, shift)


def draw(overlay: ClickThroughOverlay, session: CursorSession, show_dot: bool) -> None:
    img = overlay.blank()
    draw_status(img, [(t, LEVEL_COLORS[lvl]) for t, lvl in session.status_lines()], overlay.safe_top)
    if show_dot and session.enabled and session.dot is not None:
        draw_gaze_dot(img, (float(session.dot[0]), float(session.dot[1])))
    overlay.show(img)


def finish(session: CursorSession) -> None:
    path = session.save(QUICK_DIR)
    s = session.summary()
    med = s["click_gaze_deg_median"]
    note = (f"{s['minutes']:.1f} min, face tracked {s['tracked'] * 100:.0f}%; {s['warps']} jumps "
            f"({s['warps_refused']} refused); {s['clicks']} clicks"
            + (f"; click vs gaze just before it: median {med:.1f} deg (mean {s['click_gaze_deg_mean']:.1f}, "
               f"n {s['click_gaze_n']})" if med is not None else ""))
    append_rows([{"time": now(), "mode": "cursor_session", "confirmatory": 0, "session": session.calib,
                  "config": f"{QUICK_CONFIG} + One-Euro", "n_frames": s["frames"], "n_targets": s["clicks"],
                  "err_deg_mean": s["click_gaze_deg_mean"], "err_deg_median": med, "note": note}])
    print(f"saved {path.name}: {note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-dot", action="store_true", help="draw a faint ring where the tracker thinks you look")
    ap.add_argument("--warp-key", choices=sorted(WARP_KEYS), default=DEFAULT_WARP_KEY,
                    help="the modifier to tap (alone) to jump the cursor to your gaze")
    args = ap.parse_args()
    if not input_monitoring_granted():
        request_input_monitoring()
        print(PERMISSION_HELP)
        return 1
    cfg = ScreenConfig.from_system()
    with Camera() as cam, FaceTracker() as tracker:
        fit = calibrate(cam, tracker, cfg)
        if fit is None:
            return 0
        names = tuple(tracker.blendshape_names)
        ref = Reference.from_current(fit.posture)
        session = CursorSession(cfg, fit.model, fit.x, ref.center, fit.path.name)
        print("Cursor mode on: tap Right Command to jump to your gaze; Cmd+Option+G on/off, "
              "Cmd+Option+R recalibrate, Cmd+Option+Q quit.")
        cursor = MacCursor((cfg.width_px, cfg.height_px), HotkeyDetector(args.warp_key))
        try:
            with ClickThroughOverlay(cfg) as overlay:
                last_lines = None
                while True:
                    frame = cam.read()
                    result = tracker.process(frame) if frame is not None else None
                    process_frame(session, frame, result, names, ref)
                    action = "none"
                    for event in cursor.poll():
                        action, xy = session.handle(event, time.monotonic_ns() / 1e6)
                        if action == "warp":
                            cursor.warp(xy)
                        if action in ("quit", "recalibrate"):
                            break
                    if action == "quit":
                        break
                    if action == "recalibrate":
                        new = calibrate(cam, tracker, cfg)
                        overlay.restore_policy()
                        if new is not None:
                            ref = Reference.from_current(new.posture)
                            session.recalibrated(new.model, new.x, ref.center, new.path.name)
                        last_lines = None
                        continue
                    lines = session.status_lines()
                    if args.show_dot or lines != last_lines:   # redraw only when something changed
                        draw(overlay, session, args.show_dot)
                        last_lines = lines
                    else:
                        overlay.pump()
        except KeyboardInterrupt:
            pass
        finally:
            cursor.close()
            finish(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
