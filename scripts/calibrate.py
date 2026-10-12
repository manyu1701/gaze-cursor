"""Fullscreen calibration + held-out test recording. Saves a session to data/sessions/.

SPACE starts, ESC aborts. Look at each dot until it moves. Keep your head fairly still.
Frames are never mirrored. Screen size/geometry comes from Quartz (macOS points).

`--check-screen` only shows corner/center markers and the detected size, to verify that the
OpenCV fullscreen window maps 1:1 onto the screen's point coordinates.
"""

from __future__ import annotations

import argparse
import time

import cv2

from gaze_cursor.capture import Camera
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import CALIB, TEST, SessionRecorder
from gaze_cursor.eval.targets import calibration_grid, held_out_points
from gaze_cursor.features import extract_features
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.sinks.overlay import FullscreenCanvas

SETTLE_S = 0.7   # eyes travel to the new dot; not recorded
RECORD_S = 1.5
SEED = 0


def check_screen(cfg: ScreenConfig) -> None:
    """Draw markers at the corners and center in point coordinates; ESC/q exits."""
    with FullscreenCanvas(cfg) as canvas:
        while True:
            img = canvas.blank()
            for x in (0, cfg.width_px - 1):
                for y in (0, cfg.height_px - 1):
                    cv2.circle(img, (x, y), 24, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.drawMarker(img, (cfg.width_px // 2, cfg.height_px // 2), (0, 255, 0),
                           cv2.MARKER_CROSS, 40, 2)
            cv2.rectangle(img, (0, 0), (cfg.width_px - 1, cfg.height_px - 1), (0, 255, 0), 1)
            canvas.text(img, f"{cfg.width_px} x {cfg.height_px} pt  "
                             f"{cfg.width_cm:.1f} x {cfg.height_cm:.1f} cm  (ESC to exit)")
            if canvas.show(img) in (27, ord("q")):
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-screen", action="store_true")
    args = ap.parse_args()
    cfg = ScreenConfig.from_system()
    print(f"screen: {cfg}")
    if args.check_screen:
        check_screen(cfg)
        return
    grid = calibration_grid(cfg)
    held_out = held_out_points(cfg, grid, seed=SEED)
    plan = [(CALIB, i, p) for i, p in enumerate(grid)]
    plan += [(TEST, len(grid) + i, p) for i, p in enumerate(held_out)]
    rec = SessionRecorder(meta={
        "screen": cfg.to_dict(), "seed": SEED, "settle_s": SETTLE_S, "record_s": RECORD_S,
        "mirrored": False, "blendshape_names": [], "n_calib": len(grid), "n_test": len(held_out),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    with Camera() as cam, FaceTracker() as tracker, FullscreenCanvas(cfg) as canvas:
        # Wait for SPACE while showing the live pipeline is alive.
        while True:
            img = canvas.blank()
            canvas.text(img, "Sit ~60 cm away, head still. Press SPACE to start, ESC to abort.")
            key = canvas.show(img)
            cam.read()
            if key == 27:
                return
            if key == 32:
                break
        for n, (phase, idx, xy) in enumerate(plan):
            start = time.monotonic()
            while (elapsed := time.monotonic() - start) < SETTLE_S + RECORD_S:
                frame = cam.read()
                img = canvas.blank()
                canvas.dot(img, xy, color=(255, 255, 255) if elapsed < SETTLE_S else (0, 255, 0))
                canvas.text(img, f"{n + 1}/{len(plan)}", y=40)
                if canvas.show(img) == 27:
                    print("Aborted; nothing saved.")
                    return
                if frame is None or elapsed < SETTLE_S:
                    continue
                result = tracker.process(frame)
                if result is None:
                    rec.add(frame.timestamp_ms, phase, idx, xy, None, None, None)
                    continue
                h, w = frame.image.shape[:2]
                rec.meta["image_size"] = [w, h]
                feats = extract_features(result.points, result.transform, (w, h))
                rec.add(frame.timestamp_ms, phase, idx, xy, result.points, result.transform,
                        feats, result.blendshapes)
        rec.meta["blendshape_names"] = list(tracker.blendshape_names)
    path = rec.save()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
