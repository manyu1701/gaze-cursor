"""Live gaze dot. Fits a model from a recorded session, then draws where you're looking.

Usage: uv run scripts/gaze_overlay.py [session.npz] [--model ridge|poly2] [--features gaze|eyes|iris|blend|both]   (q / ESC quits)
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from gaze_cursor.calibration import MODELS, GazeModel
from gaze_cursor.capture import Camera
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import CALIB, latest_session, load_session
from gaze_cursor.features import FEATURE_SETS, assemble, extract_features
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.posture import Reference, posture_metrics
from gaze_cursor.sinks.overlay import FullscreenCanvas
from gaze_cursor.sinks.posture_view import draw_chip, run_posture_check

OOD_Z = 4.0  # warn when a live feature is this many calibration std-devs from its mean
SMOOTH_FRAMES = 5  # simple moving average; One-Euro comes in M2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", type=Path)
    ap.add_argument("--model", choices=MODELS, default="ridge")
    ap.add_argument("--features", choices=FEATURE_SETS, default="gaze",
                    help="default 'gaze' (iris only): raw head pose made the dot slam to the screen "
                         "edge as soon as posture changed from calibration")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the position check and the live posture hints")
    args = ap.parse_args()

    session = load_session(args.session or latest_session())
    cfg = ScreenConfig.from_system()
    if (cfg.width_px, cfg.height_px) != (session.meta["screen"]["width_px"], session.meta["screen"]["height_px"]):
        print("warning: screen size differs from the session's; recalibrate for accurate results")
    x_tr, y_tr, _ = session.select(CALIB, args.features)
    model = GazeModel(args.model).fit(x_tr, y_tr)
    recent: deque[np.ndarray] = deque(maxlen=SMOOTH_FRAMES)
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-9
    worst_z = 0.0
    # The model is only as good as the posture it was trained in: compare against that session.
    ref = None if args.no_check else Reference.from_sessions([session])

    with Camera() as cam, FaceTracker() as tracker, FullscreenCanvas(cfg) as canvas:
        if ref is not None:
            check = run_posture_check(cam, tracker, canvas, ref,
                                      title="Sit as you did when you calibrated this model",
                                      ref_lighting=session.meta.get("lighting"),
                                      override_label="run anyway (accuracy may drop)")
            if not check.proceed:
                return
        metrics = None
        while True:
            frame = cam.read()
            result = tracker.process(frame) if frame is not None else None
            metrics = None
            if frame is not None and result is not None:
                h, w = frame.image.shape[:2]
                metrics = posture_metrics(result.points, result.transform, (w, h))
                feats = extract_features(result.points, result.transform, (w, h))
                if feats is not None:
                    names = tuple(session.meta["blendshape_names"])
                    x = assemble(feats, result.blendshapes, names, args.features)
                    recent.append(model.predict(x)[0])
                    worst_z = float(np.max(np.abs((x - mu) / sd)))
            img = canvas.blank()
            if recent:
                x, y = np.mean(recent, axis=0)
                canvas.dot(img, (np.clip(x, 0, cfg.width_px), np.clip(y, 0, cfg.height_px)),
                           color=(0, 255, 0), r=20)
            if ref is not None:
                draw_chip(canvas, img, ref, metrics)
            canvas.text(img, "q / ESC to quit", y=canvas.size[1] - 30)
            if worst_z > OOD_Z:
                canvas.text(img, f"input is {worst_z:.0f} sd from calibration: sit as you did, or recalibrate",
                            y=canvas.size[1] - 70)
            if canvas.show(img) in (27, ord("q")):
                break


if __name__ == "__main__":
    main()
