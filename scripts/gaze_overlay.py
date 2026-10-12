"""Live gaze dot. Fits a model from a recorded session, then draws where you're looking.

Usage: uv run scripts/gaze_overlay.py [session.npz] [--model ridge|poly2] [--features iris|blend|both]   (q / ESC quits)
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
from gaze_cursor.sinks.overlay import FullscreenCanvas

SMOOTH_FRAMES = 5  # simple moving average; One-Euro comes in M2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", type=Path)
    ap.add_argument("--model", choices=MODELS, default="ridge")
    ap.add_argument("--features", choices=FEATURE_SETS, default="iris")
    args = ap.parse_args()

    session = load_session(args.session or latest_session())
    cfg = ScreenConfig.from_system()
    if (cfg.width_px, cfg.height_px) != (session.meta["screen"]["width_px"], session.meta["screen"]["height_px"]):
        print("warning: screen size differs from the session's; recalibrate for accurate results")
    x_tr, y_tr, _ = session.select(CALIB, args.features)
    model = GazeModel(args.model).fit(x_tr, y_tr)
    recent: deque[np.ndarray] = deque(maxlen=SMOOTH_FRAMES)

    with Camera() as cam, FaceTracker() as tracker, FullscreenCanvas(cfg) as canvas:
        while True:
            frame = cam.read()
            if frame is not None and (result := tracker.process(frame)) is not None:
                h, w = frame.image.shape[:2]
                feats = extract_features(result.points, result.transform, (w, h))
                if feats is not None:
                    names = tuple(session.meta["blendshape_names"])
                    x = assemble(feats, result.blendshapes, names, args.features)
                    recent.append(model.predict(x)[0])
            img = canvas.blank()
            if recent:
                x, y = np.mean(recent, axis=0)
                canvas.dot(img, (np.clip(x, 0, cfg.width_px), np.clip(y, 0, cfg.height_px)),
                           color=(0, 255, 0), r=20)
            canvas.text(img, "q / ESC to quit", y=cfg.height_px - 30)
            if canvas.show(img) in (27, ord("q")):
                break


if __name__ == "__main__":
    main()
