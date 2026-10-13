"""The 9-dot quick calibration screens, shared by Run (gaze_overlay.py) and Cursor (cursor_mode.py).

Shows a start screen (face tracked? lit?), then the quick dots on a fullscreen canvas, records them to
data/quick/ (record and replay) and fits the frozen round-3 winner (``quick.QUICK_CONFIG``).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gaze_cursor.calibration import GazeModel
from gaze_cursor.capture import Camera
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import CALIB, SESSIONS_DIR, SessionRecorder
from gaze_cursor.features import assemble, extract_features
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.posture import LIGHT_MIN_FACE, lighting_label, lighting_stats, posture_metrics
from gaze_cursor.quick import (
    QUICK_CONFIG, QUICK_DOTS, QUICK_LAYOUT, QUICK_RECORD_S, QUICK_SETTLE_S, fit_quick, fit_report,
    missing_dots, quick_targets)
from gaze_cursor.sinks.overlay import FullscreenCanvas

QUICK_DIR = SESSIONS_DIR.parent / "quick"   # gitignored; never read by the research analyses
# Redo the quick calibration when its mean fit on its own 9 dots exceeds this. On the 21 front-lit
# sessions the mean fit predicts held-out accuracy (r = 0.78) better than the worst dot (r = 0.66;
# the old "worst dot > 4 deg" rule fired on 57% of them); > 3.0 flags ~1 in 4, which end ~1 deg worse.
REDO_MEAN_DEG = 3.0
GREY, AMBER = (150, 150, 150), (0, 170, 255)


@dataclass
class QuickFit:
    model: GazeModel
    x: np.ndarray                      # the dots' frames (gaze features), for the input-range warning
    posture: list[dict[str, float]]    # posture during the dots: the live posture chip compares to it
    report: dict[str, float]
    path: Path
    y: np.ndarray | None = None        # the dots' screen positions per frame (points): the online prior
    idx: np.ndarray | None = None      # dot index per frame


def _lines(canvas: FullscreenCanvas, img: np.ndarray, lines: list[tuple[str, tuple]], y0: int) -> None:
    for k, (text, color) in enumerate(lines):
        cv2.putText(img, text, (60, y0 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def start_screen(cam: Camera, tracker: FaceTracker, canvas: FullscreenCanvas, message: str = "") -> str:
    """'quick', 'latest' or 'quit'. Shows whether the face is tracked and how it is lit."""
    recent: deque[float] = deque(maxlen=15)
    bg: deque[float] = deque(maxlen=15)
    while True:
        frame = cam.read()
        result = tracker.process(frame) if frame is not None else None
        if frame is None or result is None:
            recent.clear(); bg.clear()
        else:
            s = lighting_stats(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), result.points)
            recent.append(s["face"]); bg.append(s["background"])
        img = canvas.blank()
        top = canvas.safe_top + 60
        lines = [(f"Quick calibration: look at {QUICK_DOTS} dots where you sit now (~15 s). Keep your head still.",
                  (235, 235, 235))]
        if message:
            lines.append((message, AMBER))
        if recent:
            face, back = float(np.mean(recent)), float(np.mean(bg))
            label = lighting_label(face, back)
            lines.append((f"face tracked; light {label}, face {face:.0f}/255", GREY))
            if face < LIGHT_MIN_FACE:
                lines.append(("too dark for good tracking: put a lamp in front of you", AMBER))
        else:
            lines.append(("no face detected: look at the camera", AMBER))
        _lines(canvas, img, lines, top)
        canvas.text(img, "SPACE: start     L: use the latest full calibration     ESC or Q: quit",
                    y=canvas.size[1] - 40)
        key = canvas.show(img)
        if key in (27, ord("q")):
            return "quit"
        if key in (ord("l"), ord("L")):
            return "latest"
        if key == 32 and recent:
            return "quick"


def quick_calibration(cam: Camera, tracker: FaceTracker, canvas: FullscreenCanvas,
                      cfg: ScreenConfig, note: str = "quick calibration (Run)",
                      record: bool = True) -> QuickFit | Path | None:
    """Show the 9 dots and record them. Returns the fit; the saved path alone if a dot had too few
    frames; None if aborted. Saves the recording to data/quick/ (record and replay), unless
    ``record`` is False (demo for a friend: nothing about their face is written to disk)."""
    seed = int(time.time()) % 2**31
    targets = quick_targets(cfg, seed=seed)
    rec = SessionRecorder(meta={
        "screen": cfg.to_dict(), "layout": QUICK_LAYOUT, "seed": seed, "note": note,
        "settle_s": QUICK_SETTLE_S, "record_s": QUICK_RECORD_S, "mirrored": False, "blendshape_names": [],
        "n_calib": len(targets), "n_test": 0, "model": QUICK_CONFIG, "window": [*canvas.origin, *canvas.size],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    xs, ys, idx, posture, light = [], [], [], [], []
    for n, t in enumerate(targets):
        start = time.monotonic()
        while (elapsed := time.monotonic() - start) < QUICK_SETTLE_S + QUICK_RECORD_S:
            frame = cam.read()
            img = canvas.blank()
            canvas.dot(img, t.xy, color=(255, 255, 255) if elapsed < QUICK_SETTLE_S else (0, 255, 0))
            canvas.text(img, f"{n + 1}/{len(targets)}", y=canvas.safe_top + 26)
            if canvas.show(img) in (27, ord("q")):
                return None
            if frame is None or elapsed < QUICK_SETTLE_S:
                continue
            result = tracker.process(frame)
            if result is None:
                rec.add(frame.timestamp_ms, CALIB, t.idx, t.xy, None, None, None)
                continue
            h, w = frame.image.shape[:2]
            rec.meta["image_size"] = [w, h]
            feats = extract_features(result.points, result.transform, (w, h))
            rec.add(frame.timestamp_ms, CALIB, t.idx, t.xy, result.points, result.transform, feats,
                    result.blendshapes)
            light.append(lighting_stats(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), result.points))
            if feats is None:      # blink: same rule as the recorded sessions
                continue
            blend = result.blendshapes if result.blendshapes is not None else np.zeros(len(tracker.blendshape_names))
            xs.append(assemble(feats[None], blend[None], tuple(tracker.blendshape_names), "gaze")[0])
            ys.append(t.xy); idx.append(t.idx)
            posture.append(posture_metrics(result.points, result.transform, (w, h)))
    rec.meta["blendshape_names"] = list(tracker.blendshape_names)
    if light:
        rec.meta["lighting"] = {k: float(np.mean([s[k] for s in light])) for k in ("face", "background", "ratio")}
    path = rec.save(QUICK_DIR) if record else Path("demo (not saved)")
    x, y, idx_a = np.array(xs), np.array(ys, float), np.array(idx)
    if len(x) == 0 or missing_dots(idx_a, targets):
        return path
    model = fit_quick(x, y, cfg)
    return QuickFit(model, x, posture, fit_report(model, x, y, idx_a, cfg), path, y, idx_a)


def calibrate_here(cam: Camera, tracker: FaceTracker, canvas: FullscreenCanvas, cfg: ScreenConfig,
                   note: str = "quick calibration (Run)", record: bool = True) -> QuickFit | str | None:
    """Start screen + quick calibration, repeated until it succeeds. None = quit; 'latest' = L."""
    message = ""
    while True:
        choice = start_screen(cam, tracker, canvas, message)
        if choice != "quick":
            return None if choice == "quit" else "latest"
        out = quick_calibration(cam, tracker, canvas, cfg, note, record)
        if out is None:
            message = "calibration aborted"
            continue
        if isinstance(out, Path):
            print(f"quick calibration: some dots had too few frames (blinks or lost face); saved {out.name}")
            message = "some dots were missed (blinks or lost face): press SPACE to try again"
            continue
        report = out.report
        print(f"quick calibration: {QUICK_CONFIG} on {report['dots']} dots, {len(out.x)} frames, fit on its own "
              f"dots {report['mean_deg']:.1f} deg (worst {report['worst_deg']:.1f}); file: {out.path.name}")
        if report["mean_deg"] > REDO_MEAN_DEG:
            message = (f"the calibration fitted poorly ({report['mean_deg']:.1f} deg on its own dots): look "
                       f"straight at each dot's centre; SPACE to redo")
            continue
        return out
