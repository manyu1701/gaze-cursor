"""Live gaze dot. By default it starts with a 9-dot quick calibration where you sit (~15 s), then
draws where you're looking; R recalibrates at any time (e.g. after you move).

Usage:
  uv run scripts/gaze_overlay.py                    # quick 9-dot calibration, then the dot
  uv run scripts/gaze_overlay.py --latest           # use the latest full calibration instead
  uv run scripts/gaze_overlay.py session.npz [--model ridge|poly2] [--features gaze|...]

Keys: SPACE start, L latest full calibration (start screen), R recalibrate, q / ESC quit.
Why quick by default: the pre-registered round-3 test (PROGRESS.md) found that moving your head
costs ~2 deg with a fixed calibration and that a fresh 9-dot calibration recovers most of it.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gaze_cursor.calibration import MODELS, GazeModel
from gaze_cursor.capture import Camera
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import CALIB, SESSIONS_DIR, SessionRecorder, latest_session, load_session
from gaze_cursor.features import FEATURE_SETS, assemble, extract_features
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.posture import LIGHT_MIN_FACE, Reference, lighting_label, lighting_stats, posture_metrics
from gaze_cursor.quick import (
    QUICK_CONFIG, QUICK_DOTS, QUICK_LAYOUT, QUICK_RECORD_S, QUICK_SETTLE_S, fit_quick, fit_report,
    missing_dots, quick_targets)
from gaze_cursor.sinks.overlay import FullscreenCanvas
from gaze_cursor.sinks.posture_view import draw_chip, run_posture_check

OOD_Z = 4.0  # warn when a live feature is this many calibration std-devs from its mean
SMOOTH_FRAMES = 5  # simple moving average; One-Euro comes in M2
QUICK_DIR = SESSIONS_DIR.parent / "quick"   # gitignored; never read by the research analyses
REDO_DEG = 4.0  # quick fit worse than this on its own dots: suggest redoing it
GREY, AMBER = (150, 150, 150), (0, 170, 255)


@dataclass
class QuickFit:
    model: GazeModel
    x: np.ndarray                      # the dots' frames (gaze features), for the input-range warning
    posture: list[dict[str, float]]    # posture during the dots: the live posture chip compares to it
    report: dict[str, float]
    path: Path


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
                      cfg: ScreenConfig) -> QuickFit | Path | None:
    """Show the 9 dots and record them. Returns the fit; the saved path alone if a dot had too few
    frames; None if aborted. Saves the recording to data/quick/ (record and replay)."""
    seed = int(time.time()) % 2**31
    targets = quick_targets(cfg, seed=seed)
    rec = SessionRecorder(meta={
        "screen": cfg.to_dict(), "layout": QUICK_LAYOUT, "seed": seed, "note": "quick calibration (Run)",
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
    path = rec.save(QUICK_DIR)
    x, y, idx_a = np.array(xs), np.array(ys, float), np.array(idx)
    if len(x) == 0 or missing_dots(idx_a, targets):
        return path
    model = fit_quick(x, y, cfg)
    return QuickFit(model, x, posture, fit_report(model, x, y, idx_a, cfg), path)


def calibrate_here(cam: Camera, tracker: FaceTracker, canvas: FullscreenCanvas,
                   cfg: ScreenConfig) -> QuickFit | str | None:
    """Start screen + quick calibration, repeated until it succeeds. None = quit; 'latest' = L."""
    message = ""
    while True:
        choice = start_screen(cam, tracker, canvas, message)
        if choice != "quick":
            return None if choice == "quit" else "latest"
        out = quick_calibration(cam, tracker, canvas, cfg)
        if out is None:
            message = "calibration aborted"
            continue
        if isinstance(out, Path):
            print(f"quick calibration: some dots had too few frames (blinks or lost face); saved {out.name}")
            message = "some dots were missed (blinks or lost face): press SPACE to try again"
            continue
        report = out.report
        print(f"quick calibration: {QUICK_CONFIG} on {report['dots']} dots, {len(out.x)} frames, fit on its own "
              f"dots {report['mean_deg']:.1f} deg (worst {report['worst_deg']:.1f}); saved {out.path.name}")
        if report["worst_deg"] > REDO_DEG:
            message = (f"one dot fitted badly ({report['worst_deg']:.1f} deg): look straight at each dot; "
                       f"SPACE to redo")
            continue
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", type=Path, help="use this full calibration instead of a quick one")
    ap.add_argument("--latest", action="store_true", help="use the latest full calibration instead of a quick one")
    ap.add_argument("--model", choices=MODELS, default="ridge", help="(full calibration only)")
    ap.add_argument("--features", choices=FEATURE_SETS, default="gaze",
                    help="(full calibration only) default 'gaze' (iris only): raw head pose made the dot slam "
                         "to the screen edge as soon as posture changed from calibration")
    ap.add_argument("--no-check", action="store_true",
                    help="(full calibration only) skip the position check and the live posture hints")
    args = ap.parse_args()
    cfg = ScreenConfig.from_system()
    quick = args.session is None and not args.latest

    with Camera() as cam, FaceTracker() as tracker, FullscreenCanvas(cfg) as canvas:
        features, ref = "gaze", None
        if quick:
            out = calibrate_here(cam, tracker, canvas, cfg)
            if out is None:
                return
            if isinstance(out, QuickFit):
                model, x_tr, ref = out.model, out.x, Reference.from_current(out.posture)
            else:
                quick = False
        if not quick:
            try:
                session = load_session(args.session or latest_session())
            except FileNotFoundError:
                print("No full calibration yet: double-click Calibrate.command, or use the quick calibration.")
                return
            if (cfg.width_px, cfg.height_px) != (session.meta["screen"]["width_px"], session.meta["screen"]["height_px"]):
                print("warning: screen size differs from the session's; recalibrate for accurate results")
            features = args.features
            x_tr, y_tr, _ = session.select(CALIB, features)
            model = GazeModel(args.model).fit(x_tr, y_tr)
            # The model is only as good as the posture it was trained in: compare against that session.
            ref = None if args.no_check else Reference.from_sessions([session])
            if ref is not None:
                check = run_posture_check(cam, tracker, canvas, ref,
                                          title="Sit as you did when you calibrated this model",
                                          ref_lighting=session.meta.get("lighting"),
                                          override_label="run anyway (accuracy may drop)")
                if not check.proceed:
                    return
        names = tuple(tracker.blendshape_names)
        recent: deque[np.ndarray] = deque(maxlen=SMOOTH_FRAMES)
        mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-9
        worst_z = 0.0
        while True:
            frame = cam.read()
            result = tracker.process(frame) if frame is not None else None
            metrics = None
            if frame is not None and result is not None:
                h, w = frame.image.shape[:2]
                metrics = posture_metrics(result.points, result.transform, (w, h))
                feats = extract_features(result.points, result.transform, (w, h))
                if feats is not None:
                    x = assemble(feats, result.blendshapes, names, features)
                    recent.append(model.predict(x)[0])
                    worst_z = float(np.max(np.abs((x - mu) / sd)))
            img = canvas.blank()
            if recent:
                gx, gy = np.mean(recent, axis=0)
                canvas.dot(img, (np.clip(gx, 0, cfg.width_px), np.clip(gy, 0, cfg.height_px)),
                           color=(0, 255, 0), r=20)
            if ref is not None:
                draw_chip(canvas, img, ref, metrics)
            canvas.text(img, "R: recalibrate here (9 dots)     q / ESC: quit", y=canvas.size[1] - 30)
            if worst_z > OOD_Z:
                canvas.text(img, f"input is {worst_z:.0f} sd from calibration: press R to recalibrate here",
                            y=canvas.size[1] - 70)
            key = canvas.show(img)
            if key in (27, ord("q")):
                break
            if key in (ord("r"), ord("R")):
                out = calibrate_here(cam, tracker, canvas, cfg)
                if out is None:
                    break
                if isinstance(out, QuickFit):
                    model, x_tr = out.model, out.x
                    features, ref = "gaze", Reference.from_current(out.posture)
                    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-9
                    recent.clear(); worst_z = 0.0


if __name__ == "__main__":
    main()
