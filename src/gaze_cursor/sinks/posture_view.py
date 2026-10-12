"""On-screen posture check drawn on the fullscreen canvas (camera preview + gauges + status)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from gaze_cursor.capture import Camera
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.posture import (
    ADJUST_Z, GOOD_Z, LABELS, METRICS, PostureGate, Reference, lighting_stats, lighting_warning,
    posture_metrics)
from gaze_cursor.sinks.overlay import FullscreenCanvas

GREEN, AMBER, RED, GREY = (90, 200, 90), (0, 170, 255), (70, 70, 235), (110, 110, 110)
_LEVEL_COLOR = {"good": GREEN, "adjust": AMBER, "off": RED}
PREVIEW = (640, 360)


@dataclass(frozen=True)
class PostureResult:
    proceed: bool        # False if the user pressed ESC
    in_position: bool    # True if the gate was satisfied when the user continued


def _fmt(metric: str, v: float) -> str:
    return f"{v * 100:.1f}%" if metric == "distance" else f"{v:+.1f} deg" if metric in ("pitch", "yaw", "roll") else f"{v:.2f}"


def _put(img: np.ndarray, text: str, org: tuple[int, int], color=(230, 230, 230), scale: float = 0.6,
         thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_check(canvas: FullscreenCanvas, frame: np.ndarray | None, metrics: dict[str, float] | None,
               ref: Reference, in_position: bool, lighting: dict[str, float] | None,
               title: str) -> np.ndarray:
    """Compose the check screen: status, mirrored preview, one gauge per metric, lighting line."""
    img = canvas.blank()
    W = canvas.size[0]
    if metrics is None:
        status, color, level_all = "No face detected: look at the camera", RED, "off"
        z = {}
    else:
        z = ref.z(metrics)
        levels = {m: ref.level(z[m]) for m in METRICS}
        level_all = "off" if "off" in levels.values() else "adjust" if "adjust" in levels.values() else "good"
        if in_position:
            status, color = "IN POSITION: press SPACE to start", GREEN
        elif level_all == "good":
            status, color = "Hold still...", GREEN
        else:
            m, _lv, _z = ref.worst(metrics)
            status, color = f"Adjust {LABELS[m]}: {ref.hint(m, metrics[m])}", _LEVEL_COLOR[level_all]
    canvas.text(img, title, y=30)
    cv2.putText(img, status, (max((W - cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]) // 2, 10), 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    # Mirrored preview (display only; the tracker always sees the unflipped frame).
    px, py = (W - PREVIEW[0]) // 2, 100
    if frame is not None:
        img[py:py + PREVIEW[1], px:px + PREVIEW[0]] = cv2.resize(cv2.flip(frame, 1), PREVIEW)
    cv2.rectangle(img, (px - 3, py - 3), (px + PREVIEW[0] + 3, py + PREVIEW[1] + 3), color, 3)
    # Gauges: bar spans z in [-4, 4]; shaded bands show the good and adjust ranges.
    bar_w, bar_x = 360, (W - 760) // 2 + 250
    y = py + PREVIEW[1] + 45
    for m in METRICS:
        _put(img, LABELS[m], (bar_x - 250, y + 5))
        cv2.rectangle(img, (bar_x, y - 8), (bar_x + bar_w, y + 12), (50, 50, 50), -1)
        for lim, col in ((ADJUST_Z, (20, 70, 100)), (GOOD_Z, (40, 90, 40))):
            a, b = (int(bar_x + (0.5 + s * lim / 8) * bar_w) for s in (-1, 1))
            cv2.rectangle(img, (a, y - 8), (b, y + 12), col, -1)
        cv2.line(img, (bar_x + bar_w // 2, y - 10), (bar_x + bar_w // 2, y + 14), (200, 200, 200), 1)
        if metrics is not None:
            mk = int(bar_x + (0.5 + float(np.clip(z[m], -4, 4)) / 8) * bar_w)
            cv2.circle(img, (mk, y + 2), 8, _LEVEL_COLOR[ref.level(z[m])], -1, cv2.LINE_AA)
            _put(img, f"{_fmt(m, metrics[m])}  (usual {_fmt(m, ref.center[m])})", (bar_x + bar_w + 20, y + 5))
        y += 34
    if lighting is not None:
        warn = lighting_warning(lighting)
        _put(img, f"Light (informational): face {lighting['face']:.0f}/255, background "
                  f"{lighting['background']:.0f}" + (f"  WARNING: {warn}" if warn else ""),
             (bar_x - 250, y + 20), AMBER if warn else (170, 170, 170))
    _put(img, f"Reference: {ref.n_sessions} recorded session(s). SPACE: continue   ESC: cancel",
         (bar_x - 250, canvas.size[1] - 25), (150, 150, 150), 0.5)
    return img


def run_posture_check(cam: Camera, tracker: FaceTracker, canvas: FullscreenCanvas, ref: Reference,
                      title: str = "Sit as you did in the recorded sessions") -> PostureResult:
    """Show the check until the user presses SPACE (continue) or ESC (cancel).

    SPACE always continues; ``in_position`` reports whether the gate was satisfied at that moment.
    """
    gate = PostureGate()
    while True:
        frame = cam.read()
        if frame is None:
            continue
        result = tracker.process(frame)           # always the unflipped frame
        metrics = lighting = None
        if result is not None:
            h, w = frame.image.shape[:2]
            metrics = posture_metrics(result.points, result.transform, (w, h))
            lighting = lighting_stats(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), result.points)
        all_good = metrics is not None and ref.worst(metrics)[1] == "good"
        gate.update(time.monotonic(), all_good)
        key = canvas.show(draw_check(canvas, frame.image, metrics, ref, gate.in_position, lighting, title))
        if key == 27:
            return PostureResult(False, gate.in_position)
        if key == 32:
            return PostureResult(True, gate.in_position)


def draw_chip(canvas: FullscreenCanvas, img: np.ndarray, ref: Reference,
              metrics: dict[str, float] | None) -> None:
    """Small live posture status in the top-left corner (used while the overlay runs)."""
    if metrics is None:
        _put(img, "posture: no face", (20, 30), RED)
        return
    m, level, _z = ref.worst(metrics)
    if level == "good":
        _put(img, "posture: OK", (20, 30), GREEN)
    else:
        _put(img, f"posture: adjust {LABELS[m]} ({ref.hint(m, metrics[m])})", (20, 30), _LEVEL_COLOR[level])
