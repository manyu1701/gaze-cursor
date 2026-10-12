"""Live setup checklist drawn on the fullscreen canvas (camera preview + per-step status)."""

from __future__ import annotations

import textwrap
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from gaze_cursor.capture import Camera
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.posture import (
    LABELS, CheckItem, PostureGate, Reference, build_checklist, checklist_ready, lighting_stats,
    posture_metrics)
from gaze_cursor.sinks.overlay import FullscreenCanvas

GREEN, AMBER, RED, GREY, WHITE = (90, 200, 90), (0, 170, 255), (70, 70, 235), (140, 140, 140), (235, 235, 235)
_LEVEL_COLOR = {"good": GREEN, "adjust": AMBER, "off": RED, "unknown": GREY, "wait": GREY}
PREVIEW = (480, 270)
SMOOTH_FRAMES, STILL_WINDOW_FRAMES = 9, 45  # about 0.3 s and 1.5 s at 30 fps


@dataclass(frozen=True)
class PostureResult:
    proceed: bool        # False if the user cancelled
    in_position: bool    # True if every step was good (held for a second) when the user continued
    own_setup: bool = False                  # True if the user chose "use my current setup" (key U)
    reference: Reference | None = None       # the user's own reference in that case


def _put(img: np.ndarray, text: str, org: tuple[int, int], color=WHITE, scale: float = 0.6,
         thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _icon(img: np.ndarray, center: tuple[int, int], level: str) -> None:
    x, y = center
    color = _LEVEL_COLOR[level]
    cv2.circle(img, center, 18, color, 2, cv2.LINE_AA)
    if level == "good":
        cv2.polylines(img, [np.array([[x - 8, y], [x - 2, y + 7], [x + 9, y - 7]])], False, color, 3, cv2.LINE_AA)
    elif level in ("unknown", "wait"):
        cv2.line(img, (x - 7, y), (x + 7, y), color, 2, cv2.LINE_AA)
    else:
        cv2.line(img, (x - 7, y - 7), (x + 7, y + 7), color, 3, cv2.LINE_AA)
        cv2.line(img, (x - 7, y + 7), (x + 7, y - 7), color, 3, cv2.LINE_AA)


def draw_checklist(canvas: FullscreenCanvas, frame: np.ndarray | None, items: list[CheckItem],
                   ready: bool, tracked: bool, title: str) -> np.ndarray:
    """One screen: each setup step with a live status icon, its current value and what to change."""
    img = canvas.blank()
    W, H = canvas.size
    top = canvas.safe_top  # keep everything below the camera notch
    canvas.text(img, title, y=top + 26)
    left, y = 60, top + 100
    for it in items:
        _icon(img, (left + 20, y - 6), it.level)
        _put(img, it.title, (left + 60, y), WHITE, 0.8, 2)
        lines = textwrap.wrap(it.howto, 100)[:2]
        for k, line in enumerate(lines):
            _put(img, line, (left + 60, y + 28 + 20 * k), GREY, 0.5)
        dy = y + 28 + 20 * len(lines) + 14           # value line right under the instruction
        _put(img, it.detail, (left + 60, dy), _LEVEL_COLOR[it.level], 0.6)
        if it.hint:
            _put(img, "-> " + it.hint, (left + 60, dy + 26), AMBER if it.level == "adjust" else RED, 0.6, 2)
        y += 140
    # Mirrored preview (display only; the tracker always sees the unflipped frame).
    px, py = W - 60 - PREVIEW[0], top + 100
    if frame is not None:
        img[py:py + PREVIEW[1], px:px + PREVIEW[0]] = cv2.resize(cv2.flip(frame, 1), PREVIEW)
    border = GREEN if ready else AMBER if tracked else RED
    cv2.rectangle(img, (px - 3, py - 3), (px + PREVIEW[0] + 3, py + PREVIEW[1] + 3), border, 3)
    todo = [i for i in items if i.level in ("adjust", "off")]
    if not tracked:
        status, color = "No face detected: look at the camera", RED
    elif ready:
        status, color = "READY: press SPACE to start", GREEN
    elif not todo:
        status, color = "Hold still...", GREEN
    else:
        status, color = f"{len(todo)} step{'s' if len(todo) > 1 else ''} to fix", AMBER
    size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    cv2.putText(img, status, (px + (PREVIEW[0] - size[0]) // 2, py + PREVIEW[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    canvas.text(img, "These are guidelines: you can always continue, or calibrate in your own setup.", y=H - 84)
    canvas.text(img, "SPACE: continue anyway     U: use my current setup     ESC or Q: cancel", y=H - 40)
    return img


def _median(values: deque[dict[str, float]]) -> dict[str, float] | None:
    return {k: float(np.median([v[k] for v in values])) for k in values[0]} if values else None


def run_posture_check(cam: Camera, tracker: FaceTracker, canvas: FullscreenCanvas,
                      ref: Reference | None,
                      title: str = "Set up your position: each step turns green when it is right",
                      ref_lighting: dict | None = None) -> PostureResult:
    """Show the live checklist until the user presses SPACE (continue) or ESC/Q (cancel).

    The guidance never blocks: SPACE always continues (``in_position`` says whether every measured
    step was good, held for a second, at that moment), and U continues with the user's current
    posture as the reference (``own_setup``), for calibrating in the setup they will really use.
    ``ref`` may be None for the very first session.
    """
    gate = PostureGate()
    window: deque[dict[str, float]] = deque(maxlen=STILL_WINDOW_FRAMES)
    recent: deque[dict[str, float]] = deque(maxlen=SMOOTH_FRAMES)
    light: deque[dict[str, float]] = deque(maxlen=SMOOTH_FRAMES)
    while True:
        frame = cam.read()
        if frame is None:
            continue
        result = tracker.process(frame)           # always the unflipped frame
        if result is not None:
            h, w = frame.image.shape[:2]
            m = posture_metrics(result.points, result.transform, (w, h))
            window.append(m); recent.append(m)
            light.append(lighting_stats(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), result.points))
        else:
            window.clear(); recent.clear(); light.clear()
        items = build_checklist(ref, _median(recent), list(window), _median(light), ref_lighting)
        gate.update(time.monotonic(), checklist_ready(items))
        key = canvas.show(draw_checklist(canvas, frame.image, items, gate.in_position,
                                         result is not None, title))
        if key in (27, ord("q")):
            return PostureResult(False, gate.in_position)
        if key == 32:
            return PostureResult(True, gate.in_position)
        if key in (ord("u"), ord("U")) and len(window) >= STILL_MIN_FRAMES:
            return PostureResult(True, True, own_setup=True, reference=Reference.from_current(list(window)))


def draw_chip(canvas: FullscreenCanvas, img: np.ndarray, ref: Reference,
              metrics: dict[str, float] | None) -> None:
    """Small live posture status in the top-left corner (used while the overlay runs)."""
    y = canvas.safe_top + 22
    if metrics is None:
        _put(img, "posture: no face", (20, y), RED)
        return
    m, level, _z = ref.worst(metrics)
    if level == "good":
        _put(img, "posture: OK", (20, y), GREEN)
    else:
        _put(img, f"posture: {LABELS[m]} off - {ref.hint(m, metrics[m])}", (20, y), _LEVEL_COLOR[level])
