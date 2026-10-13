"""Live setup checklist drawn on the fullscreen canvas (camera preview + per-step status)."""

from __future__ import annotations

import textwrap
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from gaze_cursor.capture import Camera
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.posture import (
    LABELS, LIGHT_GOOD_FACE, LIGHT_MIN_FACE, STILL_MIN_FRAMES, CheckItem, PostureGate, Reference, build_checklist, checklist_ready, lighting_stats,
    posture_metrics)
from gaze_cursor.sinks.overlay import FullscreenCanvas

GREEN, AMBER, RED, GREY, WHITE = (90, 200, 90), (0, 170, 255), (70, 70, 235), (140, 140, 140), (235, 235, 235)
TIP = (230, 200, 120)  # light blue: advice, not a fault
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


def default_footer(mode: str, ready: bool, override_label: str) -> tuple[str, str]:
    """(note, key line) for the bottom of the screen. SPACE only starts when every step is green;
    U is the single override; in monitor mode SPACE just exits."""
    if mode == "monitor":
        return "", "SPACE or ESC: exit"
    if ready:
        return "", f"SPACE: start     U: {override_label} instead     ESC or Q: cancel"
    return ("SPACE unlocks when every step is green.", f"U: {override_label}     ESC or Q: cancel")


def _brightness_bar(img: np.ndarray, org: tuple[int, int], face: float, width: int = 300) -> None:
    """Live face-brightness gauge (0-160) with the minimum (60) and recommended (100) marks."""
    x, y = org
    cv2.rectangle(img, (x, y - 8), (x + width, y + 8), (50, 50, 50), -1)
    fill = int(width * min(face, 160.0) / 160.0)
    cv2.rectangle(img, (x, y - 8), (x + fill, y + 8),
                  RED if face < LIGHT_MIN_FACE else AMBER if face < LIGHT_GOOD_FACE else GREEN, -1)
    for mark in (LIGHT_MIN_FACE, LIGHT_GOOD_FACE):
        mx = x + int(width * mark / 160.0)
        cv2.line(img, (mx, y - 12), (mx, y + 12), WHITE, 1, cv2.LINE_AA)
        _put(img, f"{mark:.0f}", (mx - 10, y - 16), GREY, 0.45)   # above the bar, clear of the text below


def draw_checklist(canvas: FullscreenCanvas, frame: np.ndarray | None, items: list[CheckItem],
                   ready: bool, tracked: bool, title: str, mode: str = "gate",
                   override_label: str = "use my current setup", notice: str | None = None) -> np.ndarray:
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
        c = y + 28 + 20 * len(lines) + 14             # value line right under the instruction
        _put(img, it.detail, (left + 60, c), _LEVEL_COLOR[it.level], 0.6)
        if it.hint:
            c += 26
            _put(img, "-> " + it.hint, (left + 60, c), AMBER if it.level == "adjust" else RED, 0.6, 2)
        for k, line in enumerate(textwrap.wrap("tip: " + it.tip, 105) if it.tip else []):
            c += 22                                    # non-blocking advice, never red
            _put(img, line, (left + 60, c), TIP, 0.55)
        if it.key == "light" and it.value is not None:
            _brightness_bar(img, (left + 360, y - 8), it.value)
        y = c + 44                                     # rows grow with their hint and tip
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
        status, color = ("READY: press SPACE to start" if mode == "gate" else "All steps green"), GREEN
    elif not todo:
        status, color = "Hold still...", GREEN
    else:
        status, color = f"{len(todo)} step{'s' if len(todo) > 1 else ''} to fix", AMBER
    size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    cv2.putText(img, status, (px + (PREVIEW[0] - size[0]) // 2, py + PREVIEW[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    note, keys = default_footer(mode, ready, override_label)
    if notice:  # e.g. SPACE pressed before everything was green
        cv2.putText(img, notice, (60, H - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 2, cv2.LINE_AA)
    if note:
        canvas.text(img, note, y=H - 84)
    canvas.text(img, keys, y=H - 40)
    return img


def _median(values: deque[dict[str, float]]) -> dict[str, float] | None:
    return {k: float(np.median([v[k] for v in values])) for k in values[0]} if values else None


def run_posture_check(cam: Camera, tracker: FaceTracker, canvas: FullscreenCanvas,
                      ref: Reference | None,
                      title: str = "Set up your position: each step turns green when it is right",
                      ref_lighting: dict | None = None, mode: str = "gate",
                      override_label: str = "use my current setup", note: str = "",
                      slot_guide: Callable[[dict[str, float] | None], CheckItem] | None = None) -> PostureResult:
    """Show the live checklist and return what the user chose.

    ``mode="gate"``: SPACE only works once every measured step has been green for a second (it
    means "I am in position, start"); until then it shows a notice. U is the single override: it
    proceeds with the current setup (``own_setup=True``, ``reference`` = the current posture, for
    a session recorded in the user's own setup). ESC or Q cancels. ``mode="monitor"``: a plain
    viewer where SPACE or ESC just exits. ``ref`` may be None for the very first session.
    ``slot_guide`` (recording a set-B position) maps the smoothed live metrics to the slot step.
    """
    gate = PostureGate()
    window: deque[dict[str, float]] = deque(maxlen=STILL_WINDOW_FRAMES)
    recent: deque[dict[str, float]] = deque(maxlen=SMOOTH_FRAMES)
    light: deque[dict[str, float]] = deque(maxlen=SMOOTH_FRAMES)
    notice_until = 0.0
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
        live = _median(recent)
        items = build_checklist(ref, live, list(window), _median(light), ref_lighting, note,
                                slot_guide(live) if slot_guide else None)
        now = time.monotonic()
        gate.update(now, checklist_ready(items))
        key = canvas.show(draw_checklist(
            canvas, frame.image, items, gate.in_position, result is not None, title, mode,
            override_label, "Not ready yet: fix the steps marked with a cross, or press U" if now < notice_until else None))
        if key in (27, ord("q")):
            return PostureResult(False, gate.in_position)
        if key == 32:
            if mode == "monitor" or gate.in_position:
                return PostureResult(True, gate.in_position)
            notice_until = now + 2.0
        elif key in (ord("u"), ord("U")) and mode == "gate" and len(window) >= STILL_MIN_FRAMES:
            return PostureResult(True, gate.in_position, own_setup=True,
                                 reference=Reference.from_current(list(window)))


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
