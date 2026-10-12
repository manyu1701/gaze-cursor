"""Setup guidance and end-of-session feedback drawn on the fullscreen canvas."""

from __future__ import annotations

import cv2
import numpy as np

from gaze_cursor.posture import LABELS, METRICS
from gaze_cursor.sinks.overlay import FullscreenCanvas

AMBER, GREEN, RED, WHITE, GREY = (0, 170, 255), (90, 200, 90), (70, 70, 235), (235, 235, 235), (150, 150, 150)

# Reproducibility matters more than any single "ideal": the same physical setup every time.
SETUP_STEPS = (
    ("Sit all the way back", "Your back rests against the chair the whole time."),
    ("About an arm's length from the screen", "Roughly 60 cm: stretch your arm out and your fingertips should just reach it."),
    ("Laptop on a stable desk", "Not on your lap, lid at the same angle as your earlier sessions. Don't touch it while recording."),
    ("Face the screen squarely", "Head level, shoulders relaxed, chin neither up nor down."),
    ("Light on your face", "No bright window or lamp behind you; avoid glare on glasses."),
    ("Keep your head still, move only your eyes", "Blink normally and look at the centre of each ring until it moves."),
)


def _put(img: np.ndarray, text: str, org: tuple[int, int], color=WHITE, scale: float = 0.7, thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_instructions(canvas: FullscreenCanvas, title: str = "Before we start: set up your position") -> np.ndarray:
    """The setup checklist (all text below the camera notch)."""
    img = canvas.blank()
    left = max((canvas.size[0] - 1000) // 2, 40)
    y = canvas.safe_top + 40
    cv2.putText(img, title, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
    y += 60
    for i, (head, detail) in enumerate(SETUP_STEPS, 1):
        cv2.circle(img, (left + 14, y - 8), 16, AMBER, 2, cv2.LINE_AA)
        _put(img, str(i), (left + 8, y - 1), AMBER, 0.6, 2)
        _put(img, head, (left + 50, y), WHITE, 0.85, 2)
        _put(img, detail, (left + 50, y + 32), GREY, 0.62)
        y += 88
    _put(img, "Next: a position check against your earlier sessions.", (left, y + 10), GREY, 0.62)
    canvas.text(img, "SPACE: continue     ESC or Q: cancel", y=canvas.size[1] - 40)
    return img


def draw_summary(canvas: FullscreenCanvas, summary: dict, saved: str) -> np.ndarray:
    """End-of-session feedback: how well the posture was held, per metric."""
    img = canvas.blank()
    left = max((canvas.size[0] - 900) // 2, 40)
    y = canvas.safe_top + 40
    ok = summary["frames"] > 0 and summary["fraction_good"] >= 0.9
    cv2.putText(img, "Session saved", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 2, cv2.LINE_AA)
    _put(img, saved, (left, y + 32), GREY, 0.55)
    y += 100
    if summary["frames"] == 0:
        _put(img, summary["verdict"], (left, y), RED, 0.9, 2)
    else:
        color = GREEN if ok else AMBER if summary["fraction_good"] >= 0.7 else RED
        _put(img, f"In position for {summary['fraction_good'] * 100:.0f}% of the recording", (left, y), color, 0.95, 2)
        _put(img, summary["verdict"], (left, y + 40), color, 0.8)
        y += 110
        _put(img, "Share of frames each measure was out of its usual range", (left, y), GREY, 0.62)
        y += 36
        for m in METRICS:
            frac = summary["not_good_fraction"][m]
            _put(img, LABELS[m], (left, y + 5), WHITE, 0.65)
            cv2.rectangle(img, (left + 280, y - 10), (left + 280 + 360, y + 12), (50, 50, 50), -1)
            cv2.rectangle(img, (left + 280, y - 10), (left + 280 + int(360 * frac), y + 12),
                          GREEN if frac < 0.1 else AMBER if frac < 0.3 else RED, -1)
            _put(img, f"{frac * 100:.0f}%", (left + 660, y + 5), WHITE, 0.65)
            y += 38
    canvas.text(img, "SPACE or ESC: finish", y=canvas.size[1] - 40)
    return img


def wait_for_key(canvas: FullscreenCanvas, make_image, keys: tuple[int, ...], cancel: tuple[int, ...] = ()) -> bool:
    """Show ``make_image()`` until one of ``keys`` (returns True) or ``cancel`` (False) is pressed."""
    while True:
        key = canvas.show(make_image())
        if key in cancel:
            return False
        if key in keys:
            return True
