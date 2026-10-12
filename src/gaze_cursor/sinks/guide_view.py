"""End-of-session feedback drawn on the fullscreen canvas."""

from __future__ import annotations

import cv2
import numpy as np

from gaze_cursor.posture import LABELS, METRICS
from gaze_cursor.sinks.overlay import FullscreenCanvas

AMBER, GREEN, RED, WHITE, GREY = (0, 170, 255), (90, 200, 90), (70, 70, 235), (235, 235, 235), (150, 150, 150)

def _put(img: np.ndarray, text: str, org: tuple[int, int], color=WHITE, scale: float = 0.7, thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_summary(canvas: FullscreenCanvas, summary: dict | None, saved: str,
                 lighting: dict | None = None) -> np.ndarray:
    """End-of-session feedback: how well the posture was held, per metric."""
    img = canvas.blank()
    left = max((canvas.size[0] - 900) // 2, 40)
    y = canvas.safe_top + 40
    ok = summary is not None and summary["frames"] > 0 and summary["fraction_good"] >= 0.9
    cv2.putText(img, "Session saved", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 2, cv2.LINE_AA)
    _put(img, saved, (left, y + 32), GREY, 0.55)
    y += 100
    if summary is None:
        _put(img, "Position was not compared (no earlier sessions to compare with).", (left, y), GREY, 0.8)
    elif summary["frames"] == 0:
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
    if lighting is not None:
        _put(img, f"Lighting captured: {lighting['label']} (face {lighting['face']:.0f}/255, background "
                  f"{lighting['background']:.0f}); saved with the session", (left, canvas.size[1] - 110), GREY, 0.62)
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
