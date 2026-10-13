"""Drawing for Cursor mode's click-through overlay (BGRA images; alpha 0 = see-through)."""

from __future__ import annotations

import cv2
import numpy as np

GREEN, AMBER, GREY, RED = (90, 200, 90, 230), (0, 170, 255, 230), (170, 170, 170, 230), (60, 60, 230, 230)


def draw_status(img: np.ndarray, lines: list[tuple[str, tuple]], top: int, right_margin: int = 16) -> None:
    """A compact chip in the top-right corner: one text line per (text, BGRA colour)."""
    if not lines:
        return
    scale, thick, pad, step = 0.5, 1, 8, 20
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t, _ in lines)
    x1 = img.shape[1] - right_margin
    x0 = x1 - width - 2 * pad
    y0, y1 = top, top + pad + step * len(lines)
    cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20, 170), -1)
    for k, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x0 + pad, y0 + pad + 12 + step * k), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thick, cv2.LINE_AA)


def draw_gaze_dot(img: np.ndarray, xy: tuple[float, float], r: int = 7) -> None:
    """A faint ring where the tracker thinks you are looking (optional; off by default)."""
    c = (int(round(xy[0])), int(round(xy[1])))
    cv2.circle(img, c, r, (255, 255, 0, 140), 2, cv2.LINE_AA)
