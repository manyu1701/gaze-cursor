"""Drawing for the live tests in Run: metrics panel, target boxes, the path, the result screen."""

from __future__ import annotations

import cv2
import numpy as np

from gaze_cursor.livetest import Box, PathTarget
from gaze_cursor.sinks.overlay import FullscreenCanvas

WHITE, GREY, GREEN, AMBER, RED, BLUE = (235, 235, 235), (150, 150, 150), (90, 200, 90), (0, 170, 255), (70, 70, 235), (230, 160, 70)


def _put(img: np.ndarray, text: str, org: tuple[int, int], color=WHITE, scale: float = 0.55, thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_panel(canvas: FullscreenCanvas, img: np.ndarray, lines: list[tuple[str, tuple]], corner: str = "left") -> None:
    """A small dark panel of live metrics in a top corner, below the notch strip."""
    if not lines:
        return
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t, _ in lines) + 24
    h = 26 * len(lines) + 14
    x0 = 12 if corner == "left" else canvas.size[0] - w - 12
    y0 = canvas.safe_top + 6
    roi = img[y0:y0 + h, x0:x0 + w]
    img[y0:y0 + h, x0:x0 + w] = (roi * 0.35).astype(np.uint8)
    for k, (text, color) in enumerate(lines):
        _put(img, text, (x0 + 12, y0 + 24 + 26 * k), color)


def draw_box(canvas: FullscreenCanvas, img: np.ndarray, box: Box, dwell: float, inside: bool) -> None:
    """The target box; it fills from the bottom as the dwell completes."""
    x, y = canvas._px(box.center)
    h = int(box.half_px)
    color = GREEN if inside else WHITE
    if dwell > 0:
        top = int(y + h - 2 * h * dwell)
        cv2.rectangle(img, (x - h, top), (x + h, y + h), (40, 90, 40), -1)
    cv2.rectangle(img, (x - h, y - h), (x + h, y + h), color, 2, cv2.LINE_AA)
    cv2.line(img, (x - 6, y), (x + 6, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y - 6), (x, y + 6), color, 1, cv2.LINE_AA)


def draw_path(canvas: FullscreenCanvas, img: np.ndarray, path: PathTarget, t_s: float, close: bool) -> None:
    """The whole path faintly, and the moving target (green when the dot is close to it)."""
    pts = np.array([canvas._px(tuple(p)) for p in path.points], np.int32)
    cv2.polylines(img, [pts], False, (60, 60, 60), 1, cv2.LINE_AA)
    x, y = canvas._px(tuple(path.at(t_s)))
    cv2.circle(img, (x, y), 22, GREEN if close else WHITE, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 5, GREEN if close else WHITE, -1, cv2.LINE_AA)


def draw_gaze(canvas: FullscreenCanvas, img: np.ndarray, dot: np.ndarray | None, raw: np.ndarray | None) -> None:
    """The smoothed gaze dot (filled), and the raw prediction as a small grey point."""
    if raw is not None:
        cv2.circle(img, canvas._px((float(raw[0]), float(raw[1]))), 4, GREY, -1, cv2.LINE_AA)
    if dot is not None:
        cv2.circle(img, canvas._px((float(dot[0]), float(dot[1]))), 14, BLUE, -1, cv2.LINE_AA)


def result_screen(canvas: FullscreenCanvas, title: str, lines: list[tuple[str, tuple]],
                  boxes: list[dict] | None = None) -> np.ndarray:
    """End-of-test summary; for the targets test, each box is drawn with an arrow from its centre
    to where the dot actually was (red = missed)."""
    img = canvas.blank()
    if boxes:
        ppd_scale = 1.0
        for r in boxes:
            c = np.array(r["center"])
            half = r["size_deg"] / 2
            color = GREEN if r["hit"] else RED
            x, y = canvas._px(tuple(c))
            hp = int(half * _ppd(canvas))
            cv2.rectangle(img, (x - hp, y - hp), (x + hp, y + hp), (70, 70, 70), 1)
            if np.all(np.isfinite(r["offset_deg"])):
                end = c + np.array(r["offset_deg"]) * _ppd(canvas) * ppd_scale
                cv2.arrowedLine(img, (x, y), canvas._px(tuple(end)), color, 2, cv2.LINE_AA, tipLength=0.2)
    y = canvas.safe_top + 60
    w = max([cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for t, _ in lines] + [400]) + 40
    y0, y1 = y - 40, y + 44 + 32 * len(lines)
    img[y0:y1, 40:40 + w] = (img[y0:y1, 40:40 + w] * 0.25).astype(np.uint8)   # backing for the text
    cv2.putText(img, title, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
    for k, (text, color) in enumerate(lines):
        _put(img, text, (60, y + 44 + 32 * k), color, 0.7, 2)
    canvas.text(img, "SPACE: back to the gaze dot     T: targets again     P: path again     q: quit", y=canvas.size[1] - 40)
    return img


def _ppd(canvas: FullscreenCanvas) -> float:
    cfg = canvas.cfg
    return float(2 * cfg.distance_cm * np.tan(np.radians(0.5)) / cfg.cm_per_px)
