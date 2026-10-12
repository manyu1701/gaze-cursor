"""Fullscreen OpenCV canvas for calibration targets and the gaze dot (debug sink)."""

from __future__ import annotations

import cv2
import numpy as np

from gaze_cursor.config import ScreenConfig


class FullscreenCanvas:
    """Fullscreen window drawn in logical screen pixels."""

    def __init__(self, cfg: ScreenConfig, name: str = "gaze-cursor") -> None:
        self.cfg = cfg
        self.name = name

    def __enter__(self) -> FullscreenCanvas:
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return self

    def __exit__(self, *exc: object) -> None:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def blank(self) -> np.ndarray:
        return np.full((self.cfg.height_px, self.cfg.width_px, 3), 24, np.uint8)

    def dot(self, img: np.ndarray, xy: tuple[float, float], color=(255, 255, 255), r: int = 16) -> None:
        c = (int(xy[0]), int(xy[1]))
        cv2.circle(img, c, r, color, 2, cv2.LINE_AA)
        cv2.circle(img, c, 3, (0, 0, 255), -1, cv2.LINE_AA)

    def text(self, img: np.ndarray, msg: str, y: int | None = None) -> None:
        y = self.cfg.height_px // 2 if y is None else y
        size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.putText(img, msg, ((self.cfg.width_px - size[0]) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2, cv2.LINE_AA)

    def show(self, img: np.ndarray) -> int:
        """Display and return the key pressed (-1 if none)."""
        cv2.imshow(self.name, img)
        return cv2.waitKey(1) & 0xFF
