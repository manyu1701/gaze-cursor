"""Fullscreen OpenCV canvas for calibration targets and the gaze dot (debug sink).

All positions passed to the drawing helpers are in macOS screen points (origin top-left of the
main display). On notch MacBooks the fullscreen window starts below the menu bar, so the canvas
is sized to the window's real bounds (read from Quartz) and positions are shifted by its origin.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from gaze_cursor.config import ScreenConfig


def own_window_bounds() -> tuple[int, int, int, int] | None:
    """(x, y, w, h) in screen points of this process's largest on-screen window, or None."""
    import Quartz

    best = None
    for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID):
        if w.get("kCGWindowOwnerPID") != os.getpid() or w.get("kCGWindowLayer") != 0:
            continue
        b = w["kCGWindowBounds"]
        area = b["Width"] * b["Height"]
        if best is None or area > best[0]:
            best = (area, (int(b["X"]), int(b["Y"]), int(b["Width"]), int(b["Height"])))
    return best[1] if best else None


class FullscreenCanvas:
    """Fullscreen window drawn 1:1 in screen points. ``origin`` is the window's top-left."""

    def __init__(self, cfg: ScreenConfig, name: str = "gaze-cursor") -> None:
        self.cfg = cfg
        self.name = name
        self.origin = (0, 0)
        self.size = (cfg.width_px, cfg.height_px)  # (w, h) of the drawable area

    def __enter__(self) -> FullscreenCanvas:
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._locate()
        return self

    def _locate(self, timeout_s: float = 6.0) -> None:
        """Wait for the fullscreen animation to settle, then adopt the window's real bounds."""
        start, last, stable = time.monotonic(), None, 0
        while time.monotonic() - start < timeout_s:
            cv2.imshow(self.name, self.blank())
            cv2.waitKey(80)
            bounds = own_window_bounds()
            stable = stable + 1 if bounds and bounds == last else 0
            last = bounds
            # Bounds can sit at the full screen at first, then move when the animation ends,
            # so require a minimum time plus several identical readings.
            if bounds and stable >= 8 and time.monotonic() - start > 1.5 and bounds[3] > 0.8 * self.cfg.height_px:
                self.origin, self.size = (bounds[0], bounds[1]), (bounds[2], bounds[3])
                return
        print("warning: could not read the fullscreen window bounds; assuming the full screen")

    def __exit__(self, *exc: object) -> None:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def blank(self) -> np.ndarray:
        return np.full((self.size[1], self.size[0], 3), 24, np.uint8)

    def _px(self, xy: tuple[float, float]) -> tuple[int, int]:
        return int(round(xy[0] - self.origin[0])), int(round(xy[1] - self.origin[1]))

    def visible(self, xy: tuple[float, float]) -> bool:
        x, y = self._px(xy)
        return 0 <= x < self.size[0] and 0 <= y < self.size[1]

    def dot(self, img: np.ndarray, xy: tuple[float, float], color=(255, 255, 255), r: int = 16) -> None:
        """Draw a target ring at screen position ``xy``."""
        c = self._px(xy)
        cv2.circle(img, c, r, color, 2, cv2.LINE_AA)
        cv2.circle(img, c, 3, (0, 0, 255), -1, cv2.LINE_AA)

    def text(self, img: np.ndarray, msg: str, y: int | None = None) -> None:
        """Centered text; ``y`` is in canvas pixels from the top of the window."""
        y = self.size[1] // 2 if y is None else y
        size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.putText(img, msg, ((self.size[0] - size[0]) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2, cv2.LINE_AA)

    def show(self, img: np.ndarray) -> int:
        """Display and return the key pressed (-1 if none)."""
        cv2.imshow(self.name, img)
        return cv2.waitKey(1) & 0xFF
