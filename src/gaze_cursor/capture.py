"""Webcam capture with monotonic timestamps."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    """A captured BGR frame and its monotonic timestamp in milliseconds."""

    image: np.ndarray
    timestamp_ms: int


class Camera:
    """Context manager around ``cv2.VideoCapture``."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self._index = index
        self._size = (width, height)
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> Camera:
        cap = cv2.VideoCapture(self._index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open camera {self._index}. Check Camera permission for this app "
                "in System Settings > Privacy & Security > Camera."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap = cap
        return self

    def __exit__(self, *exc: object) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Frame | None:
        """Return the next frame, or None if the read failed."""
        assert self._cap is not None, "Camera used outside its context manager"
        ok, image = self._cap.read()
        if not ok:
            return None
        return Frame(image=image, timestamp_ms=time.monotonic_ns() // 1_000_000)
