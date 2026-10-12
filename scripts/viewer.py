"""Live webcam viewer: face mesh, iris landmarks, head pose, FPS. `q` quits, `s` saves a PNG."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from gaze_cursor.capture import Camera
from gaze_cursor.landmarks import (
    LEFT_IRIS,
    RIGHT_IRIS,
    FaceTracker,
    LandmarkResult,
    head_pose_from_matrix,
)

WINDOW = "gaze-cursor viewer"
DEBUG_DIR = Path(__file__).resolve().parents[1] / "debug"
_TESS = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION


def draw(image: np.ndarray, result: LandmarkResult, fps: float) -> None:
    """Draw mesh, irises, and head-pose/FPS text onto ``image`` in place."""
    h, w = image.shape[:2]
    pts = (result.points[:, :2] * (w, h)).astype(np.int32)
    for c in _TESS:
        cv2.line(image, tuple(pts[c.start]), tuple(pts[c.end]), (90, 90, 90), 1, cv2.LINE_AA)
    for iris in (RIGHT_IRIS, LEFT_IRIS):
        center = tuple(pts[iris[0]])
        radius = int(np.linalg.norm(pts[iris[1]] - pts[iris[0]]))
        cv2.circle(image, center, max(radius, 2), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(image, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
    yaw, pitch, roll = head_pose_from_matrix(result.transform)
    cv2.putText(image, f"yaw {yaw:+6.1f}  pitch {pitch:+6.1f}  roll {roll:+6.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    times: deque[float] = deque(maxlen=30)
    with Camera() as cam, FaceTracker() as tracker:
        try:
            while True:
                frame = cam.read()
                if frame is None:
                    continue
                image = cv2.flip(frame.image, 1)  # mirror for natural feel
                frame = type(frame)(image=image, timestamp_ms=frame.timestamp_ms)
                result = tracker.process(frame)
                now = time.perf_counter()
                times.append(now)
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
                if result is not None:
                    draw(image, result, fps)
                else:
                    cv2.putText(image, "no face", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"{fps:5.1f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW, image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    DEBUG_DIR.mkdir(exist_ok=True)
                    path = DEBUG_DIR / f"viewer_{int(time.time())}.png"
                    cv2.imwrite(str(path), image)
                    print(f"saved {path}")
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
