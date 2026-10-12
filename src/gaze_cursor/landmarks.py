"""MediaPipe FaceLandmarker wrapper and head-pose math."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from gaze_cursor.capture import Frame

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"

# Indices into the 478-point output: 0-467 mesh, 468-472 right iris, 473-477 left iris.
RIGHT_IRIS = (468, 469, 470, 471, 472)
LEFT_IRIS = (473, 474, 475, 476, 477)


@dataclass(frozen=True)
class LandmarkResult:
    """Landmarks for one face. ``points`` is (478, 3), x/y normalized to [0, 1]."""

    points: np.ndarray
    transform: np.ndarray  # 4x4 canonical-face -> camera matrix
    blendshapes: np.ndarray  # (52,) scores in [0, 1], order = FaceTracker.blendshape_names


def head_pose_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees from a 4x4 (or 3x3) transform.

    Assumes R = Ry(yaw) @ Rx(pitch) @ Rz(roll). Any uniform scale is removed first.
    """
    r = np.asarray(m, dtype=float)[:3, :3]
    r = r / np.cbrt(abs(np.linalg.det(r)))
    yaw = math.atan2(r[0, 2], r[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -r[1, 2])))
    roll = math.atan2(r[1, 0], r[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class FaceTracker:
    """Single-face landmark tracker (Tasks API, VIDEO running mode)."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run `uv run scripts/download_model.py` first."
            )
        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1
        self.blendshape_names: tuple[str, ...] = ()  # filled on the first detected face

    def process(self, frame: Frame) -> LandmarkResult | None:
        """Run the landmarker on a frame; None if no face is found."""
        # VIDEO mode requires strictly increasing timestamps.
        ts = max(frame.timestamp_ms, self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        points = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)
        transform = np.asarray(result.facial_transformation_matrixes[0], dtype=float)
        categories = result.face_blendshapes[0]
        if not self.blendshape_names:
            self.blendshape_names = tuple(c.category_name for c in categories)
        blend = np.array([c.score for c in categories], dtype=np.float32)
        return LandmarkResult(points=points, transform=transform, blendshapes=blend)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
