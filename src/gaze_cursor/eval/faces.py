"""Replayable face crops: native-resolution JPEG crops stored next to a session (local only).

Row i of a session has entry i here; rows without a face have an empty entry. Crops are the
unflipped frame, the face-mesh bounding box x1.5, squared. They exist so image-level fixes and
pretrained gaze models can be tested offline on recorded data. They are face images: they stay
on this machine (gitignored, file mode 0600) and are never uploaded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

QUALITY = 90
MARGIN = 1.5            # crop side = largest face-mesh bbox side * MARGIN
FPS = 30
# Measured JPEG size per thousand crop pixels (live camera, 455 px crop): q85 0.113 KB, q92 0.156 KB.
_KB_PER_KPX = {85: 0.113, 92: 0.156}


def faces_path_for(session_path: Path) -> Path:
    """``session_X.npz`` -> ``session_X.faces.npz``."""
    return session_path.with_name(session_path.stem + ".faces.npz")


def crop_box(points: np.ndarray, image_size: tuple[int, int], margin: float = MARGIN) -> tuple[int, int, int, int]:
    """Square crop (x0, y0, x1, y1) in frame pixels: face-mesh bbox x ``margin``, clamped inside
    the frame (the square shrinks if the frame is smaller than it)."""
    w, h = image_size
    xy = points[:468, :2] * (w, h)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    side = int(min(round(float((hi - lo).max()) * margin), w, h))
    cx, cy = (lo + hi) / 2
    x0 = int(np.clip(round(cx - side / 2), 0, w - side))
    y0 = int(np.clip(round(cy - side / 2), 0, h - side))
    return x0, y0, x0 + side, y0 + side


def encode_crop(image: np.ndarray, box: tuple[int, int, int, int], quality: int = QUALITY) -> bytes:
    """JPEG bytes of ``image[box]`` (BGR, native resolution, no resizing)."""
    x0, y0, x1, y1 = box
    ok, buf = cv2.imencode(".jpg", image[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def estimate_mb(n_frames: int, side: int = 316, quality: int = QUALITY) -> tuple[float, float]:
    """(low, high) MB for ``n_frames`` crops of ``side`` px, +-25% around the point estimate.
    Typical v2 session: ~750 frames, 316 px crops -> about 8-13 MB."""
    q = sorted(_KB_PER_KPX)
    lo, hi = q[0], q[-1]
    kb_per_kpx = _KB_PER_KPX[lo] + (_KB_PER_KPX[hi] - _KB_PER_KPX[lo]) * np.clip((quality - lo) / (hi - lo), 0, 1)
    mb = n_frames * kb_per_kpx * (side * side / 1000) / 1024
    return float(mb * 0.75), float(mb * 1.25)


def save_faces(path: Path, jpegs: list[bytes | None], boxes: list[tuple[int, int, int, int] | None],
               quality: int = QUALITY) -> Path:
    """Write the sidecar (``offsets``, concatenated JPEG ``data``, per-row ``box``) with mode 0600."""
    n = len(jpegs)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, j in enumerate(jpegs):
        offsets[i + 1] = offsets[i] + (len(j) if j else 0)
    data = np.frombuffer(b"".join(j for j in jpegs if j), dtype=np.uint8) if offsets[-1] else np.zeros(0, np.uint8)
    box = np.array([b if b else (-1, -1, -1, -1) for b in boxes], dtype=np.int32).reshape(n, 4)
    np.savez(path, offsets=offsets, data=data, box=box, quality=np.int16(quality))
    os.chmod(path, 0o600)
    return path


@dataclass(frozen=True)
class FaceCrops:
    """Loaded face crops of one session; ``image(i)`` decodes row i (None if it had no face)."""

    offsets: np.ndarray
    data: np.ndarray
    box: np.ndarray
    quality: int

    @classmethod
    def load(cls, path: Path) -> FaceCrops:
        with np.load(path) as d:
            return cls(d["offsets"], d["data"], d["box"], int(d["quality"]))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def has(self, i: int) -> bool:
        return self.offsets[i + 1] > self.offsets[i]

    def jpeg(self, i: int) -> bytes | None:
        return self.data[self.offsets[i]:self.offsets[i + 1]].tobytes() if self.has(i) else None

    def image(self, i: int) -> np.ndarray | None:
        j = self.jpeg(i)
        return None if j is None else cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
