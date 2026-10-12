"""Record/replay of calibration and test sessions (one .npz per session)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gaze_cursor.features import N_FEATURES, assemble

SESSIONS_DIR = Path(__file__).resolve().parents[3] / "data" / "sessions"
CALIB, TEST = 0, 1
N_BLEND = 52  # FaceLandmarker blendshape count; names are stored in meta["blendshape_names"]


@dataclass
class SessionRecorder:
    """Accumulates per-frame rows; ``save`` writes them to data/sessions/<id>.npz."""

    meta: dict
    _rows: list[tuple] = field(default_factory=list)

    def add(
        self, t_ms: int, phase: int, target_idx: int, target_xy: tuple[float, float],
        points: np.ndarray | None, transform: np.ndarray | None, features: np.ndarray | None,
        blendshapes: np.ndarray | None = None,
    ) -> None:
        self._rows.append(
            (t_ms, phase, target_idx, target_xy, points, transform, features, blendshapes)
        )

    def save(self, directory: Path = SESSIONS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"session_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        n = len(self._rows)
        nan = np.nan
        np.savez_compressed(
            path,
            t_ms=np.array([r[0] for r in self._rows], dtype=np.int64),
            phase=np.array([r[1] for r in self._rows], dtype=np.int8),
            target_idx=np.array([r[2] for r in self._rows], dtype=np.int16),
            target_xy=np.array([r[3] for r in self._rows], dtype=np.float32).reshape(n, 2),
            points=np.stack([r[4] if r[4] is not None else np.full((478, 3), nan, np.float32) for r in self._rows]) if n else np.zeros((0, 478, 3), np.float32),
            transform=np.stack([r[5] if r[5] is not None else np.full((4, 4), nan) for r in self._rows]) if n else np.zeros((0, 4, 4)),
            features=np.stack([r[6] if r[6] is not None else np.full(N_FEATURES, nan) for r in self._rows]) if n else np.zeros((0, N_FEATURES)),
            blendshapes=np.stack([r[7] if r[7] is not None else np.full(N_BLEND, nan, np.float32) for r in self._rows]) if n else np.zeros((0, N_BLEND), np.float32),
            meta=json.dumps(self.meta),
        )
        return path


@dataclass(frozen=True)
class Session:
    """A loaded session. Rows with ``valid`` False had no face or a closed eye."""

    path: Path
    meta: dict
    t_ms: np.ndarray
    phase: np.ndarray
    target_idx: np.ndarray
    target_xy: np.ndarray
    features: np.ndarray
    blendshapes: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.features).any(axis=1) & ~np.isnan(self.blendshapes).any(axis=1)

    def select(
        self, phase: int, feature_set: str = "iris"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Valid rows of one phase: (model input for ``feature_set``, target_xy, target_idx)."""
        m = self.valid & (self.phase == phase)
        names = tuple(self.meta.get("blendshape_names", ()))
        x = assemble(self.features[m], self.blendshapes[m], names, feature_set)
        return x, self.target_xy[m], self.target_idx[m]


def load_session(path: Path) -> Session:
    with np.load(path) as d:
        return Session(
            path=path, meta=json.loads(str(d["meta"])), t_ms=d["t_ms"], phase=d["phase"],
            target_idx=d["target_idx"], target_xy=d["target_xy"], features=d["features"],
            blendshapes=d["blendshapes"],
        )


def latest_session(directory: Path = SESSIONS_DIR) -> Path:
    found = sorted(directory.glob("session_*.npz"))
    if not found:
        raise FileNotFoundError(f"No sessions in {directory}. Run scripts/calibrate.py first.")
    return found[-1]
