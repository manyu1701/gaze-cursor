"""Record/replay of calibration and test sessions (one .npz per session)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gaze_cursor.features import N_FEATURES, assemble, extract_features

SESSIONS_DIR = Path(__file__).resolve().parents[3] / "data" / "sessions"
CALIB, TEST = 0, 1
CLEAN_WINDOW = [0, 0, 1470, 956]
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

    @property
    def is_clean(self) -> bool:
        """True if targets were drawn 1:1 on the full display (borderless window at 0,0,1470,956).

        Earlier sessions used an OpenCV window that squeezed or shifted the canvas, so their
        target positions are biased and must not be used for evaluation.
        """
        return list(self.meta.get("window") or []) == CLEAN_WINDOW

    @property
    def layout(self) -> str:
        return str(self.meta.get("layout", "v1"))

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean()) if len(self.valid) else 0.0

    def select_all(
        self, feature_set: str = "iris"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Valid rows of every phase: (model input, target_xy, target_idx, phase)."""
        m = self.valid
        names = tuple(self.meta.get("blendshape_names", ()))
        x = assemble(self.features[m], self.blendshapes[m], names, feature_set)
        return x, self.target_xy[m], self.target_idx[m], self.phase[m]

    def select(
        self, phase: int, feature_set: str = "iris"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Valid rows of one phase: (model input for ``feature_set``, target_xy, target_idx)."""
        m = self.valid & (self.phase == phase)
        names = tuple(self.meta.get("blendshape_names", ()))
        x = assemble(self.features[m], self.blendshapes[m], names, feature_set)
        return x, self.target_xy[m], self.target_idx[m]


def _features_or_nan(points: np.ndarray, transform: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    f = None if np.isnan(points).any() else extract_features(points, transform, size)
    return np.full(N_FEATURES, np.nan) if f is None else f


DEFAULT_IMAGE_SIZE = (1280, 720)  # camera frame size for sessions recorded before it was stored


def load_session(path: Path, recompute_features: bool = True) -> Session:
    """Load a session. Features are recomputed from the stored landmarks by default, so
    improvements to feature extraction apply to old recordings (record and replay)."""
    with np.load(path) as d:
        meta = json.loads(str(d["meta"]))
        features = d["features"]
        if recompute_features:
            size = tuple(meta.get("image_size", DEFAULT_IMAGE_SIZE))
            features = np.stack([_features_or_nan(p, t, size)
                                 for p, t in zip(d["points"], d["transform"])]) if len(features) else features
        return Session(
            path=path, meta=meta, t_ms=d["t_ms"], phase=d["phase"],
            target_idx=d["target_idx"], target_xy=d["target_xy"], features=features,
            blendshapes=d["blendshapes"],
        )


def clean_sessions(directory: Path = SESSIONS_DIR) -> list[Session]:
    """All sessions whose targets were drawn 1:1 on the display, oldest first."""
    sessions = [load_session(p) for p in sorted(directory.glob("session_*.npz"))]
    return [s for s in sessions if s.is_clean]


def latest_session(directory: Path = SESSIONS_DIR) -> Path:
    found = sorted(directory.glob("session_*.npz"))
    if not found:
        raise FileNotFoundError(f"No sessions in {directory}. Run scripts/calibrate.py first.")
    return found[-1]
