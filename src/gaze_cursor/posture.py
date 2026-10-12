"""Posture check: is the user sitting like they did in the recorded sessions? Pure functions.

The reference is computed from recorded sessions (median and robust spread per metric), so it
updates itself as sessions are added. Lighting is reported but never gates the check: sessions are
recorded in deliberately different lighting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from gaze_cursor.landmarks import head_pose_from_matrix

METRICS = ("pitch", "yaw", "roll", "distance", "x", "y")
LABELS = {"pitch": "head pitch", "yaw": "head yaw", "roll": "head roll",
          "distance": "distance (inter-iris)", "x": "face x in view", "y": "face y in view"}
# Minimum tolerance (one "scale" unit) per metric. HEURISTICS, not measured limits: a metric with
# little spread in the reference sessions would otherwise be impossible to satisfy. ``distance``
# is relative to its reference value, the others are absolute.
FLOORS = {"pitch": 2.5, "yaw": 4.0, "roll": 3.0, "distance": 0.06, "x": 0.05, "y": 0.05}
GOOD_Z, ADJUST_Z = 1.5, 3.0  # |z| <= GOOD_Z: good; <= ADJUST_Z: adjust; beyond: off


def posture_metrics(points: np.ndarray, transform: np.ndarray,
                    image_size: tuple[int, int]) -> dict[str, float]:
    """Head pose (deg, unflipped frame), distance proxy and face center for one frame.

    ``distance`` is the inter-iris distance as a fraction of the frame width (larger = closer).
    ``x``/``y`` are the mean mesh position in the unflipped frame, normalized to [0, 1].
    """
    w, h = image_size
    iod = float(np.linalg.norm((points[468, :2] - points[473, :2]) * (w, h)) / w)
    cx, cy = points[:468, :2].mean(axis=0)
    yaw, pitch, roll = head_pose_from_matrix(transform)
    return {"pitch": pitch, "yaw": yaw, "roll": roll, "distance": iod,
            "x": float(cx), "y": float(cy)}


def lighting_stats(gray: np.ndarray, points: np.ndarray) -> dict[str, float]:
    """Face and background brightness (0-255), their ratio, and the fraction of clipped face pixels."""
    h, w = gray.shape
    xs, ys = points[:468, 0] * w, points[:468, 1] * h
    x0, x1 = int(np.clip(xs.min(), 0, w - 1)), int(np.clip(xs.max(), 1, w))
    y0, y1 = int(np.clip(ys.min(), 0, h - 1)), int(np.clip(ys.max(), 1, h))
    face = gray[y0:y1, x0:x1].astype(float)
    mask = np.ones(gray.shape, bool)
    mask[y0:y1, x0:x1] = False
    bg = float(gray[mask].mean()) if mask.any() else float("nan")
    return {"face": float(face.mean()), "background": bg,
            "ratio": float(face.mean() / bg) if bg else float("nan"),
            "clipped": float((face >= 250).mean())}


def lighting_warning(stats: dict[str, float]) -> str | None:
    """Only extremes are flagged (informational, never blocks)."""
    if stats["face"] < 35:
        return "very dark: tracking may fail"
    if stats["face"] > 220 or stats["clipped"] > 0.05:
        return "face overexposed"
    return None


@dataclass(frozen=True)
class Reference:
    """Where the user sat in the reference sessions: per-metric center and tolerance scale."""

    center: dict[str, float]
    scale: dict[str, float]
    n_sessions: int = 0
    n_frames: int = 0
    session_names: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_arrays(cls, values: dict[str, np.ndarray], **kw) -> Reference:
        """Median and robust spread (1.4826 * MAD, floored) of per-frame metric arrays."""
        center, scale = {}, {}
        for m in METRICS:
            v = np.asarray(values[m], dtype=float)
            c = float(np.median(v))
            mad = 1.4826 * float(np.median(np.abs(v - c)))
            floor = FLOORS[m] * (abs(c) if m == "distance" else 1.0)
            center[m], scale[m] = c, max(mad, floor)
        return cls(center, scale, **kw)

    @classmethod
    def from_sessions(cls, sessions: Sequence, stride: int = 3) -> Reference:
        """Pool the valid frames of ``sessions`` (every ``stride``-th frame)."""
        cols: dict[str, list[float]] = {m: [] for m in METRICS}
        used = 0
        for s in sessions:
            if s.points is None or s.transform is None:
                continue
            size = tuple(s.meta.get("image_size", (1280, 720)))
            idx = np.flatnonzero(s.valid)[::stride]
            for i in idx:
                for m, v in posture_metrics(s.points[i], s.transform[i], size).items():
                    cols[m].append(v)
            used += 1
        if not cols["pitch"]:
            raise ValueError("no valid frames with landmarks in the reference sessions")
        return cls.from_arrays({m: np.array(v) for m, v in cols.items()}, n_sessions=used,
                               n_frames=len(cols["pitch"]),
                               session_names=tuple(s.path.name for s in sessions))

    def z(self, metrics: dict[str, float]) -> dict[str, float]:
        return {m: (metrics[m] - self.center[m]) / self.scale[m] for m in METRICS}

    @staticmethod
    def level(z: float) -> str:
        a = abs(z)
        return "good" if a <= GOOD_Z else "adjust" if a <= ADJUST_Z else "off"

    def worst(self, metrics: dict[str, float]) -> tuple[str, str, float]:
        """(metric, level, z) of the metric furthest from the reference."""
        z = self.z(metrics)
        m = max(z, key=lambda k: abs(z[k]))
        return m, self.level(z[m]), z[m]

    def hint(self, metric: str, live: float) -> str:
        """A short instruction for one metric. Directions are only given where unambiguous."""
        d = live - self.center[metric]
        if metric == "distance":
            return "move back a little" if d > 0 else "move closer"
        if metric == "pitch":
            return f"{live:+.0f} deg, usually {self.center[metric]:+.0f}: adjust screen angle or seat height"
        if metric == "yaw":
            return "face the screen centre"
        if metric == "roll":
            return "level your head"
        return "centre your face in the camera view"


@dataclass
class PostureGate:
    """Debounce: 'in position' after ``hold_s`` of all-good readings; lost after ``release_s`` not good."""

    hold_s: float = 1.0
    release_s: float = 0.5
    in_position: bool = False
    _good_since: float | None = None
    _bad_since: float | None = None

    def update(self, now: float, all_good: bool) -> bool:
        if all_good:
            self._bad_since = None
            if self._good_since is None:
                self._good_since = now
            if not self.in_position and now - self._good_since >= self.hold_s:
                self.in_position = True
        else:
            self._good_since = None
            if self._bad_since is None:
                self._bad_since = now
            if self.in_position and now - self._bad_since >= self.release_s:
                self.in_position = False
        return self.in_position


def summarize_recording(ref: Reference, frames: Sequence[dict[str, float]]) -> dict:
    """How well did the user hold the reference posture during a recording?

    ``frames`` holds one metrics dict per recorded frame. Returns the fraction of frames with
    every metric good, the fraction of frames each metric was not good, the metric that drifted
    most, and a one-line verdict. Purely descriptive: it never changes or filters the data.
    """
    if not frames:
        return {"frames": 0, "fraction_good": 0.0, "verdict": "no face was tracked during the recording",
                "worst_metric": None, "not_good_fraction": {}, "mean_z": {}}
    z = np.array([[ref.z(f)[m] for m in METRICS] for f in frames])
    not_good = np.abs(z) > GOOD_Z
    frac_good = float((~not_good.any(axis=1)).mean())
    per_metric = {m: float(not_good[:, i].mean()) for i, m in enumerate(METRICS)}
    mean_z = {m: float(z[:, i].mean()) for i, m in enumerate(METRICS)}
    worst = max(METRICS, key=lambda m: abs(mean_z[m]) + per_metric[m])
    if frac_good >= 0.9:
        verdict = "good session: you held your position"
    elif frac_good >= 0.7:
        verdict = f"mostly in position; {LABELS[worst]} drifted"
    else:
        verdict = f"you drifted out of position ({LABELS[worst]} most); consider recording again"
    return {"frames": len(frames), "fraction_good": frac_good, "verdict": verdict,
            "worst_metric": worst, "not_good_fraction": per_metric, "mean_z": mean_z}
