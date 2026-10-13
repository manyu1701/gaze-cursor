"""Smoothing filters for the 2-D gaze point (M2 step 6). Pure, timestamp-based, one sample at a time.

Every filter has ``update(t_ms, xy) -> xy`` and ``reset()``. A missing sample (blink, lost face) is
simply not passed in; the caller keeps showing the last output.

- ``MovingAverage``: mean of the last n samples (Run up to M2 step 6, n = 5).
- ``OneEuroFilter`` (Casiez et al. 2012): a low-pass whose cutoff rises with speed, so it smooths
  hard while the eyes fixate and follows quickly when they jump. ``min_cutoff`` (Hz) sets the
  fixation smoothing, ``beta`` (per px/s) how fast the cutoff opens with speed.
- ``snap_px`` / ``snap_frames`` (One-Euro only): if the raw point stays more than ``snap_px`` from
  the output for ``snap_frames`` samples in a row, treat it as a saccade and jump to it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


class Raw:
    """No smoothing."""

    def update(self, t_ms: float, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=float)

    def reset(self) -> None:
        pass


@dataclass
class MovingAverage:
    n: int = 5
    _buf: deque = field(default_factory=deque, repr=False)

    def update(self, t_ms: float, xy: np.ndarray) -> np.ndarray:
        self._buf.append(np.asarray(xy, dtype=float))
        while len(self._buf) > self.n:
            self._buf.popleft()
        return np.mean(self._buf, axis=0)

    def reset(self) -> None:
        self._buf.clear()


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


@dataclass
class OneEuroFilter:
    min_cutoff: float = 1.0     # Hz
    beta: float = 0.0           # per (px / s)
    d_cutoff: float = 1.0       # Hz, for the speed estimate
    snap_px: float | None = None
    snap_frames: int = 2
    _t: float | None = field(default=None, repr=False)
    _x: np.ndarray | None = field(default=None, repr=False)
    _dx: np.ndarray | None = field(default=None, repr=False)
    _far: int = field(default=0, repr=False)

    def update(self, t_ms: float, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if self._x is None or self._t is None:
            self._t, self._x, self._dx, self._far = t_ms, xy.copy(), np.zeros(2), 0
            return self._x.copy()
        if self.snap_px is not None:
            self._far = self._far + 1 if np.linalg.norm(xy - self._x) > self.snap_px else 0
            if self._far >= self.snap_frames:
                self._t, self._x, self._dx, self._far = t_ms, xy.copy(), np.zeros(2), 0
                return self._x.copy()
        dt = max((t_ms - self._t) / 1000.0, 1e-3)
        dx = (xy - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = _alpha(cutoff, dt)
        self._x = a * xy + (1 - a) * self._x
        self._t = t_ms
        return self._x.copy()

    def reset(self) -> None:
        self._t = self._x = self._dx = None
        self._far = 0


# Chosen for Run by the M2 step-6 rule (PROGRESS.md, `scripts/tune_smoothing.py`, 2026-10-13):
# lowest development jitter with median lag50 <= 100 ms and lag90 <= 250 ms.
RUN_FILTER: dict | None = {"kind": "oneeuro", "min_cutoff": 0.05, "beta": 0.001, "d_cutoff": 1.0}


def make_filter(spec: dict | None):
    """A filter from a spec dict: {"kind": "raw" | "ma" | "oneeuro", ...params}. None -> MovingAverage(5)."""
    if spec is None:
        return MovingAverage(5)
    kind = spec["kind"]
    params = {k: v for k, v in spec.items() if k != "kind"}
    if kind == "raw":
        return Raw()
    if kind == "ma":
        return MovingAverage(**params)
    if kind == "oneeuro":
        return OneEuroFilter(**params)
    raise ValueError(f"unknown filter kind {kind!r}")
