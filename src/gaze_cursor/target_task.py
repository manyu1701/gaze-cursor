"""Target practice: move the gaze dot into a box and hold it there. Pure logic, no camera or UI.

A box appears; the trial is a hit once the (smoothed) gaze dot has stayed inside it for ``dwell_s``
(brief exits shorter than ``grace_s`` are forgiven), and a miss after ``timeout_s``. The box size
adapts with a 2-down / 1-up staircase (two hits in a row -> smaller, a miss -> larger), which settles
near the size hit about 71% of the time: the smallest target you can reliably hit with your eyes.

While the box is up you are looking at it, so every frame after ``settle_s`` (time to find the box
and for the dot to arrive) is a labelled gaze sample: its offset from the box centre measures the
model's systematic bias (where the dot sits relative to where you look) and its accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gaze_cursor.config import ScreenConfig


@dataclass
class Trial:
    index: int
    center: tuple[float, float]          # screen points
    half: float                          # half the box side, screen points
    t_start: float
    t_end: float | None = None
    hit: bool | None = None              # None while running
    offsets: list[tuple[float, float]] = field(default_factory=list)   # dot - centre after settling

    @property
    def time_to_hit(self) -> float | None:
        return self.t_end - self.t_start if self.hit else None

    @property
    def mean_offset(self) -> tuple[float, float] | None:
        return tuple(np.mean(self.offsets, axis=0)) if self.offsets else None


@dataclass
class TaskConfig:
    n_trials: int = 20
    start_half: float = 110.0      # points (about 4.4 cm wide on a 13.6" Air)
    min_half: float = 15.0
    max_half: float = 300.0
    shrink: float = 0.8
    grow: float = 1.25
    dwell_s: float = 0.6
    grace_s: float = 0.15
    timeout_s: float = 8.0
    settle_s: float = 0.4           # a saccade (~0.2 s) plus the dot smoothing delay
    min_jump: float = 300.0        # points between consecutive box centres
    edge_margin: float = 20.0


class TargetTask:
    """State machine for one run of target practice. Call ``update`` once per frame."""

    def __init__(self, screen: ScreenConfig, config: TaskConfig | None = None, seed: int | None = None) -> None:
        self.screen = screen
        self.cfg = config or TaskConfig()
        self.rng = np.random.default_rng(seed)
        self.trials: list[Trial] = []
        self.half = self.cfg.start_half
        self._streak = 0
        self._inside_since: float | None = None
        self._last_inside: float | None = None

    @property
    def done(self) -> bool:
        return len(self.trials) >= self.cfg.n_trials and self.current is None

    @property
    def current(self) -> Trial | None:
        return self.trials[-1] if self.trials and self.trials[-1].hit is None else None

    def start(self, t: float) -> Trial:
        prev = self.trials[-1].center if self.trials else None
        trial = Trial(len(self.trials), self._place(prev), self.half, t)
        self.trials.append(trial)
        self._inside_since = self._last_inside = None
        return trial

    def dwell_fraction(self, t: float) -> float:
        """How far the current dwell is towards a hit, 0..1 (for drawing)."""
        if self.current is None or self._inside_since is None:
            return 0.0
        return min(1.0, (t - self._inside_since) / self.cfg.dwell_s)

    def update(self, t: float, gaze: tuple[float, float] | None) -> str | None:
        """Feed one frame (time in s, smoothed dot or None). Returns 'hit', 'miss' or None."""
        trial = self.current
        if trial is None:
            return None
        c = self.cfg
        if gaze is not None:
            dx, dy = gaze[0] - trial.center[0], gaze[1] - trial.center[1]
            if t - trial.t_start >= c.settle_s:
                trial.offsets.append((dx, dy))
            if max(abs(dx), abs(dy)) <= trial.half:
                if self._inside_since is None or (self._last_inside is not None and t - self._last_inside > c.grace_s):
                    self._inside_since = t
                self._last_inside = t
                if t - self._inside_since >= c.dwell_s:
                    return self._finish(trial, t, True)
        if self._last_inside is not None and t - self._last_inside > c.grace_s:
            self._inside_since = None
        if t - trial.t_start >= c.timeout_s:
            return self._finish(trial, t, False)
        return None

    def _finish(self, trial: Trial, t: float, hit: bool) -> str:
        trial.t_end, trial.hit = t, hit
        c = self.cfg
        if hit:
            self._streak += 1
            if self._streak >= 2:
                self.half, self._streak = max(c.min_half, self.half * c.shrink), 0
        else:
            self.half, self._streak = min(c.max_half, self.half * c.grow), 0
        return "hit" if hit else "miss"

    def _place(self, prev: tuple[float, float] | None) -> tuple[float, float]:
        s, c = self.screen, self.cfg
        m = self.half + c.edge_margin
        lo = np.array([m, s.top_inset_pt + m])
        hi = np.array([s.width_px - m, s.height_px - m])
        best, best_d = None, -1.0
        for _ in range(200):
            p = self.rng.uniform(lo, hi)
            d = math.inf if prev is None else float(np.hypot(*(p - prev)))
            if d >= c.min_jump and s.target_is_safe(tuple(p), ring_px=self.half):
                return float(p[0]), float(p[1])
            if d > best_d and s.target_is_safe(tuple(p), ring_px=self.half):
                best, best_d = p, d
        p = best if best is not None else (lo + hi) / 2
        return float(p[0]), float(p[1])

    def summary(self) -> dict:
        """Hit rate, time to hit, reliable box size, bias and accuracy (cm and deg)."""
        s = self.screen
        done = [tr for tr in self.trials if tr.hit is not None]
        if not done:
            return {"n_trials": 0}
        hits = [tr for tr in done if tr.hit]
        tail = done[-8:]                     # the staircase has settled by the end
        size_pt = 2 * float(np.median([tr.half for tr in tail]))
        offs = [tr.mean_offset for tr in done if tr.mean_offset is not None]
        frames = np.array([o for tr in done for o in tr.offsets]) if offs else np.zeros((0, 2))
        bias = np.mean(offs, axis=0) if offs else np.array([np.nan, np.nan])
        err_pt = float(np.mean(np.hypot(frames[:, 0], frames[:, 1]))) if len(frames) else math.nan
        cm = s.cm_per_px
        deg = lambda pt: math.degrees(math.atan2(pt * cm, s.distance_cm))   # noqa: E731
        return {
            "n_trials": len(done), "n_hits": len(hits), "hit_rate": len(hits) / len(done),
            "median_time_s": float(np.median([tr.time_to_hit for tr in hits])) if hits else math.nan,
            "box_cm": size_pt * cm, "box_deg": deg(size_pt),
            "bias_cm": (float(bias[0] * cm), float(bias[1] * cm)),
            "error_cm": err_pt * cm, "error_deg": deg(err_pt),
        }


def describe_bias(bias_cm: tuple[float, float], threshold_cm: float = 0.5) -> str:
    """'on average the dot lands 1.2 cm to the left and 2.0 cm below the target', or that it is centred."""
    x, y = bias_cm
    parts = []
    if abs(x) >= threshold_cm:
        parts.append(f"{abs(x):.1f} cm to the {'right' if x > 0 else 'left'}")
    if abs(y) >= threshold_cm:
        parts.append(f"{abs(y):.1f} cm {'below' if y > 0 else 'above'}")
    if not parts:
        return f"no systematic offset: on average the dot lands within {threshold_cm} cm of the target"
    return "on average the dot lands " + " and ".join(parts) + (" the target" if abs(y) >= threshold_cm else " of the target")
