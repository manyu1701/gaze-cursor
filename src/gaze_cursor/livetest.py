"""Live tests inside Run: target boxes and a path to follow, with scoring. Pure functions.

Targets test: boxes appear one at a time; a box is HIT when the gaze dot stays inside it for
DWELL_MS, and times out after TIMEOUT_MS. Measures: hit rate, time to first entry and to hit
(snappy), jitter of the dot while inside (steady), and the offset of the dot from the box centre
(accurate, with its direction).
Path test: a target glides along a loop around the screen and a figure 8 at PATH_SPEED_DEG deg/s;
the user follows it. Measures: mean distance from the target, share of time within 3 deg, and the
lag (time shift that best aligns the dot with the target).
All distances are in degrees of visual angle (``ScreenConfig.distance_cm``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gaze_cursor.config import ScreenConfig

DWELL_MS = 500.0
TIMEOUT_MS = 4000.0
BOX_SIZES_DEG = (5.0, 3.0)
N_BOXES = 12
MIN_HOP_DEG = 8.0          # consecutive boxes at least this far apart, so every box needs a real move
PATH_SPEED_DEG = 8.0       # deg/s, an easy smooth-pursuit speed
PATH_SETTLE_S = 1.0        # the first second of the path is not scored (finding the target)
WITHIN_DEG = 3.0
MAX_LAG_MS = 500.0


def px_per_deg(cfg: ScreenConfig) -> float:
    """Screen points per degree of visual angle at the screen centre."""
    return float(2 * cfg.distance_cm * np.tan(np.radians(0.5)) / cfg.cm_per_px)


def to_deg(px: np.ndarray | float, cfg: ScreenConfig) -> np.ndarray:
    return np.asarray(px, dtype=float) / px_per_deg(cfg)


@dataclass(frozen=True)
class Box:
    center: tuple[float, float]
    half_px: float
    size_deg: float

    def contains(self, xy: np.ndarray) -> bool:
        return bool(abs(xy[0] - self.center[0]) <= self.half_px and abs(xy[1] - self.center[1]) <= self.half_px)


def make_boxes(cfg: ScreenConfig, n: int = N_BOXES, sizes_deg: tuple[float, ...] = BOX_SIZES_DEG,
               seed: int = 0) -> list[Box]:
    """``n`` boxes spread over a 4 x 3 grid of screen cells (jittered, so every region is tested),
    sizes alternating, in a random order where consecutive boxes are >= MIN_HOP_DEG apart. Every box
    lies fully inside the visible area (below the menu bar / notch strip)."""
    rng = np.random.default_rng(seed)
    ppd = px_per_deg(cfg)
    margin = 12.0
    top = cfg.top_inset_pt + margin
    cols, rows = 4, 3
    boxes = []
    for k in range(n):
        cell = k % (cols * rows)
        cx, cy = cell % cols, cell // cols
        size = sizes_deg[k % len(sizes_deg)]
        half = size * ppd / 2
        x0, x1 = margin + half, cfg.width_px - margin - half
        y0, y1 = top + half, cfg.height_px - margin - half
        u = (cx + rng.uniform(0.2, 0.8)) / cols
        v = (cy + rng.uniform(0.2, 0.8)) / rows
        boxes.append(Box((float(x0 + u * (x1 - x0)), float(y0 + v * (y1 - y0))), half, size))
    for _ in range(2000):                      # an order with real hops between consecutive boxes
        order = rng.permutation(n)
        hops = [np.hypot(*np.subtract(boxes[a].center, boxes[b].center)) / ppd for a, b in zip(order, order[1:])]
        if min(hops) >= MIN_HOP_DEG:
            return [boxes[i] for i in order]
    return [boxes[i] for i in order]


@dataclass
class BoxTrial:
    """One box: call ``update`` with every new dot position; ``status`` becomes 'hit' or 'timeout'."""

    box: Box
    start_ms: float
    status: str = "waiting"                   # waiting -> inside <-> waiting -> hit | timeout
    first_entry_ms: float | None = None
    hit_ms: float | None = None
    _inside_since: float | None = None
    samples: list[tuple[float, float, float]] = field(default_factory=list)   # (t, x, y)

    @property
    def done(self) -> bool:
        return self.status in ("hit", "timeout")

    def dwell_fraction(self, t_ms: float) -> float:
        return 0.0 if self._inside_since is None else min((t_ms - self._inside_since) / DWELL_MS, 1.0)

    def update(self, t_ms: float, dot: np.ndarray | None) -> str:
        if self.done:
            return self.status
        if dot is not None:
            self.samples.append((t_ms, float(dot[0]), float(dot[1])))
            if self.box.contains(dot):
                if self._inside_since is None:
                    self._inside_since = t_ms
                    if self.first_entry_ms is None:
                        self.first_entry_ms = t_ms - self.start_ms
                self.status = "inside"
                if t_ms - self._inside_since >= DWELL_MS:
                    self.status, self.hit_ms = "hit", t_ms - self.start_ms
                    return self.status
            else:
                self._inside_since, self.status = None, "waiting"
        if t_ms - self.start_ms >= TIMEOUT_MS:
            self.status = "timeout"
        return self.status

    def result(self, cfg: ScreenConfig) -> dict:
        """Offset and jitter over the last DWELL_MS of the trial (the hold, or the final half
        second of a timeout), in degrees."""
        s = np.array(self.samples) if self.samples else np.zeros((0, 3))
        end = s[-1, 0] if len(s) else self.start_ms
        last = s[s[:, 0] >= end - DWELL_MS] if len(s) else s
        if len(last):
            off = last[:, 1:].mean(axis=0) - np.array(self.box.center)
            jitter = float(to_deg(np.sqrt(last[:, 1:].var(axis=0).sum()), cfg))
        else:
            off, jitter = np.array([np.nan, np.nan]), np.nan
        return {"size_deg": self.box.size_deg, "center": self.box.center, "hit": self.status == "hit",
                "first_entry_ms": self.first_entry_ms, "hit_ms": self.hit_ms,
                "offset_deg": tuple(float(v) for v in to_deg(off, cfg)),
                "error_deg": float(to_deg(np.linalg.norm(off), cfg)), "jitter_deg": jitter}


def summarize_boxes(results: list[dict]) -> dict:
    """Hit rate (overall and per size), median times, mean error / offset / jitter."""
    if not results:
        return {"n": 0}
    hit = np.array([r["hit"] for r in results])
    times = [r["hit_ms"] for r in results if r["hit"]]
    entry = [r["first_entry_ms"] for r in results if r["first_entry_ms"] is not None]
    off = np.array([r["offset_deg"] for r in results], float)
    out = {"n": len(results), "hit_rate": float(hit.mean()),
           "median_hit_ms": float(np.median(times)) if times else float("nan"),
           "median_entry_ms": float(np.median(entry)) if entry else float("nan"),
           "error_deg": float(np.nanmean([r["error_deg"] for r in results])),
           "offset_deg": tuple(float(v) for v in np.nanmean(off, axis=0)),
           "jitter_deg": float(np.nanmean([r["jitter_deg"] for r in results]))}
    for size in sorted({r["size_deg"] for r in results}):
        h = [r["hit"] for r in results if r["size_deg"] == size]
        out[f"hit_rate_{size:g}deg"] = float(np.mean(h))
    return out


class PathTarget:
    """A target moving at constant speed along a loop around the screen and then a figure 8."""

    def __init__(self, cfg: ScreenConfig, speed_deg: float = PATH_SPEED_DEG) -> None:
        w, h = cfg.width_px, cfg.height_px
        top = cfg.top_inset_pt
        mx, my = 0.12 * w, 0.12 * (h - top)
        x0, x1, y0, y1 = mx, w - mx, top + my, h - my
        loop = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
        loop = np.vstack([np.linspace(a, b, 60, endpoint=False) for a, b in zip(loop[:-1], loop[1:])] + [loop[-1:]])
        th = np.linspace(0, 2 * np.pi, 400)
        cx, cy, ax, ay = w / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        eight = np.column_stack([cx + ax * np.sin(th), cy + ay * np.sin(2 * th)])
        lead = np.linspace(loop[-1], eight[0], 30)       # glide from the loop's end to the 8's start
        self.points = np.vstack([loop, lead[1:], eight[1:]])
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.speed_px = speed_deg * px_per_deg(cfg)
        self.duration_s = float(self.cum[-1] / self.speed_px)

    def at(self, t_s: float | np.ndarray) -> np.ndarray:
        """Target position(s) at time ``t_s`` since the start (clamped to the end)."""
        s = np.clip(np.asarray(t_s, dtype=float) * self.speed_px, 0, self.cum[-1])
        return np.stack([np.interp(s, self.cum, self.points[:, 0]), np.interp(s, self.cum, self.points[:, 1])], axis=-1)


def pursuit_summary(t_s: np.ndarray, dot: np.ndarray, path: PathTarget, cfg: ScreenConfig) -> dict:
    """Mean distance (deg) and share within WITHIN_DEG, scored after PATH_SETTLE_S; lag = the time
    shift (0-500 ms) that minimises the mean distance between the dot and the earlier target."""
    t_s, dot = np.asarray(t_s, float), np.asarray(dot, float)
    keep = t_s >= PATH_SETTLE_S
    if keep.sum() < 10:
        return {"n": int(keep.sum())}
    t, d = t_s[keep], dot[keep]
    err = to_deg(np.linalg.norm(d - path.at(t), axis=1), cfg)
    lags = np.arange(0.0, MAX_LAG_MS + 1, 10.0)
    mean_at = [float(to_deg(np.linalg.norm(d - path.at(t - lag / 1000), axis=1), cfg).mean()) for lag in lags]
    best = int(np.argmin(mean_at))
    return {"n": int(keep.sum()), "error_deg": float(err.mean()), "median_deg": float(np.median(err)),
            "within": float((err <= WITHIN_DEG).mean()), "lag_ms": float(lags[best]),
            "error_at_lag_deg": mean_at[best]}


@dataclass
class LiveStats:
    """Rolling live metrics: fps, latency, tracking, jitter."""

    window: int = 30
    _t: list[float] = field(default_factory=list)
    _lat: list[float] = field(default_factory=list)
    _tracked: list[bool] = field(default_factory=list)
    _dot: list[tuple[float, float]] = field(default_factory=list)

    def add(self, t_ms: float, latency_ms: float, tracked: bool, dot: np.ndarray | None) -> None:
        for buf, v in ((self._t, t_ms), (self._lat, latency_ms), (self._tracked, tracked)):
            buf.append(v)
            del buf[:-self.window]
        if dot is not None:
            self._dot.append((float(dot[0]), float(dot[1])))
            del self._dot[:-15]

    def fps(self) -> float:
        return float((len(self._t) - 1) / ((self._t[-1] - self._t[0]) / 1000)) if len(self._t) > 2 else float("nan")

    def latency_ms(self) -> float:
        return float(np.median(self._lat)) if self._lat else float("nan")

    def tracked(self) -> float:
        return float(np.mean(self._tracked)) if self._tracked else 0.0

    def jitter_deg(self, cfg: ScreenConfig) -> float:
        """Spread of the dot over the last ~0.5 s (only meaningful while you hold your gaze still)."""
        if len(self._dot) < 5:
            return float("nan")
        d = np.array(self._dot)
        return float(to_deg(np.sqrt(d.var(axis=0).sum()), cfg))


# Head movement since the quick calibration. Round 3 measured ~+2 deg for moves of ~6-10 cm or
# ~15-30% in distance; the prompt threshold is a judgement call well below that.
IPD_CM = 6.3                 # rough scale for showing face shifts in cm (display only)
MOVED_CM, MOVED_RATIO = 4.0, 0.10


def head_shift(metrics: dict[str, float], ref_center: dict[str, float],
               aspect: float = 720 / 1280) -> tuple[float, float, float]:
    """(distance ratio, sideways cm, height cm) of the live face vs the calibration posture.
    ``metrics`` / ``ref_center`` are ``posture_metrics`` dicts (distance, x, y). Sideways > 0 = the
    user's left; height > 0 = lower."""
    per = IPD_CM / ref_center["distance"]
    return (metrics["distance"] / ref_center["distance"], (metrics["x"] - ref_center["x"]) * per,
            (metrics["y"] - ref_center["y"]) * aspect * per)


def head_moved(ratio: float, dx_cm: float, dy_cm: float) -> bool:
    return abs(ratio - 1) > MOVED_RATIO or float(np.hypot(dx_cm, dy_cm)) > MOVED_CM
