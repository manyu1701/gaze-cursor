"""Calibration and held-out test target layouts. Pure functions."""

from __future__ import annotations

from dataclasses import dataclass

from dataclasses import dataclass

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import CALIB, TEST
from gaze_cursor.eval.session import CALIB, TEST


def calibration_grid(cfg: ScreenConfig, inset: float = 0.1) -> np.ndarray:
    """3x3 grid in serpentine order, as (9, 2) px positions."""
    xs = np.array([inset, 0.5, 1 - inset]) * cfg.width_px
    ys = np.array([inset, 0.5, 1 - inset]) * cfg.height_px
    pts = []
    for r, y in enumerate(ys):
        row = xs if r % 2 == 0 else xs[::-1]
        pts += [(x, y) for x in row]
    return np.array(pts)


def held_out_points(
    cfg: ScreenConfig, grid: np.ndarray, n: int = 12, seed: int = 0,
    margin: float = 0.08, min_dist_px: float = 120.0,
) -> np.ndarray:
    """``n`` random points, none within ``min_dist_px`` of the grid or of each other."""
    rng = np.random.default_rng(seed)
    lo = np.array([margin * cfg.width_px, margin * cfg.height_px])
    hi = np.array([(1 - margin) * cfg.width_px, (1 - margin) * cfg.height_px])
    pts: list[np.ndarray] = []
    for _ in range(10_000):
        p = rng.uniform(lo, hi)
        if all(np.linalg.norm(p - q) >= min_dist_px for q in [*grid, *pts]):
            pts.append(p)
            if len(pts) == n:
                return np.array(pts)
    raise RuntimeError("Could not place test points; relax min_dist_px")


@dataclass(frozen=True)
class Target:
    """One calibration or held-out target, in screen points. ``idx`` is stable per position."""

    phase: int
    idx: int
    xy: tuple[float, float]


def calibration_grid_v2(cfg: ScreenConfig, n: int = 4, ring_px: float = 16.0,
                        margin_px: float = 8.0) -> np.ndarray:
    """n x n grid that reaches the safe edges: the extreme positions ``target_is_safe`` allows
    plus ``margin_px``. Row-major, (n*n, 2) points."""
    xs = np.linspace(ring_px + margin_px, cfg.width_px - ring_px - margin_px, n)
    ys = np.linspace(cfg.top_inset_pt + ring_px + margin_px, cfg.height_px - ring_px - margin_px, n)
    return np.array([(x, y) for y in ys for x in xs])


def held_out_points_v2(cfg: ScreenConfig, grid: np.ndarray, n: int = 9, seed: int = 0,
                       edge_margin_px: float = 60.0, min_dist_px: float = 100.0) -> np.ndarray:
    """``n`` random points strictly inside the calibrated rectangle (>= ``edge_margin_px`` from its
    edges), >= ``min_dist_px`` from every grid point and from each other, so they are
    interpolated and never extrapolated."""
    rng = np.random.default_rng(seed)
    lo = grid.min(axis=0) + edge_margin_px
    hi = grid.max(axis=0) - edge_margin_px
    pts: list[np.ndarray] = []
    for _ in range(20_000):
        p = rng.uniform(lo, hi)
        if all(np.linalg.norm(p - q) >= min_dist_px for q in [*grid, *pts]):
            pts.append(p)
            if len(pts) == n:
                return np.array(pts)
    raise RuntimeError("Could not place held-out points; relax min_dist_px or edge_margin_px")


def build_plan(cfg: ScreenConfig, layout: str = "v2", seed: int = 0) -> list[Target]:
    """Targets in presentation order.

    - ``v1``: 3x3 calibration grid in serpentine order, then 12 held-out points (the M1 layout).
    - ``v2``: 16 calibration targets reaching the safe edges + 9 interior held-out targets, all
      25 in one random order (seeded). Target ``idx`` is fixed before shuffling.
    """
    if layout == "v1":
        grid = calibration_grid(cfg)
        held = held_out_points(cfg, grid, seed=seed)
        return ([Target(CALIB, i, tuple(p)) for i, p in enumerate(grid)]
                + [Target(TEST, len(grid) + i, tuple(p)) for i, p in enumerate(held)])
    if layout == "v2":
        grid = calibration_grid_v2(cfg)
        held = held_out_points_v2(cfg, grid, seed=seed)
        plan = ([Target(CALIB, i, tuple(p)) for i, p in enumerate(grid)]
                + [Target(TEST, len(grid) + i, tuple(p)) for i, p in enumerate(held)])
        order = np.random.default_rng([seed, 1]).permutation(len(plan))
        return [plan[i] for i in order]
    raise ValueError(f"unknown layout {layout!r}; choose v1 or v2")
