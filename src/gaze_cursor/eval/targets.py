"""Calibration and held-out test target layouts. Pure functions."""

from __future__ import annotations

import numpy as np

from gaze_cursor.config import ScreenConfig


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
