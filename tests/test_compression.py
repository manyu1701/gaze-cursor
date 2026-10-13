"""Centre-compression exploration: dot sets and the gain summary."""

import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.compression import edge_dot_set, gain, quick_dot_set
from gaze_cursor.eval.stats import TargetStats
from gaze_cursor.eval.targets import calibration_grid_v2

SCREEN = ScreenConfig()


def test_edge_dot_set_covers_every_edge_and_quick_set_does_not() -> None:
    grid = calibration_grid_v2(SCREEN)
    def edges(dots):
        xy = grid[list(dots)]
        mid = lambda a, v: np.isclose(xy[:, a], v) & ~np.isclose(xy[:, 1 - a], grid[:, 1 - a].min()) \
            & ~np.isclose(xy[:, 1 - a], grid[:, 1 - a].max())
        return {"left": mid(0, grid[:, 0].min()).any(), "right": mid(0, grid[:, 0].max()).any(),
                "top": mid(1, grid[:, 1].min()).any(), "bottom": mid(1, grid[:, 1].max()).any()}
    edge = edge_dot_set(SCREEN)
    assert len(edge) == 9 and {0, 3, 12, 15} <= set(edge) and all(edges(edge).values())
    assert not edges(quick_dot_set(SCREEN))["left"] and len(quick_dot_set(SCREEN)) == 9


def test_gain_is_one_without_squeeze_and_below_one_with_it() -> None:
    xy = np.random.default_rng(0).uniform([0, 0], [1470, 956], (50, 2))
    centre = np.array([735, 478])
    def stats(bias):
        n = len(xy)
        z = np.zeros(n)
        return TargetStats(z.astype(int), np.arange(n), xy, z, z, z, bias, z, z.astype(int))
    assert gain(stats(np.zeros_like(xy))) == pytest.approx((1.0, 1.0))
    squeezed = (xy - centre) * np.array([0.9, 0.7]) + centre - xy       # mean prediction pulled to the centre
    gx, gy = gain(stats(squeezed))
    assert gx == pytest.approx(0.9) and gy == pytest.approx(0.7)
