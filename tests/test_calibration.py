import numpy as np
import pytest

from gaze_cursor.calibration import GazeModel
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.targets import calibration_grid, held_out_points


def test_ridge_recovers_linear_map() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 9))
    w = rng.normal(size=(9, 2))
    y = x @ w * 100 + [700, 400]
    model = GazeModel("ridge", alpha=1e-6).fit(x, y)
    assert model.predict(x) == pytest.approx(y, abs=1e-2)


def test_poly2_fits_quadratic() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 3))
    y = np.stack([x[:, 0] ** 2 + x[:, 1], x[:, 2] * x[:, 0]], axis=1) * 50
    pred = GazeModel("poly2", alpha=1e-6).fit(x, y).predict(x)
    assert np.abs(pred - y).max() < 1.0


def test_unknown_model() -> None:
    with pytest.raises(ValueError):
        GazeModel("nope")


def test_grid_and_held_out_layout() -> None:
    cfg = ScreenConfig()
    grid = calibration_grid(cfg)
    assert grid.shape == (9, 2)
    held = held_out_points(cfg, grid, n=12, seed=0)
    assert held.shape == (12, 2)
    assert min(np.linalg.norm(h - g) for h in held for g in grid) >= 120
    assert np.array_equal(held, held_out_points(cfg, grid, n=12, seed=0))  # reproducible
