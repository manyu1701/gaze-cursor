import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.metrics import cm_to_deg, error_metrics, jitter_metrics

CFG = ScreenConfig(width_px=1000, height_px=500, width_cm=40, height_cm=20, distance_cm=60)


def test_error_conversion() -> None:
    pred = np.array([[100.0, 0.0], [0.0, 0.0]])
    m = error_metrics(pred, np.zeros((2, 2)), CFG)
    assert m["err_px_mean"] == pytest.approx(50)
    assert m["err_cm_mean"] == pytest.approx(50 * 0.04)  # 0.04 cm/px


def test_degrees_geometry() -> None:
    assert cm_to_deg(60.0, CFG) == pytest.approx(45.0)
    assert cm_to_deg(0.0, CFG) == 0.0


def test_jitter_zero_when_constant_and_per_fixation() -> None:
    pred = np.array([[1.0, 1.0]] * 3 + [[5.0, 5.0], [5.0, 5.0]])
    idx = np.array([0, 0, 0, 1, 1])
    assert jitter_metrics(pred, idx, CFG)["jitter_px"] == pytest.approx(0)


def test_jitter_value() -> None:
    pred = np.array([[0.0, 0.0], [2.0, 0.0]])  # var_x (ddof=1) = 2
    assert jitter_metrics(pred, np.array([0, 0]), CFG)["jitter_px"] == pytest.approx(2 ** 0.5)
