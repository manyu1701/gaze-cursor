"""Vertical-bias diagnosis: bands, signed degrees and the pre-specified gate."""

import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.stats import TargetStats
from gaze_cursor.eval.vertical import band_edges, band_of, band_report, gate, signed_deg

SCREEN = ScreenConfig()


def _stats(xy: np.ndarray, bias: np.ndarray, spread: float = 5.0) -> TargetStats:
    n = len(xy)
    err_px = np.linalg.norm(bias, axis=1)
    return TargetStats(np.zeros(n, int), np.arange(n), xy, err_px, signed_deg(err_px, SCREEN), err_px * SCREEN.cm_per_px,
                       bias, np.full(n, spread), np.full(n, 30))


def _grid(n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(30, 1440, n), rng.uniform(60, 930, n)])


def test_bands_split_between_the_v2_grid_rows() -> None:
    assert band_edges(SCREEN, 1) == pytest.approx([202, 494, 786])
    assert band_of([56, 201, 203, 600, 932], SCREEN).tolist() == [0, 0, 1, 2, 3]
    assert signed_deg(np.array([-10.0, 10.0]), SCREEN)[0] == pytest.approx(-signed_deg(np.array([10.0]), SCREEN)[0])


def test_gate_detects_a_planted_top_row_bias_and_not_a_null() -> None:
    xy = _grid()
    planted = np.column_stack([np.zeros(len(xy)), np.where(xy[:, 1] < 202, 80.0, 0.0)])  # top: 80 pt too low
    real, crit = gate(band_report(_stats(xy, planted), SCREEN))
    assert real and all(crit.values())
    jitter = np.random.default_rng(1).normal(0, 5, xy.shape)
    assert not gate(band_report(_stats(xy, jitter), SCREEN))[0]
    uniform = np.column_stack([np.zeros(len(xy)), np.full(len(xy), 80.0)])          # everything low: an offset
    real, crit = gate(band_report(_stats(xy, uniform), SCREEN))
    assert not real and not crit["top exceeds bottom by >= 0.5 deg"]


def test_band_report_separates_bias_from_spread() -> None:
    xy = _grid()
    rows = band_report(_stats(xy, np.zeros_like(xy), spread=40.0), SCREEN)
    assert all(r["abs_bias_deg"] == 0 and r["spread_deg"] > 0.5 for r in rows if r["n"])
