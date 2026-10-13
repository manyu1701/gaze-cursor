"""Offline jitter / lag metrics on spliced fixation streams."""

import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.temporal import FRAME_MS, fixation_metrics, run_filter, splice, summarize
from gaze_cursor.filters import MovingAverage, Raw

SCREEN = ScreenConfig()


def _stream(noise=0.0, n_fix=6, frames=30, seed=0):
    rng = np.random.default_rng(seed)
    targets = np.array([[200, 200], [1200, 200], [1200, 800], [200, 800], [700, 500], [200, 200]], float)[:n_fix]
    t, pred, tgt, key = [], [], [], []
    for j, xy in enumerate(targets):
        base = 5000.0 * j                                   # recorded with gaps (the unrecorded travel)
        t += list(base + np.arange(frames) * FRAME_MS)
        pred += list(xy + rng.normal(0, noise, (frames, 2)))
        tgt += [xy] * frames; key += [j] * frames
    return splice(np.array(t), np.array(pred), np.array(tgt), np.array(key))


def test_splice_makes_contiguous_time_and_known_step_times() -> None:
    s = _stream()
    assert np.allclose(np.diff(s.t_ms), FRAME_MS)
    assert s.start_ms[1] == pytest.approx(30 * FRAME_MS) and list(np.unique(s.fix)) == list(range(6))


def test_raw_has_zero_lag_and_moving_average_half_its_window() -> None:
    s = _stream()
    raw = fixation_metrics(s, run_filter(s, Raw()), SCREEN)
    assert np.all(raw["lag50"] == 0) and np.allclose(raw["jitter"], 0)
    ma = fixation_metrics(s, run_filter(s, MovingAverage(5)), SCREEN)
    assert np.all(ma["lag50"] == 2 * FRAME_MS) and np.all(ma["lag90"] == 4 * FRAME_MS)


def test_smoothing_lowers_jitter_on_noisy_fixations() -> None:
    s = _stream(noise=25.0)
    raw = summarize([fixation_metrics(s, run_filter(s, Raw()), SCREEN)])
    ma = summarize([fixation_metrics(s, run_filter(s, MovingAverage(9)), SCREEN)])
    assert ma["jitter"] < 0.5 * raw["jitter"] and ma["lag50"] > raw["lag50"]
    assert raw["reached90"] == 1.0 and raw["n_steps"] == 5
