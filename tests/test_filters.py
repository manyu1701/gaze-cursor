"""Smoothing filters: noise reduction, step response, snapping."""

import numpy as np
import pytest

from gaze_cursor.filters import MovingAverage, OneEuroFilter, Raw, make_filter


def _run(f, xy, dt=33.0):
    f.reset()
    return np.array([f.update(i * dt, p) for i, p in enumerate(xy)])


def _step(n=60, at=30, jump=500.0):
    xy = np.zeros((n, 2)); xy[at:, 0] = jump
    return xy


def _frames_to(out, at, level):
    return int(np.argmax(out[at:, 0] >= level))


def test_one_euro_reduces_noise_on_a_fixation() -> None:
    noisy = np.random.default_rng(0).normal(0, 20, (300, 2)) + 700
    out = _run(OneEuroFilter(min_cutoff=0.3, beta=0.0), noisy)
    assert out[50:].std(axis=0).max() < 0.3 * noisy.std(axis=0).min()
    assert np.abs(out[100:].mean(axis=0) - 700).max() < 5


def test_one_euro_converges_on_a_step_and_beta_cuts_the_lag() -> None:
    xy = _step()
    slow = _run(OneEuroFilter(min_cutoff=0.3, beta=0.0), xy)
    fast = _run(OneEuroFilter(min_cutoff=0.3, beta=0.01), xy)
    assert fast[-1, 0] == pytest.approx(500, abs=5)
    assert _frames_to(fast, 30, 450) < _frames_to(slow, 30, 450) or slow[-1, 0] < 450


def test_moving_average_lags_by_half_its_window_and_raw_not_at_all() -> None:
    xy = _step()
    ma = _run(MovingAverage(5), xy)
    assert _frames_to(ma, 30, 250) == 2                      # (n - 1) / 2 frames to half way
    assert np.array_equal(_run(Raw(), xy), xy)


def test_snap_jumps_to_a_saccade_after_k_far_frames() -> None:
    out = _run(OneEuroFilter(min_cutoff=0.05, beta=0.0, snap_px=150, snap_frames=2), _step())
    assert out[31, 0] == 500 and out[30, 0] < 100


def test_make_filter_specs() -> None:
    assert isinstance(make_filter(None), MovingAverage) and make_filter(None).n == 5
    assert isinstance(make_filter({"kind": "oneeuro", "min_cutoff": 1.0, "beta": 0.01}), OneEuroFilter)
    with pytest.raises(ValueError):
        make_filter({"kind": "kalman"})


def test_run_uses_the_filter_chosen_by_the_step_6_rule() -> None:
    from gaze_cursor.filters import RUN_FILTER
    assert RUN_FILTER == {"kind": "oneeuro", "min_cutoff": 0.05, "beta": 0.001, "d_cutoff": 1.0}
    from pathlib import Path
    assert "make_filter(RUN_FILTER)" in (Path(__file__).resolve().parents[1] / "scripts" / "gaze_overlay.py").read_text()
