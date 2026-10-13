"""Self-recalibration from clicks: exact start, RLS behaviour, labels, gates, CUSUM."""

import numpy as np
import pytest

from gaze_cursor.calibration import GazeModel
from gaze_cursor.online import Cusum, OnlineRidge, click_label, prequential

RNG = np.random.default_rng(0)
W = np.array([[300.0, 10.0], [-20.0, 250.0], [280.0, 5.0], [15.0, 240.0]])


def calibration(n_dots: int = 9, per_dot: int = 30):
    dots = RNG.normal(size=(n_dots, 4))
    x = np.repeat(dots, per_dot, axis=0) + RNG.normal(scale=0.05, size=(n_dots * per_dot, 4))
    y = x @ W + np.array([735.0, 478.0])
    return x, y


def test_starts_exactly_at_the_batch_ridge_fit() -> None:
    x, y = calibration()
    m = OnlineRidge.from_calibration(x, y, alpha=10.0)
    g = GazeModel("ridge", 10.0).fit(x, y)
    q = RNG.normal(size=(20, 4))
    assert np.allclose(m.predict(q), g.predict(q), atol=1e-6)


def test_learns_a_planted_offset_and_not_noise() -> None:
    x, y = calibration()
    m = OnlineRidge.from_calibration(x, y, alpha=10.0)
    offset = np.array([0.0, -140.0])                     # gaze reads too high, like the first live try
    q = RNG.normal(size=(40, 4))
    before = np.abs(m.predict(q) - (q @ W + [735, 478] - offset)).mean()
    for xi in q:
        m.update(xi, xi @ W + [735, 478] - offset)
    after = np.abs(m.predict(q) - (q @ W + [735, 478] - offset)).mean()
    assert after < 0.25 * before
    # Unbiased noisy clicks around the truth do not drag the model away.
    m2 = OnlineRidge.from_calibration(x, y, alpha=10.0)
    for xi in q:
        m2.update(xi, xi @ W + [735, 478] + RNG.normal(scale=30, size=2))
    assert np.abs(m2.predict(q) - (q @ W + [735, 478])).mean() < 25


def test_the_calibration_counts_as_its_dots_not_its_frames() -> None:
    x, y = calibration()
    m = OnlineRidge.from_calibration(x, y, alpha=10.0)
    xi = x[:1]
    target = m.predict(xi)[0] + [0, -100.0]
    m.update(xi, target)
    moved = -(m.predict(xi)[0] - (target - [0, -100.0]))[1]
    assert 5 < moved < 60                                  # one click among ~9 prior samples: a real but partial step


def test_click_label_uses_the_window_before_the_click() -> None:
    t = np.arange(0, 1000, 33.0)
    f = np.column_stack([t, t, t, t])
    f[(t > 640) & (t < 700)] = np.nan                      # a blink inside the window is skipped
    lab = click_label(t, f, 900.0)
    assert 500 <= lab[0] <= 800
    assert click_label(t, f, 5000.0) is None


def test_prequential_predicts_before_learning_and_gates() -> None:
    x, y = calibration()
    prior = OnlineRidge.from_calibration(x, y, alpha=10.0)
    t = np.arange(0, 20000, 33.0)
    feats = np.repeat(RNG.normal(size=(1, 4)), len(t), axis=0)
    true = prior.predict(feats[:1])[0] + [0, -140.0]
    clicks = [(1000.0 * k + 500, true[0], true[1], 0) for k in range(1, 15)]
    clicks.append((16500.0, true[0] + 2000, true[1], 0))     # a click made without looking
    res = prequential(t, feats, clicks, {0: prior}, px_per_deg=50.0)
    assert res[0].err_online_deg == pytest.approx(res[0].err_frozen_deg)   # nothing learned yet
    assert res[-2].err_online_deg < 0.5 * res[-2].err_frozen_deg
    assert not res[-1].updated and not res[-1].evaluated


def test_cusum_fires_on_a_shift_not_on_noise() -> None:
    c = Cusum()
    assert not any(c.update(r, i) for i, r in enumerate(RNG.normal(scale=0.5, size=200)))
    assert any(c.update(3.0, i) for i in range(10))
