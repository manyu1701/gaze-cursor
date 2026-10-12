import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.stats import (
    bootstrap_ci, bootstrap_ci_clustered, paired_comparison, per_target, summarize)

CFG = ScreenConfig()


def test_per_target_aggregates_frames_to_targets() -> None:
    y = np.array([[0.0, 0], [0, 0], [100, 100], [100, 100]])
    pred = np.array([[3.0, 4], [3, 4], [100, 100], [100, 104]])
    t = per_target(pred, y, np.array([7, 7, 9, 9]), np.array([0, 0, 1, 1]), CFG)
    assert t.target.tolist() == [7, 9] and t.n_frames.tolist() == [2, 2]
    assert t.err_px.tolist() == pytest.approx([5.0, 2.0])
    assert t.bias_px[0].tolist() == pytest.approx([3, 4])
    assert t.spread_px[0] == pytest.approx(0.0)
    assert t.spread_px[1] == pytest.approx(np.sqrt(8))  # var of (100, 104) with ddof=1 = 8


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    v = np.random.default_rng(0).normal(2.0, 1.0, 60)
    assert bootstrap_ci(v, 2000, seed=3) == bootstrap_ci(v, 2000, seed=3)
    lo, hi = bootstrap_ci(v, 2000, seed=3)
    assert lo < v.mean() < hi and hi - lo < 1.0
    clo, chi = bootstrap_ci_clustered(v, np.repeat([0, 1, 2], 20), 500, seed=3)
    assert clo < v.mean() < chi


def _stats(err: np.ndarray, sessions: np.ndarray):
    n = len(err)
    from gaze_cursor.eval.stats import TargetStats
    return TargetStats(sessions, np.arange(n), np.zeros((n, 2)), err, err, err, np.zeros((n, 2)),
                       np.ones(n), np.full(n, 5))


def test_paired_comparison_sign_ci_and_session_counts() -> None:
    sess = np.repeat([0, 1, 2], 20)
    ref = _stats(np.full(60, 5.0), sess)
    new = _stats(np.full(60, 4.0) + np.random.default_rng(1).normal(0, 0.1, 60), sess)
    c = paired_comparison(ref, new, 2000, seed=0)
    assert c["mean_diff_deg"] == pytest.approx(1.0, abs=0.1) and c["ci_lo"] > 0.8
    assert c["sessions_new_better"] == 3 and c["targets_new_better"] == 60
    worse = paired_comparison(new, ref, 2000, seed=0)
    assert worse["mean_diff_deg"] < 0 and worse["sessions_new_better"] == 0


def test_paired_comparison_rejects_misaligned_targets() -> None:
    a = _stats(np.ones(4), np.zeros(4, int))
    b = _stats(np.ones(4), np.array([0, 0, 0, 1]))
    with pytest.raises(ValueError):
        paired_comparison(a, b)


def test_summary_has_mean_median_p90_and_jitter() -> None:
    s = summarize(_stats(np.arange(1.0, 11.0), np.zeros(10, int)), CFG)
    assert s["err_deg_mean"] == pytest.approx(5.5) and s["err_deg_median"] == pytest.approx(5.5)
    assert s["err_deg_p90"] == pytest.approx(9.1) and s["n_targets"] == 10
