"""Session quality metrics (M3 step 6b): pure-function checks on synthetic sessions."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.configs import REGISTRY
from gaze_cursor.eval.quality import (
    FIT_CONFIG, _rot_angle_deg, all_metrics, bias_vs_noise, coverage, drift_and_head,
    fixation_quality, held_out_stats, refusal,
)
from gaze_cursor.eval.session import CALIB, TEST
from gaze_cursor.eval.stats import TargetStats
from tests.helpers import fake_session

CFG = ScreenConfig(1470, 956, 29.0, 18.9, 60.0)


def _grid_session(test_xy: list[tuple[float, float]]):
    """Calibration targets at the corners and middle; test targets at ``test_xy``."""
    cal = [(100, 100), (1370, 100), (100, 850), (1370, 850), (735, 480)]
    s = fake_session(0, n_calib=len(cal), n_test=len(test_xy), frames=4)
    xy = np.repeat(np.array(cal + list(test_xy), dtype=np.float32), 4, axis=0)
    return replace(s, target_xy=xy)


def test_refusal_rules():
    s = fake_session(0)
    assert refusal(s) is None
    assert "round-4" in refusal(replace(s, meta={**s.meta, "note": "quick calibration (Cursor)"}))
    assert refusal(replace(s, meta={**s.meta, "note": "quick calibration (Run)"})) is None
    assert "unclean" in refusal(fake_session(0, window=[0, 0, 800, 600]))


def test_coverage_hull_inside_and_outside():
    inside = coverage(_grid_session([(700, 450), (300, 300)]))
    assert inside["test_outside_hull"] == 0.0
    assert 0.8 < inside["calib_extent_x"] < 0.9 and 0.7 < inside["calib_extent_y"] < 0.9
    mixed = coverage(_grid_session([(700, 450), (20, 20)]))
    assert mixed["test_outside_hull"] == 0.5


def test_coverage_without_test_phase_is_nan_not_error():
    c = coverage(fake_session(0, n_calib=9, n_test=0))
    assert np.isnan(c["test_outside_hull"]) and c["n_test"] == 0.0


def test_fixation_quality_counts_thin_targets_and_ignores_target_pauses():
    s = fake_session(0, n_calib=5, n_test=2, frames=12)
    t = np.arange(len(s.t_ms)) * 33
    t[12:] += 2000          # a long pause between target 0 and target 1 is not a dropped frame
    f = s.features.copy()
    f[24:24 + 8] = np.nan   # target 2 keeps only 4 valid frames (< 10)
    q = fixation_quality(replace(s, t_ms=t, features=f))
    assert q["thin_targets"] == 1.0
    assert q["dropped_gap_share"] == 0.0
    t[5] += 400             # a stall inside one target is
    assert fixation_quality(replace(s, t_ms=t))["dropped_gap_share"] > 0.0


def _stats(xy: np.ndarray, pred: np.ndarray, spread: float = 5.0) -> TargetStats:
    m = len(xy)
    err = np.linalg.norm(pred - xy, axis=1)
    return TargetStats(np.zeros(m, int), np.arange(m), xy, err, err / 40, err / 40, pred - xy,
                       np.full(m, spread), np.full(m, 10))


def test_compression_slope_is_one_for_perfect_and_half_for_halved():
    xy = np.array([[100, 100], [1370, 100], [100, 850], [1370, 850], [735, 478.0]])
    centre = np.array([735, 478.0])
    perfect = bias_vs_noise(_stats(xy, xy.copy()), CFG)
    assert np.isclose(perfect["slope_x"], 1.0) and np.isclose(perfect["slope_y"], 1.0)
    assert np.isclose(perfect["bias_deg"], 0.0)
    halved = bias_vs_noise(_stats(xy, centre + 0.5 * (xy - centre)), CFG)
    assert np.isclose(halved["slope_x"], 0.5) and np.isclose(halved["slope_y"], 0.5)
    assert 0.0 < halved["bias_share"] <= 1.0


def test_bias_vs_noise_needs_three_targets():
    xy = np.array([[100, 100], [200, 200.0]])
    assert np.isnan(bias_vs_noise(_stats(xy, xy), CFG)["slope_x"])


def test_held_out_stats_align_and_noise_raises_error():
    clean = fake_session(0, n_calib=10, n_test=5, frames=8)
    within, loto = held_out_stats(clean)
    assert within is not None and loto is not None
    assert len(within.err_deg) == 5 and len(loto.err_deg) == 15   # keys line up: 1 row per target
    noisy = fake_session(0, n_calib=10, n_test=5, frames=8, noise=0.2)
    assert held_out_stats(noisy)[1].err_deg.mean() > loto.err_deg.mean()


def test_held_out_never_splits_a_target_between_train_and_test():
    # Each target gets a constant random feature signature unrelated to its position. A fit that
    # saw other frames of the same target would memorise it (low error); leave-one-target-out
    # cannot, so its error must stay far above the leaky fit's.
    s = fake_session(1, n_calib=5, n_test=0, frames=20, gain=0.0)
    rng = np.random.default_rng(5)
    sig = rng.normal(0, 3.0, (5, 4))
    f = s.features.copy()
    f[:, :4] = sig[s.target_idx.astype(int)] + rng.normal(0, 0.02, (len(f), 4))
    s = replace(s, features=f)
    _, loto = held_out_stats(s)
    leaky = REGISTRY[FIT_CONFIG].make_model(CFG).fit(*s.select_all("gaze")[:2])
    x, y, _, _ = s.select_all("gaze")
    leaky_err = float(np.mean(np.linalg.norm(leaky.predict(x) - y, axis=1)))
    assert loto is not None
    loto_px = float(loto.err_px.mean())
    assert loto_px > 2.0 * leaky_err and loto_px > 150


def test_quick_style_session_has_no_heldout_but_has_loto():
    m = all_metrics(fake_session(0, n_calib=9, n_test=0, frames=8))
    assert np.isnan(m["heldout_mean"]) and np.isfinite(m["loto_mean"])
    assert m["tags_present"] == 0.0 and m["tags_expected"] == 5.0


def test_rotation_angle():
    assert np.isclose(_rot_angle_deg(np.eye(3), np.eye(3)), 0.0)
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    assert np.isclose(_rot_angle_deg(np.eye(3), rz), 90.0)


def test_head_shift_between_first_and_last_quarter():
    s = fake_session(0, n_calib=6, n_test=2, frames=10)
    n = len(s.t_ms)
    tr = np.tile(np.eye(4), (n, 1, 1))
    tr[:, 0, 3] = np.linspace(0.0, 4.0, n)          # drifts 4 cm sideways in total
    out = drift_and_head(replace(s, transform=tr), None)
    assert 2.5 < out["head_shift_cm"] < 3.5 and np.isclose(out["head_rot_deg"], 0.0)
    assert np.isnan(drift_and_head(s, None)["head_shift_cm"])   # no transform recorded
    still = np.tile(np.eye(4), (n, 1, 1))
    assert drift_and_head(replace(s, transform=still), None)["head_shift_cm"] == 0.0


def test_within_fit_trains_on_calibration_frames_only():
    # Test-phase targets sit on a different line than calibration targets; if the fit used them the
    # calibration -> test error would be near zero.
    s = fake_session(2, n_calib=10, n_test=4, frames=6)
    xy = s.target_xy.copy()
    xy[s.phase == TEST] = xy[s.phase == TEST][:, ::-1] * np.array([1.0, 0.6])  # inconsistent with features
    within, _ = held_out_stats(replace(s, target_xy=xy))
    assert within is not None and within.err_px.mean() > 100


def test_conditions_reads_lighting_and_tags():
    s = fake_session(0)
    bare = all_metrics(s)
    assert np.isnan(bare["face_brightness"]) and bare["tags_present"] == 0.0
    meta = {**s.meta, "lighting": {"face": 123.0, "ratio": 1.4, "label": "front-lit"},
            "camera": "built-in", "glasses": False, "person_id": "p01"}
    tagged = all_metrics(replace(s, meta=meta))
    assert tagged["face_brightness"] == 123.0 and tagged["lighting_label"] == "front-lit"
    assert tagged["tags_present"] == 3.0   # glasses=False still counts as present; "" / None do not
    assert (CALIB, TEST) == (0, 1)
