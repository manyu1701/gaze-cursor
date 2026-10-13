"""Quick 9-dot recalibration in Run: it must be exactly the configuration that was tested."""

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval import prereg3
from gaze_cursor.eval.correction import choose_dots
from gaze_cursor.eval.targets import calibration_grid_v2
from gaze_cursor.quick import QUICK_CONFIG, QUICK_DOTS, fit_quick, fit_report, missing_dots, quick_targets

SCREEN = ScreenConfig(width_px=1470, height_px=956, width_cm=29.06, height_cm=18.9)


def test_quick_calibration_is_the_frozen_round_3_candidate() -> None:
    assert QUICK_DOTS == prereg3.PREREG3_DOTS and QUICK_CONFIG == prereg3.PREREG3_CANDIDATE
    assert prereg3.PREREG3_METHOD == "fresh"


def test_quick_dots_are_the_ones_the_test_scored() -> None:
    grid = calibration_grid_v2(SCREEN)
    tested = set(choose_dots(grid, QUICK_DOTS, (SCREEN.width_px / 2, SCREEN.height_px / 2)).tolist())
    for seed in (0, 1, 2):
        t = quick_targets(SCREEN, seed=seed)
        assert {x.idx for x in t} == tested and len(t) == QUICK_DOTS
        assert all(np.allclose(x.xy, grid[x.idx]) and SCREEN.target_is_safe(x.xy) for x in t)
    assert [x.idx for x in quick_targets(SCREEN, seed=0)] != [x.idx for x in quick_targets(SCREEN, seed=1)]


def test_fit_quick_recovers_a_linear_mapping_and_reports_missed_dots() -> None:
    rng = np.random.default_rng(0)
    targets = quick_targets(SCREEN)
    xy = np.repeat(np.array([t.xy for t in targets]), 30, axis=0)
    idx = np.repeat([t.idx for t in targets], 30)
    a = np.array([[400.0, 0], [0, 300], [20, 0], [0, 10]])                  # 4 gaze features -> screen
    x = (xy - xy.mean(axis=0)) @ np.linalg.pinv(a) + rng.normal(0, 1e-4, (len(xy), 4))
    model = fit_quick(x, xy, SCREEN)
    rep = fit_report(model, x, xy, idx, SCREEN)
    assert rep["dots"] == QUICK_DOTS and rep["worst_deg"] < 1.0
    assert missing_dots(idx, targets) == []
    assert missing_dots(idx[idx != targets[0].idx], targets) == [targets[0].idx]
