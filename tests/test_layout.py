import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import CALIB, TEST
from gaze_cursor.eval.targets import build_plan, calibration_grid_v2, held_out_points_v2

CFG = ScreenConfig()


def test_v2_has_25_targets_with_16_calibration_and_9_held_out() -> None:
    plan = build_plan(CFG, "v2", seed=1)
    assert len(plan) == 25
    assert sum(t.phase == CALIB for t in plan) == 16 and sum(t.phase == TEST for t in plan) == 9
    assert sorted(t.idx for t in plan) == list(range(25))


def test_v2_targets_are_all_safe_and_the_grid_reaches_the_safe_edges() -> None:
    plan = build_plan(CFG, "v2", seed=2)
    assert all(CFG.target_is_safe(t.xy) for t in plan)
    grid = calibration_grid_v2(CFG)
    assert grid[:, 0].min() == 24 and grid[:, 0].max() == CFG.width_px - 24
    assert grid[:, 1].min() == CFG.top_inset_pt + 24 and grid[:, 1].max() == CFG.height_px - 24


def test_v2_held_out_are_interior_and_well_separated() -> None:
    grid = calibration_grid_v2(CFG)
    for seed in range(5):
        held = held_out_points_v2(CFG, grid, seed=seed)
        assert len(held) == 9
        assert (held >= grid.min(axis=0) + 60 - 1e-6).all() and (held <= grid.max(axis=0) - 60 + 1e-6).all()
        assert min(np.linalg.norm(h - g) for h in held for g in grid) >= 100
        pair = [np.linalg.norm(a - b) for i, a in enumerate(held) for b in held[i + 1:]]
        assert min(pair) >= 100


def test_v2_is_deterministic_per_seed_and_randomised_across_seeds() -> None:
    a, b, c = (build_plan(CFG, "v2", seed=s) for s in (3, 3, 4))
    assert a == b and a != c
    assert [t.idx for t in a] != sorted(t.idx for t in a)  # shuffled, not in idx order


def test_v1_layout_is_unchanged() -> None:
    plan = build_plan(CFG, "v1")
    assert len(plan) == 21 and sum(t.phase == CALIB for t in plan) == 9
    assert [t.idx for t in plan] == list(range(21))
