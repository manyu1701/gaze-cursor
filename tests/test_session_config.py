from pathlib import Path

import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import CALIB, TEST, SessionRecorder, load_session
from gaze_cursor.features import BLEND_FEATURE_NAMES, N_FEATURES

NAMES = BLEND_FEATURE_NAMES + tuple(f"other{i}" for i in range(52 - len(BLEND_FEATURE_NAMES)))


def test_from_system_is_plausible() -> None:
    pytest.importorskip("Quartz", reason="needs macOS (pyobjc)")
    cfg = ScreenConfig.from_system()
    assert cfg.width_px > 0 and cfg.height_px > 0
    assert 20 < cfg.width_cm < 100 and 10 < cfg.height_cm < 70
    assert cfg.width_px / cfg.height_px == pytest.approx(cfg.width_cm / cfg.height_cm, rel=0.05)


def test_session_round_trip_with_blendshapes(tmp_path: Path) -> None:
    rec = SessionRecorder(meta={"screen": ScreenConfig().to_dict(), "blendshape_names": list(NAMES)})
    feats = np.arange(N_FEATURES, dtype=float)
    blend = np.linspace(0, 1, len(NAMES)).astype(np.float32)
    rec.add(1, CALIB, 0, (10.0, 20.0), None, None, feats, blend)
    rec.add(2, CALIB, 0, (10.0, 20.0), None, None, None, blend)       # blink/no features -> invalid
    rec.add(3, TEST, 1, (30.0, 40.0), None, None, feats, None)         # no blendshapes -> invalid
    s = load_session(rec.save(tmp_path), recompute_features=False)
    assert s.valid.tolist() == [True, False, False]
    x, y, _ = s.select(CALIB, "both")
    assert x.shape == (1, 9 + len(BLEND_FEATURE_NAMES)) and y.tolist() == [[10.0, 20.0]]
    assert s.select(TEST, "iris")[0].shape == (0, 9)
    assert s.meta["blendshape_names"] == list(NAMES)


def test_features_recomputed_from_stored_landmarks(tmp_path: Path) -> None:
    from gaze_cursor.features import extract_features
    from tests.test_features import _face  # synthetic landmarks

    pts = _face(iris_dx=0.02)
    rec = SessionRecorder(meta={"screen": ScreenConfig().to_dict(), "image_size": [1000, 1000],
                                "blendshape_names": list(NAMES)})
    stale = np.zeros(N_FEATURES)  # deliberately wrong stored features
    blend = np.zeros(len(NAMES), np.float32)
    rec.add(1, CALIB, 0, (0.0, 0.0), pts, np.eye(4), stale, blend)
    s = load_session(rec.save(tmp_path))
    assert s.features[0] == pytest.approx(extract_features(pts, np.eye(4), (1000, 1000)))


def test_target_safety_rules() -> None:
    cfg = ScreenConfig()  # 1470x956, notch strip 32 pt, assumed corner radius 24
    assert cfg.target_is_safe((735, 478))
    assert cfg.target_is_safe((147, 96))                      # calibration grid corner
    assert not cfg.target_is_safe((735, 10))                  # inside the menu-bar/notch strip
    assert not cfg.target_is_safe((5, 500))                   # too close to the left edge
    assert not cfg.target_is_safe((1465, 940))                # too close to the bottom-right
    assert cfg.target_is_safe((17, 940))                      # bottom corners are square
    big = ScreenConfig(corner_radius_pt=80, top_inset_pt=0)   # arc check, with no strip to hide it
    assert not big.target_is_safe((20, 20))                   # outside the rounded top-left arc
    assert not big.target_is_safe((1450, 20))                 # ... and the top-right
    assert big.target_is_safe((80, 80))                       # at the arc center


def test_all_standard_targets_are_safe() -> None:
    from gaze_cursor.eval.targets import calibration_grid, held_out_points

    cfg = ScreenConfig()
    grid = calibration_grid(cfg)
    pts = [*grid, *held_out_points(cfg, grid)]
    assert all(cfg.target_is_safe(tuple(p)) for p in pts)


def test_clean_layout_and_valid_fraction_flags() -> None:
    from tests.helpers import fake_session

    assert fake_session(0).is_clean and fake_session(0).layout == "v1"
    assert not fake_session(0, window=[0, 33, 1470, 923]).is_clean
    assert fake_session(0, layout="v2").layout == "v2"
    assert fake_session(0).valid_fraction == 1.0
