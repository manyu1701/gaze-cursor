from pathlib import Path

import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import CALIB, TEST, SessionRecorder, load_session
from gaze_cursor.features import BLEND_FEATURE_NAMES

NAMES = BLEND_FEATURE_NAMES + tuple(f"other{i}" for i in range(52 - len(BLEND_FEATURE_NAMES)))


def test_from_system_is_plausible() -> None:
    cfg = ScreenConfig.from_system()
    assert cfg.width_px > 0 and cfg.height_px > 0
    assert 20 < cfg.width_cm < 100 and 10 < cfg.height_cm < 70
    assert cfg.width_px / cfg.height_px == pytest.approx(cfg.width_cm / cfg.height_cm, rel=0.05)


def test_session_round_trip_with_blendshapes(tmp_path: Path) -> None:
    rec = SessionRecorder(meta={"screen": ScreenConfig().to_dict(), "blendshape_names": list(NAMES)})
    feats = np.arange(9, dtype=float)
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
    stale = np.zeros(9)  # deliberately wrong stored features
    blend = np.zeros(len(NAMES), np.float32)
    rec.add(1, CALIB, 0, (0.0, 0.0), pts, np.eye(4), stale, blend)
    s = load_session(rec.save(tmp_path))
    assert s.features[0] == pytest.approx(extract_features(pts, np.eye(4), (1000, 1000)))
