"""The live tests in Run, simulated without a camera or window."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from gaze_cursor.config import ScreenConfig

CFG = ScreenConfig()


def _overlay():
    path = Path(__file__).resolve().parents[1] / "scripts" / "gaze_overlay.py"
    spec = importlib.util.spec_from_file_location("gaze_overlay_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_targets_test_scores_a_perfect_tracker_as_all_hits() -> None:
    go = _overlay()
    test = go.LiveTest("targets", CFG, 1000.0)
    t = 1000.0
    while not test.done and t < 1000 + 12 * 5000:
        t += 33.3
        c = np.array(test.target_at(t)) + [3.0, -2.0]
        test.update(t, c, c, (1.0, 0.0, 0.0), np.zeros(4))
    title, lines, s, boxes = test.summary()
    assert test.done and s["hit_rate"] == 1.0 and s["n"] == 12 and len(boxes) == 12
    assert s["error_deg"] < 0.1 and s["head_moved_frac"] == 0.0 and "hits 100%" in lines[0][0]


def test_path_test_recovers_the_lag_and_flags_head_movement() -> None:
    go = _overlay()
    test = go.LiveTest("path", CFG, 0.0)
    t = 0.0
    while not test.done:
        t += 33.3
        d = test.path.at(t / 1000 - 0.15)
        test.update(t, d, d, (1.2, 0.0, 0.0), np.zeros(4))   # head moved: 20% closer
    _, lines, s, _ = test.summary()
    assert abs(s["lag_ms"] - 150) <= 20 and s["head_moved_frac"] == 1.0
    assert any("away from the calibration position" in text for text, _ in lines)


def test_live_panel_prompts_after_head_movement() -> None:
    go = _overlay()
    from gaze_cursor.livetest import LiveStats
    st = LiveStats()
    for k in range(20):
        st.add(33.3 * k, 12.0, True, np.array([700.0, 400.0]))
    still = [t for t, _ in go.live_lines(st, CFG, (1.0, 0.5, 0.2), 0.0)]
    moved = [t for t, _ in go.live_lines(st, CFG, (1.0, 6.0, 0.0), 0.0)]
    assert not any("you moved" in t for t in still) and any("you moved" in t for t in moved)
