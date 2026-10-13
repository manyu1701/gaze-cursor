"""Live tests: box layout and hit logic, path and pursuit scoring, live stats, head shift."""

import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.livetest import (
    ACC_FROM_MS, ACC_SHOW_MS, accuracy_result, summarize_accuracy,
    DWELL_MS, MIN_HOP_DEG, TIMEOUT_MS, BoxTrial, LiveStats, PathTarget, head_moved, head_shift, make_boxes,
    pursuit_summary, px_per_deg, summarize_boxes)

CFG = ScreenConfig()


def test_boxes_are_visible_spread_and_need_real_moves() -> None:
    boxes = make_boxes(CFG, seed=3)
    ppd = px_per_deg(CFG)
    assert len(boxes) == 12 and {b.size_deg for b in boxes} == {4.0, 6.0}
    for b in boxes:
        x, y = b.center
        assert b.half_px <= x <= CFG.width_px - b.half_px and CFG.top_inset_pt + b.half_px <= y <= CFG.height_px - b.half_px
    hops = [np.hypot(*np.subtract(a.center, b.center)) / ppd for a, b in zip(boxes, boxes[1:])]
    assert min(hops) >= MIN_HOP_DEG
    xs, ys = np.array([b.center for b in boxes]).T
    assert xs.min() < 0.3 * CFG.width_px and xs.max() > 0.7 * CFG.width_px and ys.max() > 0.6 * CFG.height_px


def test_box_trial_hits_after_a_dwell_and_resets_when_leaving() -> None:
    box = make_boxes(CFG)[0]
    c = np.array(box.center)
    far = c + [box.half_px * 3, 0]
    tr = BoxTrial(box, 0.0)
    t = 0.0
    for _ in range(10):                       # look elsewhere for 330 ms
        t += 33; tr.update(t, far)
    entry = t + 33
    for _ in range(5):                        # inside briefly, then out: dwell resets
        t += 33; tr.update(t, c)
    t += 33; tr.update(t, far)
    assert tr.status == "waiting" and tr.first_entry_ms == pytest.approx(entry)
    while not tr.done:
        t += 33; tr.update(t, c + [2, -1])
    assert tr.status == "hit" and tr.hit_ms >= DWELL_MS
    r = tr.result(CFG)
    assert r["hit"] and r["error_deg"] < 0.1 and r["jitter_deg"] < 0.1


def test_box_trial_times_out_and_summary_counts_it() -> None:
    box = make_boxes(CFG)[0]
    tr = BoxTrial(box, 0.0)
    t = 0.0
    while not tr.done:
        t += 33; tr.update(t, None)
    assert tr.status == "timeout" and t >= TIMEOUT_MS
    hit = BoxTrial(box, 0.0)
    for k in range(40):
        hit.update(33.0 * k, np.array(box.center))
    s = summarize_boxes([tr.result(CFG), hit.result(CFG)])
    assert s["hit_rate"] == 0.5 and s["n"] == 2 and np.isfinite(s["median_hit_ms"])


def test_path_moves_at_constant_speed_inside_the_screen() -> None:
    p = PathTarget(CFG)
    t = np.linspace(0, p.duration_s, 2000)
    xy = p.at(t)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(t) / px_per_deg(CFG)
    assert np.median(speed) == pytest.approx(8.0, rel=0.02)
    assert xy[:, 1].min() > CFG.top_inset_pt and 15 < p.duration_s < 60


def test_pursuit_summary_recovers_a_planted_lag() -> None:
    p = PathTarget(CFG)
    t = np.arange(0, p.duration_s, 1 / 30)
    dot = p.at(t - 0.2) + np.random.default_rng(0).normal(0, 5, (len(t), 2))
    s = pursuit_summary(t, dot, p, CFG)
    assert s["lag_ms"] == pytest.approx(200, abs=20) and s["error_at_lag_deg"] < s["error_deg"]
    assert pursuit_summary(t, p.at(t), p, CFG)["within"] == 1.0


def test_live_stats_and_head_shift() -> None:
    st = LiveStats()
    for k in range(40):
        st.add(33.3 * k, 12.0, k % 4 != 0, np.array([700.0, 400.0]))
    assert st.fps() == pytest.approx(30, rel=0.02) and st.latency_ms() == 12 and st.tracked() == pytest.approx(0.75, abs=0.05)
    assert st.jitter_deg(CFG) == 0
    ref = {"distance": 0.066, "x": 0.5, "y": 0.5}
    assert head_shift(ref, ref) == (1.0, 0.0, 0.0) and not head_moved(1.0, 0.0, 0.0)
    ratio, dx, dy = head_shift({"distance": 0.066, "x": 0.56, "y": 0.5}, ref)
    assert dx == pytest.approx(5.7, abs=0.1) and head_moved(ratio, dx, dy) and head_moved(1.2, 0, 0)


def test_a_chase_at_the_end_of_a_miss_does_not_dominate_its_offset() -> None:
    box = make_boxes(CFG)[0]
    c = np.array(box.center)
    tr = BoxTrial(box, 0.0)
    t = 0.0
    while t < 3000:                                        # 3 s just outside the box, 1 deg off
        t += 33; tr.update(t, c + [box.half_px + px_per_deg(CFG), 0])
    while not tr.done:                                     # then a 1 s chase far away
        t += 33; tr.update(t, c + [12 * px_per_deg(CFG), 0])
    r = tr.result(CFG)
    assert not r["hit"] and r["error_deg"] < 0.5 * 12


def test_accuracy_result_scores_only_the_settled_window_and_finds_a_vertical_shift() -> None:
    ppd = px_per_deg(CFG)
    res = []
    for k, y in enumerate((150.0, 480.0, 850.0)):
        c = (700.0, y)
        samples = [(t, 700.0 + 300.0 * (t < ACC_FROM_MS), y - 2 * ppd) for t in np.arange(0, ACC_SHOW_MS, 33.0)]
        r = accuracy_result(samples, c, 0.0, CFG)
        assert r["error_deg"] == pytest.approx(2.0, abs=0.05) and r["offset_deg"][1] == pytest.approx(-2.0, abs=0.05)
        res.append(r)
    s = summarize_accuracy(res)
    assert s["n"] == 3 and s["within_2_5"] == 1.0 and s["dy_top_deg"] == pytest.approx(-2.0, abs=0.05)
    assert accuracy_result([], (0.0, 0.0), 0.0, CFG)["n"] == 0
