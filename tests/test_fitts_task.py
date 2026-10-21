"""The Fitts task's pure logic: ring geometry, block building, trial bookkeeping, seat check, dot schedule,
and the pointer-move event of the input layer."""

import math

import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.fitts_task import (
    Block, FittsTask, dot_schedule, line_points, index_of_difficulty, make_blocks, ring_points, seat_ok, star_order)
from gaze_cursor.sinks.cursor import FLAG_COMMAND, RIGHT_COMMAND_KEYCODE, HotkeyDetector, InputEvent

CFG = ScreenConfig(width_px=1470, height_px=956, width_cm=30.0, height_cm=19.5, distance_cm=60.0)


def ev(kind: str, t: float, xy=None, button=None) -> InputEvent:
    return InputEvent(kind, t, xy, button)


def test_index_of_difficulty_and_star_order() -> None:
    assert index_of_difficulty(300, 100) == pytest.approx(2.0)
    order = star_order(9)
    assert sorted(order) == list(range(9)) and order[:3] == [0, 5, 1]


def test_ring_points_have_the_ring_amplitude_between_consecutive_targets_and_must_fit() -> None:
    pts = ring_points(CFG, 600, 60)
    assert len(pts) == 9
    for a, b in zip(pts, pts[1:]):                       # star order: every step crosses the ring
        assert math.dist(a, b) > 0.9 * 600 * math.sin(math.radians(80))
    with pytest.raises(ValueError):
        ring_points(CFG, 2000, 60)
    with pytest.raises(ValueError):
        star_order(8)


def test_make_blocks_covers_every_cell_keeps_seats_together_and_is_seeded() -> None:
    args = (CFG, ("trackpad", "gaze_jump"), ("centre", "moved"), (500, 700), (60, 120))
    a = make_blocks(*args, seed=3)
    assert len(a) == 2 * 2 * 2 * 2
    assert {(b.method, b.seat, b.amplitude, b.width) for b in a} == {
        (m, s, am, w) for m in ("trackpad", "gaze_jump") for s in ("centre", "moved") for am in (500, 700)
        for w in (60, 120)}
    seats = [b.seat for b in a]
    assert seats in (["centre"] * 8 + ["moved"] * 8, ["moved"] * 8 + ["centre"] * 8)   # kept together, order shuffled
    assert any(make_blocks(*args, seed=k)[0].seat == "moved" for k in range(20))
    assert a == make_blocks(*args, seed=3)
    assert a != make_blocks(*args, seed=4)


def test_block_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        Block("mouse", "centre", 500, 60)


def test_a_trial_records_phases_and_the_click_decides_hit_or_miss() -> None:
    task = FittsTask(CFG, [Block("gaze_jump", "centre", 500, 60, n_trials=2)])
    task.start_block()
    trial = task.next_target(1000.0, (10.0, 10.0), (1.0, 0.0, 0.0))
    task.on_event(ev("move", 1120.0, (20, 20)))
    task.on_event(ev("warp", 1300.0))
    task.on_event(ev("landing", 1301.0, (700.0, 400.0)))
    assert trial.t_landing == 1301.0
    task.on_event(ev("warp", 1400.0))                                 # a second jump changes neither field
    task.on_event(ev("landing", 1401.0, (5.0, 5.0)))
    assert (trial.t_warp, trial.warp_landing, trial.t_landing) == (1300.0, (700.0, 400.0), 1301.0)
    assert task.on_event(ev("click", 1900.0, (0, 0), "left")) == "miss"     # a click far from the target
    assert trial.hit is False and (trial.endpoint_error_px or 0) > 30
    task.mark_shown(trial, 1016.0)
    assert (trial.t_requested, trial.t_onset) == (1000.0, 1016.0) and trial.movement_time_ms == 884.0
    assert (trial.t_first_move, trial.t_warp, trial.warp_landing) == (1120.0, 1300.0, (700.0, 400.0))
    second = task.next_target(3000.0, (5.0, 5.0))
    out = task.on_event(ev("click", 3500.0, (second.center[0] + 10, second.center[1]), "left"))
    assert out == "hit" and second.hit and task.block_done and task.done
    assert task.summary()["gaze_jump/centre"]["hit_rate"] == 0.5
    d = second.to_dict()
    assert d["id_bits"] == pytest.approx(index_of_difficulty(500, 60)) and d["hit"] is True


def test_moves_are_kept_with_their_trial_and_a_right_click_does_not_end_a_trial() -> None:
    task = FittsTask(CFG, [Block("trackpad", "centre", 500, 60, n_trials=1)])
    task.start_block()
    task.on_event(ev("move", 10.0, (1, 1)))                            # before any target: trial index -1
    task.next_target(100.0, None)
    task.on_event(ev("move", 150.0, (2, 2)))
    assert task.on_event(ev("click", 200.0, (3, 3), "right")) is None and task.current is not None
    assert task.moves_array().tolist() == [[10.0, 1, 1, -1], [150.0, 2, 2, 0]]


def test_timeout_is_a_flagged_miss_and_blocks_advance_only_when_finished() -> None:
    task = FittsTask(CFG, [Block("trackpad", "centre", 500, 60, 1), Block("gaze_jump", "centre", 500, 60, 1)],
                     timeout_ms=1000.0)
    task.start_block()
    task.next_target(0.0, None)
    assert task.update(500.0) is None and task.update(1000.0) == "miss"
    assert task.trials[0].timed_out and task.block_done and not task.done
    with pytest.raises(RuntimeError):
        task.next_target(1100.0, None)                                  # the block's single trial is used up
    task.start_block()
    task.next_target(1200.0, None)
    task.on_event(ev("click", 1300.0, (0, 0), "left"))
    with pytest.raises(RuntimeError):
        task.start_block()                                              # nothing further


def test_seat_check_uses_the_measured_head_shift() -> None:
    still = [(1.0, 0.5, 0.0)] * 10
    moved = [(1.0, 9.0, 0.0)] * 10
    assert seat_ok(still, "centre") and not seat_ok(still, "moved")
    assert seat_ok(moved, "moved") and not seat_ok(moved, "centre")
    assert not seat_ok([None] * 10, "centre") and not seat_ok([], "moved")
    assert seat_ok(still[:8] + moved[:2], "centre") and not seat_ok(still[:7] + moved[:3], "centre")
    with pytest.raises(ValueError):
        seat_ok(still, "sideways")


def test_dot_schedule_shows_every_position_once_per_mode() -> None:
    sched = dot_schedule(CFG, seed=1)
    assert len(sched) == 18
    for mode in ("eyes", "head"):
        assert sorted(d["index"] for d in sched if d["mode"] == mode) == list(range(9))
    assert any(dot_schedule(CFG, seed=k)[0]["mode"] == "head" for k in range(20))
    assert {d["index"]: d["xy"] for d in sched if d["mode"] == "eyes"} == {
        d["index"]: d["xy"] for d in sched if d["mode"] == "head"}
    assert all(CFG.target_is_safe(d["xy"], 16) for d in sched)
    with pytest.raises(ValueError):
        dot_schedule(CFG, n_dots=8)


def test_moves_are_reported_and_do_not_cancel_a_warp_tap() -> None:
    d = HotkeyDetector()
    d.flags_changed(RIGHT_COMMAND_KEYCODE, FLAG_COMMAND, 0)
    out = d.mouse_moved((3, 4), 50)
    assert [(e.kind, e.xy) for e in out] == [("move", (3.0, 4.0))]
    assert [e.kind for e in d.flags_changed(RIGHT_COMMAND_KEYCODE, 0, 100)] == ["warp"]


def test_line_layout_alternates_two_far_targets_and_must_fit() -> None:
    a, b = line_points(CFG, 1100, 140)
    assert math.dist(a, b) == pytest.approx(1100) and a[1] == b[1]
    with pytest.raises(ValueError):
        line_points(CFG, 1400, 140)
    task = FittsTask(CFG, [Block("trackpad", "centre", 1100, 140, n_trials=4, layout="line")])
    task.start_block()
    centers = []
    for k in range(4):
        tr = task.next_target(100.0 * k, None)
        centers.append(tr.center)
        task.on_event(ev("click", 100.0 * k + 50, tr.center, "left"))
    assert centers == [a, b, a, b]
    with pytest.raises(ValueError):
        Block("trackpad", "centre", 500, 60, layout="spiral")


def test_practice_blocks_come_first_one_per_method_and_are_left_out_of_the_summary() -> None:
    blocks = make_blocks(CFG, ("trackpad", "gaze_jump"), ("centre", "moved"), (600,), (70, 140),
                         line_amplitudes=(1100,), practice=True, seed=2)
    pr = [b for b in blocks if b.practice]
    assert blocks[:2] == pr and {b.method for b in pr} == {"trackpad", "gaze_jump"} and all(b.n_trials == 4 for b in pr)
    assert len(blocks) == 2 + 2 * 2 * 4 and {b.layout for b in blocks} == {"ring", "line"}
    assert len({b.seat for b in pr}) == 1 and pr[0].seat == blocks[2].seat       # warm up in the first seat
    task = FittsTask(CFG, blocks[:1])
    task.start_block()
    tr = task.next_target(0.0, None)
    task.on_event(ev("click", 10.0, tr.center, "left"))
    assert task.summary() == {}
