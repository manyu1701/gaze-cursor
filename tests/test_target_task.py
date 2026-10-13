import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.target_task import TargetTask, TaskConfig, describe_bias

SCREEN = ScreenConfig()
FPS = 30


def run(task: TargetTask, gaze_of, max_s: float = 400.0) -> None:
    """Drive the task at 30 fps; ``gaze_of(trial, t)`` gives the dot (or None)."""
    t = 0.0
    task.start(t)
    while not task.done and t < max_s:
        t += 1 / FPS
        tr = task.current
        if task.update(t, gaze_of(tr, t)) is not None and len(task.trials) < task.cfg.n_trials:
            task.start(t)


def test_perfect_gaze_hits_after_the_dwell_and_the_box_shrinks() -> None:
    task = TargetTask(SCREEN, TaskConfig(n_trials=10), seed=0)
    run(task, lambda tr, t: tr.center)
    s = task.summary()
    assert s["n_hits"] == 10 and s["hit_rate"] == 1.0
    assert s["median_time_s"] == pytest.approx(0.6, abs=2 / FPS)       # the dwell (+ frame timing), nothing else
    assert task.trials[-1].half < task.trials[0].half                    # two hits in a row -> smaller
    assert s["error_cm"] == pytest.approx(0, abs=1e-9)


def test_no_gaze_times_out_and_the_box_grows_to_its_cap() -> None:
    cfg = TaskConfig(n_trials=12)
    task = TargetTask(SCREEN, cfg, seed=0)
    run(task, lambda tr, t: None)
    s = task.summary()
    assert s["n_hits"] == 0 and all(tr.t_end - tr.t_start == pytest.approx(cfg.timeout_s, abs=0.05) for tr in task.trials)
    assert task.trials[-1].half == cfg.max_half


def test_staircase_settles_where_the_dot_noise_makes_hits_uncertain() -> None:
    rng = np.random.default_rng(1)
    noise_pt = 60.0
    task = TargetTask(SCREEN, TaskConfig(n_trials=60), seed=1)
    run(task, lambda tr, t: (tr.center[0] + rng.normal(0, noise_pt), tr.center[1] + rng.normal(0, noise_pt)))
    s = task.summary()
    assert 0.2 < s["hit_rate"] < 0.95
    assert noise_pt * 1.0 < s["box_cm"] / SCREEN.cm_per_px < noise_pt * 8   # box side in points, a few sigma


def test_bias_is_measured_after_settling() -> None:
    task = TargetTask(SCREEN, TaskConfig(n_trials=6, settle_s=0.3), seed=2)
    off = (40.0, -40.0)                                                     # right and up, in points
    run(task, lambda tr, t: (tr.center[0] + off[0], tr.center[1] + off[1]))
    bx, by = task.summary()["bias_cm"]
    assert bx == pytest.approx(off[0] * SCREEN.cm_per_px) and by == pytest.approx(off[1] * SCREEN.cm_per_px)
    assert "to the right" in describe_bias((bx, by)) and "above" in describe_bias((bx, by))
    assert "no systematic offset" in describe_bias((0.1, -0.2))


def _hit_time(outside: tuple[float, float]) -> float | None:
    """Dot on the box at 30 fps except during ``outside`` (start, end) seconds; time of the hit."""
    task = TargetTask(SCREEN, TaskConfig(n_trials=1, dwell_s=0.6, grace_s=0.15), seed=3)
    tr = task.start(0.0)
    far = (tr.center[0] + 500, tr.center[1])
    for i in range(1, 120):
        t = i / FPS
        if task.update(t, far if outside[0] <= t < outside[1] else tr.center) == "hit":
            return t
    return None


def test_brief_exits_are_forgiven_but_long_ones_restart_the_dwell() -> None:
    assert _hit_time((0.3, 0.4)) == pytest.approx(1 / FPS + 0.6, abs=1.5 / FPS)    # 0.1 s out: forgiven
    assert _hit_time((0.3, 0.6)) == pytest.approx(0.6 + 0.6, abs=1.5 / FPS)        # 0.3 s out: dwell restarts


def test_boxes_stay_on_the_visible_screen_and_jump() -> None:
    task = TargetTask(SCREEN, TaskConfig(n_trials=40), seed=4)
    run(task, lambda tr, t: tr.center)
    for a, b in zip(task.trials, task.trials[1:]):
        assert np.hypot(b.center[0] - a.center[0], b.center[1] - a.center[1]) >= task.cfg.min_jump
    for tr in task.trials:
        assert SCREEN.target_is_safe(tr.center, ring_px=tr.half)
