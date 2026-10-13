"""Round-4 replay and decision on synthetic Cursor sessions (no camera)."""

import json

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.cursor_session import CursorSession
from gaze_cursor.eval import online_eval as oe
from gaze_cursor.livetest import px_per_deg
from gaze_cursor.online import OnlineRidge
from gaze_cursor.sinks.cursor import InputEvent

CFG = ScreenConfig()
W = np.array([[300.0, 10.0], [-20.0, 250.0], [280.0, 5.0], [15.0, 240.0]])


def prior(seed: int = 0) -> OnlineRidge:
    rng = np.random.default_rng(seed)
    dots = rng.normal(size=(9, 4))
    x = np.repeat(dots, 30, axis=0) + rng.normal(scale=0.05, size=(270, 4))
    return OnlineRidge.from_calibration(x, x @ W + [735, 478], alpha=10.0)


def record(tmp_path, offset_deg: float, seed: int, started: str, n_clicks: int = 40,
           minutes: float = 6.0) -> tuple:
    """A session whose clicks land ``offset_deg`` below where the frozen model says the user looked."""
    rng = np.random.default_rng(seed)
    p = prior()
    s = CursorSession(CFG, p, rng.normal(size=(50, 4)), None, "session_q.npz", started=started)
    t, dt, gap = 0.0, 33.0, minutes * 60000 / n_clicks
    for _ in range(n_clicks):
        fx = np.clip(rng.normal(size=(1, 4)) * 0.6, -1.2, 1.2)
        for _ in range(30):                                   # one fixation (30 frames at 30 fps), then the click
            s.frame(t, fx + rng.normal(scale=0.02, size=(1, 4)), None)
            t += dt
        g = p.predict(fx)[0] + [0, offset_deg * px_per_deg(CFG)] + rng.normal(scale=15, size=2)
        s.handle(InputEvent("click", t - dt, (float(g[0]), float(g[1])), "left"), t)
        t += gap                                              # reading / typing until the next click
    path = s.save(tmp_path)
    (tmp_path / "session_q.npz").touch()
    return path, p


def test_a_planted_offset_is_learned_and_passes_the_rule(tmp_path, monkeypatch) -> None:
    reps = []
    for k in range(3):
        path, p = record(tmp_path, 3.0, k, f"20261014_10{k}000")
        monkeypatch.setattr(oe, "prior_from_quick", lambda q, p=p: p)
        reps.append(oe.replay(path, tmp_path))
    assert all(r.integrity_px[0] < 1e-6 for r in reps)
    assert all(oe.qualifies(r, "2026-10-14T09:00:00") is None for r in reps)
    res = oe.decide(reps)
    assert res["supported"] and res["stats"]["mean"] > 1.0


def test_no_offset_gives_no_gain(tmp_path, monkeypatch) -> None:
    reps = []
    for k in range(3):
        path, p = record(tmp_path, 0.0, 10 + k, f"20261014_11{k}000")
        monkeypatch.setattr(oe, "prior_from_quick", lambda q, p=p: p)
        reps.append(oe.replay(path, tmp_path))
    assert not oe.decide(reps)["supported"]


def test_a_mismatched_calibration_is_excluded_and_old_sessions_do_not_count(tmp_path, monkeypatch) -> None:
    path, _ = record(tmp_path, 3.0, 0, "20261013_203902")
    monkeypatch.setattr(oe, "prior_from_quick", lambda q: prior(seed=99))   # not the model that ran live
    rep = oe.replay(path, tmp_path)
    assert rep.excluded_segments and not rep.clicks
    assert oe.qualifies(rep, "2026-10-14T00:00:00") == "recorded before the freeze"


def test_the_log_records_which_calibration_ran_when(tmp_path) -> None:
    s = CursorSession(CFG, prior(), np.zeros((3, 4)), None, "session_a.npz")
    s.frame(100.0, None, None)
    s.recalibrated(prior(), np.zeros((3, 4)), None, "session_b.npz")
    s.frame(5000.0, None, None)
    meta = json.loads(str(np.load(s.save(tmp_path))["meta"]))
    assert meta["calibrations"] == [[100.0, "session_a.npz"], [5000.0, "session_b.npz"]]
