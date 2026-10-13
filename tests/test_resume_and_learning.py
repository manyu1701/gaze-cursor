"""Reusing the last calibration, learning from clicks live, autosave and the on-time chip."""

import json
import time

import numpy as np

from gaze_cursor import resume
from gaze_cursor.config import ScreenConfig
from gaze_cursor.cursor_session import EVENT_CODES, CursorSession
from gaze_cursor.livetest import px_per_deg
from gaze_cursor.online import OnlineRidge
from gaze_cursor.sinks.cursor import InputEvent

CFG = ScreenConfig()
RNG = np.random.default_rng(0)


def prior() -> tuple[OnlineRidge, np.ndarray]:
    x = RNG.normal(size=(40, 4))
    y = np.column_stack([700 + 80 * x[:, 0], 450 + 60 * x[:, 1]])
    return OnlineRidge.from_calibration(x, y, alpha=1.0), x


def run_session(session: CursorSession, n: int = 60, feat=(0.0, 0.0, 0.0, 0.0)) -> float:
    for k in range(n):
        session.frame(33.0 * k, np.array([feat], float), (1.0, 0.0, 0.0))
    return 33.0 * (n - 1)


def test_resume_roundtrip_keeps_the_learned_model(tmp_path) -> None:
    model, x = prior()
    model.update(x[0], np.array([650.0, 400.0]))
    resume.save(tmp_path, model, x, {"yaw": 1.0}, "session_q.npz", CFG, calibrated_at=1000.0)
    saved, why = resume.load(tmp_path, CFG, now=1000.0 + 3600)
    assert saved is not None and why == ""
    assert np.allclose(saved.model.predict(x), model.predict(x)) and saved.model.n_updates == 1
    assert saved.ref_center == {"yaw": 1.0} and saved.calib == "session_q.npz"
    assert abs(saved.age_h(1000.0 + 3600) - 1.0) < 1e-9


def test_resume_refuses_old_other_screen_or_missing(tmp_path) -> None:
    assert resume.load(tmp_path, CFG)[0] is None
    model, x = prior()
    resume.save(tmp_path, model, x, {"yaw": 1.0}, "q.npz", CFG, calibrated_at=0.0)
    assert resume.load(tmp_path, CFG, now=(resume.MAX_AGE_H + 1) * 3600)[0] is None
    other = ScreenConfig(width_px=CFG.width_px + 100)
    saved, why = resume.load(tmp_path, other, now=10.0)
    assert saved is None and "screen" in why
    (tmp_path / resume.FILE_NAME).write_bytes(b"garbage")
    assert resume.load(tmp_path, CFG, now=10.0)[0] is None


def test_click_nudges_the_model_towards_where_you_clicked() -> None:
    model, x = prior()
    s = CursorSession(CFG, model, x, {"d": 0}, "q.npz", learn=True)
    t = run_session(s)
    before = model.predict(np.zeros((1, 4)))[0]
    target = before + np.array([2 * px_per_deg(CFG), 0.0])          # 2 deg to the right: within the gate
    s.handle(InputEvent("click", t, tuple(target), "left"), t)
    after = model.predict(np.zeros((1, 4)))[0]
    assert s.learned == 1 and after[0] > before[0]
    assert s.events[-1][1] == EVENT_CODES["click"] and s.events[-1][5] == 1.0     # logged as learned


def test_stray_click_far_from_the_gaze_is_not_learned() -> None:
    model, x = prior()
    s = CursorSession(CFG, model, x, {"d": 0}, "q.npz", learn=True)
    t = run_session(s)
    before = model.predict(np.zeros((1, 4)))[0]
    s.handle(InputEvent("click", t, (before[0] + 20 * px_per_deg(CFG), before[1]), "left"), t)
    assert s.learned == 0 and model.n_updates == 0 and s.events[-1][5] == 0.0


def test_no_learning_when_off_or_without_recent_gaze() -> None:
    model, x = prior()
    off = CursorSession(CFG, model, x, {"d": 0}, "q.npz", learn=False)
    t = run_session(off)
    off.handle(InputEvent("click", t, tuple(model.predict(np.zeros((1, 4)))[0]), "left"), t)
    assert off.learned == 0
    on = CursorSession(CFG, model, x, {"d": 0}, "q.npz", learn=True)
    for k in range(60):
        on.frame(33.0 * k, None, None)                                # face lost the whole time
    on.handle(InputEvent("click", 1900.0, (700.0, 450.0), "left"), 1900.0)
    assert on.learned == 0


def test_chip_shows_time_on_learned_clicks_and_the_reused_calibration() -> None:
    model, x = prior()
    s = CursorSession(CFG, model, x, {"d": 0}, "q.npz", learn=True, resumed="3 h ago")
    s.frame(0.0, np.zeros((1, 4)), (1.0, 0.0, 0.0))
    texts = [t for t, _ in s.status_lines()]
    assert "on 0 min, learned from 0 clicks" in texts and "using your calibration from 3 h ago" in texts
    s.frame(75 * 60000.0, np.zeros((1, 4)), (1.0, 0.0, 0.0))
    texts = [t for t, _ in s.status_lines()]
    assert "on 1:15 h, learned from 0 clicks" in texts and not any("using your" in t for t in texts)


def test_save_is_atomic_and_repeatable(tmp_path) -> None:
    model, x = prior()
    s = CursorSession(CFG, model, x, {"d": 0}, "q.npz", learn=True)
    run_session(s, 10)
    a = s.save(tmp_path)
    run_session(s, 20)
    b = s.save(tmp_path)                                              # an autosave overwrites the earlier file
    assert a == b and list(tmp_path.iterdir()) == [a]                 # no temp file left behind
    d = np.load(a)
    meta = json.loads(str(d["meta"]))
    assert len(d["frames"]) == 30 + 0 or len(d["frames"]) == 30       # all frames so far
    assert meta["learning"]["on"] is True and meta["summary"]["clicks_learned"] == 0


def test_summary_counts_learned_clicks() -> None:
    model, x = prior()
    s = CursorSession(CFG, model, x, {"d": 0}, "q.npz", learn=True)
    t = run_session(s)
    s.handle(InputEvent("click", t, tuple(model.predict(np.zeros((1, 4)))[0]), "left"), t)
    assert s.summary()["clicks_learned"] == 1 and time.time() > 0


def test_autosave_does_not_make_an_old_calibration_look_fresh(tmp_path) -> None:
    model, x = prior()
    s = CursorSession(CFG, model, x, {"d": 0}, "q.npz", learn=True, calibrated_at=1000.0)
    resume.save(tmp_path, s.model, s.x_train, s.ref_center, s.calib, CFG, s.calibrated_at)
    saved, _ = resume.load(tmp_path, CFG, now=1000.0 + 3600)
    assert saved is not None and saved.calibrated_at == 1000.0
    again = CursorSession(CFG, saved.model, saved.x_train, saved.ref_center, saved.calib, learn=True,
                          calibrated_at=saved.calibrated_at)
    resume.save(tmp_path, again.model, again.x_train, again.ref_center, again.calib, CFG, again.calibrated_at)
    assert resume.load(tmp_path, CFG, now=1000.0 + (resume.MAX_AGE_H + 1) * 3600)[0] is None
    again.recalibrated(model, x, {"d": 0}, "new.npz")                    # new dots: the clock restarts
    assert again.calibrated_at > 1000.0 + 1e6
