"""Cursor mode's decisions and its record-and-replay log (no OS calls)."""

import json

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.cursor_session import EVENT_CODES, CursorSession, window_median
from gaze_cursor.livetest import px_per_deg
from gaze_cursor.sinks.cursor import InputEvent

CFG = ScreenConfig()


class Identity:
    """Predicts the screen point stored in the first two feature columns (100 x the features)."""
    def predict(self, x):
        x = np.atleast_2d(x)
        return x[:, :2] * 100


def make(x_train=None) -> CursorSession:
    x_train = np.array([[1, 1, 0, 0], [10, 8, 0, 0], [5, 4, 0, 0]], float) if x_train is None else x_train
    return CursorSession(CFG, Identity(), x_train, {"dummy": 0}, "session_q.npz")


def feed(s: CursorSession, t0: float, n: int, xy=(5.0, 4.0)) -> float:
    for k in range(n):
        s.frame(t0 + 33 * k, np.array([[xy[0], xy[1], 0, 0]]), (1.0, 0.0, 0.0))
    return t0 + 33 * (n - 1)


def test_warp_goes_to_the_smoothed_gaze_when_tracked() -> None:
    s = make()
    t = feed(s, 0, 60)
    action, xy = s.handle(InputEvent("warp", t), t + 10)
    assert action == "warp" and np.allclose(xy, (500, 400), atol=1)


def test_no_warp_when_the_face_was_lost_or_eyes_out_of_range_or_off() -> None:
    s = make()
    t = feed(s, 0, 30)
    s.frame(t + 33, None, None)                                     # blink / lost face
    action, _ = s.handle(InputEvent("warp", t + 400), t + 400)      # last good sample 400 ms ago
    assert action == "none" and "not tracked" in s.message
    s = make()
    t = feed(s, 0, 5, xy=(500.0, 4.0))                              # far outside the calibration range
    assert s.handle(InputEvent("warp", t), t)[0] == "none" and "range" in s.message
    s = make()
    t = feed(s, 0, 30)
    s.handle(InputEvent("toggle", t), t)
    assert not s.enabled and s.handle(InputEvent("warp", t), t)[0] == "none"
    assert s.status_lines()[0][1] == "off"


def test_quit_and_recalibrate_actions_and_reset() -> None:
    s = make()
    t = feed(s, 0, 10)
    assert s.handle(InputEvent("quit", t), t)[0] == "quit"
    assert s.handle(InputEvent("recalibrate", t), t)[0] == "recalibrate"
    s.recalibrated(Identity(), np.ones((3, 4)), None, "session_r.npz")
    assert s.dot is None and s.calib == "session_r.npz"


def test_window_median_uses_the_fixation_before_the_click() -> None:
    t = np.arange(0, 1000, 33.0)
    v = np.column_stack([t, t])                     # value = time, so the window shows in the median
    m = window_median(t, v, 900.0)
    assert 500 <= m[0] <= 800                       # -400..-100 ms before t = 900
    assert window_median(t, v, 5000.0) is None


def test_clicks_are_scored_against_the_gaze_just_before_and_everything_is_saved(tmp_path) -> None:
    s = make()
    t = feed(s, 0, 60)                                                  # gaze at (500, 400)
    off = px_per_deg(CFG) * 2                                           # click 2 deg to the right
    s.handle(InputEvent("warp", t), t)
    s.handle(InputEvent("click", t + 20, (500 + off, 400), "left"), t + 30)
    err = s.click_errors_deg()
    assert len(err) == 1 and abs(err[0] - 2) < 0.1
    summ = s.summary()
    assert summ["clicks"] == 1 and summ["warps"] == 1 and summ["warps_refused"] == 0
    path = s.save(tmp_path)
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    assert d["frames"].shape == (60, len(CursorSession.FRAME_COLUMNS))
    assert d["events"].shape == (2, len(CursorSession.EVENT_COLUMNS))
    assert d["events"][1, 1] == EVENT_CODES["click"] and meta["calibration"] == "session_q.npz"
    assert meta["summary"]["clicks"] == 1


def test_a_task_can_save_the_same_log_under_its_own_prefix_with_extras_and_landings_are_logged(tmp_path) -> None:
    s = make()
    t = feed(s, 0, 10)
    s.handle(InputEvent("landing", t + 5, (500.0, 400.0)), t + 5)
    path = s.save(tmp_path, prefix="fitts", kind="fitts_task", extra_meta={"trials": [{"index": 0}]},
                  extra_arrays={"moves": np.zeros((3, 4))})
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    assert path.name.startswith("fitts_") and meta["kind"] == "fitts_task" and meta["trials"] == [{"index": 0}]
    assert d["moves"].shape == (3, 4) and d["events"][0, 1] == EVENT_CODES["landing"]
    assert s.summary()["warps"] == 0                                    # a landing is not a warp


def test_status_key_ignores_the_running_time_but_changes_with_what_matters() -> None:
    s = make()
    t = feed(s, 0, 5)
    k0 = s.status_key()
    feed(s, t + 33, 2000)                       # ~66 s later: the 'on N min' info line changes, the key must not
    assert s.status_key() == k0
    s.handle(InputEvent("toggle", 0), 0)        # gaze warp OFF
    assert s.status_key() != k0


def test_a_lasting_warning_is_not_a_one_off_message_or_the_banner() -> None:
    s = make()
    feed(s, 0, 5)
    s.banner = "DEMO: nothing is saved"
    s.message = "no jump: gaze is stale"
    assert not s.lasting_warning()
    s.dot = None                                 # face lost: lasts until it is back
    assert s.lasting_warning()
