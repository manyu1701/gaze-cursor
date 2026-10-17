"""Cursor mode's face crops (the label window before each click) and the 6c closing fixes."""

import json

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.cursor_session import FACE_PER_CLICK, CursorSession
from gaze_cursor.eval.faces import FaceCrops, faces_path_for
from gaze_cursor.sinks.cursor import InputEvent

CFG = ScreenConfig()
IMG = np.zeros((40, 40, 3), np.uint8)
X = np.array([[5.0, 4.0, 0, 0]])


class Identity:
    def predict(self, x):
        return np.atleast_2d(x)[:, :2] * 100


def make(keep: bool) -> CursorSession:
    x_train = np.array([[1, 1, 0, 0], [10, 8, 0, 0], [5, 4, 0, 0]], float)
    s = CursorSession(CFG, Identity(), x_train, {"dummy": 0}, "session_q.npz")
    s.keep_faces = keep
    return s


def run_frames(s: CursorSession, n: int, x=X) -> None:
    for k in range(n):                                  # 33 ms frames, a 20 x 20 face box
        s.offer_crop(33.0 * k, IMG, (5, 5, 25, 25))
        s.frame(33.0 * k, x, (1.0, 0.0, 0.0))


def test_crops_come_from_the_label_window_before_a_click_and_are_saved_locally(tmp_path) -> None:
    s = make(True)
    run_frames(s, 60)
    assert len(s._recent) < 25                          # only the last ~0.5 s is held, nothing is encoded yet
    t_click = 33.0 * 59 + 20
    s.handle(InputEvent("click", t_click, xy=(500.0, 400.0), button="left"), t_click)
    assert len(s.face_jpegs) == FACE_PER_CLICK
    ms = [r[2] for r in s.face_rows]
    assert all(-400 <= m <= -100 for m in ms) and ms == sorted(ms)
    path = s.save(tmp_path)
    faces = faces_path_for(path)
    assert (faces.stat().st_mode & 0o777) == 0o600
    crops = FaceCrops.load(faces)
    assert len(crops) == FACE_PER_CLICK and crops.image(0).shape[:2] == (20, 20)
    with np.load(faces) as z:
        assert list(z["event_row"]) == [len(s.events) - 1] * FACE_PER_CLICK and z["features_valid"].all()
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
    assert meta["face_crops"]["file"] == faces.name and meta["face_crops"]["rows"] == FACE_PER_CLICK


def test_nothing_is_kept_unless_asked_and_no_file_without_clicks(tmp_path) -> None:
    s = make(False)                                     # a demo or --no-images
    run_frames(s, 30)
    s.handle(InputEvent("click", 1000.0, xy=(500.0, 400.0), button="left"), 1000.0)
    assert not s.face_jpegs and not s._recent
    path = s.save(tmp_path)
    with np.load(path) as z:
        assert json.loads(str(z["meta"]))["face_crops"] is None
    assert not faces_path_for(path).exists()
    s2 = make(True)
    run_frames(s2, 5)
    path2 = s2.save(tmp_path)                           # asked for, but nothing clicked: no file
    assert not faces_path_for(path2).exists()


def test_a_backfilled_distance_estimate_is_not_a_measured_tag() -> None:
    from gaze_cursor.conditions import build_conditions, validate
    block = build_conditions(person_id="p01", camera={"name": "cam", "width": 1280, "height": 720},
                             screen_layout={"displays": [{"name": "d"}]}, assumed_distance_cm=60.0, iris_px=20.0,
                             image_width=1280, glasses="no")
    assert "distance" not in {i.field for i in validate(block)}
    assert "distance" in {i.field for i in validate({**block, "backfilled": True})}
