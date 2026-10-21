"""L2: every learned model state is archived once, immutable, with provenance and a walkable parent chain."""

import json

import numpy as np

from gaze_cursor import model_history as mh
from gaze_cursor import resume
from gaze_cursor.config import ScreenConfig
from gaze_cursor.online import OnlineRidge

CFG = ScreenConfig()
RNG = np.random.default_rng(1)


def prior() -> tuple[OnlineRidge, np.ndarray]:
    x = RNG.normal(size=(40, 4))
    y = np.column_stack([700 + 80 * x[:, 0], 450 + 60 * x[:, 1]])
    return OnlineRidge.from_calibration(x, y, alpha=1.0), x


def put(root, model, x, reason, parent=None, **kw) -> str:
    return mh.archive(root, model, x, {"yaw": 1.0}, "session_q.npz", CFG, 1000.0, reason, parent_id=parent, **kw)


def test_archive_roundtrip_is_exact_and_private(tmp_path) -> None:
    model, x = prior()
    model.update(x[0], np.array([650.0, 400.0]))
    sid = put(tmp_path, model, x, "autosave", cursor_log="cursor_1.npz", person_id="p01", code="abc123")
    (path,) = (tmp_path / mh.DIR_NAME).glob("state_*.npz")
    assert sid.endswith("-1") and path.stat().st_mode & 0o777 == 0o600
    s = mh.load(path)
    assert np.array_equal(s.model.theta, model.theta) and np.array_equal(s.model.P, model.P)
    assert s.model.n_updates == 1 and s.ref_center == {"yaw": 1.0} and np.array_equal(s.x_train, x)
    assert (s.meta["reason"], s.meta["person_id"], s.meta["code"], s.meta["cursor_log"]) == (
        "autosave", "p01", "abc123", "cursor_1.npz")
    assert mh.verify(path)


def test_same_state_is_stored_once_and_never_overwritten(tmp_path) -> None:
    model, x = prior()
    a = put(tmp_path, model, x, "calibration")
    b = put(tmp_path, model, x, "quit", parent="other")
    assert a == b
    (path,) = (tmp_path / mh.DIR_NAME).glob("state_*.npz")
    assert mh.load(path).meta["reason"] == "calibration"          # the first writer's provenance stays


def test_a_learned_state_is_a_new_state(tmp_path) -> None:
    model, x = prior()
    first = put(tmp_path, model, x, "calibration")
    model.update(x[0], np.array([650.0, 400.0]))
    assert put(tmp_path, model, x, "autosave", parent=first) != first
    assert len(mh.list_states(tmp_path)) == 2


def test_lineage_walks_parents_and_survives_a_replaced_resume_file(tmp_path) -> None:
    model, x = prior()
    ids = [put(tmp_path, model, x, "calibration")]
    for k in range(3):
        model.update(x[k], np.array([650.0 + k, 400.0]))
        ids.append(put(tmp_path, model, x, "autosave", parent=ids[-1]))
    resume.save(tmp_path, model, x, {"yaw": 1.0}, "q.npz", CFG, 1000.0, ids[-1])
    fresh, xf = prior()                                              # "--fresh" / Cmd+Option+R: a new chain
    new = put(tmp_path, fresh, xf, "calibration")
    resume.save(tmp_path, fresh, xf, {"yaw": 1.0}, "q2.npz", CFG, 2000.0, new)
    assert [m["state_id"] for m in mh.lineage(tmp_path, ids[-1])] == ids      # the old chain is still there
    assert [m["state_id"] for m in mh.lineage(tmp_path, new)] == [new]
    assert len(mh.list_states(tmp_path)) == 5


def test_verify_detects_a_damaged_file_and_list_flags_unreadable(tmp_path) -> None:
    model, x = prior()
    put(tmp_path, model, x, "calibration")
    d = tmp_path / mh.DIR_NAME
    (path,) = d.glob("state_*.npz")
    s = mh.load(path)
    np.savez_compressed(path, mean=s.model.mean, scale=s.model.scale, theta=s.model.theta + 1e-6, P=s.model.P,
                        x_train=s.x_train, meta=json.dumps(s.meta))
    assert not mh.verify(path)
    (d / "state_bad_x-0.npz").write_bytes(b"garbage")
    assert any("error" in m for m in mh.list_states(tmp_path))


def test_resume_file_remembers_the_state_id_and_old_files_still_load(tmp_path) -> None:
    model, x = prior()
    resume.save(tmp_path, model, x, {"yaw": 1.0}, "q.npz", CFG, 1000.0, "abc-0")
    saved, _ = resume.load(tmp_path, CFG, now=1000.0 + 60)
    assert saved is not None and saved.state_id == "abc-0"
    resume.save(tmp_path, model, x, {"yaw": 1.0}, "q.npz", CFG, 1000.0)       # no id: as before L2
    loaded, _ = resume.load(tmp_path, CFG, now=1060.0)
    assert loaded is not None and loaded.state_id is None
    stale, screen = resume.read_unchecked(tmp_path)                            # no age / screen check
    assert stale is not None and screen == [CFG.width_px, CFG.height_px]
    assert resume.read_unchecked(tmp_path / "nowhere") == (None, None)
