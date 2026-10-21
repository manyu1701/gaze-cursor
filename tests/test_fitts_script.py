"""Drive scripts/fitts_task.py end to end with stand-ins for the camera, tracker, canvas and cursor."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.sinks import tags as condition_tags
from gaze_cursor.sinks.cursor import InputEvent
from tests.test_features import _face

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _tags_without_the_os(monkeypatch):
    monkeypatch.setattr(condition_tags, "resolve", lambda *a, **k: condition_tags.Tags("p01", "lamp", "no", "upright"))
    monkeypatch.setattr(condition_tags, "conditions_for", lambda tags, *a, **k: {"person_id": tags.person_id, "distance": {}})


def _load():
    spec = importlib.util.spec_from_file_location("fitts_task_script", ROOT / "scripts" / "fitts_task.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Model:
    def predict(self, x):
        return np.array([[700.0, 450.0]])


def test_the_task_runs_both_methods_logs_trials_moves_and_landings_and_saves_one_file(tmp_path, monkeypatch) -> None:
    gc = _load()
    t = {"ms": 0}
    frame_img = np.full((720, 1280, 3), 120, np.uint8)
    result = SimpleNamespace(points=_face(), transform=np.eye(4), blendshapes=np.zeros(52, np.float32))

    class Cam:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self):
            t["ms"] += 33
            return SimpleNamespace(image=frame_img, timestamp_ms=t["ms"])

    class Tracker:
        blendshape_names = tuple(f"b{i}" for i in range(52))
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def process(self, f): return result

    made: dict = {}

    class Task(gc.FittsTask):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made["task"] = self

    warps, polls = [], {"n": 0}

    class Cursor:
        def __init__(self, size, detector=None, track_moves=False):
            assert track_moves
        def pointer(self): return (10.0, 20.0)
        def poll(self):
            polls["n"] += 1
            task, now = made["task"], t["ms"]
            out = [InputEvent("move", now, (11.0, 21.0))]
            trial = task.current
            if trial is not None and polls["n"] % 6 == 0:
                if task.blocks[trial.block].method == "gaze_jump":
                    out.append(InputEvent("warp", now))
                cx, cy = trial.center
                out.append(InputEvent("click", now + 1, (cx + 5, cy), "left"))
            return out
        def warp(self, xy):
            warps.append(xy)
            return xy
        def close(self): pass

    class Canvas:
        def __init__(self, cfg):
            self.size, self.origin, self.safe_top, self.keep_pointer = (cfg.width_px, cfg.height_px), (0, 0), 40, False
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def blank(self): return np.zeros((10, 10, 3), np.uint8)
        def circle(self, *a, **k): pass
        def dot(self, *a, **k): pass
        def text(self, *a, **k): pass
        def show(self, img): return 32

    x = np.random.default_rng(0).normal(scale=0.05, size=(50, 4))
    feats = gc.cm.assemble(gc.cm.extract_features(result.points, result.transform, (1280, 720)), result.blendshapes,
                           Tracker.blendshape_names, "gaze")
    xs = np.vstack([x + feats, feats])
    fit = SimpleNamespace(model=Model(), x=xs, y=np.tile([700.0, 450.0], (len(xs), 1)),
                          posture=[{"dummy": 0.0}], path=Path("session_quick.npz"))
    rows = []
    monkeypatch.setattr(gc, "Camera", Cam)
    monkeypatch.setattr(gc, "FaceTracker", Tracker)
    monkeypatch.setattr(gc, "MacCursor", Cursor)
    monkeypatch.setattr(gc, "FullscreenCanvas", Canvas)
    monkeypatch.setattr(gc, "FittsTask", Task)
    monkeypatch.setattr(gc.cm, "calibrate", lambda cam, tracker, cfg: fit)
    monkeypatch.setattr(gc, "input_monitoring_granted", lambda: True)
    monkeypatch.setattr(gc, "QUICK_DIR", tmp_path / "quick")
    monkeypatch.setattr(gc.cm, "QUICK_DIR", tmp_path / "quick")
    monkeypatch.setattr(gc, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(gc, "append_rows", rows.extend)
    monkeypatch.setattr(gc.ScreenConfig, "from_system", classmethod(lambda cls: ScreenConfig()))
    center = gc.cm.posture_metrics(result.points, result.transform, (1280, 720))     # head never moves
    monkeypatch.setattr(gc.Reference, "from_current", classmethod(lambda cls, p: SimpleNamespace(center=center)))
    monkeypatch.setattr(gc.time, "monotonic_ns", lambda: t["ms"] * 1_000_000)

    assert gc.main(["--seats", "centre", "--widths", "140", "--no-dots", "--no-seat-gate"]) == 0
    (saved,) = [p for p in (tmp_path / "tasks").glob("fitts_*.npz") if not p.name.endswith(".faces.npz")]
    d = np.load(saved)
    meta = json.loads(str(d["meta"]))
    task = meta["task"]
    assert meta["kind"] == "fitts_task" and task["aborted"] is False and task["learning"] is False
    assert len(task["blocks"]) == 2 and {b["method"] for b in task["blocks"]} == {"trackpad", "gaze_jump"}
    assert len(task["trials"]) == 18 and all(tr["hit"] for tr in task["trials"])
    jump = [tr for tr in task["trials"] if task["blocks"][tr["block"]]["method"] == "gaze_jump"]
    assert all(tr["t_warp_ms"] is not None and tr["t_first_move_ms"] is not None for tr in jump)
    assert all(tr["t_warp_ms"] is None for tr in task["trials"] if tr not in jump)    # trackpad: the jump is off
    assert d["moves"].shape[1] == 4 and len(d["moves"]) > 18
    assert meta["conditions"]["person_id"] == "p01" and task["model_state_id"]
    assert len(warps) == 9 and np.allclose(warps[0], (700, 450), atol=1)
    assert rows[0]["mode"] == "fitts_task" and rows[0]["n_targets"] == 18
    assert (tmp_path / "tasks" / (saved.stem + ".faces.npz")).exists()
    assert not list((tmp_path / "tasks").glob(".*tmp*"))
