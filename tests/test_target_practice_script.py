"""Drive scripts/target_practice.py end to end with stand-ins for the camera, tracker, canvas and model."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gaze_cursor.config import ScreenConfig
from tests.test_features import _face

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("target_practice", ROOT / "scripts" / "target_practice.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        self.t += 1 / 30
        return self.t

    def strftime(self, fmt: str) -> str:
        return "2026-10-14T10:00:00"


def test_target_practice_runs_saves_and_logs(tmp_path, monkeypatch) -> None:
    tp = _load()
    screen = ScreenConfig()
    pts = _face()
    frame = SimpleNamespace(image=np.full((720, 1280, 3), 120, np.uint8), timestamp_ms=0)
    result = SimpleNamespace(points=pts, transform=np.eye(4), blendshapes=np.zeros(52, np.float32))
    holder = {}

    class Model:  # perfect gaze: the dot sits on the current box
        def predict(self, x):
            task = holder["task"]
            tr = task.current or task.trials[-1]
            return np.array([tr.center])

    class Canvas:
        def __init__(self, cfg):
            self.origin, self.size, self.keys = (0, 0), (1470, 956), [32]
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def blank(self): return np.zeros((956, 1470, 3), np.uint8)
        def text(self, *a, **k): pass
        def box(self, *a, **k): pass
        def marker(self, *a, **k): pass
        def show(self, img):
            if holder["task"].done and not self.keys:
                self.keys.append(32)              # close the summary screen
            return self.keys.pop(0) if self.keys else -1

    class Cam:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return frame

    class Tracker:
        blendshape_names = tuple(f"b{i}" for i in range(52))
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def process(self, f): return result

    real_task = tp.TargetTask
    def make_task(*a, **k):
        holder["task"] = real_task(*a, **k)
        return holder["task"]

    session = SimpleNamespace(meta={"blendshape_names": list(Tracker.blendshape_names)}, path=Path("session_x.npz"))
    live = SimpleNamespace(model=Model(), feature_set="gaze", sessions=[session], x_train=np.zeros((1, 4)), label="fake")
    logged = []
    monkeypatch.setattr(tp, "time", FakeClock())
    monkeypatch.setattr(tp, "TargetTask", make_task)
    monkeypatch.setattr(tp, "Camera", Cam)
    monkeypatch.setattr(tp, "FaceTracker", Tracker)
    monkeypatch.setattr(tp, "FullscreenCanvas", Canvas)
    monkeypatch.setattr(tp.ScreenConfig, "from_system", classmethod(lambda cls: screen))
    monkeypatch.setattr(tp, "load_session", lambda p: session)
    monkeypatch.setattr(tp, "latest_session", lambda: Path("session_x.npz"))
    monkeypatch.setattr(tp, "fit_live_model", lambda *a, **k: live)
    monkeypatch.setattr(tp, "append_rows", logged.extend)
    monkeypatch.setattr(tp, "TASKS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["target_practice.py", "--trials", "6", "--no-check", "--seed", "0"])

    assert tp.main() == 0
    task = holder["task"]
    assert len(task.trials) == 6 and all(tr.hit for tr in task.trials)
    assert task.trials[-1].half < task.trials[0].half              # it shrank
    saved = list(tmp_path.glob("session_*.npz"))
    assert len(saved) == 1
    from gaze_cursor.eval.session import load_session
    s = load_session(saved[0])
    assert s.meta["kind"] == "target_task" and len(s.meta["trials"]) == 6 and s.meta["summary"]["n_hits"] == 6
    assert set(np.unique(s.target_idx)) == set(range(6))
    assert logged and logged[0]["mode"] == "target_task" and "hits 6/6" in logged[0]["note"]
