"""Drive scripts/gaze_cursor.py end to end with stand-ins for the camera, tracker, overlay and cursor."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.sinks.cursor import InputEvent
from tests.test_features import _face

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("gaze_cursor_script", ROOT / "scripts" / "gaze_cursor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Model:
    def predict(self, x):
        return np.array([[700.0, 450.0]])


def test_cursor_mode_warps_on_the_hotkey_logs_clicks_and_quits(tmp_path, monkeypatch) -> None:
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

    script = {40: [InputEvent("warp", 0)], 45: [InputEvent("click", 0, (705.0, 452.0), "left")],
              50: [InputEvent("toggle", 0)], 52: [InputEvent("warp", 0)], 60: [InputEvent("quit", 0)]}
    warps, polls = [], {"n": 0}

    class Cursor:
        def __init__(self, size): self.size = size
        def poll(self):
            polls["n"] += 1
            out = script.get(polls["n"], [])
            return [InputEvent(e.kind, t["ms"], e.xy, e.button) for e in out]
        def warp(self, xy): warps.append(xy)
        def close(self): pass

    class Overlay:
        def __init__(self, cfg): self.safe_top = 40
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def blank(self): return np.zeros((956, 1470, 4), np.uint8)
        def show(self, img): pass
        def pump(self): pass
        def restore_policy(self): pass

    x = np.random.default_rng(0).normal(scale=0.05, size=(50, 4))
    feats = gc.assemble(gc.extract_features(result.points, result.transform, (1280, 720)), result.blendshapes,
                        Tracker.blendshape_names, "gaze")
    fit = SimpleNamespace(model=Model(), x=np.vstack([x + feats, feats]), posture=[{"dummy": 0.0}],
                          path=Path("session_quick.npz"))
    rows = []
    monkeypatch.setattr(gc, "Camera", Cam)
    monkeypatch.setattr(gc, "FaceTracker", Tracker)
    monkeypatch.setattr(gc, "MacCursor", Cursor)
    monkeypatch.setattr(gc, "ClickThroughOverlay", Overlay)
    monkeypatch.setattr(gc, "calibrate", lambda cam, tracker, cfg: fit)
    monkeypatch.setattr(gc, "input_monitoring_granted", lambda: True)
    monkeypatch.setattr(gc, "QUICK_DIR", tmp_path)
    monkeypatch.setattr(gc, "append_rows", rows.extend)
    monkeypatch.setattr(gc.ScreenConfig, "from_system", classmethod(lambda cls: ScreenConfig()))
    center = gc.posture_metrics(result.points, result.transform, (1280, 720))     # head never moves
    monkeypatch.setattr(gc.Reference, "from_current", classmethod(lambda cls, p: SimpleNamespace(center=center)))
    monkeypatch.setattr(gc.time, "monotonic_ns", lambda: t["ms"] * 1_000_000)
    monkeypatch.setattr(sys, "argv", ["gaze_cursor.py"])

    assert gc.main() == 0
    assert len(warps) == 1 and np.allclose(warps[0], (700, 450), atol=1)   # second warp: toggled off
    (saved,) = tmp_path.glob("cursor_*.npz")
    d = np.load(saved)
    meta = json.loads(str(d["meta"]))
    assert len(d["frames"]) == 60 and meta["summary"]["clicks"] == 1 and meta["summary"]["warps"] == 1
    assert rows[0]["mode"] == "cursor_session" and rows[0]["n_targets"] == 1
    assert rows[0]["err_deg_median"] < 0.2                                  # click 5 pt from the gaze


def test_cursor_mode_explains_the_permission_when_missing(monkeypatch, capsys) -> None:
    gc = _load()
    monkeypatch.setattr(gc, "input_monitoring_granted", lambda: False)
    monkeypatch.setattr(gc, "request_input_monitoring", lambda: None)
    monkeypatch.setattr(sys, "argv", ["gaze_cursor.py"])
    assert gc.main() == 1 and "Input Monitoring" in capsys.readouterr().out
