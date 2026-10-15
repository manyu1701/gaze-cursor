"""Drive scripts/cursor_mode.py end to end with stand-ins for the camera, tracker, overlay and cursor."""

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
    spec = importlib.util.spec_from_file_location("cursor_mode_script", ROOT / "scripts" / "cursor_mode.py")
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
        def __init__(self, size, detector=None): self.size = size
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
    xs = np.vstack([x + feats, feats])
    fit = SimpleNamespace(model=Model(), x=xs, y=np.tile([700.0, 450.0], (len(xs), 1)),
                          posture=[{"dummy": 0.0}], path=Path("session_quick.npz"))
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
    monkeypatch.setattr(sys, "argv", ["cursor_mode.py"])

    assert gc.main() == 0
    assert len(warps) == 1 and np.allclose(warps[0], (700, 450), atol=1)   # second warp: toggled off
    (saved,) = tmp_path.glob("cursor_*.npz")
    d = np.load(saved)
    meta = json.loads(str(d["meta"]))
    assert len(d["frames"]) == 60 and meta["summary"]["clicks"] == 1 and meta["summary"]["warps"] == 1
    assert rows[0]["mode"] == "cursor_session" and rows[0]["n_targets"] == 1
    assert rows[0]["err_deg_median"] < 0.2                                  # click 5 pt from the gaze
    assert meta["summary"]["clicks_learned"] == 1 and meta["learning"]["on"] is True   # learning is the default
    assert (tmp_path / "last_calibration.npz").exists() and not list(tmp_path.glob(".*tmp*"))

    # The next start reuses the calibration (and what was learned): no dots, and the quit still saves.
    def no_calibration(*a, **k): raise AssertionError("must not calibrate again")
    monkeypatch.setattr(gc, "calibrate", no_calibration)
    polls["n"] = 0
    rows.clear()
    assert gc.main() == 0
    meta2 = json.loads(str(np.load(sorted(tmp_path.glob("cursor_*.npz"))[-1])["meta"]))
    assert meta2["learning"]["resumed"] and meta2["summary"]["frames"] > 0


def test_cursor_mode_explains_the_permission_when_missing(monkeypatch, capsys) -> None:
    gc = _load()
    monkeypatch.setattr(gc, "input_monitoring_granted", lambda: False)
    monkeypatch.setattr(gc, "request_input_monitoring", lambda: None)
    monkeypatch.setattr(sys, "argv", ["cursor_mode.py"])
    assert gc.main() == 1 and "Input Monitoring" in capsys.readouterr().out


def test_demo_mode_writes_nothing_and_never_touches_the_saved_calibration(tmp_path, monkeypatch, capsys) -> None:
    gc = _load()
    calls = []
    fit = SimpleNamespace(model=Model(), x=np.zeros((20, 4)), y=np.zeros((20, 2)), posture=[{"d": 0.0}],
                          path=Path("demo (not saved)"))

    class Cam:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return None

    class Tracker:
        blendshape_names = ()
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class Cursor:
        def __init__(self, size, detector=None): pass
        def poll(self): return [InputEvent("quit", 0)]
        def close(self): pass

    class Overlay:
        def __init__(self, cfg): self.safe_top = 40
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def blank(self): return np.zeros((956, 1470, 4), np.uint8)
        def show(self, img): pass
        def pump(self): pass
        def restore_policy(self): pass

    def calibrate(cam, tracker, cfg, record=True):
        calls.append(record)
        return fit

    rows = []
    monkeypatch.setattr(gc, "Camera", Cam)
    monkeypatch.setattr(gc, "FaceTracker", Tracker)
    monkeypatch.setattr(gc, "MacCursor", Cursor)
    monkeypatch.setattr(gc, "ClickThroughOverlay", Overlay)
    monkeypatch.setattr(gc, "calibrate", calibrate)
    monkeypatch.setattr(gc, "input_monitoring_granted", lambda: True)
    monkeypatch.setattr(gc, "QUICK_DIR", tmp_path)
    monkeypatch.setattr(gc, "append_rows", rows.extend)
    monkeypatch.setattr(gc.ScreenConfig, "from_system", classmethod(lambda cls: ScreenConfig()))
    monkeypatch.setattr(gc.Reference, "from_current", classmethod(lambda cls, p: SimpleNamespace(center={"d": 0.0})))
    monkeypatch.setattr(sys, "argv", ["cursor_mode.py", "--demo"])
    # Even with a saved calibration from yesterday, a demo asks the friend for their own dots.
    gc.resume.save(tmp_path, gc.OnlineRidge.from_calibration(fit.x + np.arange(80).reshape(20, 4), fit.y, 1.0),
                   fit.x, {"d": 0.0}, "mine.npz", ScreenConfig(), calibrated_at=gc.time.time())
    before = sorted(p.name for p in tmp_path.iterdir())
    assert gc.main() == 0
    assert calls == [False]                                              # calibrated, with recording off
    assert sorted(p.name for p in tmp_path.iterdir()) == before and rows == []   # nothing new written
    assert "nothing was saved" in capsys.readouterr().out


def test_cursor_mode_quits_by_itself_when_no_face_is_seen_for_five_minutes(tmp_path, monkeypatch) -> None:
    gc = _load()
    clock = {"s": 1000.0}
    processed = {"n": 0}

    class Cam:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self):
            clock["s"] += 1.0                       # one frame per simulated second keeps the test short
            return SimpleNamespace(image=np.zeros((720, 1280, 3), np.uint8), timestamp_ms=int(clock["s"] * 1000))

    class Tracker:
        blendshape_names = tuple(f"b{i}" for i in range(52))
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def process(self, f):
            processed["n"] += 1
            return None

    class Cursor:
        def __init__(self, size, detector=None): pass
        def poll(self): return []
        def close(self): pass

    class Overlay:
        def __init__(self, cfg): self.safe_top = 40
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def blank(self): return np.zeros((956, 1470, 4), np.uint8)
        def show(self, img): pass
        def pump(self): pass
        def restore_policy(self): pass

    x = np.random.default_rng(0).normal(size=(60, 4))
    fit = SimpleNamespace(model=Model(), x=x, y=np.tile([700.0, 450.0], (60, 1)), posture=[{"dummy": 0.0}],
                          path=Path("session_quick.npz"))
    notes = []
    monkeypatch.setattr(gc, "Camera", Cam)
    monkeypatch.setattr(gc, "FaceTracker", Tracker)
    monkeypatch.setattr(gc, "MacCursor", Cursor)
    monkeypatch.setattr(gc, "ClickThroughOverlay", Overlay)
    monkeypatch.setattr(gc, "calibrate", lambda cam, tracker, cfg: fit)
    monkeypatch.setattr(gc, "input_monitoring_granted", lambda: True)
    monkeypatch.setattr(gc, "QUICK_DIR", tmp_path)
    monkeypatch.setattr(gc, "append_rows", lambda rows: None)
    monkeypatch.setattr(gc, "notify", notes.append)
    chips = []
    monkeypatch.setattr(gc, "draw_status", lambda img, lines, top: chips.append((clock["s"] - 1000.0, lines)))
    monkeypatch.setattr(gc.ScreenConfig, "from_system", classmethod(lambda cls: ScreenConfig()))
    monkeypatch.setattr(gc.Reference, "from_current", classmethod(lambda cls, p: SimpleNamespace(center={})))
    monkeypatch.setattr(gc.time, "monotonic", lambda: clock["s"])
    monkeypatch.setattr(sys, "argv", ["cursor_mode.py", "--fresh"])

    assert gc.main() == 0
    assert any("battery" in n for n in notes)
    texts = [(t, " | ".join(s for s, _ in lines)) for t, lines in chips if lines]
    assert any("low power" in s for t, s in texts if 30 <= t < 240)          # asleep: said once
    assert any("closing in" in s for t, s in texts if 240 <= t)              # last minute: a countdown
    assert not any("closing in" in s for t, s in texts if t < 239)
    assert 300 <= clock["s"] - 1000.0 < 302                 # stopped right after the 5 minutes
    assert processed["n"] < 60 + 270 // 10 + 5              # awake 30 s at full rate, then 1 frame in 10
    assert list(tmp_path.glob("cursor_*.npz"))              # and the session was saved
