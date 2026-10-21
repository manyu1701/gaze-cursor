"""L2 wiring in scripts/cursor_mode.py: every state is archived with a parent chain, nothing is lost on replace."""

import importlib.util
from pathlib import Path

import numpy as np

from gaze_cursor import model_history, resume
from gaze_cursor.config import ScreenConfig
from gaze_cursor.cursor_session import CursorSession
from gaze_cursor.online import OnlineRidge

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("cursor_mode_script", ROOT / "scripts" / "cursor_mode.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_remember_archives_each_state_with_a_parent_chain_and_keeps_the_replaced_file(tmp_path, monkeypatch) -> None:
    gc = _load()
    monkeypatch.setattr(gc, "QUICK_DIR", tmp_path)
    cfg = ScreenConfig()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 4))
    y = np.column_stack([700 + 80 * x[:, 0], 450 + 60 * x[:, 1]])
    resume.save(tmp_path, OnlineRidge.from_calibration(x[10:], y[10:], alpha=1.0), x[10:], {"yaw": 0.0}, "q0.npz",
                cfg, 1.0)
    gc.preserve_previous(cfg)                                    # a file from before L2 (no state id)
    assert len(model_history.list_states(tmp_path)) == 1

    s = CursorSession(cfg, OnlineRidge.from_calibration(x, y, alpha=1.0), x, {"yaw": 0.0}, "q1.npz", learn=True)
    s.conditions = {"person_id": "p01", "git_hash": "abc"}
    gc.remember(s, cfg, "calibration")
    s.model.update(x[0], np.array([650.0, 400.0]))
    gc.remember(s, cfg, "autosave")
    gc.remember(s, cfg, "superseded")                            # same state again: stored once
    s.recalibrated(OnlineRidge.from_calibration(x[:30], y[:30], alpha=1.0), x[:30], {"yaw": 0.0}, "q2.npz")
    assert s.state_id is None
    gc.remember(s, cfg, "calibration")

    states = model_history.list_states(tmp_path)
    assert [m["reason"] for m in states] == ["found on disk at start", "calibration", "autosave", "calibration"]
    chain = model_history.lineage(tmp_path, states[2]["state_id"])
    assert [m["reason"] for m in chain] == ["calibration", "autosave"] and chain[1]["person_id"] == "p01"
    assert states[3]["parent_id"] is None and states[3]["cursor_log"] == f"cursor_{s.started}.npz"
    saved, _ = resume.load(tmp_path, cfg)
    assert saved is not None and saved.state_id == states[3]["state_id"]
