"""Condition tags for the recorders (L1 stages 3-4): camera parsing, tag resolution, Cursor-log fields."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from gaze_cursor.conditions import validate, verdict
from gaze_cursor.cursor_session import CursorSession
from gaze_cursor.online import OnlineRidge
from gaze_cursor.sinks import platform_info, tags
from gaze_cursor.sinks.cursor import InputEvent
from tests.test_cursor_session import CFG


def ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{"person": None, "lighting": None, "glasses": None, "posture_note": None, **kw})


def paths(tmp_path):
    return {"tags_path": tmp_path / "tags.json", "person_path": tmp_path / "person.json"}


def test_camera_profile_prefers_the_built_in_camera_over_an_iphone() -> None:
    profile = {"SPCameraDataType": [{"_name": "Someone's iPhone Camera"}, {"_name": "MacBook Air Camera"}]}
    info = platform_info.parse_camera_profile(profile)
    assert info["name"] == "MacBook Air Camera" and info["built_in"] is True and len(info["available"]) == 2
    ext = platform_info.parse_camera_profile({"SPCameraDataType": [{"_name": "Logitech C920"}]})
    assert ext["name"] == "Logitech C920" and ext["built_in"] is False
    none = platform_info.parse_camera_profile({})
    assert none["name"] == "unknown" and none["built_in"] is None


def test_display_record_converts_mm_to_cm() -> None:
    d = platform_info.display_record((0, 0, 1470, 956), (290.57, 188.97), "Built-in", True, True)
    assert d["size_cm"] == [29.1, 18.9] and d["main"] is True


def test_no_person_id_refuses_to_record(tmp_path) -> None:
    with pytest.raises(SystemExit, match="No person id"):
        tags.resolve(ns(), {}, interactive=False, **paths(tmp_path))


def test_flags_beat_environment_and_environment_beats_last_answers(tmp_path) -> None:
    p = paths(tmp_path)
    p["tags_path"].write_text(json.dumps({"lighting": "dark", "glasses": "yes", "posture_note": "old"}))
    env = {"GAZE_PERSON_ID": "p07", "GAZE_LIGHTING_TAG": "lamp", "GAZE_POSTURE_NOTE": "upright"}
    t = tags.resolve(ns(glasses="no"), env, interactive=False, **p)
    assert t == tags.Tags("p07", "lamp", "no", "upright")


def test_non_interactive_falls_back_to_last_answers_then_unknown(tmp_path) -> None:
    p = paths(tmp_path)
    t = tags.resolve(ns(person="p01"), {}, interactive=False, **p)
    assert t.lighting == "unknown" and t.glasses == "unknown"          # nothing known: validate will warn
    t2 = tags.resolve(ns(person="p01", lighting="day", glasses="no", posture_note="upright"), {},
                      interactive=False, **p)
    t3 = tags.resolve(ns(person="p01"), {}, interactive=False, **p)
    assert (t3.glasses, t3.posture_note) == (t2.glasses, t2.posture_note)   # remembered
    assert t3.lighting == "unknown"                                          # lighting is never reused silently


def test_interactive_prompts_validate_and_remember_the_person(tmp_path) -> None:
    p = paths(tmp_path)
    answers = iter(["bob", "p02", "sunny", "day", "no", "upright"])
    t = tags.resolve(ns(), {}, interactive=True, ask=lambda q: next(answers), **p)
    assert t == tags.Tags("p02", "day", "no", "upright")
    assert json.loads(p["person_path"].read_text())["person_id"] == "p02"


def test_a_recording_with_all_tags_passes_the_validator(monkeypatch) -> None:
    monkeypatch.setattr(platform_info, "camera_info",
                        lambda size=None: {"name": "MacBook Air Camera", "width": 1280, "height": 720})
    monkeypatch.setattr(platform_info, "screen_layout", lambda: {"displays": [{"name": "Built-in"}]})
    monkeypatch.setattr(platform_info, "git_hash", lambda repo=None: "abc1234")
    block = tags.conditions_for(tags.Tags("p01", "lamp", "no", "upright"), CFG, frame_size=(1280, 720), iris_px=24.0)
    assert verdict(validate(block)) == "pass"
    assert block["distance"]["est_cm"] > 0


def test_cursor_log_carries_conditions_model_version_and_update_counter(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(30, 4)), rng.normal(size=(30, 2)) * 100 + 500
    model = OnlineRidge.from_calibration(x, y, 1.0)
    s = CursorSession(CFG, model, x, None, "q.npz", learn=True, conditions={"person_id": "p01"})
    s.handle(InputEvent("toggle", 0), 0)
    d = np.load(s.save(tmp_path))
    meta = json.loads(str(d["meta"]))
    assert d["events"].shape[1] == len(CursorSession.EVENT_COLUMNS) and CursorSession.EVENT_COLUMNS[-1] == "n_updates"
    assert meta["conditions"]["person_id"] == "p01" and meta["conditions"]["model_version"] == meta["model_versions"][0]
    assert len(meta["model_versions"][0]) == 16
