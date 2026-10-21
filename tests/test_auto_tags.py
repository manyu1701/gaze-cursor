"""Step 1 of automatic tagging: lighting from the camera's label, posture from the head pose, glasses remembered."""

import pytest

from gaze_cursor.conditions import validate, verdict
from gaze_cursor.config import ScreenConfig
from gaze_cursor.sinks import platform_info, tags

CFG = ScreenConfig()
POSTURE = {"yaw": 1.2, "pitch": -3.4, "roll": 0.5, "distance": 0.123, "x": 0.5, "y": 0.5}


@pytest.fixture(autouse=True)
def _no_os(monkeypatch):
    monkeypatch.setattr(platform_info, "camera_info",
                        lambda size=None: {"name": "MacBook Air Camera", "width": 1280, "height": 720})
    monkeypatch.setattr(platform_info, "screen_layout", lambda: {"displays": [{"name": "Built-in"}]})
    monkeypatch.setattr(platform_info, "git_hash", lambda repo=None: "abc1234")


def block(t: tags.Tags, label: str | None, posture=POSTURE) -> dict:
    light = None if label is None else {"label": label, "face": 90.0, "background": 200.0, "ratio": 0.45}
    return tags.conditions_for(t, CFG, frame_size=(1280, 720), iris_px=24.0, lighting_measured=light, posture=posture)


@pytest.mark.parametrize("label, tag", [("dim", "dim"), ("backlit", "backlit"), ("front-lit", "front-lit"),
                                        ("mixed", "mixed")])
def test_lighting_tag_comes_from_the_sensor_label_when_nobody_typed_one(label, tag) -> None:
    b = block(tags.Tags("p01", glasses="no"), label)
    assert (b["lighting_tag"], b["lighting_tag_source"]) == (tag, "sensor")
    assert b["lighting_measured"]["label"] == label          # the measurement stays next to the tag


def test_a_typed_lighting_tag_wins_and_both_are_kept() -> None:
    b = block(tags.Tags("p01", "lamp", "no"), "backlit")
    assert (b["lighting_tag"], b["lighting_tag_source"]) == ("lamp", "typed")
    assert b["lighting_measured"]["label"] == "backlit"      # a person-vs-sensor disagreement is recoverable


def test_no_measurement_and_no_typed_tag_stays_unknown() -> None:
    b = block(tags.Tags("p01", glasses="no"), None)
    assert b["lighting_tag"] == "unknown" and "lighting_tag" in {i.field for i in validate(b)}


def test_posture_note_is_the_measured_head_pose_unless_typed() -> None:
    auto = block(tags.Tags("p01", glasses="no"), "front-lit")
    assert auto["posture_note_source"] == "measured" and auto["posture_note"].startswith("measured: yaw +1")
    assert auto["posture_measured"]["pitch"] == -3.4
    typed = block(tags.Tags("p01", "day", "no", "leaning in"), "front-lit")
    assert (typed["posture_note"], typed["posture_note_source"]) == ("leaning in", "typed")


def test_a_fully_automatic_recording_passes_the_validator_with_glasses_remembered() -> None:
    assert verdict(validate(block(tags.Tags("p01", glasses="no"), "front-lit"))) == "pass"


def test_posture_center_is_a_median_and_ignores_nan() -> None:
    frames = [{"yaw": 1.0, "pitch": 0.0}, {"yaw": 3.0, "pitch": float("nan")}, {"yaw": 100.0, "pitch": 2.0}]
    c = tags.posture_center(frames)
    assert c is not None and c["yaw"] == 3.0 and c["pitch"] == 1.0 and "roll" not in c
    assert tags.posture_center([]) is None
