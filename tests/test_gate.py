import pytest

from gaze_cursor.posture import (
    BACKLIT_RATIO, LIGHT_GOOD_FACE, LIGHT_MIN_FACE, build_checklist, checklist_ready)
from tests.test_posture import STILL, _m, _ref


def _light(face: float, background: float = 140.0, clipped: float = 0.0) -> dict[str, float]:
    return {"face": face, "background": background, "ratio": face / background, "clipped": clipped}


def _item(face: float, background: float = 140.0, **kw):
    items = build_checklist(_ref(), _m(), STILL, _light(face, background, **kw))
    return [i for i in items if i.key == "light"][0], items


def test_gate_constants_are_the_preregistered_values() -> None:
    assert (LIGHT_MIN_FACE, LIGHT_GOOD_FACE, BACKLIT_RATIO) == (60.0, 100.0, 0.6)


@pytest.mark.parametrize("face,level", [(59.0, "off"), (60.0, "good"), (99.0, "good"), (100.0, "good")])
def test_gate_levels_around_the_thresholds(face: float, level: str) -> None:
    item, items = _item(face)
    assert item.level == level
    assert checklist_ready(items) == (level == "good")           # below the gate SPACE stays locked


def test_below_the_gate_gives_front_light_guidance() -> None:
    item, _ = _item(37.0, 75.0)                                  # the dim session
    assert item.level == "off" and "lamp or window in front of you" in item.hint
    assert "aimed at your face" in item.hint and "more front light" in item.tip


def test_tip_is_non_blocking_between_the_gate_and_the_recommended_level() -> None:
    item, items = _item(78.0, 146.0)                             # the backlit session
    assert item.level == "good" and checklist_ready(items)       # passes the gate
    assert "more front light" in item.tip and "aim for 100+" in item.tip
    assert "bright light behind you" in item.tip                 # ratio 0.53 < 0.6


def test_no_tip_when_well_lit_and_not_backlit() -> None:
    item, _ = _item(124.0, 79.0)                                 # the front-lit session
    assert item.level == "good" and item.tip == ""


def test_backlight_tip_depends_on_the_ratio_but_the_gate_never_does() -> None:
    item, items = _item(110.0, 300.0)                            # bright face, much brighter background
    assert item.level == "good" and checklist_ready(items)
    assert "bright light behind you" in item.tip and "more front light" not in item.tip


def test_overexposure_still_blocks() -> None:
    assert _item(230.0)[0].level == "off" and _item(120.0, clipped=0.1)[0].level == "off"


def test_the_live_value_for_the_brightness_bar_is_exposed() -> None:
    assert _item(88.0)[0].value == 88.0
    assert [i for i in build_checklist(_ref(), None, [], None) if i.key == "light"][0].value is None
