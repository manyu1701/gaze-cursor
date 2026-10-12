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


def test_note_mismatch_flags_a_wrong_condition_word() -> None:
    from gaze_cursor.posture import note_mismatch

    assert note_mismatch("dim D1, screen only", "front-lit") == "note says dim but the lighting is front-lit"
    assert note_mismatch("Front-lit, lamp close", "dim") == "note says front-lit but the lighting is dim"
    assert note_mismatch("backlit ambient", "mixed") == "note says backlit but the lighting is mixed"
    assert note_mismatch("dim D1", "dim") is None and note_mismatch("front lit", "front-lit") is None
    assert note_mismatch("evening, lamp", "dim") is None and note_mismatch("", "dim") is None   # no condition word


def test_checklist_warns_live_when_the_note_disagrees_with_the_lighting() -> None:
    items = build_checklist(_ref(), _m(), STILL, _light(120.0, 74.0), None, "dim D1, screen only")
    light = [i for i in items if i.key == "light"][0]
    assert "(front-lit)" in light.detail and "CHECK YOUR NOTE: note says dim but the lighting is front-lit" in light.tip
    assert light.level == "good"                                   # a warning, never a block
    ok = [i for i in build_checklist(_ref(), _m(), STILL, _light(120.0, 74.0), None, "front-lit") if i.key == "light"][0]
    assert "CHECK YOUR NOTE" not in ok.tip


def test_summary_screen_with_notes_keeps_the_notch_clear() -> None:
    from gaze_cursor.config import ScreenConfig
    from gaze_cursor.sinks.guide_view import draw_summary
    from gaze_cursor.sinks.overlay import FullscreenCanvas

    cfg = ScreenConfig()
    canvas = FullscreenCanvas(cfg)
    img = draw_summary(canvas, None, "s.npz", {"label": "mixed", "face": 93.0, "background": 148.0},
                       ["Round-2 class by captured lighting: none, outside every class range (would be excluded)",
                        "CHECK YOUR NOTE: note says dim but the lighting is mixed"])
    strip = int(cfg.top_inset_pt) + 2
    assert (img[:strip] == canvas.blank()[:strip]).all()
