"""Hotkey logic, clamping and event timing of the Cursor-mode input layer (no OS calls), and the
click-through overlay window."""

import os

import pytest

from gaze_cursor.sinks.cursor import (
    FLAG_CONTROL, FLAG_OPTION, KEY_G, KEY_Q, RIGHT_OPTION_KEYCODE, HotkeyDetector, clamp, event_time_ms)


def test_a_quick_tap_of_right_option_warps() -> None:
    d = HotkeyDetector()
    assert d.flags_changed(RIGHT_OPTION_KEYCODE, FLAG_OPTION, 1000) == []
    out = d.flags_changed(RIGHT_OPTION_KEYCODE, 0, 1150)
    assert [e.kind for e in out] == ["warp"] and out[0].t_ms == 1150


def test_a_long_hold_option_with_a_key_or_option_click_do_not_warp() -> None:
    d = HotkeyDetector()
    d.flags_changed(RIGHT_OPTION_KEYCODE, FLAG_OPTION, 0)
    assert d.flags_changed(RIGHT_OPTION_KEYCODE, 0, 900) == []          # held, not tapped
    d.flags_changed(RIGHT_OPTION_KEYCODE, FLAG_OPTION, 1000)
    d.key_down(0, FLAG_OPTION, 1050)                                    # Option+A types a character
    assert d.flags_changed(RIGHT_OPTION_KEYCODE, 0, 1100) == []
    d.flags_changed(RIGHT_OPTION_KEYCODE, FLAG_OPTION, 2000)
    d.mouse_down((5, 5), "left", 2050)                                  # Option-click
    assert d.flags_changed(RIGHT_OPTION_KEYCODE, 0, 2100) == []


def test_other_modifiers_do_not_warp() -> None:
    d = HotkeyDetector()
    d.flags_changed(58, FLAG_OPTION, 0)                                 # left Option
    assert d.flags_changed(58, 0, 100) == []


def test_toggle_quit_and_clicks() -> None:
    d = HotkeyDetector()
    both = FLAG_CONTROL | FLAG_OPTION
    assert [e.kind for e in d.key_down(KEY_G, both, 1)] == ["toggle"]
    assert [e.kind for e in d.key_down(KEY_Q, both, 2)] == ["quit"]
    assert d.key_down(KEY_Q, FLAG_CONTROL, 3) == []                     # Ctrl+Q alone is not ours
    (c,) = d.mouse_down((10.5, 20.0), "left", 4)
    assert (c.kind, c.xy, c.button, c.t_ms) == ("click", (10.5, 20.0), "left", 4)


def test_clamp_keeps_the_cursor_on_the_display() -> None:
    assert clamp((-5, 2000), (1470, 956)) == (0.0, 955.0)
    assert clamp((100.4, 200.6), (1470, 956)) == (100.4, 200.6)


def test_event_time_uses_the_event_stamp_only_when_plausible() -> None:
    assert event_time_ms(9_950_000_000, 10_000.0) == 9_950.0            # 50 ms before the callback
    assert event_time_ms(0, 10_000.0) == 10_000.0                       # missing stamp
    assert event_time_ms(20_000_000_000, 10_000.0) == 10_000.0          # in the future: other clock


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="opens a (transparent) window on a real display")
def test_the_cursor_overlay_never_takes_the_mouse_or_focus() -> None:
    from gaze_cursor.config import ScreenConfig
    from gaze_cursor.sinks.overlay import ClickThroughOverlay

    with ClickThroughOverlay(ScreenConfig()) as o:
        o.show(o.blank())
        assert o._window.ignoresMouseEvents() and not o._window.isOpaque() and not o._window.isKeyWindow()
