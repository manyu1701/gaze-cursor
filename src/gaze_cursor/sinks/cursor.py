"""Real cursor control and global input for Cursor mode (M3). macOS-specific; the core loop only sees
``CursorSink`` (warp + poll) and ``InputEvent``s, so a fake can stand in for tests.

- ``warp(xy)`` jumps the system cursor to a main-display point (top-left origin, the same space the
  gaze model predicts in) via ``CGWarpMouseCursorPosition``. No synthetic clicks are posted.
- A **listen-only** Quartz event tap reports the hotkeys and every mouse-down without blocking or
  changing any event. It needs **Input Monitoring** permission for the app running Python (Terminal).

Hotkeys (defaults): tap **Right Command** alone = warp to gaze (the author's choice after the first live
try; Right Option before); Cmd+Option+G = on / off;
Cmd+Option+R = recalibrate (9 quick dots); Cmd+Option+X = quit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

RIGHT_OPTION_KEYCODE, RIGHT_COMMAND_KEYCODE = 61, 54
KEY_G, KEY_X, KEY_R = 5, 7, 15  # ANSI virtual key codes (X, not Q: the tap is listen-only, so Terminal also sees the key and Cmd+Option+X is its "Quit and Keep Windows")
TAP_MAX_MS = 400              # a modifier press longer than this is a hold (e.g. Cmd-drag), not a tap
FLAG_OPTION = 1 << 19         # kCGEventFlagMaskAlternate
FLAG_CONTROL = 1 << 18        # kCGEventFlagMaskControl
FLAG_COMMAND = 1 << 20        # kCGEventFlagMaskCommand
# Tap-to-warp modifier: its key code and the flag that is set while it is down.
WARP_KEYS = {"right_command": (RIGHT_COMMAND_KEYCODE, FLAG_COMMAND), "right_option": (RIGHT_OPTION_KEYCODE, FLAG_OPTION)}
DEFAULT_WARP_KEY = "right_command"


@dataclass(frozen=True)
class InputEvent:
    kind: str                            # "warp", "toggle", "recalibrate", "quit", "click", "move", "landing"
    t_ms: float                          # monotonic milliseconds (the camera frames' clock)
    xy: tuple[float, float] | None = None  # click / pointer position in main-display points
    button: str | None = None            # "left" / "right" for clicks


class CursorSink(Protocol):
    def warp(self, xy: tuple[float, float]) -> tuple[float, float]: ...
    def poll(self) -> list[InputEvent]: ...


def clamp(xy: tuple[float, float], size: tuple[int, int]) -> tuple[float, float]:
    """Keep a point inside the display (0 .. size - 1)."""
    return (min(max(float(xy[0]), 0.0), size[0] - 1.0), min(max(float(xy[1]), 0.0), size[1] - 1.0))


class HotkeyDetector:
    """Turns raw key / modifier / mouse events into ``InputEvent``s. Pure: no OS calls.

    The warp key is a *tap* of a modifier on its own: pressed and released within ``TAP_MAX_MS`` with no
    other key or click in between (so Cmd+C, Cmd-click etc. keep working as usual)."""

    def __init__(self, warp_key: str = DEFAULT_WARP_KEY, tap_max_ms: float = TAP_MAX_MS) -> None:
        self.warp_keycode, self.warp_flag = WARP_KEYS[warp_key]
        self.tap_max_ms = tap_max_ms
        self._down_at: float | None = None

    def flags_changed(self, keycode: int, flags: int, t_ms: float) -> list[InputEvent]:
        if keycode != self.warp_keycode:
            self._down_at = None
            return []
        if flags & self.warp_flag and self._down_at is None:   # pressed
            self._down_at = t_ms
            return []
        start, self._down_at = self._down_at, None              # released
        if start is not None and t_ms - start <= self.tap_max_ms:
            return [InputEvent("warp", t_ms)]
        return []

    def key_down(self, keycode: int, flags: int, t_ms: float) -> list[InputEvent]:
        self._down_at = None                                   # the modifier was used with a key
        if flags & FLAG_COMMAND and flags & FLAG_OPTION:   # Cmd+Option+G / R / X (the author)
            if keycode == KEY_G:
                return [InputEvent("toggle", t_ms)]
            if keycode == KEY_X:
                return [InputEvent("quit", t_ms)]
            if keycode == KEY_R:
                return [InputEvent("recalibrate", t_ms)]
        return []

    def mouse_down(self, xy: tuple[float, float], button: str, t_ms: float) -> list[InputEvent]:
        self._down_at = None                                   # Cmd-click is not a warp tap
        return [InputEvent("click", t_ms, (float(xy[0]), float(xy[1])), button)]

    def mouse_moved(self, xy: tuple[float, float], t_ms: float) -> list[InputEvent]:
        """A pointer move (trackpad or drag). It does not cancel a warp tap: moving the pointer while the
        modifier is down is not using the modifier."""
        return [InputEvent("move", t_ms, (float(xy[0]), float(xy[1])))]


def event_time_ms(event_ns: int, now_ms: float) -> float:
    """The event's own timestamp in monotonic ms when it is plausible (on macOS both count from boot),
    else ``now_ms``: a callback can run up to a frame after the event, and click labels need the
    true click time."""
    t = event_ns / 1e6
    return t if 0 < now_ms - t < 2000 else now_ms


def input_monitoring_granted() -> bool:
    import Quartz
    return bool(Quartz.CGPreflightListenEventAccess())


INPUT_MONITORING_PANE = "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"


def request_input_monitoring() -> None:
    """Ask macOS to list Terminal under Input Monitoring, and open that settings page."""
    import subprocess

    import Quartz
    Quartz.CGRequestListenEventAccess()
    subprocess.run(["open", INPUT_MONITORING_PANE], check=False)


PERMISSION_HELP = (
    "Cursor mode needs Input Monitoring for the app running Python (Terminal). The settings page is open:\n"
    "  1. Switch Terminal ON under Input Monitoring (add it with + if it is not listed).\n"
    "  2. Quit Terminal completely (Cmd+Q) and reopen it: macOS only applies this permission after a restart.\n"
    "  3. Run Cursor.command again.\n"
    "  (It only listens for the hotkeys and your clicks; it never blocks or changes them.)")


class MacCursor:
    """``CursorSink`` on macOS: warps the cursor and collects hotkeys and clicks from a listen-only tap.
    Call ``poll()`` once per frame from the main thread: it pumps the run loop and returns new events."""

    def __init__(self, size: tuple[int, int], detector: HotkeyDetector | None = None,
                 track_moves: bool = False) -> None:
        """``track_moves`` also reports every pointer move (``move`` events, about 100 per second while the
        pointer moves) and each warp's measured landing (``landing``): for tasks that time the pointer path."""
        import Quartz

        self._q = Quartz
        self.size = size
        self.detector = detector or HotkeyDetector()
        self.track_moves = track_moves
        self._events: list[InputEvent] = []
        kinds = [Quartz.kCGEventFlagsChanged, Quartz.kCGEventKeyDown,
                 Quartz.kCGEventLeftMouseDown, Quartz.kCGEventRightMouseDown]
        if track_moves:
            kinds += [Quartz.kCGEventMouseMoved, Quartz.kCGEventLeftMouseDragged, Quartz.kCGEventRightMouseDragged]
        mask = 0
        for kind in kinds:
            mask |= Quartz.CGEventMaskBit(kind)
        self._tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                            Quartz.kCGEventTapOptionListenOnly, mask, self._callback, None)
        if self._tap is None:
            raise PermissionError(PERMISSION_HELP)
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(self._tap, True)

    def _callback(self, proxy, kind, event, refcon):
        q = self._q
        if kind in (q.kCGEventTapDisabledByTimeout, q.kCGEventTapDisabledByUserInput):
            q.CGEventTapEnable(self._tap, True)               # macOS disables slow taps; re-arm
            return event
        t = event_time_ms(q.CGEventGetTimestamp(event), time.monotonic_ns() / 1e6)
        flags = int(q.CGEventGetFlags(event))
        if kind == q.kCGEventFlagsChanged:
            code = int(q.CGEventGetIntegerValueField(event, q.kCGKeyboardEventKeycode))
            self._events += self.detector.flags_changed(code, flags, t)
        elif kind == q.kCGEventKeyDown:
            code = int(q.CGEventGetIntegerValueField(event, q.kCGKeyboardEventKeycode))
            self._events += self.detector.key_down(code, flags, t)
        elif kind in (q.kCGEventLeftMouseDown, q.kCGEventRightMouseDown):
            loc = q.CGEventGetLocation(event)
            button = "left" if kind == q.kCGEventLeftMouseDown else "right"
            self._events += self.detector.mouse_down((loc.x, loc.y), button, t)
        elif kind in (q.kCGEventMouseMoved, q.kCGEventLeftMouseDragged, q.kCGEventRightMouseDragged):
            loc = q.CGEventGetLocation(event)
            self._events += self.detector.mouse_moved((loc.x, loc.y), t)
        return event                                           # listen-only: never altered

    def poll(self) -> list[InputEvent]:
        self._q.CFRunLoopRunInMode(self._q.kCFRunLoopDefaultMode, 0, True)
        out, self._events = self._events, []
        return out

    def pointer(self) -> tuple[float, float]:
        """Where the system cursor is now (main-display points, top-left origin)."""
        loc = self._q.CGEventGetLocation(self._q.CGEventCreate(None))
        return float(loc.x), float(loc.y)

    def warp(self, xy: tuple[float, float]) -> tuple[float, float]:
        """Jump the cursor to ``xy`` (clamped to the display); returns the position the system reports
        right after (the landing). With ``track_moves`` it is also queued as a ``landing`` event."""
        x, y = clamp(xy, self.size)
        self._q.CGWarpMouseCursorPosition((x, y))
        self._q.CGAssociateMouseAndMouseCursorPosition(True)  # the trackpad moves it again at once
        landed = self.pointer()
        if self.track_moves:
            self._events.append(InputEvent("landing", time.monotonic_ns() / 1e6, landed))
        return landed

    def close(self) -> None:
        if self._tap is not None:
            self._q.CGEventTapEnable(self._tap, False)
            self._tap = None
