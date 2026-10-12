"""Borderless fullscreen canvas for calibration targets and the gaze dot (debug sink).

A plain borderless NSWindow covers the whole main display (including the menu-bar/notch strip),
so drawing is exactly 1:1 with macOS screen points. OpenCV's fullscreen window cannot be used:
on notch MacBooks it sits below the menu bar and rescales the image (measured: y=33, 923 pt
tall, with a further title strip). All positions given to the drawing helpers are screen points
(origin top-left of the main display); frames are built with OpenCV and shown as layer contents.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from gaze_cursor.config import ScreenConfig


def own_window_bounds() -> tuple[int, int, int, int] | None:
    """(x, y, w, h) in screen points of this process's largest on-screen window, or None."""
    import Quartz

    best = None
    for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID):
        if w.get("kCGWindowOwnerPID") != os.getpid():
            continue
        b = w["kCGWindowBounds"]
        area = b["Width"] * b["Height"]
        if best is None or area > best[0]:
            best = (area, (int(b["X"]), int(b["Y"]), int(b["Width"]), int(b["Height"])))
    return best[1] if best else None


def _make_window_class():
    import AppKit
    import objc

    class _CanvasWindow(AppKit.NSWindow):
        keys: list[int] = []

        def canBecomeKeyWindow(self) -> bool:
            return True

        def constrainFrameRect_toScreen_(self, frame, screen):  # allow covering the notch strip
            return frame

        def keyDown_(self, event) -> None:
            chars = event.characters()
            if chars:
                _CanvasWindow.keys.append(ord(chars[0]))

    return _CanvasWindow


class FullscreenCanvas:
    """Borderless fullscreen window drawn 1:1 in screen points. ``origin`` is its top-left."""

    def __init__(self, cfg: ScreenConfig, name: str = "gaze-cursor") -> None:
        self.cfg = cfg
        self.name = name
        self.origin = (0, 0)
        self.size = (cfg.width_px, cfg.height_px)  # (w, h) of the drawable area
        self._window = None
        self._app = None

    def __enter__(self) -> FullscreenCanvas:
        import AppKit
        import Quartz

        self._app = AppKit.NSApplication.sharedApplication()
        self._app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
        screen = AppKit.NSScreen.mainScreen().frame()
        window_cls = _make_window_class()
        window_cls.keys.clear()
        win = window_cls.alloc().initWithContentRect_styleMask_backing_defer_(
            screen, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
        win.setLevel_(Quartz.CGShieldingWindowLevel())
        win.setOpaque_(True)
        win.setBackgroundColor_(AppKit.NSColor.blackColor())
        win.setHasShadow_(False)
        win.contentView().setWantsLayer_(True)
        win.makeKeyAndOrderFront_(None)
        self._app.activateIgnoringOtherApps_(True)
        AppKit.NSCursor.hide()
        self._window = win
        for _ in range(10):  # let the window appear, then read its real bounds
            self.show(self.blank())
        bounds = own_window_bounds()
        if bounds:
            self.origin = (bounds[0], bounds[1])
            if (bounds[2], bounds[3]) != self.size:
                print(f"warning: window is {bounds[2]}x{bounds[3]}, expected {self.size}")
        else:
            print("warning: could not read the window bounds; assuming the full screen")
        return self

    def __exit__(self, *exc: object) -> None:
        import AppKit

        AppKit.NSCursor.unhide()
        if self._window is not None:
            self._window.orderOut_(None)
            self._window.close()
            self._window = None
        self._pump()

    def blank(self) -> np.ndarray:
        return np.full((self.size[1], self.size[0], 3), 24, np.uint8)

    def _px(self, xy: tuple[float, float]) -> tuple[int, int]:
        return int(round(xy[0] - self.origin[0])), int(round(xy[1] - self.origin[1]))

    def visible(self, xy: tuple[float, float]) -> bool:
        x, y = self._px(xy)
        return 0 <= x < self.size[0] and 0 <= y < self.size[1]

    def dot(self, img: np.ndarray, xy: tuple[float, float], color=(255, 255, 255), r: int = 16) -> None:
        """Draw a target ring at screen position ``xy``."""
        c = self._px(xy)
        cv2.circle(img, c, r, color, 2, cv2.LINE_AA)
        cv2.circle(img, c, 3, (0, 0, 255), -1, cv2.LINE_AA)

    def text(self, img: np.ndarray, msg: str, y: int | None = None) -> None:
        """Centered text; ``y`` is in canvas pixels from the top of the window."""
        y = self.size[1] // 2 if y is None else y
        size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.putText(img, msg, ((self.size[0] - size[0]) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2, cv2.LINE_AA)

    def _pump(self) -> None:
        import AppKit

        while True:
            event = self._app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                AppKit.NSEventMaskAny, None, AppKit.NSDefaultRunLoopMode, True)
            if event is None:
                return
            self._app.sendEvent_(event)

    def show(self, img: np.ndarray) -> int:
        """Display ``img`` (BGR, canvas-sized) and return the key pressed (-1 if none)."""
        import Quartz

        rgba = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
        h, w = rgba.shape[:2]
        provider = Quartz.CGDataProviderCreateWithData(None, rgba.tobytes(), rgba.nbytes, None)
        image = Quartz.CGImageCreate(
            w, h, 8, 32, w * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
            Quartz.kCGImageAlphaPremultipliedLast, provider, None, False,
            Quartz.kCGRenderingIntentDefault)
        self._window.contentView().layer().setContents_(image)
        self._pump()
        keys = type(self._window).keys
        return keys.pop(0) if keys else -1
