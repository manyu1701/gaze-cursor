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


_WINDOW_CLASS = None


def _make_window_class():
    """The NSWindow subclass, created once per process: Objective-C refuses to define a class name
    twice, so a second canvas in the same run (e.g. the light check, then the calibration) reuses it."""
    global _WINDOW_CLASS
    if _WINDOW_CLASS is not None:
        return _WINDOW_CLASS
    import AppKit

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

    _WINDOW_CLASS = _CanvasWindow
    return _CanvasWindow


class FullscreenCanvas:
    """Borderless fullscreen window drawn 1:1 in screen points. ``origin`` is its top-left."""

    def __init__(self, cfg: ScreenConfig, name: str = "gaze-cursor", alpha: float = 1.0) -> None:
        self.cfg = cfg
        self.name = name
        self.alpha = alpha  # < 1 only for tests/diagnostics
        self.origin = (0, 0)
        self.size = (cfg.width_px, cfg.height_px)  # (w, h) of the drawable area
        self._window = None
        self._app = None
        self._cursor_hidden: bool | None = None

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
        win.setAlphaValue_(self.alpha)
        win.contentView().setWantsLayer_(True)
        win.makeKeyAndOrderFront_(None)
        self._window = win
        self._activate()
        for _ in range(10):  # let the window appear, then read its real bounds
            self.show(self.blank())
        self._cursor_hidden = None
        self._sync_cursor()
        bounds = own_window_bounds()
        if bounds:
            self.origin = (bounds[0], bounds[1])
            if (bounds[2], bounds[3]) != self.size:
                print(f"warning: window is {bounds[2]}x{bounds[3]}, expected {self.size}")
        else:
            print("warning: could not read the window bounds; assuming the full screen")
        return self

    def _activate(self) -> None:
        """Ask macOS to make this process the active app. From a terminal this can be refused
        (recent macOS only lets the frontmost app yield focus), so callers must not rely on it."""
        import AppKit

        if hasattr(self._app, "activate"):
            self._app.activate()
        self._app.activateIgnoringOtherApps_(True)
        AppKit.NSRunningApplication.currentApplication().activateWithOptions_(
            AppKit.NSApplicationActivateIgnoringOtherApps)
        self._window.makeKeyAndOrderFront_(None)

    @property
    def has_focus(self) -> bool:
        """True if key presses will reach this window (otherwise they go to the Terminal)."""
        return bool(self._app.isActive() and self._window.isKeyWindow())

    def _sync_cursor(self) -> None:
        """Hide the mouse pointer only while we have focus; show it when the user must click."""
        import AppKit

        hide = self.has_focus
        if hide != self._cursor_hidden:
            (AppKit.NSCursor.hide if hide else AppKit.NSCursor.unhide)()
            self._cursor_hidden = hide

    def __exit__(self, *exc: object) -> None:
        import AppKit

        if self._cursor_hidden:
            AppKit.NSCursor.unhide()
        if self._window is not None:
            self._window.orderOut_(None)
            self._window.close()
            self._window = None
        self._pump()

    @property
    def safe_top(self) -> int:
        """First canvas row below the menu-bar/notch strip (plus a margin). The window covers the
        whole display, so anything drawn above this can be hidden behind the camera notch."""
        return int(self.cfg.top_inset_pt) + 14

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

    def box(self, img: np.ndarray, center: tuple[float, float], half: float, color=(255, 255, 255),
            thickness: int = 3) -> None:
        """A square of side ``2 * half`` screen points centred at ``center``; ``thickness=-1`` fills it."""
        x, y = self._px(center)
        h = int(round(half))
        cv2.rectangle(img, (x - h, y - h), (x + h, y + h), color, thickness, cv2.LINE_AA)

    def marker(self, img: np.ndarray, xy: tuple[float, float], color=(255, 255, 0), r: int = 10) -> None:
        """A filled dot with a dark rim (the live gaze position)."""
        c = self._px(xy)
        cv2.circle(img, c, r + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, c, r, color, -1, cv2.LINE_AA)

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

        self._frames = getattr(self, "_frames", 0) + 1
        if self._frames % 15 == 0:  # about twice a second
            self._sync_cursor()
        if not self._window.isKeyWindow() or not self._app.isActive():
            if self._frames % 15 == 0:
                self._activate()
            img = img.copy()
            cv2.rectangle(img, (0, self.size[1] - 110), (self.size[0], self.size[1] - 50), (0, 90, 140), -1)
            self.text(img, "Keys are not reaching this window: click anywhere on the screen once",
                      y=self.size[1] - 72)
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


class ClickThroughOverlay:
    """Transparent fullscreen layer for Cursor mode: drawn above normal windows, never takes focus,
    never receives the mouse (every click goes to the app underneath), leaves the system cursor alone.
    Draw on ``blank()`` (BGRA, alpha 0 = see-through) and pass it to ``show``."""

    def __init__(self, cfg: ScreenConfig) -> None:
        self.cfg = cfg
        self.size = (cfg.width_px, cfg.height_px)
        self._window = None
        self._app = None

    def __enter__(self) -> ClickThroughOverlay:
        import AppKit

        self._app = AppKit.NSApplication.sharedApplication()
        self._app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)  # no Dock icon, no focus
        screen = AppKit.NSScreen.mainScreen().frame()
        win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            screen, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
        win.setLevel_(AppKit.NSStatusWindowLevel)
        win.setOpaque_(False)
        win.setBackgroundColor_(AppKit.NSColor.clearColor())
        win.setHasShadow_(False)
        win.setIgnoresMouseEvents_(True)
        win.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                   | AppKit.NSWindowCollectionBehaviorStationary
                                   | AppKit.NSWindowCollectionBehaviorIgnoresCycle)
        win.contentView().setWantsLayer_(True)
        win.orderFrontRegardless()
        self._window = win
        return self

    def __exit__(self, *exc: object) -> None:
        if self._window is not None:
            self._window.orderOut_(None)
            self._window.close()
            self._window = None
        self.pump()

    @property
    def safe_top(self) -> int:
        return int(self.cfg.top_inset_pt) + 14

    def blank(self) -> np.ndarray:
        return np.zeros((self.size[1], self.size[0], 4), np.uint8)

    def pump(self) -> None:
        import AppKit

        while True:
            event = self._app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                AppKit.NSEventMaskAny, None, AppKit.NSDefaultRunLoopMode, True)
            if event is None:
                return
            self._app.sendEvent_(event)

    def show(self, bgra: np.ndarray) -> None:
        import Quartz

        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        a = rgba[..., 3:4].astype(np.uint16)
        rgba[..., :3] = (rgba[..., :3].astype(np.uint16) * a // 255).astype(np.uint8)  # premultiplied alpha
        rgba = np.ascontiguousarray(rgba)
        h, w = rgba.shape[:2]
        provider = Quartz.CGDataProviderCreateWithData(None, rgba.tobytes(), rgba.nbytes, None)
        image = Quartz.CGImageCreate(
            w, h, 8, 32, w * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
            Quartz.kCGImageAlphaPremultipliedLast, provider, None, False, Quartz.kCGRenderingIntentDefault)
        self._window.contentView().layer().setContents_(image)
        self.pump()
