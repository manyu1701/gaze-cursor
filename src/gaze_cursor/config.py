"""Screen geometry used to convert pixels to cm and degrees of visual angle."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScreenConfig:
    """Display geometry. Pixels are logical points (what macOS windows use).

    Use ``from_system()`` for the real display. The defaults are the MacBook Air 13.6"
    (Mac17,3, M5): 2560x1664 native = 1470x956 points, used for tests and as a fallback.
    """

    width_px: int = 1470
    height_px: int = 956
    width_cm: float = 29.0
    height_cm: float = 18.9
    distance_cm: float = 60.0  # assumed eye-to-screen distance
    # Display shape. Apple: the 13.6" Air has rounded TOP corners only (bottom corners square) and
    # a notch. Radius is not published and has no API, so this is a deliberately generous
    # ASSUMPTION used only to keep targets away from the corners.
    corner_radius_pt: float = 24.0
    top_inset_pt: float = 32.0  # menu bar / notch strip (NSScreen.safeAreaInsets.top, measured)
    notch_x0_pt: float = 646.0  # notch left/right edge (from NSScreen.auxiliaryTop*Area, measured)
    notch_x1_pt: float = 825.0

    @classmethod
    def from_system(cls, distance_cm: float = 60.0) -> ScreenConfig:
        """Main display geometry from Quartz: size in points, physical size from the EDID (mm)."""
        import Quartz

        display = Quartz.CGMainDisplayID()
        bounds = Quartz.CGDisplayBounds(display).size
        mm = Quartz.CGDisplayScreenSize(display)
        extra: dict[str, float] = {}
        try:  # notch geometry; absent on displays without a notch
            import AppKit

            screen = AppKit.NSScreen.mainScreen()
            top = screen.safeAreaInsets().top
            left, right = screen.auxiliaryTopLeftArea(), screen.auxiliaryTopRightArea()
            if top > 0:
                extra["top_inset_pt"] = float(top)
            if left.size.width > 0 and right.size.width > 0:
                extra["notch_x0_pt"] = float(left.origin.x + left.size.width)
                extra["notch_x1_pt"] = float(right.origin.x)
        except Exception:  # pragma: no cover - fall back to the defaults
            pass
        return cls(
            width_px=int(bounds.width), height_px=int(bounds.height),
            width_cm=mm.width / 10, height_cm=mm.height / 10, distance_cm=distance_cm, **extra,
        )

    def target_is_safe(self, xy: tuple[float, float], ring_px: float = 16.0) -> bool:
        """True if a target ring of radius ``ring_px`` centered at ``xy`` (screen points) is fully
        on the visible display: clear of the screen edges, the menu-bar/notch strip, and the
        rounded top corners."""
        x, y = xy
        r = self.corner_radius_pt
        if not (ring_px <= x <= self.width_px - ring_px and
                self.top_inset_pt + ring_px <= y <= self.height_px - ring_px):
            return False
        # Rounded top corners: inside a corner square, stay inside the arc (corner centers at r, r).
        if y < r and (x < r or x > self.width_px - r):
            cx = r if x < r else self.width_px - r
            if math.hypot(x - cx, y - r) > r - ring_px:
                return False
        return True

    @property
    def cm_per_px(self) -> float:
        return self.width_cm / self.width_px

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
