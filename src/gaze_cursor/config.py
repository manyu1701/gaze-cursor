"""Screen geometry used to convert pixels to cm and degrees of visual angle."""

from __future__ import annotations

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

    @classmethod
    def from_system(cls, distance_cm: float = 60.0) -> ScreenConfig:
        """Main display geometry from Quartz: size in points, physical size from the EDID (mm)."""
        import Quartz

        display = Quartz.CGMainDisplayID()
        bounds = Quartz.CGDisplayBounds(display).size
        mm = Quartz.CGDisplayScreenSize(display)
        return cls(
            width_px=int(bounds.width), height_px=int(bounds.height),
            width_cm=mm.width / 10, height_cm=mm.height / 10, distance_cm=distance_cm,
        )

    @property
    def cm_per_px(self) -> float:
        return self.width_cm / self.width_px

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
