"""Screen geometry used to convert pixels to cm and degrees of visual angle."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScreenConfig:
    """Display geometry. Pixels are logical points (what macOS windows use).

    Defaults: MacBook Air 13.6" (Mac17,3, M5), 2560x1664 native = 1470x956 points.
    Physical size is derived from the 13.6" diagonal and is accurate to about 1%.
    """

    width_px: int = 1470
    height_px: int = 956
    width_cm: float = 29.0
    height_cm: float = 18.9
    distance_cm: float = 60.0  # assumed eye-to-screen distance

    @property
    def cm_per_px(self) -> float:
        return self.width_cm / self.width_px

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
