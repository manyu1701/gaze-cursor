"""Named model configurations for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass

from gaze_cursor.calibration import GazeModel
from gaze_cursor.config import ScreenConfig


@dataclass(frozen=True)
class Config:
    """A feature set plus a model and its regularization strength."""

    name: str
    feature_set: str
    model: str = "ridge"
    alpha: float = 1.0

    def make_model(self, screen: ScreenConfig | None = None) -> GazeModel:
        return GazeModel(self.model, self.alpha, screen)


def _c(feature_set: str, alpha: float, model: str = "ridge") -> Config:
    return Config(f"{feature_set}_a{alpha:g}", feature_set, model, alpha)


REGISTRY: dict[str, Config] = {c.name: c for c in (
    _c("iris", 1), _c("iris", 100),            # iris = iris + raw head pose + openness (M1 baseline)
    _c("eyes", 10),
    _c("gaze", 1), _c("gaze", 10), _c("gaze", 100),
    _c("gaze_blend", 10), _c("gaze_blend", 100),
    _c("both", 100),
    _c("head", 1), _c("head", 10), _c("head", 100),   # M2 step 3a: pooled, head pose + position
    _c("ray", 1), _c("ray", 10), _c("ray", 100),      # M2 step 3b: linearised gaze ray -> screen
    _c("geo", 1, "geo"), _c("geo", 10, "geo"), _c("geo", 100, "geo"),  # 3b constrained: fixed geometry
)}

# Pre-registered comparison (see PREREGISTRATIONS.md): reference first, candidate second.
PREREG_REFERENCE = "gaze_a10"
PREREG_CANDIDATE = "gaze_blend_a100"
PREREG_DIAGNOSTICS = ("gaze_a100", "gaze_blend_a10")


def get_configs(names: list[str]) -> list[Config]:
    """Resolve config names (or ``["all"]``); raise a helpful error for unknown names."""
    if names == ["all"]:
        return list(REGISTRY.values())
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown config(s) {unknown}; choose from {sorted(REGISTRY)}")
    return [REGISTRY[n] for n in names]
