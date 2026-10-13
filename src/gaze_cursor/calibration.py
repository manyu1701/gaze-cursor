"""Feature -> screen (x, y) regression models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from gaze_cursor.config import ScreenConfig

MODELS = ("ridge", "poly2", "geo")


class GazeModel:
    """Maps a feature matrix (n, n_features) to screen positions (n, 2) in pixels.

    ``ridge`` / ``poly2`` are free regressions on any feature set. ``geo`` is the constrained gaze-ray
    model (see ``_geo_targets``); it needs the ``geo`` feature set and the screen geometry.
    """

    def __init__(self, kind: str = "ridge", alpha: float = 1.0, screen: ScreenConfig | None = None) -> None:
        if kind not in MODELS:
            raise ValueError(f"unknown model {kind!r}; choose from {MODELS}")
        if kind == "geo" and screen is None:
            raise ValueError("the geo model needs the screen geometry")
        self.kind = kind
        self.alpha = alpha
        self.screen = screen
        steps: list = [StandardScaler()]
        if kind == "poly2":
            steps += [PolynomialFeatures(2, include_bias=False), StandardScaler()]
        self._pipe: Pipeline = make_pipeline(*steps, Ridge(alpha=alpha))

    def fit(self, features: np.ndarray, targets_px: np.ndarray) -> GazeModel:
        if self.kind == "geo":
            design, u, weight = _geo_design(features), *self._geo_targets(features, targets_px)
            self._pipe.fit(design, u, ridge__sample_weight=weight)
        else:
            self._pipe.fit(features, targets_px)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(features)
        if self.kind != "geo":
            return self._pipe.predict(features)
        theta = self._pipe.predict(_geo_design(features))
        hx, hy, hz = features[:, 6], features[:, 7], features[:, 8]
        c, w = self.screen.cm_per_px, self.screen.width_px
        return np.stack([w / 2 - (hx + hz * theta[:, 0]) / c, (hy + hz * theta[:, 1]) / c], axis=1)

    def _geo_targets(self, features: np.ndarray, targets_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The gaze-ray model, solved for what is left to learn.

        The camera is in the screen plane at its top centre and faces the user, so a screen point
        (sx, sy) is at camera x = (W/2 - sx) * c, y = sy * c (cm; c = cm per point; the camera's
        exact offset is learned via the 1/z term). A ray from the eye at (hx, hy, z) hits it when
        x - hx = z * tan(a), y - hy = z * tan(b). Head position therefore enters with its FIXED
        geometric effect, and only the gaze angles are learned:
        tan(a), tan(b) = linear(iris offsets, head yaw, head pitch) + offset / z.
        Fitting in angle space and weighting by z^2 keeps the loss in screen units.
        """
        hx, hy, hz = features[:, 6], features[:, 7], features[:, 8]
        c, w = self.screen.cm_per_px, self.screen.width_px
        u = np.stack([((w / 2 - targets_px[:, 0]) * c - hx) / hz, (targets_px[:, 1] * c - hy) / hz], axis=1)
        return u, hz ** 2


def _geo_design(features: np.ndarray) -> np.ndarray:
    """Learned part of the geo model: iris offsets (4), yaw and pitch (radians), 1 / depth."""
    f = np.atleast_2d(features)
    return np.concatenate([f[:, 0:4], np.radians(f[:, 4:6]), 1.0 / f[:, 8:9]], axis=1)
