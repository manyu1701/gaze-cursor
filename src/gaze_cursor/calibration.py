"""Feature -> screen (x, y) regression models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

MODELS = ("ridge", "poly2")


class GazeModel:
    """Maps a feature matrix (n, n_features) to screen positions (n, 2) in pixels."""

    def __init__(self, kind: str = "ridge", alpha: float = 1.0) -> None:
        if kind not in MODELS:
            raise ValueError(f"unknown model {kind!r}; choose from {MODELS}")
        self.kind = kind
        self.alpha = alpha
        steps: list = [StandardScaler()]
        if kind == "poly2":
            steps += [PolynomialFeatures(2, include_bias=False), StandardScaler()]
        self._pipe: Pipeline = make_pipeline(*steps, Ridge(alpha=alpha))

    def fit(self, features: np.ndarray, targets_px: np.ndarray) -> GazeModel:
        self._pipe.fit(features, targets_px)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._pipe.predict(np.atleast_2d(features))
