"""Few-dot correction of a pooled model on a new session. Pure functions.

A model trained on earlier calibrations is applied to a new session; the user then looks at k
quick dots, and a small correction is fitted on them: a constant offset (k = 1-3) or an affine map
(k >= 3; 6 numbers). The dots are chosen from the session's calibration targets, the error is
measured on its held-out targets, so the dots are never test data.
"""

from __future__ import annotations

import numpy as np


def choose_dots(xy: np.ndarray, k: int, center: tuple[float, float]) -> np.ndarray:
    """Indices of ``k`` points of ``xy``: the one closest to ``center``, then farthest-point sampling."""
    xy = np.asarray(xy, dtype=float)
    picked = [int(np.argmin(np.linalg.norm(xy - center, axis=1)))]
    while len(picked) < min(k, len(xy)):
        d = np.min(np.linalg.norm(xy[:, None] - xy[picked][None], axis=2), axis=1)
        picked.append(int(np.argmax(d)))
    return np.array(picked)


def fit_correction(pred: np.ndarray, truth: np.ndarray, kind: str) -> np.ndarray:
    """A 3x2 matrix M with corrected = [pred, 1] @ M, fitted on per-dot mean predictions.

    ``offset``: corrected = pred + mean(truth - pred). ``affine``: least-squares affine map
    (needs >= 3 non-collinear dots)."""
    pred, truth = np.atleast_2d(pred), np.atleast_2d(truth)
    if kind == "offset":
        m = np.vstack([np.eye(2), (truth - pred).mean(axis=0)])
        return m
    if kind == "affine":
        if len(pred) < 3:
            raise ValueError("an affine correction needs at least 3 dots")
        m, *_ = np.linalg.lstsq(np.column_stack([pred, np.ones(len(pred))]), truth, rcond=None)
        return m
    raise ValueError(f"unknown correction {kind!r}; choose offset or affine")


def apply_correction(pred: np.ndarray, m: np.ndarray) -> np.ndarray:
    pred = np.atleast_2d(pred)
    return np.column_stack([pred, np.ones(len(pred))]) @ m
