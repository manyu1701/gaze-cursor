"""Accuracy and jitter metrics. Pure functions."""

from __future__ import annotations

import math

import numpy as np

from gaze_cursor.config import ScreenConfig


def px_to_cm(px: np.ndarray | float, cfg: ScreenConfig) -> np.ndarray | float:
    return px * cfg.cm_per_px


def cm_to_deg(cm: np.ndarray | float, cfg: ScreenConfig) -> np.ndarray | float:
    """Visual angle subtended by ``cm`` on screen, viewed head-on at cfg.distance_cm."""
    return np.degrees(np.arctan2(cm, cfg.distance_cm))


def error_metrics(pred: np.ndarray, target: np.ndarray, cfg: ScreenConfig) -> dict[str, float]:
    """Euclidean error between predictions and targets (both (n, 2) px): mean/median in px, cm, deg."""
    err_px = np.linalg.norm(pred - target, axis=1)
    err_cm = px_to_cm(err_px, cfg)
    err_deg = cm_to_deg(err_cm, cfg)
    return {
        "err_px_mean": float(err_px.mean()), "err_px_median": float(np.median(err_px)),
        "err_cm_mean": float(err_cm.mean()), "err_deg_mean": float(err_deg.mean()),
        "err_deg_median": float(np.median(err_deg)),
    }


def jitter_metrics(
    pred: np.ndarray, target_idx: np.ndarray, cfg: ScreenConfig
) -> dict[str, float]:
    """Mean over fixations of the prediction spread (sqrt(var_x + var_y)) within a fixation."""
    spreads = []
    for t in np.unique(target_idx):
        p = pred[target_idx == t]
        if len(p) > 1:
            spreads.append(math.sqrt(float(p.var(axis=0, ddof=1).sum())))
    px = float(np.mean(spreads)) if spreads else float("nan")
    return {"jitter_px": px, "jitter_deg": float(cm_to_deg(px_to_cm(px, cfg), cfg))}
