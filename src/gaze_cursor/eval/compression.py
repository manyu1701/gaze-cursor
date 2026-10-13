"""Centre compression of the quick calibration (EXPLORATORY, after the vertical-bias diagnosis).

Predictions were squeezed toward the screen centre on both axes. Variants of the quick 9-dot fit
are compared on identical scored targets per session:
- ridge strength (``alpha``);
- fitting on one averaged feature vector per dot instead of every frame (less noise in the inputs,
  so less regression dilution);
- an affine stretch fitted on the dots' mean predictions (undoes a uniform squeeze);
- which 9 grid targets are the dots.
``gain`` summarises the squeeze: the slope of mean prediction on target position, per axis (1 = none).
Pure functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gaze_cursor.calibration import GazeModel
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.correction import apply_correction, choose_dots, fit_correction
from gaze_cursor.eval.session import CALIB, TEST, Session
from gaze_cursor.eval.stats import TargetStats, per_target
from gaze_cursor.eval.targets import calibration_grid_v2


@dataclass(frozen=True)
class Variant:
    name: str
    alpha: float = 10.0
    fit_on: str = "frames"     # "frames" (every valid frame) or "means" (one row per dot)
    stretch: bool = False      # affine correction fitted on the dots' mean predictions
    feature_set: str = "gaze"  # "eyes" adds eyelid openness, "gaze_blend" the eye blendshapes


def quick_dot_set(cfg: ScreenConfig, k: int = 9) -> tuple[int, ...]:
    """Today's quick dots (``choose_dots``: centre first, then farthest-point), as grid indices."""
    grid = calibration_grid_v2(cfg)
    return tuple(sorted(int(i) for i in choose_dots(grid, k, (cfg.width_px / 2, cfg.height_px / 2))))


def edge_dot_set(cfg: ScreenConfig) -> tuple[int, ...]:
    """9 of the 4 x 4 v2 grid targets that cover every edge: the 4 corners, one middle point per
    edge in a pinwheel (top 1, right 7, bottom 14, left 8), and the inner point today's rule starts
    from (nearest the screen centre). Today's set has no left, right or bottom middle point."""
    grid = calibration_grid_v2(cfg)
    if len(grid) != 16:
        raise ValueError("edge_dot_set assumes the 4 x 4 v2 grid")
    centre = int(choose_dots(grid, 1, (cfg.width_px / 2, cfg.height_px / 2))[0])
    return tuple(sorted({0, 3, 12, 15, 1, 7, 14, 8, centre}))


def _fit(v: Variant, x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> GazeModel:
    if v.fit_on == "means":
        keys = np.unique(idx)
        x = np.array([x[idx == t].mean(axis=0) for t in keys])
        y = np.array([y[idx == t][0] for t in keys])
    elif v.fit_on != "frames":
        raise ValueError(f"unknown fit_on {v.fit_on!r}")
    return GazeModel("ridge", v.alpha).fit(x, y)


def score_session(s: Session, i: int, dots: tuple[int, ...], v: Variant, scored: np.ndarray,
                  screen: ScreenConfig) -> TargetStats:
    """Fit ``v`` on the ``dots`` of session ``s`` and score the target indices ``scored``."""
    xc, yc, tc = s.select(CALIB, v.feature_set)
    xt, yt, tt = s.select(TEST, v.feature_set)
    x, y, t = np.concatenate([xc, xt]), np.concatenate([yc, yt]), np.concatenate([tc, tt]).astype(int)
    on = np.isin(t, dots)
    model = _fit(v, x[on], y[on], t[on])
    te = np.isin(t, scored)
    pred = model.predict(x[te])
    if v.stretch:
        p_dot = model.predict(x[on])
        keys = np.unique(t[on])
        mean_pred = np.array([p_dot[t[on] == k].mean(axis=0) for k in keys])
        truth = np.array([y[on][t[on] == k][0] for k in keys])
        pred = apply_correction(pred, fit_correction(mean_pred, truth, "affine"))
    return per_target(pred, y[te], i * 1000 + t[te], np.full(int(te.sum()), i), screen)


def gain(t: TargetStats) -> tuple[float, float]:
    """Slope of mean prediction on target position, per axis (least squares). 1 = no squeeze."""
    mean_pred = t.xy + t.bias_px
    out = []
    for a in (0, 1):
        slope, _ = np.polyfit(t.xy[:, a], mean_pred[:, a], 1)
        out.append(float(slope))
    return out[0], out[1]
