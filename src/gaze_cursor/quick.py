"""Quick recalibration where the user sits: the M2 round-3 winner (PROGRESS.md). Pure functions.

A fresh ``gaze_a10`` fitted on 9 dots of the current position beat the model pooled over every
earlier calibration by +1.29 deg (pre-registered H5, set B). The dots, their timing and the model
are exactly what was tested: the 9 of the 16 v2 calibration targets picked by ``choose_dots``
(centre first, then farthest-point), 0.7 s settle + 1.0 s recording each, ``gaze_a10``.
"""

from __future__ import annotations

import numpy as np

from gaze_cursor.calibration import GazeModel
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.configs import REGISTRY
from gaze_cursor.eval.correction import choose_dots
from gaze_cursor.eval.session import CALIB
from gaze_cursor.eval.stats import per_target
from gaze_cursor.eval.targets import Target, calibration_grid_v2

QUICK_DOTS = 9
QUICK_CONFIG = "gaze_a10"
QUICK_SETTLE_S, QUICK_RECORD_S = 0.7, 1.0   # same as calibrate.py, layout v2
MIN_DOT_FRAMES = 10                          # fewer valid frames on a dot (blinks, lost face): redo it
QUICK_LAYOUT = "quick9"


def quick_targets(cfg: ScreenConfig, k: int = QUICK_DOTS, seed: int = 0) -> list[Target]:
    """The ``k`` quick dots in a random (seeded) order. ``idx`` is the v2 calibration index, so a
    quick session lines up with the full calibrations target by target."""
    grid = calibration_grid_v2(cfg)
    picked = choose_dots(grid, k, (cfg.width_px / 2, cfg.height_px / 2))
    order = np.random.default_rng([seed, 2]).permutation(len(picked))
    return [Target(CALIB, int(picked[i]), (float(grid[picked[i], 0]), float(grid[picked[i], 1]))) for i in order]


def fit_quick(x: np.ndarray, y: np.ndarray, screen: ScreenConfig | None = None) -> GazeModel:
    """The frozen quick model on the dots' frames (``gaze`` features)."""
    return REGISTRY[QUICK_CONFIG].make_model(screen).fit(x, y)


def missing_dots(target_idx: np.ndarray, targets: list[Target], min_frames: int = MIN_DOT_FRAMES) -> list[int]:
    """Indices (``Target.idx``) of dots with fewer than ``min_frames`` valid frames."""
    counts = {t.idx: int(np.sum(target_idx == t.idx)) for t in targets}
    return [i for i, n in counts.items() if n < min_frames]


def fit_report(model: GazeModel, x: np.ndarray, y: np.ndarray, target_idx: np.ndarray,
               screen: ScreenConfig) -> dict[str, float]:
    """How well the model fits its own dots (in-sample, so optimistic): mean and worst per-dot error
    in degrees. A large value means a dot was not looked at properly; redo the calibration."""
    t = per_target(model.predict(x), y, target_idx, np.zeros(len(y), int), screen)
    return {"mean_deg": float(t.err_deg.mean()), "worst_deg": float(t.err_deg.max()), "dots": int(len(t.err_deg))}
