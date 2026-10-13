"""Run a config over an evaluation mode and aggregate per target."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.configs import Config
from gaze_cursor.eval.metrics import cm_to_deg, px_to_cm
from gaze_cursor.eval.session import Session
from gaze_cursor.eval.splits import Dataset, build_dataset, make_splits
from gaze_cursor.eval.stats import TargetStats, per_target


def common_screen(sessions: Sequence[Session]) -> ScreenConfig:
    """The ScreenConfig shared by all sessions (error units need one geometry)."""
    cfgs = {tuple(sorted(s.meta["screen"].items())) for s in sessions}
    if len(cfgs) != 1:
        raise ValueError("sessions were recorded with different screen geometries")
    return ScreenConfig(**sessions[0].meta["screen"])


def build_datasets(sessions: Sequence[Session], configs: Sequence[Config]) -> dict[str, Dataset]:
    """One dataset per feature set used by ``configs`` (rows align across feature sets)."""
    return {fs: build_dataset(sessions, fs) for fs in sorted({c.feature_set for c in configs})}


def evaluate(datasets: dict[str, Dataset], config: Config, mode: str, screen: ScreenConfig,
             held_out: Sequence[int] | None = None) -> tuple[TargetStats, float]:
    """Fit ``config`` on each split of ``mode`` and score the test frames.

    Returns the per-target stats and, for cross-checking against benchmark.py, the frame-weighted
    mean error in degrees (the per-target mean is the unit of analysis everywhere else).
    """
    ds = datasets[config.feature_set]
    preds, ys, targets, sessions = [], [], [], []
    for sp in make_splits(ds, mode, held_out):
        model = config.make_model(screen).fit(ds.x[sp.train], ds.y[sp.train])
        preds.append(model.predict(ds.x[sp.test]))
        ys.append(ds.y[sp.test]); targets.append(ds.target[sp.test]); sessions.append(ds.session[sp.test])
    if not preds:
        raise ValueError(f"mode {mode!r} produced no splits (need more sessions or targets)")
    pred, y, tg, se = (np.concatenate(a) for a in (preds, ys, targets, sessions))
    frame_err = np.linalg.norm(pred - y, axis=1)
    frame_mean_deg = float(np.mean(cm_to_deg(px_to_cm(frame_err, screen), screen)))
    return per_target(pred, y, tg, se, screen), frame_mean_deg
