"""Synthetic sessions for harness tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gaze_cursor.eval.session import CALIB, CLEAN_WINDOW, TEST, Session
from gaze_cursor.features import BLEND_FEATURE_NAMES

NAMES = BLEND_FEATURE_NAMES + tuple(f"other{i}" for i in range(52 - len(BLEND_FEATURE_NAMES)))


def fake_session(i: int, n_calib: int = 6, n_test: int = 3, frames: int = 5, seed: int = 0,
                 window: list[int] | None = None, layout: str | None = None) -> Session:
    """Frames whose features are a noiseless linear function of the target position."""
    rng = np.random.default_rng(seed + i)
    rows, targets = [], []
    for t in range(n_calib + n_test):
        xy = rng.uniform([30, 60], [1440, 930])
        targets.append((CALIB if t < n_calib else TEST, t, xy))
    phase = np.repeat([p for p, _, _ in targets], frames).astype(np.int8)
    idx = np.repeat([t for _, t, _ in targets], frames).astype(np.int16)
    xy = np.repeat([x for _, _, x in targets], frames, axis=0).astype(np.float32)
    feats = np.zeros((len(phase), 9))
    feats[:, 0] = xy[:, 0] / 1470 - 0.5
    feats[:, 1] = xy[:, 1] / 956 - 0.5
    feats[:, 2] = feats[:, 0]; feats[:, 3] = feats[:, 1]
    feats[:, 7:] = 0.25
    blend = np.tile(rng.uniform(0, 1, 52), (len(phase), 1)).astype(np.float32)
    meta = {"screen": {"width_px": 1470, "height_px": 956, "width_cm": 29.0, "height_cm": 18.9,
                       "distance_cm": 60.0},
            "window": CLEAN_WINDOW if window is None else window, "blendshape_names": list(NAMES)}
    if layout:
        meta["layout"] = layout
    return Session(path=Path(f"session_fake_{i:02d}.npz"), meta=meta,
                   t_ms=np.arange(len(phase)), phase=phase, target_idx=idx, target_xy=xy,
                   features=feats, blendshapes=blend)
