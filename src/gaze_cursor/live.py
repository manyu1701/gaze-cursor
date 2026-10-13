"""Build the model a live script uses: the latest (or given) calibration alone, or the pool."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaze_cursor.calibration import GazeModel
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.configs import REGISTRY
from gaze_cursor.eval.session import CALIB, Session, load_session, session_files
from gaze_cursor.pool import POOL_CONFIG, pool_sessions


@dataclass
class LiveModel:
    model: GazeModel
    feature_set: str
    sessions: list[Session]      # what it was trained on (the posture check uses these)
    x_train: np.ndarray          # training inputs (for the out-of-distribution warning)
    label: str                   # one line for the terminal / results log


def fit_live_model(session: Session, screen: ScreenConfig, pool: bool = False, feature_set: str = "gaze",
                   kind: str = "ridge") -> LiveModel:
    """``pool=False``: ``kind`` on ``feature_set``, trained on the session's calibration targets
    (today's Run). ``pool=True``: the round-3 candidate on every target of every clean calibration
    recorded in the same lighting as ``session`` (see ``pool.py``)."""
    if not pool:
        x, y, _ = session.select(CALIB, feature_set)
        name = session.path.stem.removeprefix("session_")
        return LiveModel(GazeModel(kind, screen=screen).fit(x, y), feature_set, [session], x,
                         f"{kind}/{feature_set} on {name}")
    sessions = pool_sessions([load_session(p) for p in session_files()], session)
    cfg = REGISTRY[POOL_CONFIG]
    parts = [s.select_all(cfg.feature_set) for s in sessions]
    x, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    light = (session.meta.get("lighting") or {}).get("label", "unknown")
    names = ", ".join(s.path.stem.removeprefix("session_") for s in sessions)
    return LiveModel(cfg.make_model(screen).fit(x, y), cfg.feature_set, sessions, x,
                     f"{cfg.name} pooled over {len(sessions)} '{light}' calibration(s): {names}")
