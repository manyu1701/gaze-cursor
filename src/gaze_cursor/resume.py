"""Remember the last calibration (and what click learning added to it) so the next start needs no dots.

``save`` writes one small npz (calibration features, the learned model state, the posture reference,
screen size, wall-clock time); ``load`` returns it only if it is still trustworthy: same screen, not older
than ``MAX_AGE_H``. Anything else means "calibrate again". Positions and numbers only, no images.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.online import OnlineRidge

MAX_AGE_H = 12.0            # a calibration from another part of the day (light, posture) is not reused
FILE_NAME = "last_calibration.npz"


@dataclass
class Saved:
    model: OnlineRidge
    x_train: np.ndarray
    ref_center: dict[str, float]
    calib: str
    calibrated_at: float        # epoch seconds when the 9 dots were done (NOT when this file was last written)
    state_id: str | None = None  # the model_history state this file holds (None: written before L2)

    def age_h(self, now: float | None = None) -> float:
        return ((time.time() if now is None else now) - self.calibrated_at) / 3600


def save(directory: Path, model: OnlineRidge, x_train: np.ndarray, ref_center: dict[str, float], calib: str,
         cfg: ScreenConfig, calibrated_at: float, state_id: str | None = None) -> Path:
    """Atomically (re)write the resume file in ``directory``. ``calibrated_at`` is when the dots were done:
    autosaves keep it, so the 12 h limit counts from the calibration, not from the last use."""
    directory.mkdir(parents=True, exist_ok=True)
    path, tmp = directory / FILE_NAME, directory / ".last_calibration.tmp.npz"
    meta = {"ref_center": ref_center, "calib": calib, "forget": model.forget, "n_updates": model.n_updates,
            "screen": [cfg.width_px, cfg.height_px], "calibrated_at": calibrated_at,
            "state_id": state_id}
    np.savez_compressed(tmp, mean=model.mean, scale=model.scale, theta=model.theta, P=model.P,
                        x_train=x_train, meta=json.dumps(meta))
    tmp.replace(path)
    return path


def read_unchecked(directory: Path) -> tuple[Saved | None, list[int] | None]:
    """The file's content with no age or screen check (None if absent or unreadable), plus the screen it
    was made for. Used to archive whatever is on disk before it can be replaced."""
    path = directory / FILE_NAME
    if not path.exists():
        return None, None
    try:
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        saved = Saved(OnlineRidge(d["mean"], d["scale"], d["theta"], d["P"], float(meta["forget"]),
                                  int(meta["n_updates"])),
                      d["x_train"], {k: float(v) for k, v in meta["ref_center"].items()}, str(meta["calib"]),
                      float(meta["calibrated_at"]), meta.get("state_id"))
        return saved, list(meta["screen"])
    except (OSError, EOFError, KeyError, ValueError):
        return None, None


def load(directory: Path, cfg: ScreenConfig, now: float | None = None) -> tuple[Saved | None, str]:
    """(saved state, "") if it can be reused, else (None, the reason, in words for the user)."""
    if not (directory / FILE_NAME).exists():
        return None, "no earlier calibration"
    saved, screen = read_unchecked(directory)
    if saved is None:
        return None, "the saved calibration could not be read"
    if screen != [cfg.width_px, cfg.height_px]:
        return None, "the screen changed since the last calibration"
    if saved.age_h(now) > MAX_AGE_H or saved.age_h(now) < 0:
        return None, f"the last calibration is {saved.age_h(now):.0f} h old"
    return saved, ""
