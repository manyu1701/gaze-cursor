"""Keep every learned model state (PLAN step L2): the live model may start fresh, history is never lost.

``resume.py`` holds the one *current* state in a single overwritten file. This module adds an archive next
to it: each state is written once to ``model_states/state_<UTC time>_<state id>.npz`` and never changed or
deleted. A state is the online model (standardizer, coefficients, inverse information, forgetting, update
count) plus the calibration frames and posture reference it came from, and its provenance:

- ``state_id``: ``<model version>-<n_updates>``. The model version is the hash that sessions already store
  (``conditions.model_version``), so a fresh ``cursor_*`` log's starting version links to the state with 0
  updates (a resumed log's starting version belongs to a state with updates; match on ``model_version``).
- ``parent_id``: the state this one was learned from (None for a fresh calibration), so the chain of
  learning is walkable. Identical states are stored once (the archive is idempotent).
- ``reason``: why it was written (calibration, autosave, quit, superseded, resumed, ...), the cursor log
  of that run, person, code hash, screen, calibration time.

Positions and numbers only, no images. A state is rebuildable exactly: ``load`` returns the same model.
Pure numpy and files: portable, no OS code.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gaze_cursor.conditions import model_version
from gaze_cursor.config import ScreenConfig
from gaze_cursor.online import OnlineRidge

DIR_NAME = "model_states"
SCHEMA_VERSION = 1


@dataclass
class State:
    model: OnlineRidge
    x_train: np.ndarray
    ref_center: dict[str, float]
    meta: dict


def state_id(model: OnlineRidge) -> str:
    """``<model version>-<n_updates>``: any change to the arrays or the update count gives a new id."""
    return f"{model_version(model.mean, model.scale, model.theta, model.P)}-{model.n_updates}"


def _find(directory: Path, sid: str) -> Path | None:
    hits = sorted(directory.glob(f"state_*_{sid}.npz"))
    return hits[0] if hits else None


def archive(root: Path, model: OnlineRidge, x_train: np.ndarray, ref_center: dict[str, float], calib: str,
            cfg: ScreenConfig, calibrated_at: float, reason: str, parent_id: str | None = None,
            cursor_log: str | None = None, person_id: str | None = None, code: str | None = None) -> str:
    """Write the state to ``root / model_states`` unless that exact state is already there; return its id.

    ``root`` is the directory that holds ``last_calibration.npz`` (``data/quick``). Never overwrites a file:
    the first writer of a state keeps its provenance. The file is private (mode 0600).
    """
    sid = state_id(model)
    directory = root / DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    if _find(directory, sid) is not None:
        return sid
    now = time.time()
    meta = {
        "schema_version": SCHEMA_VERSION, "state_id": sid, "model_version": sid.rsplit("-", 1)[0],
        "parent_id": parent_id, "reason": reason, "archived_at": now,
        "archived_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "n_updates": model.n_updates, "forget": model.forget, "calib": calib, "calibrated_at": calibrated_at,
        "screen": [cfg.width_px, cfg.height_px], "ref_center": ref_center, "cursor_log": cursor_log,
        "person_id": person_id, "code": code,
    }
    path = directory / f"state_{time.strftime('%Y%m%d_%H%M%S', time.gmtime(now))}_{sid}.npz"
    tmp = directory / f".{path.stem}.tmp.npz"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        np.savez_compressed(f, mean=model.mean, scale=model.scale, theta=model.theta, P=model.P,
                            x_train=x_train, meta=json.dumps(meta))
    tmp.replace(path)
    return sid


def load(path: Path) -> State:
    """Read one archived state back (the same model, bit for bit)."""
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        model = OnlineRidge(d["mean"], d["scale"], d["theta"], d["P"], float(meta["forget"]),
                            int(meta["n_updates"]))
        x_train = d["x_train"]
    return State(model, x_train, {k: float(v) for k, v in meta["ref_center"].items()}, meta)


def verify(path: Path) -> bool:
    """True if the stored arrays still hash to the id in the file (the file was not altered or damaged)."""
    try:
        s = load(path)
    except (OSError, EOFError, KeyError, ValueError):
        return False
    return state_id(s.model) == s.meta.get("state_id")


def list_states(root: Path) -> list[dict]:
    """Meta of every archived state, oldest first (reads only the small meta, with the file name added).
    Unreadable files are listed with ``error`` instead of being skipped silently."""
    out: list[dict] = []
    for path in sorted((root / DIR_NAME).glob("state_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as d:
                meta = json.loads(str(d["meta"]))
        except (OSError, EOFError, KeyError, ValueError):
            out.append({"file": path.name, "error": "could not be read"})
            continue
        out.append({**meta, "file": path.name})
    return sorted(out, key=lambda m: (m.get("archived_at", 0.0), m["file"]))


def lineage(root: Path, sid: str) -> list[dict]:
    """The chain of states that led to ``sid``, oldest first (stops at a state with no archived parent)."""
    by_id = {m["state_id"]: m for m in list_states(root) if "state_id" in m}
    chain: list[dict] = []
    seen: set[str] = set()
    while sid in by_id and sid not in seen:
        seen.add(sid)
        chain.append(by_id[sid])
        sid = by_id[sid].get("parent_id") or ""
    return chain[::-1]
