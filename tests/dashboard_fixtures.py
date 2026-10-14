"""A synthetic data directory for the dashboard tests (no camera, no real recordings)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.conditions import build_conditions
from gaze_cursor.dashboard.catalog import Roots
from gaze_cursor.features import N_FEATURES
from gaze_cursor.online import OnlineRidge
from gaze_cursor import resume

SCREEN = {"width_px": 1470, "height_px": 956, "width_cm": 29.0, "height_cm": 18.9, "distance_cm": 60.0}


def tagged_block(person: str | None = "p01") -> dict:
    return build_conditions(
        person_id=person, camera={"name": "FaceTime HD Camera", "width": 1280, "height": 720, "built_in": True},
        screen_layout={"displays": [{"w_pt": 1470, "h_pt": 956, "w_cm": 30.0, "h_cm": 19.5, "main": True}]},
        assumed_distance_cm=60.0, iris_px=19.6, image_width=1280, lighting_tag="lamp", glasses="no",
        posture_note="upright", git_hash="abc1234")


def write_session(path: Path, meta: dict, n_targets: int = 4, frames_per: int = 5, minutes: float = 1.0) -> Path:
    n = n_targets * frames_per
    np.savez_compressed(
        path, t_ms=np.linspace(0, minutes * 60000, n).astype(np.int64),
        phase=np.zeros(n, np.int8), target_idx=np.repeat(np.arange(n_targets), frames_per).astype(np.int16),
        target_xy=np.zeros((n, 2), np.float32), features=np.zeros((n, N_FEATURES)),
        blendshapes=np.zeros((n, 52), np.float32), meta=json.dumps(meta))
    return path


def write_cursor(path: Path, minutes: float = 30.0, clicks: int = 40, learned: int = 25, median: float | None = 2.4,
                 started: str = "20261014_100000") -> Path:
    meta = {"kind": "cursor_session", "screen": SCREEN, "started": started,
            "summary": {"frames": 1000, "minutes": minutes, "tracked": 0.9, "warps": 12, "warps_refused": 1,
                        "clicks": clicks, "click_gaze_deg_median": median, "click_gaze_deg_mean": 3.0,
                        "click_gaze_n": clicks - 2, "clicks_learned": learned},
            "learning": {"on": True, "resumed": "10:00", "gate_deg": 6.0, "forget": 0.98}}
    np.savez_compressed(path, frames=np.zeros((3, 15)), events=np.zeros((3, 7)), meta=json.dumps(meta))
    return path


def make_data(tmp: Path) -> Roots:
    """sessions/: legacy + tagged + backfilled-sidecar + a face sidecar to ignore; quick/: Cursor calibration,
    Cursor log, last_calibration; tasks/: one practice task; results.csv; a repo folder with the three docs."""
    data, repo = tmp / "data", tmp / "repo"
    for d in ("sessions", "quick", "tasks"):
        (data / d).mkdir(parents=True)
    (repo / "models").mkdir(parents=True)
    light = {"face": 120.0, "background": 80.0, "ratio": 1.5, "label": "front-lit"}
    base = {"screen": SCREEN, "window": [0, 0, 1470, 956], "layout": "v2", "lighting": light}
    write_session(data / "sessions" / "session_20261013_100000.npz", {**base, "note": "pos C", "started": "2026-10-13T10:00:00"})
    write_session(data / "sessions" / "session_20261014_090000.npz",
                  {**base, "note": "pos N", "started": "2026-10-14T09:00:00", "conditions": tagged_block()})
    legacy = write_session(data / "sessions" / "session_20261012_080000.npz", {**base, "note": "evening, lamp"})
    legacy.with_name(legacy.stem + ".conditions.json").write_text(json.dumps({**tagged_block("p01"), "backfilled": True}))
    (data / "sessions" / "session_20261012_080000.faces.npz").write_bytes(b"not a real archive")
    write_session(data / "quick" / "session_20261014_095500.npz",
                  {"screen": SCREEN, "note": "quick calibration (Cursor)", "started": "2026-10-14T09:55:00"})
    write_cursor(data / "quick" / "cursor_20261014_100000.npz")
    write_cursor(data / "quick" / "cursor_20261013_200000.npz", minutes=60, clicks=100, learned=0, median=None,
                 started="20261013_200000")
    write_session(data / "tasks" / "task_20261014_110000.npz", {"started": "2026-10-14T11:00:00", "kind": "practice"})
    rng = np.random.default_rng(0)
    model = OnlineRidge.from_calibration(rng.normal(size=(30, 4)), rng.uniform(0, 900, (30, 2)), alpha=1.0)
    resume.save(data / "quick", model, rng.normal(size=(30, 4)), {"x": 0.0, "y": 0.0}, "quick calibration",
                ScreenConfig(), calibrated_at=time.time() - 3600)
    (data / "results.csv").write_text(
        "time,mode,confirmatory,config,session,n_targets,err_deg_mean,err_deg_median,jitter_deg,ci_lo,ci_hi,note\n"
        "2026-10-13T02:41:12,within,0,,session_a.npz,9,5.4,5.0,2.2,,,\n"
        "2026-10-14T01:00:00,h6_click_learning,1,,,,0.64,,,0.44,0.85,round 4\n")
    (repo / "PREREGISTRATIONS.md").write_text(
        "# Pre-registrations and results\n\n# Round 1: features (H1)\n\n### Pre-registration: H1 (frozen)\n\n"
        "### RESULT of the test\n\n# Round 2: lighting (H2)\n\n### Pre-registration, round 2 (frozen)\n")
    (repo / "PLAN.md").write_text("# Plan\n\n**Status (2026-10-14):** step 6c next.\n\n**Status: M2 DONE.**\n")
    (repo / "RESULTS.md").write_text("# Results\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    return Roots.from_data_dir(data, repo=repo, cache=tmp / "cache" / "dashboard_cache.json")
