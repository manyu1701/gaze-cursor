"""Backfilled condition sidecars for recordings made before the tags existed (L1 stage 5).

Recorded data is never edited: each ``session_X.npz`` / ``cursor_X.npz`` gets a ``X.conditions.json`` next to
it. Only what is known is written: the person id and glasses answer come from the author's flags, the
camera, resolution and screen layout from the recording's own ``meta`` plus the stated camera name, and the
iris size from the stored landmarks (a Cursor log borrows its calibration's). Lighting tag, posture note and
code version stay absent (they were not recorded), so the validator keeps warning about them: a
backfilled block can never fail and never claims more than was measured.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from gaze_cursor.conditions import (
    build_conditions, iris_diameter_px, read_meta, sidecar_path,
)

DEFAULT_IMAGE_SIZE = (1280, 720)    # Camera() default; sessions record the real one in meta["image_size"]


def median_iris_px(npz: Path) -> float:
    """Median iris diameter (px) over the tracked frames of a session file; NaN if it stores no landmarks."""
    meta = read_meta(npz)
    size = tuple(meta.get("image_size") or DEFAULT_IMAGE_SIZE)
    with np.load(npz, allow_pickle=False) as d:
        if "points" not in d.files:
            return float("nan")
        points = d["points"]
    values = [iris_diameter_px(p, size) for p in points if np.isfinite(p).all()]
    values = [v for v in values if not np.isnan(v)]
    return float(np.median(values)) if values else float("nan")


def screen_layout_from_meta(meta: Mapping[str, object]) -> dict[str, object]:
    """No layout claim: the arrangement of displays was not recorded (an external monitor may have been
    attached). ``displays`` stays empty, so the validator warns; the screen the recording was made on is kept
    as ``recorded_screen`` (``meta["screen"]``) for reference only."""
    s = meta.get("screen")
    return {"displays": [], "recorded_screen": dict(s) if isinstance(s, Mapping) else None,
            "note": "backfilled: display arrangement was not recorded"}


def backfill_block(meta: Mapping[str, object], npz: Path, *, person_id: str, camera_name: str,
                   glasses: str = "unknown") -> dict[str, object]:
    """The sidecar block for one recording (``backfilled`` true: it can only warn)."""
    source, source_meta = npz, meta
    if meta.get("kind") == "cursor_session":
        # A Cursor log has no landmarks or frame size: borrow from the FIRST calibration it ran on, the same
        # rule the live recorder uses, and record which file the measurement came from.
        calibrations = meta.get("calibrations") or []
        first = calibrations[0][1] if calibrations and len(calibrations[0]) > 1 else meta.get("calibration", "")
        source = npz.with_name(str(first))
        try:
            source_meta = read_meta(source) if source.is_file() else {}
        except (OSError, ValueError, KeyError):
            source_meta = {}
    try:
        iris = median_iris_px(source) if source.is_file() else float("nan")
    except (OSError, ValueError, KeyError):
        iris = float("nan")
    size = tuple(source_meta.get("image_size") or ()) or None
    cam_meta = {"name": camera_name, "built_in": None, "available": []}
    if size:
        cam_meta.update(width=int(size[0]), height=int(size[1]))
    screen = meta.get("screen") if isinstance(meta.get("screen"), Mapping) else {}
    light = source_meta.get("lighting")
    block = build_conditions(
        person_id=person_id, camera=cam_meta, screen_layout=screen_layout_from_meta(meta),
        assumed_distance_cm=float(screen.get("distance_cm", 60.0)), iris_px=iris,
        image_width=int(size[0]) if size else DEFAULT_IMAGE_SIZE[0], glasses=glasses,
        lighting_measured=light if isinstance(light, Mapping) else None, backfilled=True)
    if source != npz:
        block["distance"]["measured_in"] = source.name
        block["lighting_measured_in"] = source.name
    if not size:      # the pixel size was not recorded: the distance used the default, say so
        block["distance"]["image_size_assumed"] = list(DEFAULT_IMAGE_SIZE)
    block["sources"] = {"camera.name": "author flag, same for every backfilled recording",
                        "glasses": "author flag, same for every backfilled recording" if glasses != "unknown" else None}
    block["lighting_tag"] = None        # not recorded at the time: left absent so the validator warns
    block["git_hash"] = None
    return block


def write_sidecar(npz: Path, block: Mapping[str, object]) -> Path:
    """Write ``X.conditions.json`` next to the recording (never overwrites one that exists)."""
    path = sidecar_path(npz)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(block, indent=2))
    return path


def recordings(root: Path) -> list[Path]:
    """Session and Cursor-log files under ``root`` (face-image sidecars and temp saves excluded)."""
    out = []
    for pattern in ("session_*.npz", "cursor_*.npz"):
        out += [p for p in root.glob(pattern) if not p.name.endswith(".faces.npz") and not p.name.startswith(".")]
    return sorted(out)
