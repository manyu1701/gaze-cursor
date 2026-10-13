"""Condition tags of a recorded session (the L1 data contract). Pure: numpy and the standard library only.

Every recorder writes one versioned ``meta["conditions"]`` block so sessions can be pooled across days,
setups and people later (CLAUDE.md "Learning loop and data strategy"). ``validate`` is the one shared check
used by ``session_report`` check 6. ``screen`` and ``lighting`` stay where they are in ``meta``; this block
adds what they do not say (who, which camera, which layout, which model).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
IRIS_DIAMETER_CM = 1.17       # human iris, ~11.7 mm (CLAUDE.md "Generalisation readiness")
DEFAULT_FOV_DEG = 65.0        # ASSUMED horizontal field of view of a laptop webcam; stored with every estimate
LIGHTING_TAGS = ("day", "lamp", "dark", "backlit", "unknown")
GLASSES_TAGS = ("yes", "no", "unknown")
PERSON_ID_PATTERN = re.compile(r"^p\d{2,}$")
PERSON_ENV = "GAZE_PERSON_ID"
PERSON_FILE = Path.home() / ".gaze-cursor" / "person.json"   # outside the repo: personal

# MediaPipe iris ring landmarks (centre excluded): opposite points are 0-2 and 1-3.
RIGHT_IRIS_RING = (469, 470, 471, 472)
LEFT_IRIS_RING = (474, 475, 476, 477)

PASS, WARN, FAIL = "pass", "warn", "fail"
N_CHECKED = 9    # fields ``validate`` can flag: person_id, camera, camera.resolution, screen_layout, distance,
                 # lighting_tag, glasses, posture_note, git_hash


@dataclass(frozen=True)
class Issue:
    """One finding of ``validate``: the field, its severity (``warn`` or ``fail``) and a short reason."""

    field: str
    severity: str
    message: str


def iris_diameter_px(points: np.ndarray, image_size: tuple[int, int]) -> float:
    """Mean iris diameter in pixels of one frame (both eyes, both ring diameters); NaN without a full ring.

    ``points`` are normalised landmarks ``(N, >=2)``. Image x and y are scaled separately, so a
    non-square frame is handled.
    """
    if points.shape[0] <= max(LEFT_IRIS_RING):
        return float("nan")
    scale = np.array(image_size[:2], dtype=float)
    diameters = []
    for ring in (RIGHT_IRIS_RING, LEFT_IRIS_RING):
        for a, b in ((ring[0], ring[2]), (ring[1], ring[3])):
            diameters.append(float(np.linalg.norm((points[a, :2] - points[b, :2]) * scale)))
    d = float(np.mean(diameters))
    return d if d > 0 else float("nan")


def estimate_distance_cm(iris_px: float, image_width: int, fov_deg: float = DEFAULT_FOV_DEG) -> float:
    """Eye-to-camera distance from the iris size: ``f_px * iris_cm / iris_px``, pinhole camera.

    ``f_px = (width / 2) / tan(fov / 2)``. The field of view is an assumption (the camera is not
    calibrated), so the result is an estimate; ``iris_px`` is stored next to it so it can be recomputed.
    """
    if not (iris_px > 0) or image_width <= 0 or not (0 < fov_deg < 180):
        return float("nan")
    f_px = (image_width / 2) / math.tan(math.radians(fov_deg) / 2)
    return f_px * IRIS_DIAMETER_CM / iris_px


def model_version(mean: np.ndarray, scale: np.ndarray, theta: np.ndarray, P: np.ndarray) -> str:
    """Short hash of a model's starting state (its bytes), the link from a session to the model that began it.

    Identical arrays give an identical id; any change gives a different one. The state after later
    clicks is not hashed: it is rebuilt by replaying the logged clicks.
    """
    h = hashlib.sha256()
    for a in (mean, scale, theta, P):
        arr = np.ascontiguousarray(a, dtype="<f8")
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def resolve_person_id(flag: str | None, environ: Mapping[str, str], config_path: Path = PERSON_FILE) -> str | None:
    """The person id: ``--person`` flag, else ``GAZE_PERSON_ID``, else the saved file; ``None`` if none is set."""
    for candidate in (flag, environ.get(PERSON_ENV)):
        if candidate and candidate.strip():
            return candidate.strip()
    try:
        value = json.loads(config_path.read_text()).get("person_id")
    except (OSError, ValueError, AttributeError):
        return None
    return str(value).strip() if value and str(value).strip() else None


def save_person_id(person_id: str, config_path: Path = PERSON_FILE) -> None:
    """Remember the person id for the next recording (local file, outside the repo)."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"person_id": person_id.strip()}))


def build_conditions(
    *,
    person_id: str,
    camera: Mapping[str, object],
    screen_layout: Mapping[str, object],
    assumed_distance_cm: float,
    iris_px: float,
    image_width: int,
    lighting_tag: str = "unknown",
    lighting_measured: Mapping[str, object] | None = None,
    glasses: str = "unknown",
    posture_note: str = "",
    git_hash: str = "unknown",
    model_version_id: str | None = None,
    fov_deg: float = DEFAULT_FOV_DEG,
    backfilled: bool = False,
) -> dict[str, object]:
    """Assemble the ``meta["conditions"]`` block (plain JSON types). Measured values sit beside their tags."""
    est = estimate_distance_cm(iris_px, image_width, fov_deg)
    return {
        "schema_version": SCHEMA_VERSION,
        "backfilled": bool(backfilled),
        "person_id": person_id,
        "camera": dict(camera),
        "screen_layout": dict(screen_layout),
        "distance": {
            "assumed_cm": float(assumed_distance_cm),
            "iris_px": None if math.isnan(iris_px) else float(iris_px),
            "est_cm": None if math.isnan(est) else float(est),
            "fov_deg_assumed": float(fov_deg),
        },
        "lighting_tag": lighting_tag,
        "lighting_measured": dict(lighting_measured) if lighting_measured else None,
        "glasses": glasses,
        "posture_note": posture_note,
        "git_hash": git_hash,
        "model_version": model_version_id,
    }


def validate(block: Mapping[str, object] | None) -> list[Issue]:
    """Check a conditions block; empty list = all good.

    Severity rule (decided 2026-10-14): a missing ``person_id`` **fails** a session written with
    ``schema_version >= 1`` and ``backfilled`` exactly ``False`` (what the writers set); everything else that is
    missing only **warns**. Anything else is treated as legacy and can never fail: no block, a missing, bool,
    string or < 1 ``schema_version``, or ``backfilled`` true / absent / not a real boolean. Their tags were not
    recorded at the time.
    """
    if not block:
        return [Issue("conditions", WARN, "no conditions block (legacy session)")]
    version = block.get("schema_version")
    legacy = (not isinstance(version, int) or isinstance(version, bool) or version < 1
              or block.get("backfilled") is not False)
    issues: list[Issue] = []

    def need(field: str, ok: bool, message: str, severity: str = WARN) -> None:
        if not ok:
            issues.append(Issue(field, WARN if legacy else severity, message))

    if not legacy and version > SCHEMA_VERSION:
        issues.append(Issue("schema_version", WARN, f"newer than this code knows ({version} > {SCHEMA_VERSION})"))
    pid = block.get("person_id")
    need("person_id", isinstance(pid, str) and bool(pid.strip()), "person_id missing", FAIL)
    if isinstance(pid, str) and pid.strip() and not PERSON_ID_PATTERN.match(pid):
        issues.append(Issue("person_id", WARN, f"'{pid}' does not look like p01, p02, ..."))
    camera = block.get("camera")
    cam = camera if isinstance(camera, Mapping) else {}
    need("camera", bool(cam.get("name")) and cam.get("name") != "unknown", "camera name missing or unknown")
    need("camera.resolution", bool(cam.get("width")) and bool(cam.get("height")), "camera resolution missing")
    layout = block.get("screen_layout")
    displays = layout.get("displays") if isinstance(layout, Mapping) else None
    need("screen_layout", isinstance(displays, list) and len(displays) > 0, "screen layout (displays) missing")
    dist = block.get("distance")
    d = dist if isinstance(dist, Mapping) else {}
    need("distance", d.get("est_cm") is not None, "distance estimate (iris size) missing")
    need("lighting_tag", block.get("lighting_tag") in LIGHTING_TAGS[:-1],
         f"lighting tag missing or unknown (one of {', '.join(LIGHTING_TAGS[:-1])})")
    need("glasses", block.get("glasses") in GLASSES_TAGS[:-1], "glasses missing or unknown (yes or no)")
    need("posture_note", isinstance(block.get("posture_note"), str) and bool(str(block.get("posture_note")).strip()),
         "posture note missing")
    need("git_hash", bool(block.get("git_hash")) and block.get("git_hash") != "unknown", "code git hash missing")
    return issues


def verdict(issues: list[Issue]) -> str:
    """Overall result of ``validate``: ``fail`` if any issue fails, else ``warn`` if any, else ``pass``."""
    if any(i.severity == FAIL for i in issues):
        return FAIL
    return WARN if issues else PASS


def sidecar_path(npz: Path) -> Path:
    """Where the backfilled block of ``session_X.npz`` lives: ``session_X.conditions.json`` next to it."""
    return npz.with_name(npz.stem + ".conditions.json")


def read_meta(npz: Path) -> dict:
    """The ``meta`` dict of a session or Cursor log, read without loading its arrays."""
    with np.load(npz, allow_pickle=False) as d:
        return json.loads(str(d["meta"]))


def conditions_block(meta: Mapping[str, object], npz: Path | None = None) -> dict | None:
    """The conditions block of a recording: its own ``meta["conditions"]``, else a backfilled sidecar, else None."""
    own = meta.get("conditions")
    if isinstance(own, dict) and own:
        return own
    if npz is not None:
        try:
            side = json.loads(sidecar_path(npz).read_text())
        except (OSError, ValueError):
            return None
        # a sidecar is a backfill by definition, whatever it says about itself: it can warn, never fail
        return {**side, "backfilled": True} if isinstance(side, dict) and side else None
    return None


def tags_present(issues: list[Issue]) -> int:
    """How many of the ``N_CHECKED`` fields are fine (all of them missing for a recording with no block)."""
    if any(i.field == "conditions" for i in issues):
        return 0
    flagged = {i.field for i in issues} - {"schema_version"}
    return max(0, N_CHECKED - len(flagged))
