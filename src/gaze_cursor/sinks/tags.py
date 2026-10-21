"""Condition tags for the recorders (L1 data contract, stage 4): flags, environment, last answers, prompts.

Shared by ``calibrate``, Run's quick calibration and Cursor mode so one recording kind cannot drift from
another. Sources, first match wins: command-line flag, environment variable (the dashboard launch form
sets these), last answer (``~/.gaze-cursor/tags.json``, outside the repo), a prompt at the terminal. A
missing person id refuses to record (decided 2026-10-14); the other tags fall back to "unknown", which
``conditions.validate`` warns about.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gaze_cursor import conditions as cond
from gaze_cursor.config import ScreenConfig
from gaze_cursor.sinks import platform_info

TAGS_FILE = Path.home() / ".gaze-cursor" / "tags.json"
AUTO_FIELDS = {"lighting": "auto", "posture_note": ""}
ENV = {"lighting": "GAZE_LIGHTING_TAG", "glasses": "GAZE_GLASSES", "posture_note": "GAZE_POSTURE_NOTE"}


@dataclass(frozen=True)
class Tags:
    """What the person tells us about this recording (the rest is measured or read from the OS)."""

    person_id: str
    lighting: str = "auto"      # "auto": the sensor's label after calibration; a valid tag = typed by a person
    glasses: str = "unknown"
    posture_note: str = ""


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """The shared tag flags (all optional: missing ones come from the environment, last answers or a prompt)."""
    g = ap.add_argument_group("condition tags (stored with the recording)")
    g.add_argument("--person", help="person id, e.g. p01 (also GAZE_PERSON_ID or ~/.gaze-cursor/person.json)")
    g.add_argument("--lighting", choices=("day", "lamp", "dark", "backlit"), help="override the camera's own lighting label: day, lamp, dark or backlit "
                                                                    "(left out: taken from the camera)")
    g.add_argument("--glasses", choices=cond.GLASSES_TAGS[:-1], help="wearing glasses: yes or no")
    g.add_argument("--posture", dest="posture_note", help='posture note, e.g. "upright, laptop on desk"')


def _last_answers(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {k: str(v) for k, v in data.items() if isinstance(v, str)} if isinstance(data, dict) else {}


def resolve(args: argparse.Namespace, environ: Mapping[str, str] | None = None, *,
            interactive: bool | None = None, ask: Callable[[str], str] = input,
            tags_path: Path = TAGS_FILE, person_path: Path = cond.PERSON_FILE) -> Tags:
    """Collect the tags. Raises ``SystemExit`` if no person id can be found (not even by prompting)."""
    environ = os.environ if environ is None else environ
    tty = sys.stdin.isatty() if interactive is None else interactive
    last = _last_answers(tags_path)

    def pick(field: str, valid: tuple[str, ...] | None, question: str) -> str:
        value = getattr(args, field, None) or environ.get(ENV[field], "").strip() or ""
        if value and (valid is None or value in valid):
            return value
        if field in AUTO_FIELDS:      # measured from the camera after calibration: never asked, never reused
            return AUTO_FIELDS[field]
        default = last.get(field, "")
        if default and not tty:
            return default
        if not tty:
            return "unknown" if valid else ""
        while True:
            options = f" ({'/'.join(valid)})" if valid else ""
            answer = ask(f"{question}{options}{f' [{default}]' if default else ''}: ").strip() or default
            if valid is None or answer in valid:
                return answer
            print(f"  please answer one of: {', '.join(valid)}")

    person = cond.resolve_person_id(getattr(args, "person", None), environ, person_path)
    if not person and tty:
        while True:
            person = ask("Person id for this recording (p01, p02, ...; saved for next time): ").strip()
            if cond.PERSON_ID_PATTERN.match(person):
                cond.save_person_id(person, person_path)
                break
            print("  use the form p01, p02, ...")
    if not person:
        raise SystemExit("No person id: pass --person p01, set GAZE_PERSON_ID, or run once in a terminal to be asked. "
                         "Recordings without one cannot be pooled, so nothing is recorded.")
    tags = Tags(person, pick("lighting", cond.LIGHTING_TAGS[:-1], "Lighting"),
                pick("glasses", cond.GLASSES_TAGS[:-1], "Glasses"),
                pick("posture_note", None, "Posture note (one short phrase)"))
    try:
        tags_path.parent.mkdir(parents=True, exist_ok=True)
        tags_path.write_text(json.dumps({k: v for k, v in asdict(tags).items()
                                         if k not in ("person_id", "lighting", "posture_note") and v}))
    except OSError:
        pass
    return tags


def conditions_for(tags: Tags, cfg: ScreenConfig, *, frame_size: tuple[int, int] | None,
                   iris_px: float, lighting_measured: Mapping[str, object] | None = None,
                   model_version_id: str | None = None,
                   posture: Mapping[str, float] | None = None) -> dict[str, object]:
    """The ``meta["conditions"]`` block for a recording made now.

    Lighting not typed by a person comes from the camera's own label (``lighting_measured["label"]``); a posture
    note not typed becomes the measured head pose and face distance (``posture``). The source of each is stored.
    """
    width = frame_size[0] if frame_size else 0
    lighting_tag, lighting_source = tags.lighting, "typed"
    if lighting_tag not in cond.LIGHTING_TAGS[:-1]:
        lighting_tag = cond.SENSOR_LIGHTING.get(str((lighting_measured or {}).get("label", "")), "unknown")
        lighting_source = "sensor"
    note, note_source = tags.posture_note, "typed" if tags.posture_note.strip() else "none"
    if not note.strip() and posture:
        note, note_source = posture_text(posture), "measured"
    return cond.build_conditions(
        person_id=tags.person_id, camera=platform_info.camera_info(frame_size),
        screen_layout=platform_info.screen_layout(), assumed_distance_cm=cfg.distance_cm,
        iris_px=iris_px, image_width=width, lighting_tag=lighting_tag, lighting_measured=lighting_measured,
        glasses=tags.glasses, posture_note=note, git_hash=platform_info.git_hash(),
        model_version_id=model_version_id, lighting_tag_source=lighting_source,
        posture_measured=posture, posture_note_source=note_source)


def posture_center(per_frame: Sequence[Mapping[str, float]]) -> dict[str, float] | None:
    """Median of each head-pose metric over the recorded frames (NaN ignored); None if there are none."""
    out: dict[str, float] = {}
    for key in ("yaw", "pitch", "roll", "distance", "x", "y"):
        vals = np.array([f[key] for f in per_frame if key in f], float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[key] = float(np.median(vals))
    return out or None


def posture_text(p: Mapping[str, float]) -> str:
    """The measured posture as a short phrase for the posture-note field."""
    nan = float("nan")
    return (f"measured: yaw {p.get('yaw', nan):+.0f}, pitch {p.get('pitch', nan):+.0f}, "
            f"roll {p.get('roll', nan):+.0f} deg; face {p.get('distance', nan):.2f} of frame width")


def median_iris_px(per_frame: list[float]) -> float:
    """Median of the per-frame iris diameters, ignoring NaN; NaN if there are none."""
    arr = np.array(per_frame, float)
    arr = arr[~np.isnan(arr)]
    return float(np.median(arr)) if len(arr) else float("nan")


def print_tag_verdict(block: Mapping[str, object] | None) -> str:
    """Print the tag check (the same validator as ``session_report`` check 6); returns pass / warn / fail."""
    issues = cond.validate(dict(block) if block else None)
    result = cond.verdict(issues)
    print(f"condition tags: {result.upper()} ({cond.tags_present(issues)}/{cond.N_CHECKED} fields)")
    for i in issues:
        print(f"  {i.severity}: {i.field}: {i.message}")
    return result


def run_report(path: Path) -> None:
    """Run the session quality report on a fresh recording and show its verdict. Never raises, never blocks."""
    import subprocess
    script = Path(__file__).resolve().parents[3] / "scripts" / "session_report.py"
    try:
        out = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True, timeout=180)
        print(out.stdout.strip() or out.stderr.strip()[-400:])
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"(quality report skipped: {exc})")
