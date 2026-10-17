"""What the OS says about this setup: camera, screen layout, code version (L1 data contract, stage 3).

OS-specific code stays here (CLAUDE.md "Generalisation readiness"). The parsers are pure so they can be
tested without a Mac; the thin wrappers call ``system_profiler`` / AppKit / git and fall back to
"unknown" rather than failing a recording.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

BUILT_IN_HINTS = ("macbook", "built-in", "facetime", "imac", "studio display")
CONTINUITY_HINTS = ("iphone", "ipad")


def parse_camera_profile(profile: dict) -> dict[str, object]:
    """Camera facts from ``system_profiler SPCameraDataType -json`` output.

    The recorder opens camera index 0, which on macOS is the built-in camera unless an external one is
    preferred, so the name is the first camera that is not an iPhone / iPad (Continuity Camera). All names
    are kept in ``available``. ``built_in`` is True / False when the name makes it clear, else None.
    """
    entries = [str(c.get("_name", "")).strip() for c in profile.get("SPCameraDataType", []) if isinstance(c, dict)]
    entries = [n for n in entries if n]
    local = [n for n in entries if not any(h in n.lower() for h in CONTINUITY_HINTS)]
    name = (local or entries or ["unknown"])[0]
    built_in = True if any(h in name.lower() for h in BUILT_IN_HINTS) else (None if name == "unknown" else False)
    return {"name": name, "built_in": built_in, "available": entries}


def camera_info(frame_size: tuple[int, int] | None = None) -> dict[str, object]:
    """Camera name, built-in or external, plus the measured resolution ``(width, height)`` if known."""
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True,
                             timeout=10, check=True).stdout
        info = parse_camera_profile(json.loads(out))
    except (OSError, ValueError, subprocess.SubprocessError):
        info = {"name": "unknown", "built_in": None, "available": []}
    if frame_size:
        info["width"], info["height"] = int(frame_size[0]), int(frame_size[1])
    return info


def display_record(frame: tuple[float, float, float, float], size_mm: tuple[float, float], name: str,
                   built_in: bool, main: bool) -> dict[str, object]:
    """One display as a plain dict: origin and size in points, physical size in cm (EDID millimetres)."""
    x, y, w, h = frame
    return {"name": name, "origin_pt": [x, y], "size_pt": [w, h],
            "size_cm": [round(size_mm[0] / 10, 1), round(size_mm[1] / 10, 1)],
            "built_in": built_in, "main": main}


def screen_layout() -> dict[str, object]:
    """All displays and how macOS arranges them (points, cm); ``{"displays": []}`` if AppKit is missing."""
    try:
        import AppKit
        import Quartz
    except ImportError:
        return {"displays": []}
    screens = list(AppKit.NSScreen.screens())
    main_id = Quartz.CGMainDisplayID()
    displays = []
    for s in screens:
        did = s.deviceDescription()["NSScreenNumber"]
        f = s.frame()
        mm = Quartz.CGDisplayScreenSize(did)
        name = str(s.localizedName()) if hasattr(s, "localizedName") else f"display {did}"
        displays.append(display_record((f.origin.x, f.origin.y, f.size.width, f.size.height),
                                       (mm.width, mm.height), name, bool(Quartz.CGDisplayIsBuiltin(did)),
                                       did == main_id))
    return {"displays": displays}


def git_hash(repo: Path | None = None) -> str:
    """Short git hash of the code (``+dirty`` if the tree has uncommitted changes); "unknown" outside git."""
    cwd = repo or Path(__file__).resolve().parent

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, cwd=cwd,
                              check=True).stdout.strip()
    try:
        h = run("rev-parse", "--short", "HEAD")
        # data/ is ignored: a results row is not a code change
        return h + ("+dirty" if run("status", "--porcelain", "--", ":(top)", ":(top,exclude)data") else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
