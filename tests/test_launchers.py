import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ("Calibrate.command", "Run.command", "Practice.command")


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_is_executable_and_well_formed(name: str) -> None:
    path = ROOT / name
    text = path.read_text()
    assert os.stat(path).st_mode & stat.S_IXUSR, "double-click needs the executable bit"
    assert text.startswith("#!/bin/bash")
    assert 'cd "$(dirname "$0")"' in text                      # works from Finder, any current directory
    assert ".local/bin" in text and "command -v uv" in text    # uv found, friendly message if not
    assert "Press Enter to close" in text                      # a failure stays readable
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0


def test_calibrate_launcher_runs_the_guided_calibration_with_a_note() -> None:
    text = (ROOT / "Calibrate.command").read_text()
    assert "scripts/calibrate.py --guided" in text and '--note "$NOTE"' in text


def test_run_launcher_starts_the_overlay_without_needing_a_full_calibration() -> None:
    text = (ROOT / "Run.command").read_text()
    assert "scripts/gaze_overlay.py" in text and "No calibration found yet" not in text   # quick 9 dots first


def test_practice_launcher_starts_target_practice() -> None:
    text = (ROOT / "Practice.command").read_text()
    assert "scripts/target_practice.py" in text and "No calibration found yet" in text
