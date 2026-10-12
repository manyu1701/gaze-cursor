import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ("Calibrate.command", "Run.command")


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


def test_run_launcher_starts_the_overlay_with_the_latest_session() -> None:
    assert "scripts/gaze_overlay.py" in (ROOT / "Run.command").read_text()


def test_run_launcher_explains_when_there_is_no_calibration(tmp_path: Path) -> None:
    shutil.copy(ROOT / "Run.command", tmp_path / "Run.command")          # it uses the folder it lives in
    (tmp_path / "data" / "sessions").mkdir(parents=True)
    (tmp_path / "data" / "sessions" / "session_x.faces.npz").write_bytes(b"")   # an image sidecar is not a calibration
    r = subprocess.run(["bash", str(tmp_path / "Run.command")], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "No calibration found yet" in r.stdout and "Calibrate.command" in r.stdout


def test_window_class_is_created_once_per_process() -> None:
    """Regression: the guided calibration opens two fullscreen canvases in one process, and a second
    ``_CanvasWindow`` class definition raised "overriding existing Objective-C class"."""
    from gaze_cursor.sinks.overlay import _make_window_class

    assert _make_window_class() is _make_window_class()


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="opens (invisible) windows on a real display")
def test_two_canvases_can_be_opened_one_after_the_other_in_one_process() -> None:
    from gaze_cursor.config import ScreenConfig
    from gaze_cursor.sinks.overlay import FullscreenCanvas

    for _ in range(2):                                   # light-check screen, then the calibration screen
        with FullscreenCanvas(ScreenConfig(), alpha=0.0) as canvas:
            assert canvas.show(canvas.blank()) == -1
