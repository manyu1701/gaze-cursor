"""Scripts must import the package, not themselves.

Python puts a script's own folder first on sys.path, so a script named like a package (e.g.
scripts/gaze_cursor.py) shadows it: ``import gaze_cursor.capture`` then fails with "'gaze_cursor' is not a
package". Tests that load scripts via importlib under another name do not see this, so check it directly.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {p.name for p in (ROOT / "src").iterdir() if (p / "__init__.py").exists()}


def test_no_script_is_named_like_a_package() -> None:
    clash = sorted(p.name for p in (ROOT / "scripts").glob("*.py") if p.stem in PACKAGES)
    assert not clash, f"rename these scripts, they shadow the package: {clash}"


@pytest.mark.parametrize("script", ["cursor_mode.py", "gaze_overlay.py", "target_practice.py", "evaluate_online.py"])
def test_script_starts_as_a_real_script(script: str) -> None:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage" in r.stdout.lower()
