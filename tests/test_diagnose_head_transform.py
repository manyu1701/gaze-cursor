"""The head-transform check recovers a planted head effect and maps the transform's axes."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnose_head_transform", ROOT / "scripts" / "diagnose_head_transform.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CM_PER_PX = 1 / 50.6


def synthetic(scale: float, n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Screen = iris mapping + ``scale`` x the geometric head effect (x: -1/cm_per_px, y: +1/cm_per_px)."""
    rng = np.random.default_rng(seed)
    f = np.zeros((n, 12))
    f[:, 0:4] = rng.normal(size=(n, 4))
    f[:, 4:6] = rng.normal(size=(n, 2))
    head = rng.normal(scale=5, size=(n, 2))
    k = 1 / CM_PER_PX
    screen = np.column_stack([300 * f[:, 0] - scale * k * head[:, 0], 200 * f[:, 1] + scale * k * head[:, 1]])
    return f, head, screen + rng.normal(scale=5, size=screen.shape)


@pytest.mark.parametrize("scale", [1.0, 0.1])
def test_head_ratios_recover_the_planted_scale(scale: float) -> None:
    f, head, screen = synthetic(scale)
    ratios, rms = mod.head_ratios(f, head, screen, CM_PER_PX)
    assert ratios == pytest.approx([scale, scale], abs=0.02) and rms < 10
    assert mod.within(ratios) == (scale == 1.0)


def test_axis_signs_flip_an_inverted_axis() -> None:
    rng = np.random.default_rng(1)
    corner = rng.normal(size=(500, 2))
    signs, r = mod.axis_signs(corner * np.array([2.0, -3.0]), corner)
    assert list(signs) == [1.0, -1.0] and r[0] > 0.99 and r[1] < -0.99


def test_transform_translation_is_the_last_column() -> None:
    m = np.tile(np.eye(4), (3, 1, 1))
    m[:, :3, 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert mod.transform_translation(m).tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
