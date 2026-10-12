import numpy as np
import pytest

from gaze_cursor.landmarks import head_pose_from_matrix


def _rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = np.radians([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def test_identity() -> None:
    assert head_pose_from_matrix(np.eye(4)) == pytest.approx((0, 0, 0))


@pytest.mark.parametrize("angles", [(30, 0, 0), (0, -20, 0), (0, 0, 15), (25, -10, 5)])
def test_round_trip(angles: tuple[float, float, float]) -> None:
    m = np.eye(4)
    m[:3, :3] = _rot(*angles)
    m[:3, 3] = [1, 2, -40]  # translation must be ignored
    assert head_pose_from_matrix(m) == pytest.approx(angles, abs=1e-6)


def test_scale_removed() -> None:
    m = np.eye(4)
    m[:3, :3] = 1.7 * _rot(10, 5, -3)
    assert head_pose_from_matrix(m) == pytest.approx((10, 5, -3), abs=1e-6)
