import numpy as np
import pytest

from gaze_cursor.features import (
    FEATURE_NAMES, LEFT_EYE, MIN_OPENNESS, RIGHT_EYE, eye_features, extract_features,
)

SIZE = (1000, 1000)  # square so normalized == pixels


def _face(iris_dx: float = 0.0, iris_dy: float = 0.0, gap: float = 0.03) -> np.ndarray:
    """Synthetic landmarks: each eye 0.1 wide, centered at x=0.35 / 0.65, y=0.5."""
    pts = np.zeros((478, 3), np.float32)
    for (a, b, up, lo, iris), cx in ((RIGHT_EYE, 0.35), (LEFT_EYE, 0.65)):
        pts[a] = (cx - 0.05, 0.5, 0)
        pts[b] = (cx + 0.05, 0.5, 0)
        pts[up] = (cx, 0.5 - gap / 2, 0)
        pts[lo] = (cx, 0.5 + gap / 2, 0)
        pts[iris] = (cx + iris_dx, 0.5 + iris_dy, 0)
    return pts


def test_centered_iris_is_zero() -> None:
    x, y, _ = eye_features(_face(), RIGHT_EYE, SIZE)
    assert (x, y) == pytest.approx((0, 0), abs=1e-6)


def test_iris_offset_normalized_by_eye_width() -> None:
    x, y, _ = eye_features(_face(iris_dx=0.02, iris_dy=0.01), LEFT_EYE, SIZE)
    assert (x, y) == pytest.approx((0.2, 0.1), abs=1e-6)  # 0.02 / 0.1 and 0.01 / 0.1


def test_openness_is_gap_over_width() -> None:
    assert eye_features(_face(gap=0.03), RIGHT_EYE, SIZE)[2] == pytest.approx(0.3)


def test_corner_order_does_not_flip_sign() -> None:
    pts = _face(iris_dx=0.02)
    a, b = RIGHT_EYE[0], RIGHT_EYE[1]
    pts[[a, b]] = pts[[b, a]]
    assert eye_features(pts, RIGHT_EYE, SIZE)[0] == pytest.approx(0.2, abs=1e-6)


def test_extract_features_vector_and_blink() -> None:
    f = extract_features(_face(), np.eye(4), SIZE)
    assert f is not None and f.shape == (len(FEATURE_NAMES),)
    assert extract_features(_face(gap=0.1 * MIN_OPENNESS * 0.5), np.eye(4), SIZE) is None


def test_blend_features_selected_by_name_in_fixed_order() -> None:
    from gaze_cursor.features import BLEND_FEATURE_NAMES, blend_features

    names = tuple(reversed(BLEND_FEATURE_NAMES)) + ("_neutral", "jawOpen")  # arbitrary model order
    scores = np.arange(len(names), dtype=float)
    out = blend_features(scores, names)
    assert out.shape == (len(BLEND_FEATURE_NAMES),)
    assert out[0] == names.index(BLEND_FEATURE_NAMES[0])


def test_assemble_feature_sets() -> None:
    from gaze_cursor.features import BLEND_FEATURE_NAMES, assemble

    base = np.arange(9, dtype=float)[None, :]
    blend = np.zeros((1, len(BLEND_FEATURE_NAMES)))
    nb = len(BLEND_FEATURE_NAMES)
    assert assemble(base, blend, BLEND_FEATURE_NAMES, "iris").shape == (1, 9)
    assert assemble(base, blend, BLEND_FEATURE_NAMES, "blend").shape == (1, 3 + nb)
    assert assemble(base, blend, BLEND_FEATURE_NAMES, "both").shape == (1, 9 + nb)
    assert assemble(base, blend, BLEND_FEATURE_NAMES, "blend")[0, :3].tolist() == [4, 5, 6]
    with pytest.raises(ValueError):
        assemble(base, blend, BLEND_FEATURE_NAMES, "nope")


def test_pose_free_feature_sets() -> None:
    from gaze_cursor.features import BLEND_FEATURE_NAMES, assemble

    base = np.arange(9, dtype=float)[None, :]
    blend = np.zeros((1, len(BLEND_FEATURE_NAMES)))
    assert assemble(base, blend, BLEND_FEATURE_NAMES, "eyes")[0].tolist() == [0, 1, 2, 3, 7, 8]
    assert assemble(base, blend, BLEND_FEATURE_NAMES, "gaze")[0].tolist() == [0, 1, 2, 3]
