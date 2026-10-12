"""Gaze features from face landmarks. Pure functions."""

from __future__ import annotations

import numpy as np

from gaze_cursor.landmarks import head_pose_from_matrix

FEATURE_NAMES = (
    "r_iris_x", "r_iris_y", "l_iris_x", "l_iris_y",
    "yaw", "pitch", "roll", "r_open", "l_open",
)
N_FEATURES = len(FEATURE_NAMES)

# MediaPipe indices: (corner_a, corner_b, upper_lid, lower_lid, iris_center).
RIGHT_EYE = (33, 133, 159, 145, 468)
LEFT_EYE = (362, 263, 386, 374, 473)

# Blendshape scores used as candidate features (names as emitted by FaceLandmarker).
BLEND_FEATURE_NAMES = tuple(
    f"eye{kind}{side}"
    for kind in ("LookIn", "LookOut", "LookUp", "LookDown", "Blink", "Squint", "Wide")
    for side in ("Left", "Right")
)
FEATURE_SETS = ("iris", "blend", "both")

MIN_OPENNESS = 0.15  # below this the eye is treated as closed (blink)


def eye_features(
    points: np.ndarray, eye: tuple[int, int, int, int, int], image_size: tuple[int, int]
) -> tuple[float, float, float]:
    """Return (iris_x, iris_y, openness) for one eye.

    Iris offset is measured from the eye-corner midpoint along the eye axis (x, positive
    toward image-right) and perpendicular to it (y, positive toward image-down), both divided
    by eye width. Openness is the lid gap divided by eye width.
    """
    a, b, up, lo, iris = eye
    scale = np.array(image_size, dtype=float)  # (w, h): make x/y isotropic
    pa, pb, pu, pl, pi = (points[i, :2] * scale for i in (a, b, up, lo, iris))
    if pa[0] > pb[0]:
        pa, pb = pb, pa
    axis = pb - pa
    width = float(np.linalg.norm(axis))
    u = axis / width
    v = np.array([-u[1], u[0]])  # points image-down when u points image-right
    offset = pi - (pa + pb) / 2
    return float(offset @ u / width), float(offset @ v / width), float(np.linalg.norm(pu - pl) / width)


def extract_features(
    points: np.ndarray, transform: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray | None:
    """Return the feature vector (see FEATURE_NAMES), or None if an eye is closed.

    ``image_size`` is (width, height) in pixels of the frame the landmarks came from.
    """
    rx, ry, r_open = eye_features(points, RIGHT_EYE, image_size)
    lx, ly, l_open = eye_features(points, LEFT_EYE, image_size)
    if min(r_open, l_open) < MIN_OPENNESS:
        return None
    yaw, pitch, roll = head_pose_from_matrix(transform)
    return np.array([rx, ry, lx, ly, yaw, pitch, roll, r_open, l_open])


def blend_features(blendshapes: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Pick BLEND_FEATURE_NAMES out of a (..., 52) blendshape array, in a fixed order."""
    cols = [names.index(n) for n in BLEND_FEATURE_NAMES]
    return np.asarray(blendshapes)[..., cols]


def assemble(
    features: np.ndarray, blendshapes: np.ndarray, names: tuple[str, ...], feature_set: str
) -> np.ndarray:
    """Build the model input matrix for a feature set from base features and blendshapes.

    - ``iris``: the 9 base features (iris position, head pose, eye openness).
    - ``blend``: head pose (yaw/pitch/roll) + the blendshape subset, no iris geometry.
    - ``both``: iris + blendshape subset.
    """
    if feature_set == "iris":
        return features
    blend = blend_features(blendshapes, names)
    if feature_set == "blend":
        return np.concatenate([features[..., 4:7], blend], axis=-1)
    if feature_set == "both":
        return np.concatenate([features, blend], axis=-1)
    raise ValueError(f"unknown feature set {feature_set!r}; choose from {FEATURE_SETS}")
