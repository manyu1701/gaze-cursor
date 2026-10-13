"""Gaze features from face landmarks. Pure functions."""

from __future__ import annotations

import math

import numpy as np

from gaze_cursor.landmarks import head_pose_from_matrix

FEATURE_NAMES = (
    "r_iris_x", "r_iris_y", "l_iris_x", "l_iris_y",
    "yaw", "pitch", "roll", "r_open", "l_open",
    "head_x", "head_y", "head_z",
)
N_FEATURES = len(FEATURE_NAMES)
N_BASE = 9  # the M1 features (iris, pose, openness); the head geometry follows them

# Eye-midpoint position from the distance between the eye centres (~ the inter-pupil distance, IPD). The adult mean is about 6.3 cm; the
# exact value only rescales the head position (a linear model absorbs it), so it is not per person.
IPD_CM = 6.3
MIN_COS_YAW = 0.5  # cap the yaw correction of the apparent IPD (beyond 60 deg it is meaningless)

# MediaPipe indices: (corner_a, corner_b, upper_lid, lower_lid, iris_center).
RIGHT_EYE = (33, 133, 159, 145, 468)
LEFT_EYE = (362, 263, 386, 374, 473)

# Blendshape scores used as candidate features (names as emitted by FaceLandmarker).
BLEND_FEATURE_NAMES = tuple(
    f"eye{kind}{side}"
    for kind in ("LookIn", "LookOut", "LookUp", "LookDown", "Blink", "Squint", "Wide")
    for side in ("Left", "Right")
)
FEATURE_SETS = ("iris", "blend", "both", "eyes", "gaze", "gaze_blend", "head", "ray")

# Below this the eye is treated as closed. Kept low on purpose: lids narrow naturally when looking
# down (median openness ~0.15 at the bottom of the screen on a laptop webcam), so a higher cutoff
# discards real fixations there. Measured on the first real session; see PROGRESS.md.
MIN_OPENNESS = 0.05


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
    hx, hy, hz = head_geometry(points, yaw, image_size)
    return np.array([rx, ry, lx, ly, yaw, pitch, roll, r_open, l_open, hx, hy, hz])


def head_geometry(points: np.ndarray, yaw_deg: float, image_size: tuple[int, int]) -> tuple[float, float, float]:
    """Eye-midpoint position in camera space from the apparent distance between the eyes.

    Uses the eye centres (midpoints of the eye corners), not the irises: the irises move with gaze,
    the corners only with the head. Pinhole camera, principal point at the image centre, x toward
    image-right, y toward image-down, z away from the camera. With the eye centres ``d_px`` apart
    (corrected for head yaw), depth is ``z = f * IPD / d_px``, so the lateral offsets need no focal
    length:

    - ``head_x``, ``head_y``: offset of the eye midpoint from the optical axis, in cm.
    - ``head_z``: depth in units of ``f / image width`` (``IPD * width / d_px``), i.e. proportional
      to the distance in cm with an unknown camera-specific factor. No camera value is assumed.
    """
    w, h = image_size
    scale_xy = np.array([w, h], dtype=float)
    a = (points[RIGHT_EYE[0], :2] + points[RIGHT_EYE[1], :2]) / 2 * scale_xy
    b = (points[LEFT_EYE[0], :2] + points[LEFT_EYE[1], :2]) / 2 * scale_xy
    d_px = float(np.linalg.norm(a - b)) / max(math.cos(math.radians(yaw_deg)), MIN_COS_YAW)
    u, v = (a + b) / 2
    scale = IPD_CM / d_px
    return (u - w / 2) * scale, (v - h / 2) * scale, w * scale


def blend_features(blendshapes: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Pick BLEND_FEATURE_NAMES out of a (..., 52) blendshape array, in a fixed order."""
    cols = [names.index(n) for n in BLEND_FEATURE_NAMES]
    return np.asarray(blendshapes)[..., cols]


def assemble(
    features: np.ndarray, blendshapes: np.ndarray, names: tuple[str, ...], feature_set: str
) -> np.ndarray:
    """Build the model input matrix for a feature set from base features and blendshapes.

    - ``iris``: the 9 base features (iris position, head pose, eye openness).
    - ``eyes``: iris position + eye openness, no head pose.
    - ``gaze``: iris position only (4), no head pose and no openness. Robust to posture changes.
    - ``gaze_blend``: iris position (4) + the BLEND_FEATURE_NAMES blendshapes (14); pose-free.
    - ``blend``: head pose (yaw/pitch/roll) + the blendshape subset, no iris geometry.
    - ``both``: iris + blendshape subset.
    - ``head`` (M2 step 3a): iris position (4) + head pose (3) + head position (3); a plain pooled
      regression that can learn how posture shifts the mapping.
    - ``ray`` (M2 step 3b): the linearised gaze ray hitting the screen plane (see ``ray_features``).
    """
    if feature_set == "iris":
        return features[..., :N_BASE]
    if feature_set == "head":
        return features[..., [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]]
    if feature_set == "ray":
        return ray_features(features)
    if feature_set == "eyes":
        return features[..., [0, 1, 2, 3, 7, 8]]
    if feature_set == "gaze":
        return features[..., :4]
    if feature_set == "gaze_blend":
        return np.concatenate([features[..., :4], blend_features(blendshapes, names)], axis=-1)
    blend = blend_features(blendshapes, names)
    if feature_set == "blend":
        return np.concatenate([features[..., 4:7], blend], axis=-1)
    if feature_set == "both":
        return np.concatenate([features[..., :N_BASE], blend], axis=-1)
    raise ValueError(f"unknown feature set {feature_set!r}; choose from {FEATURE_SETS}")


def ray_features(features: np.ndarray) -> np.ndarray:
    """Linear terms of a gaze ray from the eye intersected with the screen plane.

    The camera sits in the screen plane (built into the lid), so the screen is the plane z = 0 of
    the camera frame. A ray from the eye at (x, y, z) with horizontal / vertical angles (a, b) hits
    it at (x + z tan a, y + z tan b). The gaze angle is head rotation plus eye-in-head rotation,
    and the eye-in-head angle is close to linear in the iris offset, so to first order the hit
    point is linear in: eye position (x, y, z), z * iris offsets (4), and z * head yaw / pitch
    (radians). A linear model on these terms is that intersection with learned coefficients
    (the per-person eye gain, the unknown focal length and the camera-to-screen offset).
    """
    f = np.asarray(features, dtype=float)
    hz = f[..., 11:12]
    return np.concatenate([f[..., 9:12], hz * f[..., 0:4], hz * np.radians(f[..., 4:6])], axis=-1)
