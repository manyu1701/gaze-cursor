"""Precision limit of the iris landmark (EXPLORATORY; prints only, writes nothing).

How many image pixels does the iris landmark move per degree of gaze, and how noisy is it
within a fixation? For every clean layout-v2 session and each eye: the iris centre's offset
from the eye-corner midpoint, in image pixels, is regressed on the target's visual angle
(from the session's own screen geometry and assumed distance); the within-fixation noise is
the spread of that offset around each fixation's mean. Background: docs/research/02-camera.md.

No model is fitted and nothing is appended to the results log: this is a measurement of the
input signal, not an accuracy claim.

  uv run scripts/explore_precision.py
"""

from __future__ import annotations

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import DEFAULT_IMAGE_SIZE, load_session, session_files
from gaze_cursor.features import LEFT_EYE, RIGHT_EYE

IRIS_EDGE = (469, 471)  # left and right edge of one iris (MediaPipe iris landmarks)


def eye_offsets(points_px: np.ndarray, eye: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iris centre minus eye-corner midpoint (x, y) in image px, and eye width in px, per frame."""
    a, b, i = points_px[:, eye[0], :2], points_px[:, eye[1], :2], points_px[:, eye[4], :2]
    mid = (a + b) / 2
    return i[:, 0] - mid[:, 0], i[:, 1] - mid[:, 1], np.linalg.norm(b - a, axis=1)


def gaze_angles(target_xy: np.ndarray, screen: ScreenConfig) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical visual angle (deg) of each target from the screen centre."""
    cx, cy = screen.width_px / 2, screen.height_px / 2
    to_deg = lambda d_px: np.degrees(np.arctan(d_px * screen.cm_per_px / screen.distance_cm))  # noqa: E731
    return to_deg(target_xy[:, 0] - cx), to_deg(target_xy[:, 1] - cy)


def within_fixation_std(values: np.ndarray, groups: np.ndarray) -> float:
    """Std of values around their own fixation's mean (pooled over fixations)."""
    res = np.concatenate([values[groups == g] - values[groups == g].mean() for g in np.unique(groups)])
    return float(res.std())


def session_precision(sess) -> dict[str, float] | None:
    """Per-session medians over both eyes; None if the session has no usable frames."""
    pts = sess.points
    ok = ~np.isnan(pts).any(axis=(1, 2))
    if ok.sum() < 50:
        return None
    w, h = sess.meta.get("image_size", DEFAULT_IMAGE_SIZE)
    p = pts[ok] * np.array([w, h, w])
    screen = ScreenConfig(**sess.meta["screen"])
    ax, ay = gaze_angles(sess.target_xy[ok], screen)
    fix = sess.target_idx[ok].astype(int) * 10 + sess.phase[ok].astype(int)  # one group per fixation
    per_eye = []
    for eye in (RIGHT_EYE, LEFT_EYE):
        dx, dy, width = eye_offsets(p, eye)
        per_eye.append((abs(np.polyfit(ax, dx, 1)[0]), abs(np.polyfit(ay, dy, 1)[0]),
                        within_fixation_std(dx, fix), within_fixation_std(dy, fix), float(np.median(width))))
    sx, sy, nx, ny, ew = np.mean(per_eye, axis=0)
    iris = float(np.median(np.linalg.norm(p[:, IRIS_EDGE[0], :2] - p[:, IRIS_EDGE[1], :2], axis=1)))
    ipd = float(np.median(np.linalg.norm(p[:, RIGHT_EYE[4], :2] - p[:, LEFT_EYE[4], :2], axis=1)))
    return {"px_deg_x": sx, "px_deg_y": sy, "noise_x": nx, "noise_y": ny, "eye_w": ew, "iris": iris, "ipd": ipd}


def main() -> int:
    rows = []
    for path in session_files():
        sess = load_session(path, recompute_features=False)
        if not sess.is_clean or sess.layout != "v2" or "screen" not in sess.meta:
            continue
        r = session_precision(sess)
        if r is not None:
            rows.append((path.stem.removeprefix("session_"), r))
    if not rows:
        print("no clean layout-v2 sessions found")
        return 1
    print(f"{'session':16s} px/deg x  px/deg y | noise px x    y | eye w | iris | IPD | per-frame deg x   y")
    for name, r in rows:
        print(f"{name:16s} {r['px_deg_x']:8.3f} {r['px_deg_y']:9.3f} | {r['noise_x']:10.2f} {r['noise_y']:4.2f} |"
              f" {r['eye_w']:5.1f} | {r['iris']:4.1f} | {r['ipd']:3.0f} |"
              f" {r['noise_x'] / r['px_deg_x']:13.1f} {r['noise_y'] / r['px_deg_y']:3.1f}")
    med = {k: float(np.median([r[k] for _, r in rows])) for k in rows[0][1]}
    print(f"\nMEDIAN over {len(rows)} sessions: px/deg x {med['px_deg_x']:.3f}, y {med['px_deg_y']:.3f}; "
          f"noise px x {med['noise_x']:.2f}, y {med['noise_y']:.2f}; eye width {med['eye_w']:.1f} px; "
          f"iris {med['iris']:.1f} px; IPD {med['ipd']:.0f} px")
    print(f"one px of landmark error ~ {1 / med['px_deg_x']:.1f} deg horizontally, {1 / med['px_deg_y']:.1f} deg vertically")
    print("NOTE: angles use each session's assumed screen distance (60 cm); the medians are unweighted over a mixed set "
          "(dim, head-position and mixed-light sessions included) and the sessions are not independent samples. "
          "See docs/research/02-camera.md, item 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
