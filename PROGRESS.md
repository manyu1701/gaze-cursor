# Progress log

Read this first in a new session, then `CLAUDE.md` and `PLAN.md`. Update it at the end of every work session.

## Status
- **M0: DONE, confirmed by Abhimanyu on 2026-10-13.**
  - Done when: `uv run scripts/viewer.py` shows iris tracking at >= 20 fps, `q` quits cleanly.
  - Verified so far: `uv run pytest` (6 pass, head-pose math); model downloaded; `FaceTracker` runs on a blank frame (returns None); viewer imports.
  - Live run (user, 2026-10-13): 7 PNGs in `debug/`; the 3 inspected show 30.0 fps, mesh aligned, iris circles centered on pupils, `s` dump works. Head turn showed yaw -32 when face turned toward image-left (mirrored view, so user's left = negative yaw). Only a head turn was checked; pitch/roll signs and the quit check not confirmed.
- **M1: code written and unit-tested, NOT yet run live.** Plan approved (9 calibration + 12 held-out points). 19 tests pass; benchmark ran twice on a synthetic session with identical output.
  - **Next (user):** `uv run scripts/calibrate.py` -> `uv run scripts/benchmark.py` -> `uv run scripts/gaze_overlay.py`. Done when the dot follows gaze and the benchmark prints a reproducible baseline.
  - Unverified live: fullscreen window sizing/aspect on the 1470x956 logical screen, calibrate flow, features on real data.
- M0 committed and pushed to origin/main (87c5e33). M1 decision: 9 calibration points + separate held-out test points.

## What exists
- `pyproject.toml` (uv, Python 3.12, hatchling src layout), `.python-version`, `.gitignore`
- `src/gaze_cursor/capture.py`: `Camera` (AVFoundation, 1280x720@30), `Frame`
- `src/gaze_cursor/landmarks.py`: `FaceTracker` (Tasks API, VIDEO mode), `LandmarkResult`, `head_pose_from_matrix`, iris index constants
- `scripts/download_model.py` (idempotent), `scripts/viewer.py`
- `tests/test_landmarks.py`
- `models/face_landmarker.task` (gitignored), `data/sessions/` (gitignored)

## Gotchas / decisions
- Hardware: MacBook Air Mac17,3, Apple M5, 16 GB, 2560x1664 native = 1470x956 logical points; assumed 29.0x18.9 cm physical, 60 cm distance (`config.py`). Pixels everywhere are logical points.
- Viewer sign conventions (mirrored view, from debug frames): pitch negative = head tilted back/looking up (-49 seen); roll positive = top of head toward image-left (+47 seen); yaw negative = turned to user's left. Mesh and irises stayed locked even at those extremes. Webcam sits low, so neutral pitch reads about -2 to -13.
- `calibrate.py` does not mirror frames (viewer does); sessions record `mirrored: false`.
- `eval/targets.py` test-point function is `held_out_points` (a `test_` name would be collected by pytest).
- No model save/load yet: overlay refits from a session on start.
- Deps: mediapipe 1.1.0, opencv-python 5.0.0.93 (mediapipe also pulls opencv-contrib-python; harmless), numpy 2.5.4.
- Landmarks: 478 points; right iris 468-472 (center 468), left iris 473-477 (center 473); x/y normalized.
- Head pose assumes R = Ry(yaw) Rx(pitch) Rz(roll) from MediaPipe's 4x4 matrix, scale removed. Sign conventions vs. real head motion not yet checked live; confirm in the viewer (turn head right/left, nod, tilt).
- VIDEO mode needs strictly increasing timestamps; `FaceTracker.process` enforces this.
- Viewer mirrors the image (flip) before processing, so landmarks and text match what the user sees.
- scikit-learn was added at M0 (needed from M1).
