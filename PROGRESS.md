# Progress log

Read this first in a new session, then `CLAUDE.md` and `PLAN.md`. Update it at the end of every work session.

## Status
- **M0: DONE, confirmed by Abhimanyu on 2026-10-13.**
  - Done when: `uv run scripts/viewer.py` shows iris tracking at >= 20 fps, `q` quits cleanly.
  - Verified so far: `uv run pytest` (6 pass, head-pose math); model downloaded; `FaceTracker` runs on a blank frame (returns None); viewer imports.
  - Live run (user, 2026-10-13): 7 PNGs in `debug/`; the 3 inspected show 30.0 fps, mesh aligned, iris circles centered on pupils, `s` dump works. Head turn showed yaw -32 when face turned toward image-left (mirrored view, so user's left = negative yaw). Only a head turn was checked; pitch/roll signs and the quit check not confirmed.
- **M1: code written and unit-tested, NOT yet run live.** Plan approved (9 calibration + 12 held-out points). 19 tests pass; benchmark ran twice on a synthetic session with identical output.
  - **Next (user):** `uv run scripts/calibrate.py` -> `uv run scripts/benchmark.py` -> `uv run scripts/gaze_overlay.py`. Done when the dot follows gaze and the benchmark prints a reproducible baseline.
  - Unverified live: run `calibrate.py --check-screen` first (window mapping), then calibrate flow, features on real data.
  - M1 amendments (user-requested, done): raw frames everywhere with display-only mirroring; screen geometry from Quartz; blendshapes recorded as candidate features.
- M0 committed and pushed to origin/main (87c5e33). M1 decision: 9 calibration points + separate held-out test points.

## First real session (session_20261013_024055.npz, 2026-10-13)
- Screen-mapping check (`--check-screen`) showed an OFFSET. Cause (measured via Quartz window bounds): the OpenCV fullscreen window on this notch MacBook is x=0,y=33, 1470x923 pt (menu bar/notch strip excluded), while the old canvas was 1470x956 and got squeezed/shifted, so targets were drawn up to ~30 pt away from their recorded positions. FIXED: `FullscreenCanvas` reads its real bounds from Quartz after the fullscreen animation settles (waits >1.5 s plus 8 identical readings) and sizes the canvas to them; drawing takes screen points and subtracts the window origin; calibrate.py aborts if a target would be hidden and stores `meta['window']`. The session below predates this, so its errors are biased and NOT trustworthy; recalibrate. OpenCV's `getWindowImageRect` reports 1470x956 and cannot be trusted here. Screenshots are blocked (no Screen Recording permission), so the fix is checked only via Quartz bounds, not visually.
- Found: openness < 0.15 was dropping real downward fixations (bottom-row median openness 0.145; blendshape eyeBlink also rises when looking down, so it can't separate blinks). Cutoff lowered to 0.05; `load_session` now recomputes features from stored landmarks (record and replay). Camera frame confirmed 1280x720.
- Baseline after the fix, all 541 test / 407 train frames valid (ridge, alpha 1): iris 335 px / 6.27 deg (jitter 3.05 deg); blend 370 px / 6.92 deg; both 361 px / 6.77 deg; poly2 alpha 10 on iris: 380 px / 7.10 deg (jitter 4.4 deg). Before the fix (38% of test frames dropped): iris 5.42 deg. Blendshapes did not help over iris features on this one session.
- Not yet addressed: remaining blink/outlier frames inflate jitter; no smoothing (M2).

- Offset still present after the Quartz-bounds fix (user: top corners "a few cm short"). Second session (session_20261013_025134, window recorded as [0,33,1470,923]): iris ridge 263 px / 4.94 deg mean (median 4.49), jitter 4.57 deg, train err 187 px; top targets biased +128 px in y vs +54 bottom, consistent with a hidden title strip squeezing the canvas. That session is still biased. Replaced the OpenCV window with a borderless NSWindow at shielding level covering the whole display (`sinks/overlay.py`, pyobjc-framework-Cocoa added): Quartz reports bounds (0,0,1470,956), exactly 1:1. NOT yet viewed by the user: rerun `--check-screen`, then recalibrate.
- Observed within-fixation scatter is huge (per-target prediction std ~100-340 px): noise, not offset, dominates; M2 (smoothing, outlier/blink rejection, head compensation) is where this gets addressed.

## What exists
- `pyproject.toml` (uv, Python 3.12, hatchling src layout), `.python-version`, `.gitignore`
- `src/gaze_cursor/capture.py`: `Camera` (AVFoundation, 1280x720@30), `Frame`
- `src/gaze_cursor/landmarks.py`: `FaceTracker` (Tasks API, VIDEO mode), `LandmarkResult`, `head_pose_from_matrix`, iris index constants
- `scripts/download_model.py` (idempotent), `scripts/viewer.py`
- `tests/test_landmarks.py`
- `models/face_landmarker.task` (gitignored), `data/sessions/` (gitignored)

## Gotchas / decisions
- Hardware: MacBook Air Mac17,3, Apple M5, 16 GB, 2560x1664 native = 1470x956 logical points. `ScreenConfig.from_system()` reads size (points) and physical size (Quartz: 29.06x18.90 cm) at run time; viewing distance stays an assumed 60 cm. Quartz was pulled in at M1 (CLAUDE.md said M3) for this; pixels everywhere are logical points.
- Pose sign conventions (UNFLIPPED frame, same as recordings; viewer now processes the raw frame and mirrors for display only): the earlier values (pitch negative = looking up; yaw negative = turned to user's left; roll positive = top of head toward image-left) were read from the old mirrored viewer, so yaw and roll signs are inverted relative to recordings; pitch is unchanged. Re-check in the viewer. Webcam sits low, so neutral pitch reads about -2 to -13.
- No component mirrors the frame it processes; only the viewer's display is flipped (landmarks drawn at 1 - x). Sessions record `mirrored: false`.
- Blendshapes (52) are recorded per frame in sessions; names in `meta["blendshape_names"]`. Candidate subset = `BLEND_FEATURE_NAMES` (eyeLook In/Out/Up/Down, Blink, Squint, Wide, L/R = 14). Feature sets: `iris` (9, baseline), `blend` (yaw/pitch/roll + 14), `both`; `benchmark.py --features`. Cost measured live: tracker median 10.1 ms with blendshapes vs 7.2 ms without (p95 12.6 vs 13.4), fine for 30 fps.
- `uv run scripts/calibrate.py --check-screen` shows corner/center markers to verify the OpenCV fullscreen window maps 1:1 to points (notch/menu bar); NOT yet checked by the user.
- `eval/targets.py` test-point function is `held_out_points` (a `test_` name would be collected by pytest).
- No model save/load yet: overlay refits from a session on start.
- Deps: mediapipe 1.1.0, opencv-python 5.0.0.93 (mediapipe also pulls opencv-contrib-python; harmless), numpy 2.5.4.
- Landmarks: 478 points; right iris 468-472 (center 468), left iris 473-477 (center 473); x/y normalized.
- Head pose assumes R = Ry(yaw) Rx(pitch) Rz(roll) from MediaPipe's 4x4 matrix, scale removed. Sign conventions vs. real head motion not yet checked live; confirm in the viewer (turn head right/left, nod, tilt).
- VIDEO mode needs strictly increasing timestamps; `FaceTracker.process` enforces this.
- Viewer mirrors the image (flip) before processing, so landmarks and text match what the user sees.
- scikit-learn was added at M0 (needed from M1).
