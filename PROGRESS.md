# Progress log

Read this first in a new session, then `CLAUDE.md` and `PLAN.md`. Update it at the end of every work session.

## Status
- **Next:** plan M2 (plan first, short; no code until approved). Unpushed local commits may exist: check `git status -sb`.
- **M0: DONE, confirmed by Abhimanyu on 2026-10-13.**
  - Done when: `uv run scripts/viewer.py` shows iris tracking at >= 20 fps, `q` quits cleanly.
  - Verified so far: `uv run pytest` (6 pass, head-pose math); model downloaded; `FaceTracker` runs on a blank frame (returns None); viewer imports.
  - Live run (user, 2026-10-13): 7 PNGs in `debug/`; the 3 inspected show 30.0 fps, mesh aligned, iris circles centered on pupils, `s` dump works. Head turn showed yaw -32 when face turned toward image-left (mirrored view, so user's left = negative yaw). Only a head turn was checked; pitch/roll signs and the quit check not confirmed.
- **M1: DONE, confirmed by Abhimanyu on 2026-10-13.** Dot follows gaze; benchmark reproducible (best baseline: `gaze` ridge 3.12 deg on session_20261013_030654). Original plan: 9 calibration + 12 held-out points.
  - **M1: baseline reproducible (3 real sessions benchmarked); close it once Abhimanyu confirms the overlay dot follows his gaze.** **Next:** M2 per PLAN.md, starting with the evaluation foundation and calibration redesign.
- (old) **Next (user):** `uv run scripts/calibrate.py` -> `uv run scripts/benchmark.py` -> `uv run scripts/gaze_overlay.py`. Done when the dot follows gaze and the benchmark prints a reproducible baseline.
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

## Third session and the "dot stuck at the bottom" regression (2026-10-13)
- session_20261013_025905 (borderless window [0,0,1470,956], first clean one): ridge all-9 `iris` 5.10 deg (272 px, signed y +190: predictions only span y 458..922 vs true 79..809); `eyes` (iris+openness) 4.20 deg; `gaze` (iris xy only) 3.99 deg (212 px, jitter ~3.0 deg).
- Cause of the stuck overlay: raw yaw/pitch/roll as linear features. Calibration pose variance is tiny (pitch std 0.75 deg, roll 0.9 deg) so ridge puts huge weights on them (roll coef -262 on y). At overlay time the user's pose was +8.3 sd pitch / +3.4 sd yaw from calibration (about 6 deg pitch shift, probably laptop tilt/sitting), so the all-9 model predicted y=959 (off-screen) -> dot pinned to the bottom edge. Pose-free models stayed on screen.
- Not conclusive for feature choice: across the three sessions pose hurts in two and helps in one (4.94 vs 5.94 deg), and cross-session (train session 2, test session 3) is ~6-7 deg for every set. So the lesson is head pose needs proper compensation (M2), not removal; `gaze` is just the robust live default. Do NOT read the 3.99 deg as the project's baseline: it was chosen after seeing test results on one session.
- Added feature sets `eyes` (6) and `gaze` (4); overlay defaults to `gaze` and shows a warning when any live feature is > 4 calibration sd from its mean ("sit as you did, or recalibrate"). Benchmark default stays `iris` (all 9) so the baseline row stays comparable.
- Practical: calibrate and run the overlay in the same laptop tilt/posture; recalibrate if the warning appears.

## Independent analysis (claude.ai, 2026-10-13, session_20261013_025905 = first unbiased session)
Ridge with numpy, iris(9) vs both(9+14 blendshapes), errors in degrees on the 12 held-out targets:
- **Regularization:** both, alpha 100 -> **3.65 mean / 3.01 median / p90 7.1** (vs 4.25 / 2.98 at alpha 1). iris alone ~4.8-5.1. Blendshapes DO help on unbiased data (the earlier "no help" came from the offset-biased session).
- **Bias dominates noise:** averaging each fixation's predictions (perfect smoothing) only goes 3.65 -> 3.23. Every top-third target is predicted 260-500 px too LOW (dy +261, +294, +337, +340, +499); bottom-half targets are within ~100 px. Vertical gaze is the weak axis, and some test targets (y=79) sit outside the calibration grid (inset 10% = 96 pt), so the model extrapolates.
- **Effective n = 9:** 407 calibration frames come from only 9 targets. Leave-one-target-out CV ~5.2-5.4 deg and noisy; poly2 overfits for this reason. Need more targets and sessions, grouped CV, strong regularization.
- **Time within fixation:** error 4.2 (0-300 ms after settle), 3.9 (300-1200 ms), 4.8 (>1200 ms). Shorter per-target recording, more targets.
- Next: M2 step 1 + 2 in PLAN.md (evaluation foundation, calibration redesign), then the vertical-bias work.

- Latest session (session_20261013_030654, recalibrated in the same posture; user: overlay "better now"): ridge `gaze` 3.12 deg mean / 2.65 median (166 px, 3.28 cm), jitter 2.75 deg; baseline `iris` (all 9, with head pose) 6.91 deg (369 px). Benchmark output is bit-identical on repeated runs (reproducible). Head pose hurt again (third of four sessions).

## Display facts (locked in `ScreenConfig`)
- Panel (Apple tech specs, support.apple.com/en-us/126320): 13.6" Liquid Retina, 2560x1664 at 224 ppi (= 29.0 x 18.9 cm, matches Quartz/EDID 29.06 x 18.90). Apple: only the TOP corners are rounded (bottom square) and the actual viewable area is smaller than the 13.6" rectangle.
- Default mode is "looks like 1470x956" (rendered at 2x = 2940x1912, downscaled to the panel), so 1 pt = 0.01975 cm.
- Measured via AppKit on this Mac: `safeAreaInsets.top` = 32 pt (menu bar/notch strip); notch spans x = 646..825 pt (179 pt wide, 32 pt tall). `ScreenConfig.from_system()` reads these.
- Corner radius: NOT published by Apple and no public API; `corner_radius_pt = 24` is a generous ASSUMPTION used only to keep targets away. `target_is_safe()` rejects targets within ring radius of the edges, the 32 pt strip or the rounded corners; `calibrate.py` aborts if any are unsafe (grid inset 10% = 147x96 pt and held-out margin 8% easily pass). `--check-screen` draws the four extreme-safe rings plus the notch outline.

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
