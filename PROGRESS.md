# Progress log

Read this first in a new session, then `CLAUDE.md` and `PLAN.md`. Update it at the end of every work session.

## Status
- **M0: DONE** (confirmed by Abhimanyu, 2026-10-13). Live viewer at 30 fps, mesh and irises aligned.
- **M1: DONE** (confirmed by Abhimanyu, 2026-10-13). Dot follows gaze; the benchmark is bit-identical on repeated runs. Best exploratory number so far: `gaze` ridge 3.12 deg mean on `session_20261013_030654` (not a project baseline: chosen after seeing test results on one session).
- **M2 steps 1 and 2: BUILT, awaiting Abhimanyu's confirmation and the 3 new recordings.** `scripts/evaluate.py` (modes within / loto / losos, per-target stats, bootstrap CIs over targets, paired comparison, per-target bias/spread, `--preregistered`) and calibration layout `v2` (`calibrate.py --layout v2 --note ...`) are implemented and tested (47 tests). Nothing from M2 step 3 onward until confirmed.
- **Next (Abhimanyu):** (1) dry-run `uv run scripts/calibrate.py` once and confirm the 25 dots are random, reach the edges and take ~45 s; (2) record 3 v2 sessions at different times of day/lighting, each with `--note "..."`, same posture each time; (3) then Claude runs `uv run scripts/evaluate.py --preregistered` exactly once.
- Git: `main` is ahead of `origin/main` (only M0 was pushed, `87c5e33`); nothing is pushed without Abhimanyu asking. Commit `b0dcb48` ("Record latest M1 benchmark results") also contains Abhimanyu's edits to CLAUDE.md, PLAN.md and PROGRESS.md, swept in by `git add -A`.

## Session inventory
| Session | Window | Clean? | Notes |
| --- | --- | --- | --- |
| `session_20261013_024055` | none recorded | no | OpenCV fullscreen window squeezed the canvas; targets biased |
| `session_20261013_025134` | [0,33,1470,923] | no | still squeezed (title strip) |
| `session_20261013_025905` | [0,0,1470,956] | yes | first unbiased session |
| `session_20261013_030424` | [0,0,1470,956] | yes | |
| `session_20261013_030654` | [0,0,1470,956] | yes | recalibrated in the same posture |

**Clean** means window `[0,0,1470,956]` (borderless NSWindow, 1:1 with screen points). By session start times (02:58 to 03:06) all three clean sessions were recorded within minutes of each other in the same room, so they do not yet cover different lighting, posture or times of day. Layout v1: 9 calibration + 12 held-out targets.

## Findings so far (all exploratory)

### First sessions: offset and blink filter
- **Offset** (sessions 1 and 2): on this notch MacBook the OpenCV fullscreen window sits at y=33 (then 923 pt tall, with a further title strip), so the canvas was squeezed and targets were drawn away from their recorded positions; top targets were predicted ~128 px too low. Fixed by replacing the OpenCV window with a borderless NSWindow at shielding level (`sinks/overlay.py`); Quartz reports bounds (0,0,1470,956). Screenshots are blocked (no Screen Recording permission), so the mapping was checked via Quartz bounds and by Abhimanyu looking at `--check-screen`.
- **Blink filter bug:** `MIN_OPENNESS = 0.15` dropped real downward fixations (bottom-row median openness 0.145; the blendshape `eyeBlink` also rises when looking down, so it cannot separate blinks). Lowered to 0.05; `load_session` now recomputes features from stored landmarks (record and replay). Camera frames confirmed 1280x720.

### "Dot stuck at the bottom" regression
- Cause: raw yaw/pitch/roll as linear features. Calibration pose variance is tiny (pitch std 0.75 deg, roll 0.9 deg), so ridge put huge weights on them (roll coefficient -262 on y). At overlay time the user's pose was +8.3 sd pitch / +3.4 sd yaw from calibration (~6 deg pitch shift: laptop tilt/sitting), so the all-features model predicted y=959 (off-screen) and the dot pinned to the bottom.
- Added pose-free feature sets `eyes` (iris + openness, 6) and `gaze` (iris xy, 4). The overlay defaults to `gaze` and warns when a live feature is > 4 calibration sd from its mean. The benchmark default stays `iris` (all 9) so that baseline row stays comparable.
- Practical rule: calibrate and run the overlay in the same laptop tilt/posture. Head pose needs proper compensation (M2), not just removal.

### Latest numbers (`session_20261013_030654`, ridge alpha 1)
- `gaze` 3.12 deg mean / 2.65 median (166 px, 3.28 cm), jitter 2.75 deg; `iris` (all 9, with pose) 6.91 deg (369 px).

### Independent analysis (claude.ai, 2026-10-13, session_20261013_025905 = first unbiased session)
Ridge with numpy, iris(9) vs both(9+14 blendshapes), errors in degrees on the 12 held-out targets:
- **Regularization:** both, alpha 100 -> **3.65 mean / 3.01 median / p90 7.1** (vs 4.25 / 2.98 at alpha 1). iris alone ~4.8-5.1. Blendshapes DO help on unbiased data (the earlier "no help" came from the offset-biased session).
- **Bias dominates noise:** averaging each fixation's predictions (perfect smoothing) only goes 3.65 -> 3.23. Every top-third target is predicted 260-500 px too LOW (dy +261, +294, +337, +340, +499); bottom-half targets are within ~100 px. Vertical gaze is the weak axis, and some test targets (y=79) sit outside the calibration grid (inset 10% = 96 pt), so the model extrapolates.
- **Effective n = 9:** 407 calibration frames come from only 9 targets. Leave-one-target-out CV ~5.2-5.4 deg and noisy; poly2 overfits for this reason. Need more targets and sessions, grouped CV, strong regularization.
- **Time within fixation:** error 4.2 (0-300 ms after settle), 3.9 (300-1200 ms), 4.8 (>1200 ms). Shorter per-target recording, more targets.
- Next: M2 step 1 + 2 in PLAN.md (evaluation foundation, calibration redesign), then the vertical-bias work.

### Independent analysis #2 (claude.ai, 2026-10-13, 3 clean sessions: 025905, 030424, 030654)
Numpy ridge, held-out 12 targets, mean deg per session (025905 / 030424 / 030654):
- Within-session (train that session's 9 calib targets): `gaze` a=10 3.96 / 5.11 / 3.04; `iris` (with raw pose) a=1 5.10 / 6.90 / 6.91; **`gaze`+14 eye blendshapes (pose-free) a=100: 3.54 / 5.27 / 2.74** (best on average). Session 030424 is ~1.5 deg worse than the others for every model: session-to-session variance is as big as model differences, so judge models on the average over sessions, never one session.
- **Cross-session pooling works:** train on ALL 21 targets of the OTHER two sessions, test this session's held-out 12: `gaze`+blendshapes a=100 **4.18 / 4.21 / 3.93**, close to within-session, and better than within-session for 030424. More targets beat per-session recalibration. (The earlier "cross-session 6-7 deg" was single-session training with raw pose features.) Caveat: all three sessions were recorded within ~30 min in the same room, so this is not yet a lighting/posture test.
- Vertical bias is smaller with the pose-free set but still there: on 030654 most top/middle targets are predicted 90-190 px too low; bottom targets are near zero. Calibration y spans 96-860 pt; a test target at y=79 is slightly outside it.
- Caution: these feature/alpha choices were picked after looking at test results. Treat `gaze`+blendshapes, alpha 100 as the hypothesis to confirm on NEW sessions (pre-registered), not as a result.

## Pre-registration: M2 hypothesis H1 (written 2026-10-13, BEFORE recording any layout-v2 session)
- **H1:** with pose-free features, `gaze_blend` (iris xy + the 14 eye blendshapes = 18 features) with ridge alpha=100 has lower cross-session error than `gaze` (iris xy, 4 features) with ridge alpha=10.
- **Data:** three new sessions recorded with calibration layout `v2` (25 targets), at different times of day/lighting, each tagged with `--note`. **Inclusion rule, fixed now:** `meta.layout == "v2"` and window `[0,0,1470,956]`. No session is dropped after seeing results; a session is dropped only for an objective failure (< 80% valid frames, or the target-safety abort), logged with the reason.
- **Primary (confirmatory) analysis:** leave-one-session-out. For each v2 session, train on the pooled targets (calibration and held-out) of every *other* clean session (the older clean sessions plus the other v2 sessions) and test on **all 25 targets of the held-out session**. Unit of analysis = target. Per-target error = mean over frames of Euclidean error in degrees of visual angle. Both configs are evaluated on identical targets (paired).
- **Primary statistic:** mean of the per-target paired differences d = err(`gaze` a=10) - err(`gaze_blend` a=100) (positive favors H1), with a 95% bootstrap CI resampling targets (10,000 resamples, fixed seed).
- **Success criterion (all three required):** (a) the CI lower bound is > 0; (b) the mean improvement is >= 0.25 deg; (c) `gaze_blend` is better in at least 2 of the 3 held-out sessions. Otherwise H1 is reported as **not supported**, with no re-tuning on these sessions.
- **Caveats, stated up front:**
  - The three clean M1 sessions (including Independent analysis #2) suggested the hypothesis, and alpha=100 was also suggested by them; they are exploratory and never confirmatory.
  - The comparison changes features and alpha together, so a win does not say which caused it. Secondary diagnostic configs (reported, not part of the criterion): `gaze` alpha=100 and `gaze_blend` alpha=10.
  - Three sessions carry little session-level information. A secondary sensitivity analysis uses a two-level bootstrap (resample sessions, then targets).
  - Everything else computed is exploratory and labeled so (`confirmatory=0` in `results.csv`).

## M2 steps 1-2: implementation notes (2026-10-13)
- Feature set `gaze_blend` (iris xy + 14 eye blendshapes, 18 cols, pose-free) added; registry configs in `eval/configs.py` (`iris_a1`, `iris_a100`, `eyes_a10`, `gaze_a1/10/100`, `gaze_blend_a10/100`, `both_a100`); pre-registered pair `gaze_a10` (ref) vs `gaze_blend_a100`.
- Layout v2: 4x4 calibration grid at x 24..1446, y 56..932 pt (safe edges + 8 pt margin) and 9 held-out points inside it (>= 60 pt from its edges, >= 100 px from grid points and each other), all 25 in one random order from a stored seed, settle 0.7 s + record 1.0 s. Session format unchanged; meta gains `layout`, `seed`, `note`.
- `--preregistered` takes the earliest 3 clean v2 sessions with >= 80% valid frames as held-out, trains on the older clean sessions plus the other held-out v2 sessions, ignores extra v2 sessions entirely, and refuses (exit 2) with fewer than 3. It was run only to confirm the refusal; no confirmatory result exists.
- `results.csv` now has one shared schema (`mode`, `confirmatory`, `config`, CI columns...); the old file was migrated in place (old rows: mode=within, confirmatory=0). Verified: `evaluate.py --modes within` on session 030654 gives frame-mean 3.12 deg for `gaze_a1`, equal to `benchmark.py`; repeated runs are identical.
- EXPLORATORY output on the 3 existing clean sessions (these sessions suggested H1, so this is not evidence for it), mean deg over targets [95% CI]: within `gaze_a10` 4.04 [3.52, 4.57] vs `gaze_blend_a100` 3.86 [3.33, 4.44] (paired d = +0.18, CI [-0.23, +0.61]); leave-one-session-out over all 21 targets per held-out session: `gaze_a10` 5.57 vs `gaze_blend_a100` 4.52 (d = +1.05, CI [+0.70, +1.39], 3/3 sessions); `iris_a1` (with raw pose) 4.78 under pooled training, better than `gaze` there (pose becomes learnable once training pools sessions with different postures; keep in mind for step 3+).
- Known design limits: LOSO tests all 21 v1 targets, including the grid targets that sit near the training pool's edge; within-session numbers use 12 test targets per session, so CIs are wide; three sessions give weak session-level inference (hence the two-level bootstrap column in the paired report).

## Display facts (locked in `ScreenConfig`)
- Panel (Apple tech specs, support.apple.com/en-us/126320): 13.6" Liquid Retina, 2560x1664 at 224 ppi (= 29.0 x 18.9 cm; Quartz/EDID gives 29.06 x 18.90). Apple: only the TOP corners are rounded (bottom square) and the actual viewable area is smaller than the 13.6" rectangle.
- Default mode is "looks like 1470x956" (rendered at 2x = 2940x1912, downscaled to the panel), so 1 pt = 0.01975 cm. Pixels everywhere in this project are logical points.
- Measured via AppKit on this Mac (MacBook Air Mac17,3, M5, 16 GB): `safeAreaInsets.top` = 32 pt (menu bar/notch strip); notch spans x = 646..825 pt (179 pt wide, 32 pt tall). `ScreenConfig.from_system()` reads these and the physical size (Quartz). Viewing distance is an assumed 60 cm.
- Corner radius: NOT published by Apple and no public API; `corner_radius_pt = 24` is a generous ASSUMPTION used only to keep targets away. `target_is_safe()` rejects targets within the ring radius of the edges, the 32 pt strip or the rounded corners; `calibrate.py` aborts if any are unsafe. `--check-screen` draws the four extreme-safe rings plus the notch outline.

## What exists
- Packaging: `pyproject.toml` (uv, Python 3.12, hatchling src layout), `.python-version`, `.gitignore`, `README.md`, `uv.lock`.
- `src/gaze_cursor/`: `capture.py` (`Camera`, `Frame`; AVFoundation, 1280x720 at 30), `landmarks.py` (`FaceTracker` Tasks API VIDEO mode with blendshapes, `LandmarkResult`, `head_pose_from_matrix`), `features.py` (iris/pose/openness features, blendshape subset, feature sets `iris`, `eyes`, `gaze`, `blend`, `both`), `calibration.py` (`GazeModel`: ridge, poly2), `config.py` (`ScreenConfig`, Quartz/AppKit geometry, `target_is_safe`), `eval/` (`session.py` record/replay + `is_clean`/`layout`/`select_all`/`clean_sessions`, `targets.py` v1 and v2 layouts + `build_plan`, `metrics.py` px/cm/deg and jitter, `configs.py` named configs and the pre-registered pair, `splits.py` the three modes, `stats.py` per-target aggregation, bootstrap, paired comparison, `runner.py`, `results_log.py` shared CSV writer with in-place migration), `sinks/overlay.py` (borderless NSWindow canvas).
- `scripts/`: `download_model.py`, `viewer.py`, `calibrate.py` (`--check-screen`, `--layout v1|v2`, `--seed`, `--note`), `benchmark.py`, `evaluate.py`, `gaze_overlay.py`.
- `tests/`: head pose, features, metrics, calibration, session/config, splits (no train/test leakage per mode), stats, layout v2, results-log migration (47 tests; synthetic sessions in `tests/helpers.py`).
- Data: `models/face_landmarker.task` (gitignored), `data/sessions/` (gitignored), `data/results.csv` (tracked, benchmark log), `debug/` (gitignored PNG dumps).

## Gotchas and decisions
- Pose sign conventions on the UNFLIPPED frame (what recordings use): pitch negative = head tilted back/looking up; yaw and roll signs are inverted relative to the old mirrored viewer. The viewer now processes the raw frame and mirrors for display only (landmarks drawn at 1 - x). The webcam sits low, so neutral pitch reads about -2 to -13. Head pose assumes R = Ry(yaw) Rx(pitch) Rz(roll) with scale removed.
- No component mirrors the frame it processes; sessions record `mirrored: false`.
- Landmarks: 478 points; right iris 468-472 (center 468), left iris 473-477 (center 473); x/y normalized. VIDEO mode needs strictly increasing timestamps (`FaceTracker.process` enforces this).
- Blendshapes (52) are recorded per frame; names are in `meta["blendshape_names"]`. Candidate subset = `BLEND_FEATURE_NAMES` (eyeLook In/Out/Up/Down, Blink, Squint, Wide, L/R = 14). Tracker cost measured live: median 10.1 ms with blendshapes vs 7.2 ms without (p95 12.6 vs 13.4), fine for 30 fps.
- Quartz/Cocoa pyobjc were pulled in at M1 (CLAUDE.md originally said M3) for screen geometry and the overlay window.
- `eval/targets.py` held-out function is `held_out_points` (a `test_` name would be collected by pytest).
- No model save/load yet: the overlay refits from a session on start.
- Deps: mediapipe 1.1.0, opencv-python 5.0.0.93 (mediapipe also pulls opencv-contrib-python; harmless), numpy 2.5.4.
- Settle frames (0.7 s per target) are not recorded; M2 step 6 (lag) would need re-recording.
