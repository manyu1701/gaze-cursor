# Plan

Core technology first; application direction decided after M3.
Each milestone ends with a clear "done when" check. Don't start the next until it passes.

## Checkpoints: when to stop and plan (agreed 2026-10-13)
Within a written step, work goes ahead without asking. **Stop, report plainly (where we are, what the numbers say, what next) and agree the next steps with the author** at any of these checkpoints:
1. **A milestone's done-when is reached**: the author confirms it before the next milestone starts.
2. **A decision gate resolves** (pre-registered test or pre-written rule) and the result changes the route.
3. **New data is needed**: a recording or live test only the author can do at the camera.
4. **The written plan runs out**, or the next step needs a design choice the plan does not make (then propose the steps, with a recommendation).
5. **Before anything hard to undo or outward-facing**: pushing to `main`, deleting branches or data, sending data off the machine.
Between checkpoints: commit as work goes, keep the private progress notes current (and write every new pre-registration into PREREGISTRATIONS.md, committed before its data), and do not build ahead of the current step.

## M0: See the face (one evening)

- Project scaffold with uv, Python 3.12, `.gitignore`, layout from `CLAUDE.md`.
- Download the MediaPipe Face Landmarker model to `models/`.
- `scripts/viewer.py`: live webcam window with face mesh, iris landmarks, and head-pose angles drawn; FPS counter.

**Done when:** `uv run scripts/viewer.py` shows landmarks tracking the irises in real time at ≥ 20 fps, and `q` quits cleanly.

## M1: First gaze estimate and first accuracy number

- Feature extraction (iris-in-eye position, head pose, eye openness), with unit tests.
- Fullscreen calibration routine: dots on a grid (e.g. 9 calibration + separate test points); record each session to `data/sessions/`.
- Fit a baseline model (ridge / polynomial regression).
- Live on-screen gaze dot overlay.
- `scripts/benchmark.py`: offline evaluation on held-out points → px / cm / degrees, jitter; append to results log.

**Done when:** a gaze dot visibly follows where you look, and the benchmark prints a reproducible baseline error from a recorded session.

## M2: Make it good (ML/DS track)

Ordered by what the M1 data says matters most (exploratory analyses of the first sessions, kept in the author's project notes).

**Status (2026-10-13):** steps 1 and 2 are **done**. The pre-registered H1 test (richer pose-free features beat iris-only across sessions) was run once and is **not supported** (paired gain +0.12 deg, 95% CI [-0.43, +0.67]). *Exploratory, found afterwards:* lighting dominates the error (dim 7.3-9 deg even trained on itself; front-lit 1.3-1.7 deg with its own calibration but 5.3 deg when trained on a pool from other lighting); model-variant differences (0.1-0.6 deg) are small next to that. Step 2b follows from this.

1. **DONE. Evaluation foundation:** grouped CV by target (leave-one-target-out) and leave-one-session-out; bootstrap CIs over targets; paired model comparisons; per-target bias vs spread report; screen error heatmap. Tune ridge alpha with grouped CV (alpha ~100 already beat alpha 1 on held-out targets).
2. **DONE. Calibration redesign:** more targets (16-25) that reach close to the safe screen edges (no extrapolation at test time); shorter recording per target (~1 s; error rises after ~1.2 s); randomized order.
2b. **DONE. Lighting and replayable images** (pre-registered round 2: **H2 supported**, d = +2.42 deg [+1.43, +3.41]; **H3 supported**, gap +4.08 deg [+2.83, +5.38]; caveats in PREREGISTRATIONS.md).
   - **Replayable images:** for every frame in new sessions save a native-resolution face crop (JPEG, local only, gitignored, size estimate per session) so image-level fixes and pretrained gaze models can be tested offline on recorded data.
   - **Lighting gate:** a data-based minimum face brightness (proposed 60/255; recommended 100) in the setup checklist with clear front-light guidance; the U override stays.
   - **Pre-registered round 2:** H2 = calibrating in the lighting you will use beats a pool recorded in other lighting; H3 = sessions that pass the brightness gate are more accurate than sessions below it. Smallest design: 4 new sessions (front-lit, backlit, 2 dim) plus 2 existing sessions reused as training twins. Protocol and success criteria are written in PREREGISTRATIONS.md before anything is recorded.
3. **Head position and the vertical bias** (pre-registered round 3: H4 = the current model loses accuracy when the head moves; H5 = a frozen step-3 candidate trained on all calibrations beats the reference on a held-out position; protocol in PREREGISTRATIONS.md). **Round 3 result (2026-10-13): H4 SUPPORTED (+1.94 deg, CI [+1.13, +2.65]); H5, amended before set B to "a fresh 9-dot recalibration beats the pooled model", SUPPORTED (+1.29 deg, CI [+0.80, +1.77], 5/5 positions). The head-geometry models were worse on set A and are parked.** (An earlier, separate freeze of H5 on a diverged branch, `dfa75ff`: `gaze_a100` pooled vs today's centre-only Run; both are reported in PREREGISTRATIONS.md, "Two freezes of round 3".) **Vertical bias (2026-10-13, exploratory diagnosis with a pre-specified gate): NOT REAL in layout v2 (top band +0.49 deg, CI [-0.50, +1.62]); step 3 closed. Post-hoc: predictions are compressed toward the centre on both axes.** What remains is that centre compression: measured in M3 step 7 (head-and-eyes test), addressed by M3 step 8 (pretrained gaze direction). **Preferred direction: a 3D gaze ray (eye position from iris-size distance + head pose, gaze direction) intersected with the screen plane**: it fixes head-pose sensitivity and is the foundation for M5. **Head-transform check (2026-10-13 evening):** the head position is measured well (set B moves recovered to ~1-2 cm), but the iris features respond only ~0.2x to head position, so the ray needs a better gaze direction (step 8), not a better head estimate.
4. **Multi-session dataset:** record sessions across lighting, posture, glasses, time of day; train on some, test on others. Cross-session error is the headline metric.
5. **Robustness to outliers:** blink/outlier rejection and robust regression (Huber/RANSAC); handle the heavy tail (mean >> median).
6. **Temporal model:** fixation/saccade classification (velocity threshold first, HMM later) and One-Euro smoothing tuned per state; measure jitter vs lag and end-to-end latency. **Status (2026-10-13): One-Euro tuned offline by a rule written first (min_cutoff 0.05 Hz, beta 0.001): held-out jitter 0.77 vs 0.83 deg, lag90 97 vs 125 ms against the old moving average; now in Run, with live per-stage latency logging. The fixation/saccade HMM is not done (a snap-on-saccade variant did not lower jitter).**
7. **Active calibration (stretch):** pick the next calibration point where GP uncertainty is highest; fewer points for the same accuracy.
8. **Transfer learning (promoted to M3 step 8, 2026-10-14):** pretrained gaze model (e.g. L2CS-Net via Hugging Face) as a prior; personal calibration learns a small correction.
9. **Calibration-cost curve (later; depends on step 1, and on step 8 and the saved face images from step 2b for its zero-dot arm):** held-out error vs number of calibration dots for (i) a zero-dot pretrained gaze model, (ii) zero-dot pooled own earlier sessions (train on other sessions, no dots for the new one), and (iii) 1/3/5/9/25-dot calibration on the new session (nested subsets of the v2 targets chosen by farthest-point sampling, repeated over seeds), tested on held-out targets, with bootstrap CIs and a per-lighting breakdown. Answers: how many dots are really needed, and can a pretrained or pooled start replace them?

10. **Automatic session quality report (planned 2026-10-13; now M3 step 6b):** a deterministic script (later a hook in calibrate / Run) that runs after every recorded session and flags problems, so "was this calibration done properly?" is answered by numbers, not by eye. Checks: (i) **coverage**: calibration targets span the tested area incl. edges (no extrapolation at test time); (ii) **fixation quality**: per-target within-fixation spread, blink / look-away / dropped-frame share; (iii) **held-out fit**: leave-one-target-out error (mean / median / p90, deg) and outlier targets; (iv) **bias vs noise**: per-target bias against spread, plus the centre-compression check; (v) **drift and head movement**: early vs late error, head shift vs calibration; (vi) **conditions**: lighting gate result and that the structured condition tags are present. Output: a pass / warn / fail line per check plus a PNG of targets and predictions, saved next to the session; a summary row in the results log. Thresholds are written down (from the existing sessions) before the report is used to accept or reject a new session. Not required for M2's done-when; built as M3 step 6b, required for new recordings from M3 step 7 on (not applied to round 4).

11. **Core requirement: follow the head, with no input (added 2026-10-13 at the author's request).** The head is never still, so the tracker must (a) **stay accurate as the head moves, re-adjusting by itself**: no R key, no extra calibration dots; and (b) **count head turns as looking**: real gaze = head orientation + eye-in-head rotation, and people turn their heads as well as their eyes to look around the screen. **Today neither holds:** the model uses iris-in-eye position only (raw head-pose features made the dot unstable), so moving costs ~2-5 deg until R, and a look made partly with the head under-moves the dot (hypothesis, to test: this contributes to the centre compression). Route:
    1. **Measure first:** a recorded / live "head-and-eyes" task: the same targets looked at (i) eyes only, head still, (ii) turning the head naturally, (iii) after moving 10-20 cm, with no recalibration; error per condition with CIs (also a check in the step-10 report and a Run live test).
    2. **Proper fix:** a 3D gaze ray = head pose (rotation + position: already tracked well, head-transform check) combined with an eye-in-head direction from a pretrained, pose-invariant gaze model (step 8), intersected with the screen plane; the personal calibration learns only a small correction. This handles (a) and (b) together and is the M5 foundation (other cameras and screens).
    3. **Bridge until then (M3):** implicit self-recalibration from natural use (clicks you make anyway, recursive least squares, drift detection) and an automatic re-adjust when Run detects the head has moved past threshold (the detection exists).
    **Done when:** after turning the head to look or moving 10-20 cm, error stays within ~1 deg of the calibrated-position error with no user input, reported with CIs over held-out targets.

**Done when:** a results table with CIs shows, on held-out targets *and* held-out sessions, which changes improved accuracy and by how much (the portfolio write-up starts here).

**Status: M2 DONE (confirmed by the author, 2026-10-13 evening; results in `RESULTS.md`).** Steps 1, 2, 2b, 3 and 6 done; step 4 partly; the open steps are re-homed below. Step 11 (follow the head, no input) is the main accuracy goal, via M3 steps 7-8 and the round-4 bridge. Run now has the quick 9-dot calibration, One-Euro smoothing and live tests (T / A / P). Pre-registrations and their results: `PREREGISTRATIONS.md`.

**Where the open M2 steps went (agreed 2026-10-14):**
- **4 multi-session dataset:** ongoing (structured condition tags on every session); the cross-session headline moves to **M5** (data design).
- **5 outliers / robust regression:** **M3, after round 4**; exploratory unless it becomes a claim (then pre-registered).
- **6 rest (fixation/saccade HMM):** **M4** (dwell clicks and snap-to-target need it).
- **7 active calibration:** **M5** (few dots per new setup), stretch.
- **8 pretrained gaze model:** **promoted to M3 step 8.**
- **9 calibration-cost curve:** after M3 step 8; evidence for the M4 decision.
- **10 quality report:** **M3 step 6b, build now.** Ordering slip on record: step 10 said "build before recording M3 sessions", but round 4 was frozen and recording began without it; round 4 stays as frozen (see 6b).
- **11 follow the head:** measured in **M3 step 7**, bridged by round 4 (click learning), properly fixed by **M3 step 8**; it stays in M3's done-when.

## M3: Control the real cursor

**Decisions (the author, 2026-10-13):** warp = **tap a hotkey** (Right Command, chosen by the author after the first live try; Right Option before) and the cursor jumps to the gaze point once; the trackpad finishes. **Clicks stay on the trackpad** in the first pass; every click is logged as a labelled gaze sample and drives input-free self-recalibration. Dwell / blink clicking comes later as a measured experiment. Build order: quick-calibration refactor -> cursor + input platform module (listen-only event tap, Input Monitoring) -> click-through status overlay -> `Cursor.command` -> click log -> online RLS + drift (pre-registered before recording) -> head-and-eyes test H + Fitts task. (Accessibility turned out unnecessary: no synthetic clicks; only Input Monitoring.)

**Status (2026-10-14):** steps 1-5 built and working live (`Cursor.command`; jump = Right Command tap, shortcuts Cmd+Option+G/R/X). Step 6 (click learning, `online.py`) built and its test **frozen as round 4** (PREREGISTRATIONS.md); **1 of 3 qualifying sessions recorded** (counts only by the frozen data rule, via `--status`); waiting for 2 more. Step 7 (head-and-eyes test H, Fitts task) not started. **2026-10-14:** steps re-written below (6b quality report next; step 8 pretrained gaze model promoted from M2).

**Steps (re-written 2026-10-14; one at a time, checkpoints as in "Checkpoints"):**
1-5. **DONE.** Quick-calibration refactor; cursor warp + listen-only event tap (Input Monitoring, no Accessibility); click-through status overlay; `Cursor.command` (tap Right Command = jump, Cmd+Option+G / R / X (quit moved from Q on 2026-10-14: Terminal also sees the listen-only key and Cmd+Option+Q is its "Quit and Keep Windows")); click log (`data/quick/cursor_*.npz`).
6. **Online learning from clicks (round 4, H6):** built and frozen; **waiting for 3 qualifying sessions** (checkpoint 3). Intake uses `evaluate_online.py --status` only; the test runs once with `--preregistered`. Drift (CUSUM) is logged only.
6b. **Session quality report (M2 step 10), build now while round 4 waits.** Thresholds written down from existing sessions before first use. Required for every new recording from step 7 on. **Not applied to round 4:** round 4 was frozen without it, so round-4 sessions are classified only by its frozen data rule; report lines for them are informational and only after the H6 test has run.
7. **Measure head and eyes + usability:** live test H (eyes only / turning the head / moved 10-20 cm, no R) and a Fitts' law task (ISO 9241-411 style; trackpad vs gaze-jump + trackpad), each pre-registered; recordings save face crops so step 8 can be evaluated on them.
8. **Pretrained, pose-invariant gaze direction (promoted from M2 step 8):** an L2CS-Net-style model gives eye-in-head direction; combined with the tracked head pose it gives a 3D gaze ray intersected with the screen plane; the personal calibration learns only a small correction. Evaluated offline first on recorded sessions with face crops (exploratory), then a pre-registered comparison against the frozen quick calibration. This is the proper fix for M2 step 11 and the M5 foundation.
Later (M4): dwell / blink clicking as a sequential test (confidence accumulates) rather than a fixed timer, measured against trackpad clicks.

**Done when:** you can complete basic tasks (switch windows, click large buttons) with gaze, and the warp mode feels useful, **while sitting naturally (head moving and turning) without pressing R** (M2 step 11, bridge route at least).

## M4: Choose a direction

- Precision tricks: snap-to-target (macOS Accessibility API to find clickable elements), gaze + head refinement.
- Fixation / saccade classification (HMM; the rest of M2 step 6), needed for dwell clicks and snap-to-target; dwell / blink clicking (from M3) measured against trackpad clicks.
- **Bayesian snap-to-target:** P(element | gaze) proportional to likelihood of the gaze under the model's error distribution (GP uncertainty) x prior (element size, usage frequency). Local first; optionally benchmark Jev as an experiment (paired against the local model on the same recorded click tasks: selection accuracy, latency, cost; only gaze coordinates and UI element data leave the machine, never face data). Jev is not used for evaluation or quality checks: those stay deterministic local statistics (decided 2026-10-13).
- Decide: productivity tool, accessibility tool, or portfolio-complete. Then plan packaging (signed Mac app, onboarding, permissions UX).

**Done when:** a direction is chosen with evidence from M1–M3 measurements.

## M5 (later): Generalise beyond this MacBook

Goal: **works with any webcam and any screen layout after a ~30-second calibration, and stays accurate as you move.** Zero-calibration for anyone is research-grade and out of scope.

- **Architecture:** separate person -> 3D gaze ray, camera geometry (intrinsics + pose), screen geometry (one plane per display, from macOS display arrangement + a few calibration dots per screen). New setup = locate camera/screens, not relearn the eyes.
- **5a. Your own variations (cheapest real test):** external webcam (on top of a monitor), second/extended monitor, different desk distances and chair heights.
- **5b. Other people:** friends/family with consent; public datasets for pretraining or evaluation (MPIIFaceGaze, GazeCapture, ETH-XGaze, Gaze360; mostly non-commercial research licences, so check before any product use). Few-shot personal calibration on top of a general model (domain adaptation).
- **5c. Other OSes:** Windows / Linux sinks for cursor control and screen geometry; core unchanged.
- **Active calibration (from M2 step 7, stretch):** pick the next calibration point where the model is least certain, so a new setup needs fewer dots.
- **Data design** (continues M2 step 4; cross-session error stays the headline): planned (fractional) condition matrix over camera x screen x lighting x glasses x posture rather than ad-hoc recording; every session tagged.
- **Evaluation:** leave-one-condition-out, leave-one-person-out, and per-group results (glasses, face types, lighting) as a fairness check; an accessibility tool that works worse for some faces is a real problem.

**Done when:** on held-out setups and held-out people, a short calibration reaches accuracy close to the single-setup result, with per-condition and per-group numbers reported.
