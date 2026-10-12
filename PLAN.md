# Plan

Core technology first; application direction decided after M3.
Each milestone ends with a clear "done when" check. Don't start the next until it passes.

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

Ordered by what the M1 data says matters most (see PROGRESS.md, "Independent analysis").

1. **Evaluation foundation:** grouped CV by target (leave-one-target-out) and leave-one-session-out; bootstrap CIs over targets; paired model comparisons; per-target bias vs spread report; screen error heatmap. Tune ridge alpha with grouped CV (alpha ~100 already beat alpha 1 on held-out targets).
2. **Calibration redesign:** more targets (16-25) that reach close to the safe screen edges (no extrapolation at test time); shorter recording per target (~1 s; error rises after ~1.2 s); randomized order.
3. **Fix the vertical bias:** top-of-screen targets are predicted far too low. Try separate x/y models, regularized nonlinear terms for y, Gaussian Process regression (also gives per-point uncertainty), and better vertical features (eyelid/blendshape-based). **Preferred direction: a 3D gaze ray (eye position from iris-size distance + head pose, gaze direction) intersected with the screen plane**: it fixes head-pose sensitivity and is the foundation for M5.
4. **Multi-session dataset:** record sessions across lighting, posture, glasses, time of day; train on some, test on others. Cross-session error is the headline metric.
5. **Robustness to outliers:** blink/outlier rejection and robust regression (Huber/RANSAC); handle the heavy tail (mean >> median).
6. **Temporal model:** fixation/saccade classification (velocity threshold first, HMM later) and One-Euro smoothing tuned per state; measure jitter vs lag and end-to-end latency.
7. **Active calibration (stretch):** pick the next calibration point where GP uncertainty is highest; fewer points for the same accuracy.
8. **Transfer learning (stretch):** pretrained gaze model (e.g. L2CS-Net via Hugging Face) as a prior; personal calibration learns a small correction.

**Done when:** a results table with CIs shows, on held-out targets *and* held-out sessions, which changes improved accuracy and by how much (the portfolio write-up starts here).

## M3: Control the real cursor

- Cursor sink using Quartz CGEvent (Accessibility permission).
- Click triggers: dwell, blink, and/or hotkey (compare).
- Simple on/off toggle and a safe exit hotkey.
- **Online learning:** every click is a labeled sample (you looked where you clicked); update the model continuously (recursive least squares) and detect drift from rising residuals (e.g. CUSUM).
- Dwell clicks as a sequential test (confidence accumulates over time) rather than a fixed timer.
- Usability metric: Fitts' law throughput (ISO 9241-411 style task) vs trackpad.
- "Warp" mode: cursor jumps to gaze on a hotkey, trackpad finishes.

**Done when:** you can complete basic tasks (switch windows, click large buttons) with gaze, and the warp mode feels useful.

## M4: Choose a direction

- Precision tricks: snap-to-target (macOS Accessibility API to find clickable elements), gaze + head refinement.
- **Bayesian snap-to-target:** P(element | gaze) proportional to likelihood of the gaze under the model's error distribution (GP uncertainty) x prior (element size, usage frequency). Local first; optionally benchmark Jev as an experiment.
- Decide: productivity tool, accessibility tool, or portfolio-complete. Then plan packaging (signed Mac app, onboarding, permissions UX).

**Done when:** a direction is chosen with evidence from M1–M3 measurements.

## M5 (later): Generalise beyond this MacBook

Goal: **works with any webcam and any screen layout after a ~30-second calibration, and stays accurate as you move.** Zero-calibration for anyone is research-grade and out of scope.

- **Architecture:** separate person -> 3D gaze ray, camera geometry (intrinsics + pose), screen geometry (one plane per display, from macOS display arrangement + a few calibration dots per screen). New setup = locate camera/screens, not relearn the eyes.
- **5a. Your own variations (cheapest real test):** external webcam (on top of a monitor), second/extended monitor, different desk distances and chair heights.
- **5b. Other people:** friends/family with consent; public datasets for pretraining or evaluation (MPIIFaceGaze, GazeCapture, ETH-XGaze, Gaze360; mostly non-commercial research licences, so check before any product use). Few-shot personal calibration on top of a general model (domain adaptation).
- **5c. Other OSes:** Windows / Linux sinks for cursor control and screen geometry; core unchanged.
- **Data design:** planned (fractional) condition matrix over camera x screen x lighting x glasses x posture rather than ad-hoc recording; every session tagged.
- **Evaluation:** leave-one-condition-out, leave-one-person-out, and per-group results (glasses, face types, lighting) as a fairness check; an accessibility tool that works worse for some faces is a real problem.

**Done when:** on held-out setups and held-out people, a short calibration reaches accuracy close to the single-setup result, with per-condition and per-group numbers reported.
