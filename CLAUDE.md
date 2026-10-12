# gaze-cursor

Move the macOS cursor by tracking eye gaze through a standard webcam.

## Goal and direction

- **Now:** a learning and portfolio project. Build the core gaze-estimation engine well and **measure it rigorously**.
- **Later (undecided):** a daily productivity tool (gaze warps the cursor, trackpad finishes the move) and/or an accessibility tool (hands-free). Keep the core modular so either path stays open. Don't make product decisions yet.
- See `PLAN.md` for milestones. Work one milestone at a time; don't build ahead.

## Reality check (design around this)

- Webcam gaze accuracy is typically **2–5° (~2–5 cm on screen at 60 cm)**. It's good for coarse positioning, bad for pixel-precise pointing. Don't promise mouse-level precision.
- Main problems: jitter (fixational eye movements), head movement breaking calibration, lighting/glasses, drift over a session, and the "Midas touch" problem (looking ≠ clicking).
- Known fixes to keep in mind: smoothing (One-Euro filter), head-pose compensation, snap-to-target, gaze + head/key refinement, explicit click triggers (dwell/blink/hotkey).

## Platform and stack

- macOS (Apple Silicon MacBook Air), built-in webcam.
- **Python 3.12** managed with **uv** (`uv run ...`, `uv add ...`). Don't use 3.13 (MediaPipe wheel support).
- `mediapipe`: face and iris landmarks. **Use the Tasks API (`mediapipe.tasks.python.vision.FaceLandmarker`), not the legacy `mp.solutions.face_mesh`.** Check current API via Context7 before writing MediaPipe code.
- `opencv-python`: capture and debug drawing. `numpy`, `scikit-learn`: features and calibration models.
- `pyobjc-framework-Quartz` / `-Cocoa`: screen geometry and the calibration overlay window (since M1); cursor control via CGEvent (M3+).
- Needs macOS permissions: **Camera** (M0+) and **Accessibility** (M3+) for the app running Python (Terminal / VS Code).

## Architecture

Pipeline of swappable stages with clean interfaces:

```
capture → landmarks → features → calibration model → smoothing → output sink
```

- **capture:** webcam frames + timestamps.
- **landmarks:** MediaPipe face/iris landmarks + head pose (transformation matrix).
- **features:** iris position relative to eye corners, head yaw/pitch/roll, eye openness. Pure functions; unit-test these.
- **calibration:** maps features → screen (x, y). Ridge baseline; see "Modeling principles" for what comes next.
- **smoothing:** One-Euro filter (tunable).
- **output sinks:** on-screen gaze dot (debug), real cursor (M3+), log file. The core must not depend on any specific sink.

Suggested layout (adjust if there's a good reason):

```
src/gaze_cursor/   capture.py landmarks.py features.py calibration.py filters.py sinks/ eval/
scripts/           runnable entry points (viewer, calibrate, benchmark)
tests/
data/sessions/     recorded sessions (gitignored)
models/            downloaded model files e.g. face_landmarker.task (gitignored)
```

## Measurement principles (the core of this project)

- **Record and replay:** every calibration/test session saves timestamps, landmarks/features, and the ground-truth target positions to `data/sessions/`. Models must be trainable and evaluable offline from recordings without the camera.
- **Held-out evaluation:** calibrate on some targets, test on others. Report error in **px, cm, and degrees of visual angle**, plus **jitter** (std during fixation) and **end-to-end latency**.
- Every benchmark run appends to a results log (CSV/JSON) with config + metrics, so improvements are traceable.

## Modeling principles (learned from the M1 data, keep to these)

- **The unit of data is a target (fixation), not a frame.** Frames within one fixation are highly correlated; 400 frames from 9 targets is ~9 samples. Size models accordingly and prefer strong regularization / few parameters until there are many more targets or sessions.
- **Never split frames randomly for validation.** Use grouped CV (`GroupKFold` / leave-one-target-out by `target_idx`) within a session, and leave-one-session-out across sessions. Random frame splits leak and give meaningless scores.
- **Report the honest number:** held-out targets, and ideally held-out *sessions* (different lighting/posture/time). Cross-session error is the real-world accuracy.
- **Don't use tree models (random forest, gradient boosting) for the screen mapping.** They cannot extrapolate and flatten at the edges, which is where errors already are. Prefer ridge, low-order/regularized nonlinear terms, Gaussian processes, or a pretrained gaze model + small personal correction.
- **Diagnose bias vs noise separately:** per-target bias (mean prediction − target) and within-fixation spread. Smoothing only fixes noise; bias needs better features, calibration coverage, or models.
- **Calibration must cover the region it's tested on** (including near edges); otherwise the model extrapolates.
- **Comparisons are paired and come with uncertainty:** compare models on the same targets (paired differences), report bootstrap CIs (resample targets, not frames) and effect sizes, not just a single mean.
- Use mean, median and p90 error; a mean far above the median means heavy-tailed outliers worth handling explicitly (robust losses, outlier/blink rejection).

## Conventions

- Typed Python, small pure functions where possible, docstrings on public functions.
- Tests with pytest for feature extraction, filters, and calibration math.
- Debug aids: be able to dump annotated frames as PNG so issues can be inspected visually.
- Don't send camera frames or face data anywhere off-device.
- Keep the loop real-time: target ~30 fps; measure before optimizing.

## Ideas parked for later

- Pretrained appearance-based gaze models (e.g. L2CS-Net) and gaze datasets (MPIIGaze, GazeCapture) at M2 (via Hugging Face MCP).
- Intent / target selection at M4: local probabilistic model first; optionally benchmark TypeSafe's Jev as an experiment.
- Packaging as a proper Mac app (possibly Swift) only after M3 proves value.
