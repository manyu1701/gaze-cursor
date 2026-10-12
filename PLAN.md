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

## M2: Make it good

- One-Euro smoothing (tune jitter vs lag).
- Head-pose compensation (features and/or model).
- Model comparison on the same recordings (linear vs polynomial vs small NN vs a pretrained gaze model).
- Robustness checks: lighting, glasses on/off, posture shift; drift over 10+ minutes; quick recalibration.
- Measure end-to-end latency.

**Done when:** a results table shows which changes improved accuracy and by how much (the portfolio write-up starts here).

## M3: Control the real cursor

- Cursor sink using Quartz CGEvent (Accessibility permission).
- Click triggers: dwell, blink, and/or hotkey (compare).
- Simple on/off toggle and a safe exit hotkey.
- "Warp" mode: cursor jumps to gaze on a hotkey, trackpad finishes.

**Done when:** you can complete basic tasks (switch windows, click large buttons) with gaze, and the warp mode feels useful.

## M4: Choose a direction

- Precision tricks: snap-to-target (macOS Accessibility API to find clickable elements), gaze + head refinement.
- Intent / target selection model (local first; optionally benchmark Jev).
- Decide: productivity tool, accessibility tool, or portfolio-complete. Then plan packaging (signed Mac app, onboarding, permissions UX).

**Done when:** a direction is chosen with evidence from M1–M3 measurements.
