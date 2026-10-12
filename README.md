# gaze-cursor

Move the macOS cursor with your eyes, using only the built-in webcam.

A learning and portfolio project: build a webcam gaze-estimation engine, then **measure it rigorously**. Accuracy is reported in pixels, centimetres and degrees of visual angle, with jitter, on held-out targets from recorded sessions.

> **Status:** M0 (face and iris tracking) and M1 (first gaze estimate and baseline accuracy) are done. M2 is under way: the evaluation harness and the improved calibration layout are built, and a pre-registered hypothesis (see `PROGRESS.md`) is waiting on three new recording sessions. Early numbers (roughly 3 to 5 degrees of error depending on session and method) are exploratory; there is no confirmed result yet. See [Roadmap](#roadmap).

## What to expect

Webcam gaze is coarse. Typical accuracy is **2–5° (about 2–5 cm on screen at 60 cm)**. That is fine for rough positioning and poor for pixel-precise pointing. The hard parts are jitter, head movement breaking calibration, lighting and glasses, drift over a session, and looking not being the same as clicking. This project measures how much each fix helps rather than promising mouse-level precision.

## Requirements

- macOS on Apple Silicon, with a built-in webcam. Developed on a MacBook Air 13.6" (M5, 2560×1664).
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12, which is required because MediaPipe wheels do not support 3.13.
- **Camera** permission for the app that runs Python (Terminal or VS Code). You are prompted on first run. Accessibility permission is only needed from M3, for cursor control.

## Quick start

```bash
uv sync                                  # install dependencies
uv run scripts/download_model.py         # fetch the MediaPipe face model (~4 MB, one time)
uv run scripts/viewer.py                 # live face mesh, irises, head pose, FPS
```

In the viewer, `q` quits and `s` saves an annotated frame to `debug/`.

## Measuring gaze (M1)

```bash
uv run scripts/calibrate.py --check-screen   # once: verify the fullscreen window maps onto the screen
uv run scripts/calibrate.py --note "evening, lamp"   # fullscreen: 16 calibration + 9 held-out dots (layout v2)
uv run scripts/benchmark.py              # fit on calibration, report error on held-out dots
uv run scripts/evaluate.py               # all clean sessions: within-session, leave-one-target-out, leave-one-session-out
uv run scripts/check_setup.py            # are you sitting like in the recorded sessions? (live hints)
uv run scripts/gaze_overlay.py           # live gaze dot, using the latest session
```

- `calibrate.py [--layout v1|v2] [--seed N] [--note TEXT]`: sit about 60 cm away with your head fairly still. `SPACE` starts and `ESC` aborts. The session is saved to `data/sessions/`. Layout `v2` (default) shows 25 dots in random order, about 1 s of recording each (about 45 s in total): 16 calibration dots on a grid that reaches the safe screen edges, and 9 held-out dots strictly inside it. `v1` is the original 9 + 12 layout. The held-out points and order come from a seed that is stored in the session, and `--note` tags the lighting or time of day.
- `benchmark.py [session.npz] [--model ridge|poly2] [--alpha 1.0] [--features iris|eyes|gaze|blend|both]`: runs offline from a recording, with no camera. `iris` is the baseline (iris position, head pose, eye openness); `eyes` drops head pose; `gaze` is iris position only; `blend` uses head pose plus the face blendshapes; `both` combines iris and blendshapes. It prints error (px, cm, degrees; mean and median) and jitter, and appends a row to `data/results.csv`.
- `evaluate.py [--configs NAME ...|all] [--modes within loto losos] [--compare REF NEW] [--per-target]`: the evaluation harness. It runs over all clean sessions (those drawn on the full display; earlier sessions with a shifted window are excluded) and reports error per target, because frames within a fixation are not independent. The three modes are `within` (calibration targets to held-out targets), `loto` (leave one target out) and `losos` (leave one session out, training on the others pooled). Each result has a bootstrap 95% confidence interval that resamples targets, and `--compare` gives a paired comparison. `--per-target` shows each target's bias and spread. Every run is logged with its mode to `data/results.csv`. `--preregistered` runs the pre-registered test once three layout-v2 sessions exist, and refuses until then.
- `check_setup.py [--last N]`: the posture check. It shows your camera view, a gauge per measure (head pitch, yaw, roll, distance, face position) against where you sat in the recorded sessions, hints on what to adjust, and a green "IN POSITION" once you have held it for a second. The reference is computed from your recorded sessions, so it updates itself as you add sessions; lighting is shown but never blocks. `calibrate.py` also shows a short setup checklist first (sit all the way back, about an arm's length from the screen, laptop on a stable desk, head still) and, after saving, a summary of how well you held your position. The same check runs automatically before `calibrate.py` (against the earlier sessions) and `gaze_overlay.py` (against the session its model was trained on), and the overlay keeps a small live posture hint in the corner. Use `--no-check` to skip it.
- `gaze_overlay.py [session.npz] [--model ridge|poly2] [--features gaze|eyes|iris|blend|both]`: `q` or `ESC` quits. It defaults to `gaze` (iris only) because raw head pose makes the dot jump when your posture drifts from calibration. Calibrate and use it in the same posture; it warns when the live input is far from the calibration data.

Screen size (in macOS points), physical size, and the notch/menu-bar strip are read from macOS at run time, and targets are kept clear of the notch and the rounded top corners, and the viewing distance defaults to 60 cm (`src/gaze_cursor/config.py`). Frames are never mirrored for processing or recording; the viewer flips its display only. Face blendshapes are recorded in every session as candidate features.

## How it works

```
capture → landmarks → features → calibration model → smoothing → output sink
```

| Stage | Module | What it does |
| --- | --- | --- |
| capture | `capture.py` | Webcam frames with monotonic timestamps |
| landmarks | `landmarks.py` | MediaPipe FaceLandmarker (Tasks API): 478 face and iris points, 52 blendshape scores, and head pose from the transformation matrix |
| features | `features.py` | Iris position within each eye, head yaw/pitch/roll, eye openness, plus an optional blendshape subset. Pure functions; blink frames are dropped |
| calibration | `calibration.py` | Ridge or polynomial-ridge regression from features to screen (x, y) |
| smoothing | | One-Euro filter (planned, M2) |
| output sinks | `sinks/` | On-screen gaze dot today; real cursor from M3 |

Each stage is swappable, and the core does not depend on any particular sink.

### Measurement principles

- **Record and replay.** Every session stores timestamps, landmarks, features and the ground-truth target positions, so models can be trained and evaluated offline.
- **Held-out evaluation.** Calibrate on one set of targets and test on different ones.
- **Traceable results.** Every benchmark and evaluation run appends its config, mode and metrics to `data/results.csv`.
- **Pre-registration.** The M2 hypothesis, protocol and success criterion were written down in `PROGRESS.md` before the confirming sessions were recorded.
- **Privacy.** Camera frames and face data never leave the device. Recorded sessions stay in the gitignored `data/sessions/`.

## Project layout

```
src/gaze_cursor/   capture, landmarks, features, calibration, config, eval/, sinks/
scripts/           viewer, download_model, calibrate, benchmark, evaluate, gaze_overlay
tests/             pytest suite
data/sessions/     recorded sessions (gitignored)
data/results.csv   benchmark and evaluation log
models/            downloaded model files (gitignored)
```

## Development

```bash
uv run pytest                            # unit tests (features, metrics, calibration, head pose)
```

## Roadmap

See [`PLAN.md`](PLAN.md) for details and each milestone's "done when" check.

| Milestone | Goal | Status |
| --- | --- | --- |
| M0 | See the face: live landmarks, head pose, FPS | Done |
| M1 | First gaze estimate and first accuracy number | Done |
| M2 | Make it good: evaluation harness, calibration redesign, vertical bias, robustness, smoothing | In progress (steps 1 and 2 built) |
| M3 | Control the real cursor: Quartz CGEvent, click triggers, warp mode | Planned |
| M4 | Choose a direction: precision tricks, intent model, productivity vs accessibility | Planned |
