# gaze-cursor

Move the macOS cursor with your eyes, using only the built-in webcam.

A learning and portfolio project: build a webcam gaze-estimation engine, then **measure it rigorously**. Accuracy is reported in pixels, centimetres and degrees of visual angle, with jitter, on held-out targets from recorded sessions.

> **Status:** M0 (face and iris tracking) is done. M1 (first gaze estimate and first accuracy number) is implemented but has not yet been run on a real calibration session, so there are **no accuracy results yet**. See [Roadmap](#roadmap).

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
uv run scripts/calibrate.py              # fullscreen: 9 calibration dots, then 12 held-out test dots
uv run scripts/benchmark.py              # fit on calibration, report error on test dots
uv run scripts/gaze_overlay.py           # live gaze dot, using the latest session
```

- `calibrate.py`: sit about 60 cm away with your head fairly still. `SPACE` starts and `ESC` aborts. The session is saved to `data/sessions/`.
- `benchmark.py [session.npz] [--model ridge|poly2] [--alpha 1.0] [--features iris|eyes|gaze|blend|both]`: runs offline from a recording, with no camera. `iris` is the baseline (iris position, head pose, eye openness); `eyes` drops head pose; `gaze` is iris position only; `blend` uses head pose plus the face blendshapes; `both` combines iris and blendshapes. It prints error (px, cm, degrees; mean and median) and jitter, and appends a row to `data/results.csv`.
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
- **Traceable results.** Every benchmark run appends its config and metrics to `data/results.csv`.
- **Privacy.** Camera frames and face data never leave the device. Recorded sessions stay in the gitignored `data/sessions/`.

## Project layout

```
src/gaze_cursor/   capture, landmarks, features, calibration, config, eval/, sinks/
scripts/           viewer, download_model, calibrate, benchmark, gaze_overlay
tests/             pytest suite
data/sessions/     recorded sessions (gitignored)
data/results.csv   benchmark log
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
| M1 | First gaze estimate and first accuracy number | Implemented, awaiting a real session |
| M2 | Make it good: smoothing, head-pose compensation, model comparison, robustness, latency | Planned |
| M3 | Control the real cursor: Quartz CGEvent, click triggers, warp mode | Planned |
| M4 | Choose a direction: precision tricks, intent model, productivity vs accessibility | Planned |
