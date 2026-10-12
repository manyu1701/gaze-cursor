# gaze-cursor

Move the macOS cursor with your eyes, using only the built-in webcam.

A learning and portfolio project: build a webcam gaze-estimation engine, then **measure it rigorously**. Accuracy is reported in pixels, centimetres and degrees of visual angle, with jitter, on held-out targets from recorded sessions.

> **Status:** M0 (face and iris tracking) and M1 (first gaze estimate and baseline accuracy) are done. M2 is under way.
> - The first pre-registered test (do richer, pose-free features beat iris-only features across sessions?) came out **not supported**: a gain of 0.12° with a 95% CI of −0.43° to +0.67°.
> - The main finding so far, from *exploratory* analysis, is that **lighting matters far more than the model**: with a front light and a calibration made in that light the error was about 1.5°, in dim light 7 to 9°. A second pre-registered round tests this and is waiting on new recordings (see [`PROGRESS.md`](PROGRESS.md)).
> - The live dot works. Real cursor control is M3 and is not built yet. See the [roadmap](#roadmap).

## What to expect

Webcam gaze is coarse. Typical accuracy is **2–5° (about 2–5 cm on screen at 60 cm)**. That is fine for rough positioning and poor for pixel-precise pointing. The hard parts are jitter, head movement breaking calibration, lighting and glasses, drift over a session, and looking not being the same as clicking. This project measures how much each fix helps instead of promising mouse-level precision.

## Requirements

- macOS on Apple Silicon with a built-in webcam. Developed on a MacBook Air 13.6" (M5, 2560×1664).
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12, which is required because MediaPipe wheels do not support 3.13.
- **Camera** permission for the app that runs Python (Terminal or VS Code); you are prompted on first run. Accessibility permission is only needed from M3, for cursor control.

## Quick start

```bash
uv sync                                  # install dependencies
uv run scripts/download_model.py         # fetch the MediaPipe face model (~4 MB, one time)
uv run scripts/viewer.py                 # live face mesh, irises, head pose, FPS (q quits, s saves a frame)
```

### One-click launchers

Double-click these in Finder. They open Terminal, which already has the Camera permission. If macOS will not open one, run `chmod +x Calibrate.command Run.command` once.

| Launcher | What it does |
| --- | --- |
| **`Calibrate.command`** | Asks for an optional note, then (1) a dark, calibration-like screen with a live lighting reading, the brightness gate and front-light tips (SPACE continues, ESC quits), (2) the live setup checklist, (3) the 25-dot calibration. The lighting is captured automatically and saved with the session. |
| **`Run.command`** | Starts the live gaze dot with your latest calibration, after a position and lighting check. If you have not calibrated yet it tells you to run `Calibrate.command` first. It will switch to real cursor control when M3 is built. |

## Recording a session

```bash
uv run scripts/calibrate.py --note "evening, lamp"
```

Sit about an arm's length (60 cm) from the screen, back against the chair, and keep your head still. 25 dots appear in random order, about 1 s of recording each (about 45 s in all): 16 calibration dots on a grid that reaches the safe screen edges, and 9 held-out dots strictly inside it. The session is saved to `data/sessions/`.

| Option | Meaning |
| --- | --- |
| `--note TEXT` | Free-text tag, e.g. the time of day. If it names a lighting condition (dim, backlit, front-lit) that the camera does not measure, you are warned. |
| `--guided` | First set up your lighting on a dark calibration-like screen (this is what `Calibrate.command` uses). |
| `--check-light` | Only show the live lighting as a recording would see it, then exit. |
| `--check-screen` | Only verify that the fullscreen window maps onto the screen (notch, corners). |
| `--layout v1\|v2` | `v2` (default) is the 25-dot layout; `v1` is the original 9 + 12 layout. |
| `--dry-run` | Save to `data/dry_runs/`, so a practice run is never mistaken for a real session. |
| `--no-images` | Do not save the face images (see below). |
| `--no-check` | Skip the setup guidance and the position check. |
| `--seed N` | Seed for the held-out points and the order (stored in the session). |

### The setup checklist

Before recording, a live checklist shows five steps, each turning green as you get it right, with a hint when it is not:

1. Sit all the way back, at about an arm's length (only your distance is measured, not the chair).
2. Laptop on a stable desk at the usual lid angle (shows up as head pitch).
3. Face the screen squarely.
4. Light on your face.
5. Keep your head still and move only your eyes.

It is guidance, not a gate you cannot pass:

| Key | Does |
| --- | --- |
| **SPACE** | Starts, once every step is green. |
| **U** | Uses your current setup instead (calibrate in the setup and lighting you really use). The session records that you chose this. |
| **ESC / Q** | Cancels. |

### Lighting

Lighting is measured on every recorded frame (face and background brightness, 0 to 255) and saved with the session with a label: *dim* (face below 50), *backlit* (face much darker than the background), *front-lit* (face at least 100 and at least 0.9 times the background brightness), or *mixed*.

- The checklist **blocks SPACE below a face brightness of 60** and tells you to put a lamp or window in front of you, aimed at your face. 100 or more is recommended. U still overrides.
- The 60 comes from three exploratory sessions, so it is a hypothesis that the second pre-registered round tests, not a measured optimum.
- Brightness behind you is never a gate, only a tip.

### Face images

Each session also saves a compressed crop of your face for every frame (`<session>.faces.npz`, about 10 to 15 MB, **local only**, gitignored, readable only by you), so image-level fixes and pretrained gaze models can be tested offline later. Re-running the landmarker on a saved crop gives slightly different features than the live run (about 14 to 24% of their usual spread), so always compare variants against the replayed baseline, not against live numbers.

## Using and checking a calibration

```bash
uv run scripts/gaze_overlay.py           # live gaze dot, using the latest session (q or ESC quits)
uv run scripts/check_setup.py            # live posture gauges against your recorded sessions
uv run scripts/benchmark.py              # fit on calibration, report error on held-out dots
uv run scripts/evaluate.py               # all clean sessions: within, leave-one-target-out, leave-one-session-out
```

- **`gaze_overlay.py [session.npz] [--model ridge|poly2] [--features ...]`**: the live dot. It defaults to the `gaze` feature set (iris only) because raw head pose makes the dot jump when your posture drifts from calibration. Before it starts it runs the position and lighting check against the session its model was trained on (U runs anyway, with lower accuracy), and it keeps a small posture hint in the corner while you use it.
- **`check_setup.py [--last N]`**: the posture check on its own. The reference position is computed from your recorded sessions, so it updates itself as you add sessions.
- **`benchmark.py [session.npz] [--model ridge|poly2] [--alpha A] [--features ...]`**: offline, no camera. Prints error (px, cm, degrees; mean and median) and jitter, and appends a row to `data/results.csv`.
- **`evaluate.py`**: the evaluation harness. It works per *target* (frames within one fixation are not independent) over all clean sessions, with bootstrap 95% confidence intervals that resample targets.

Feature sets (`--features`): `iris` (iris position, head pose, eye openness; the baseline), `eyes` (drops head pose), `gaze` (iris position only), `blend` (head pose plus face blendshapes), `gaze_blend` (iris position plus eye blendshapes, pose-free), `both` (iris plus blendshapes).

### Evaluation harness options

| Option | Meaning |
| --- | --- |
| `--configs NAME ... \| all` | Which model configurations to run. |
| `--modes within loto losos` | `within`: calibration targets to held-out targets of the same session. `loto`: leave one target out. `losos`: leave one session out, training on the others pooled. |
| `--compare REF NEW` | Paired comparison on identical targets. |
| `--per-target` | Each target's bias and spread. |
| `--preregistered` | The pre-registered round-1 test (H1), run once; refuses until three layout-v2 sessions exist. |
| `--preregistered-2` | The pre-registered round-2 lighting tests (H2, H3); refuses until the right sessions exist. |

Every run is logged with its mode to `data/results.csv`.

## How it works

```
capture → landmarks → features → calibration model → smoothing → output sink
```

| Stage | Module | What it does |
| --- | --- | --- |
| capture | `capture.py` | Webcam frames with monotonic timestamps |
| landmarks | `landmarks.py` | MediaPipe FaceLandmarker (Tasks API): 478 face and iris points, 52 blendshape scores, head pose from the transformation matrix |
| features | `features.py` | Iris position within each eye, head yaw/pitch/roll, eye openness, an optional blendshape subset. Pure functions |
| calibration | `calibration.py` | Ridge or polynomial-ridge regression from features to screen (x, y) |
| smoothing | | One-Euro filter (planned, M2) |
| output sinks | `sinks/` | On-screen gaze dot today; real cursor from M3 |

Around the pipeline: `posture.py` (setup checklist, reference posture, lighting), `config.py` (screen geometry from macOS, notch and rounded-corner aware), and `eval/` (sessions, layouts, splits, statistics, the pre-registered analyses). Each stage is swappable, and the core does not depend on any particular sink. Frames are never mirrored for processing or recording; the viewer flips its display only.

### Measurement principles

- **Record and replay.** Every session stores timestamps, landmarks, features, blendshapes and the ground-truth target positions, so models can be trained and evaluated offline.
- **Held-out evaluation.** Calibrate on one set of targets and test on different ones; the unit of analysis is the target, not the frame.
- **Traceable results.** Every benchmark and evaluation run appends its config, mode and metrics to `data/results.csv`.
- **Pre-registration.** Hypotheses, protocols and success criteria are written in `PROGRESS.md` before the confirming sessions are recorded, and the analysis refuses to run until the right sessions exist.
- **Privacy.** Camera frames and face data never leave the device. Sessions, face images and practice runs stay in gitignored folders.

## Project layout

```
Calibrate.command, Run.command   one-click launchers (macOS)
src/gaze_cursor/                 capture, landmarks, features, calibration, posture, config, eval/, sinks/
scripts/                         viewer, download_model, calibrate, check_setup, gaze_overlay, benchmark, evaluate
tests/                           pytest suite
data/sessions/                   recorded sessions and face images (gitignored)
data/dry_runs/, data/excluded/   practice and excluded sessions (gitignored)
data/results.csv                 benchmark and evaluation log
models/                          downloaded model files (gitignored)
PLAN.md, PROGRESS.md             milestones; findings, decisions and pre-registrations
```

## Development

```bash
uv run pytest                            # 115 unit tests
```

## Roadmap

See [`PLAN.md`](PLAN.md) for details and each milestone's "done when" check.

| Milestone | Goal | Status |
| --- | --- | --- |
| M0 | See the face: live landmarks, head pose, FPS | Done |
| M1 | First gaze estimate and first accuracy number | Done |
| M2 | Make it good: evaluation harness, calibration redesign, lighting and replayable images, vertical bias, robustness, smoothing | In progress: steps 1, 2 and 2b built; H1 not supported; round-2 lighting tests waiting on recordings |
| M3 | Control the real cursor: Quartz CGEvent, click triggers, warp mode | Planned |
| M4 | Choose a direction: precision tricks, intent model, productivity vs accessibility | Planned |
