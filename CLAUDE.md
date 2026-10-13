# gaze-cursor

Move the macOS cursor by tracking eye gaze through a standard webcam.

## Session start (this repo is public)

- The author's private progress notes live outside this repo, in `gaze-cursor-notes/PROGRESS.md`, a sibling of the main checkout (`../gaze-cursor-notes` from the main checkout, not from a worktree; private GitHub repo `gaze-cursor-notes`). If it exists, read its "Start here" first and update it at the end of every session (commit and push in that repo). Never copy personal notes into this repo.
- Every new pre-registration goes into `PREREGISTRATIONS.md` here and is committed **before** its data exists; results are appended there and summarised in `RESULTS.md`.
- Never commit recordings, face data or anything personal: the repo is public. `main` is the only branch; push fast-forwards only (force-push and branch deletion are blocked by rulesets).

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
- Needs macOS permissions: **Camera** (M0+) and **Input Monitoring** (M3+, for the listen-only key/click tap; macOS applies it only after Terminal is restarted) for the app running Python (Terminal / VS Code). Accessibility is not needed while no clicks are synthesised.

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

## Generalisation readiness (future M5; keep these now)

Long term the tracker should work with other webcams, extra/external screens, other people, lighting and OSes. The current 2D "iris -> this screen" mapping silently bakes in this exact camera, screen, distance and person, so:

- **Structured condition tags on every session** (in `meta`, not only the free-text `--note`): camera (name, resolution, built-in/external), screen layout (displays, sizes in points and cm, arrangement from macOS), estimated face distance, lighting (`day` / `lamp` / `dark` / `backlit`), glasses (yes/no), posture note, person id (`p01`...). Untagged sessions can't answer "does it work in condition X?" later.
- **Prefer geometric representations:** for M2 step 3, aim for a 3D gaze ray (eye position in camera space + gaze direction) intersected with a screen plane, rather than only a richer 2D regression. Face distance can be estimated monocularly from iris size (human iris diameter ~11.7 mm). This is also the proper fix for head-pose sensitivity, and multi-monitor becomes "several planes".
- **Keep the three parts separable:** person -> gaze direction (learnable across people), camera geometry (per camera), screen geometry (per layout). Never hard-code camera or screen values; everything comes from config / the OS / calibration.
- **OS-specific code stays in sinks / platform modules** (cursor control, screen geometry, overlay window); the core pipeline stays portable Python.
- Face recordings are personal data: other people's sessions need consent, stay local and gitignored.

## Process guardrails (added 2026-10-14)

These exist because two slips happened on 2026-10-13 (two sessions on diverged branches; a plan prerequisite skipped). Follow them:

- **One Claude Code session at a time on this repo.** The session-start hook warns if another session holds `.claude/state/session.lock`; if so, tell the author before changing anything.
- **Skills** (in `.claude/skills/`): `checkpoint` at every PLAN checkpoint (verify the done-when, record, then stop and report); `preregister` for any claim that will become a finding; `session-intake` for every new recording; `results-update` for any change to RESULTS.md; `plan-audit` at the start of a working session and before closing a milestone.
- **Reviewers** (in `.claude/agents/`): ask `stats-reviewer` before committing any analysis, evaluation code or results table; ask `plan-reviewer` before reporting a step as done.
- **Hooks** (`.claude/settings.json`, `.claude/hooks/guard.py`): tests run before every `git commit` in this repo (main checkout or a worktree; the tree being committed is tested; tests that open windows are skipped so a commit cannot disturb a live recording; commits in other repos such as the notes repo are ignored) and block it on failure; recorded data and models cannot be edited; PREREGISTRATIONS.md is append-only; a reminder fires when code changed but PLAN.md / notes were not updated.
- **Permissions:** force-push, `reset --hard`, `git clean`, branch deletion and history rewrites are denied; `rm -r`, push, merge and rebase ask first.
- Keep pre-registration for claims; small engineering choices do not need a frozen protocol.

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
- Generalisation (M5): external webcams, multi-monitor, other people, other OSes. See "Generalisation readiness".
