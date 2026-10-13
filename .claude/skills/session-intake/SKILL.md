---
name: session-intake
description: Use whenever the author records a new session (calibration, quick calibration, Cursor-mode log, set/round recording, live test). Checks integrity, classifies it against any active data rule, logs it, and never fits a model on data reserved for a pre-registered test.
---

# Intake a new recording

## Rules
- **Integrity only.** Do not fit or evaluate a model on a session that may be test data for an active pre-registration. If unsure whether it is test data, treat it as test data.
- Never edit recorded files (a hook blocks it). Exclusions = move to `data/excluded/` with a logged reason, only with the author's OK.
- Never open stored face images or camera frames (privacy convention); numeric processing only.
- Recordings live only in the author's checkout (`data/` is gitignored). From a worktree, read them there (or symlink), never copy them into git.

## Steps
1. Identify the file(s) and timestamp; which pre-registration's data rule, if any, it falls under (e.g. started after a freeze time).
2. Integrity checks, by session type:
   - **Calibration session** (`data/sessions/session_*.npz`): file loads; target / frame counts; valid-frame share; window geometry clean; lighting captured and its class; posture summary; face-crop sidecar (`*.faces.npz`) aligned row for row; file modes.
   - **Quick calibration / live test** (`data/quick/session_*.npz`, `livetest_*.npz`): file loads; dots / frames; valid-frame share; lighting; quick-fit error as logged.
   - **Cursor-mode log** (`data/quick/cursor_*.npz`, positions only, no face crops, no window layout): minutes awake (frame clock), labelled and usable clicks, calibration segments and their integrity. For round 4 use **only** `uv run scripts/evaluate_online.py --status` (prints the data-rule numbers and COUNTS / does not count, no errors). **Never** run `--develop` or `--preregistered` on round-4 logs during intake.
   - **Quality report (PLAN M3 step 6b, once built):** run it on new recordings from M3 step 7 on and include its pass / warn / fail lines. **Not for round 4:** round 4 was frozen (2026-10-13 ~20:56) without it, so its classification uses only the frozen data rule. Report lines for round-4 logs may be added only after the H6 test has run, labelled informational; the model-fitting checks (held-out fit, bias vs noise) never run on round-4 logs before then.
3. Classify: COUNTS / DOES NOT COUNT (which rule, with the measured value) / AMBIGUOUS (stop and ask the author; lay out the options, do not decide alone).
4. Log it in the private progress notes and, for a pre-registered round, in that round's session log in PREREGISTRATIONS.md. That file is **append-only** (a hook blocks changes to existing lines): add **one new bullet per session** below the last log entry, e.g. `- **Round-4 session N:** name, start time, note, measured conditions, verdict, "no model fitted"; still needed: k`. Never edit an earlier bullet, including its "still needed" count.
5. Tell the author in two or three lines: counts or not, why, and how many sessions the round still needs. If the round's data rule is now met, say so and use the `checkpoint` skill before running the test.
