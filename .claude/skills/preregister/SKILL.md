---
name: preregister
description: Use whenever a claim will go into RESULTS.md or a write-up as a finding (a hypothesis test, a model comparison, a threshold choice). Writes and freezes the protocol in PREREGISTRATIONS.md before the data exists, with a run-once guard. Not needed for small engineering choices.
---

# Pre-register a test

Use for claims, not for every tweak. A small engineering choice (a UI default, a refactor) needs no pre-registration; a statement like "X improves accuracy" does.

## 1. Write the protocol (append to PREREGISTRATIONS.md, new dated section)
- **Hypothesis** (one sentence, directional) and its id (H7, H8, ...).
- **Candidate and reference** exactly as code (config names, commit hash once committed).
- **Data rule:** which sessions count (time window after this commit, lighting class, minimum length / valid frames / clicks), how many, chosen in time order; what disqualifies a session and that disqualified sessions are logged, never silently dropped. No session may be selected by its results.
- **Unit of analysis:** target / fixation / click, never frame. Split: leave-one-target-out, leave-one-session-out, or the named scheme.
- **Statistic:** paired difference on identical units, 95% bootstrap CI resampling units (10,000, fixed seed), plus a two-level (sessions then units) CI when there are several sessions.
- **Success criteria:** all written as numbers before data (CI lower bound > 0, minimum effect in deg, consistency across sessions). Anything else = NOT SUPPORTED, no re-tuning.
- **Caveats up front:** one person / room / camera, how the hypothesis was suggested, confounds.
- **Reported, not criteria:** secondary numbers.

## 2. Freeze
1. Commit the protocol ALONE first (message: "Pre-register HN: ..."), before any evaluation data is recorded. Record the commit hash in the section.
2. Implement the evaluation with a guarded entry point (e.g. `--preregistered-N`) that refuses until the data rule is met and is meant to run once. Add tests that plant a known effect (detected) and a null (not detected).
3. Commit the code before any evaluation session.

## 3. Collect and run
- New sessions go through the `session-intake` skill (integrity only, no model fitted).
- When the data rule is met, run the guarded command exactly once. Append the RESULT (numbers, verdict, caveats) to PREREGISTRATIONS.md; log rows with `confirmatory=1`. Then the `checkpoint` skill.

PREREGISTRATIONS.md is append-only (enforced by a hook): amendments are new dated notes, made before the relevant data exists, explaining what changed and why.
