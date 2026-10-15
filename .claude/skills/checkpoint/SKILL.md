---
name: checkpoint
description: Use at the end of any PLAN.md step, when a pre-registered test or pre-written rule resolves, when new data is needed from the author, when the written plan runs out, or before anything hard to undo. Verifies the step against its done-when, updates the records, then STOPS and reports to the author instead of starting the next step.
---

# Checkpoint: verify, record, stop, report

PLAN.md's "Checkpoints" section defines when this applies. Within a written step, keep working; at a checkpoint, follow every step below in order and then stop.

## 1. Verify
1. Run the full test suite: `uv run pytest -q`. Note the count. Never report a step done with failing tests.
2. Re-read the step in PLAN.md and its **done-when**. For each condition, write PASS / NOT YET with the evidence (a command output, a results.csv row, a file). No condition passes on intention or "should work".
3. Check you did not build ahead: anything committed that belongs to a later step? List it.

## 2. Record
1. PLAN.md: update the step's **Status** line (date, what is done, what is open, numbers with CIs where they exist).
2. Private progress notes: what was done, decisions and who made them, gotchas, next steps.
3. If a pre-registered test ran: its RESULT goes into PREREGISTRATIONS.md (append only) and RESULTS.md via the `results-update` skill.
4. Commit (the pre-commit hook runs the tests; a commit that adds a DONE to a Status line is blocked unless this skill ran since the last commit). Push only if the author has asked for pushes in this session.

## 3. Stop and report
Reply to the author in this shape, plain words, no jargon without a gloss:
- **Where we are:** step, done-when PASS / NOT YET.
- **What the numbers say:** the key results with CIs, and whether they were pre-registered or exploratory.
- **What changed in the plan, if anything, and why.**
- **What is next:** the next step(s) with a recommendation, and anything only the author can do (recordings, permissions, decisions).
Then wait. Do not start the next step until the author agrees.
