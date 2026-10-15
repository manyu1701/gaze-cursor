---
name: ds-strategy
description: Use when planning any new step or milestone, at milestone boundaries, after a pre-registered result, or when the author asks "what are we missing?". A proactive data-science and ML-engineering lens: finds opportunities the plan is not taking (learning loops, data strategy, generalisation, labels, monitoring, model lifecycle) and proposes them ranked by value and effort. It proposes; it never changes the plan by itself.
---

# Data-science strategy lens

The plan says what to build next; this skill asks what the project should be learning from, and what it is leaving on the table. The author is a data scientist and wants every step judged from this angle too. Run it while planning a step (before the plan is shown to the author) and at milestone boundaries.

## 1. Read the state
PLAN.md (current milestone and step), RESULTS.md, the latest RESULT sections in PREREGISTRATIONS.md, CLAUDE.md "Learning loop and data strategy", and what data exists (`data/sessions`, `data/quick`, results.csv modes). Then the research base: `docs/research/ACTIONS.md` (which R-actions belong to this step or are still open) and the brief(s) for the topics the step touches (`docs/research/README.md` maps problems to briefs). Do not open face images.

## 2. Ask the questions (answer each in one or two lines; "n/a" is allowed)

**Learning loop (the three levels: session -> personal -> population)**
1. Does this step produce **labelled data** (calibration dots, clicks, task targets) that a model could learn from later? Is it saved in the standard session format with structured condition tags and a person id?
2. Which level does it feed or improve: session adaptation, the long-term personal model, the population model? Does anything in the step **throw history away** (resets, forgetting, overwritten files) that a higher level would need?
3. Could the step's model **start from a better prior** (pooled own history, a pretrained gaze model) instead of from scratch?

**Data strategy**
4. What is the **cheapest new label** we are not collecting yet (implicit signals such as clicks, dwell, corrections, recalibration presses)?
5. Coverage: which **conditions** (lighting, posture, distance, glasses, time of day, screens, people) are missing from the data, and does the step make that gap better or worse? Is a planned condition matrix needed?
6. **Label quality:** where are labels noisy (clicks without looking, blinks, drift), is that measured, and is any label evaluated against a model that already learned from it?

**Generalisation**
7. Is anything **specific to this person / camera / screen** baked in that blocks pooling with other people's data later? (hard-coded values, features in screen pixels instead of normalised or angular units, missing tags)
8. What would **leave-one-person-out** evaluation need, and is the step compatible with it?

**Evaluation and decisions**
9. Is the success metric the one that matters to a user (live accuracy at the click, time to target, Fitts throughput), not only an offline proxy?
10. Is there a decision this step could make **with data instead of opinion** (a small pre-registered A/B on the author's own use, paired on the same targets or clicks, bootstrap CI resampling those units, two-level across sessions; never frame-level or random splits)?

**ML engineering and lifecycle**
11. **Model lifecycle:** are personal / learned models versioned, reproducible from logged data, and safe to roll back? Is there a record of which model produced which session?
12. **Monitoring:** would we notice if accuracy degraded in real use (drift alarms, a daily summary of click error)?
13. **Privacy and consent by design:** if other people's sessions are added, what stays on-device, what (anonymous numeric features and labels only, opt-in) could be shared, and could learning be federated?

## 3. Output
A short table of **opportunities**: what, why (one line, with evidence from RESULTS / data where possible, and the R-number and brief when it comes from `docs/research`), which level or area, value (high / medium / low), effort (S / M / L), and where it belongs (this step, a later PLAN step, a new pre-registered round, or parked). Then **at most three recommendations** for the current step. Present them to the author with the plan; never change PLAN.md or code because of this skill without the author's agreement. Claims that would become findings go through `preregister`.
