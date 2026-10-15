# Research base

The science behind gaze-cursor, written so it can be used while building. Each topic follows the same shape:

- **Touches:** which of our problems it explains.
- **The science:** the finding, in plain words.
- **What it means for us:** what it predicts about *this* tracker.
- **What to try:** concrete changes or tests, numbered `R1`, `R2`, ... and collected in [ACTIONS.md](ACTIONS.md).
- **Confidence and sources:** how solid the claim is, and where it comes from.

**How this is used.** Claude Code reads the brief for any topic a step touches before planning it. The `ds-strategy` skill and the `ds-lead` agent read [ACTIONS.md](ACTIONS.md) and propose the relevant actions with a plan; the author decides what goes into PLAN.md. Nothing here is a result of this project: the science gives *hypotheses* for us. Anything that would become a claim about our tracker still goes through `preregister` and our own data.

## The briefs

| File | Topics | Our problems it touches |
| --- | --- | --- |
| [01-eye.md](01-eye.md) | Optical vs visual axis (kappa), fixational eye movements, saccade latency and pursuit, eye-head coordination, eyelid and vertical gaze, iris size | Why calibration exists; jitter floor; path lag; head movement; weak vertical axis; distance estimate |
| [02-camera.md](02-camera.md) | **The precision limit (computed on our data)**, exposure and lighting, camera model, how infrared trackers work | What accuracy is even possible; lighting effect; 3D gaze ray |
| [03-gaze-methods.md](03-gaze-methods.md) | Families of gaze estimation, what MediaPipe gives us, appearance-based models and normalisation, few-shot personalisation, head pose | Step 8; the three-level learning loop |
| [04-ml-stats.md](04-ml-stats.md) | Regression dilution (centre squeeze), clicks as labels, online learning and drift, hierarchical models, fixation detection and filtering, statistics with few sessions | Centre compression; the 4-5 deg real-use gap; round 4 and L2/L3; smoothing; CIs |
| [05-interaction.md](05-interaction.md) | MAGIC pointing, Midas touch and dwell, Fitts' law and ISO 9241-411, target expansion (bubble cursor) | Warp design; clicking; step 7; snap-to-target (M4) |
| [06-people-and-product.md](06-people-and-product.md) | Needs of eye-gaze users, privacy and data law, on-device inference | Accessibility direction; L6 / M5; step 8 runtime |
| [ACTIONS.md](ACTIONS.md) | Every "what to try", ranked, with where it belongs in the plan | Planning |

**Headline findings, in one paragraph.** On this camera the iris moves only about 0.16 px per degree of gaze horizontally and 0.08 px vertically, so one pixel of landmark error is about 6 degrees horizontally and 13 vertically (02); these degree figures assume the screen is 60 cm away, which the sessions' own iris size questions (02, item 6). Measured jitter is 0.8-1.3 degrees against a median error of 2.4 (mean 2.7), so the error is mostly *systematic bias*, not noise, which matches RESULTS.md. In dim light the iris landmark stops following the eye at all (four sessions), so a different input signal, not a different regression, is the likely fix. The centre squeeze is the expected, error-minimising shrinkage of a regression on noisy inputs (04), so the cure is a cleaner input signal (more pixels per degree, better landmarks, a pretrained gaze model), not stretching the output. Our warp design is a known, well-tested technique, MAGIC pointing (05), and our click-label window (400-100 ms before the click) overlaps the 100-250 ms before the click where gaze and pointer are closest in published data (04).
