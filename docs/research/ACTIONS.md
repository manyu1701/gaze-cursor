# Research actions (R-numbers)

Every "What to try" from the briefs, in one place. **Status: all proposed.** Nothing here is in the plan until the author agrees; `ds-strategy` / `ds-lead` pick the ones relevant to the step being planned and present them with the plan. Exploratory items only print or report numbers; anything that would become a claim goes through `preregister`. When an action is adopted, done or dropped, update its status here (with the date and the PLAN step), so this file stays the index.

## Suggested order (fits the agreed plan order: 6c -> L2 -> step 7 -> step 8 + L5)

1. **Now, process only:** R24 (two-level CIs and a minimum detectable effect in every pre-registration).
2. **With 6c / the quality report:** R11 (distance from the median iris size), R7 + R14 (signal strength per session: px/deg and the predicted squeeze).
3. **Cheap exploratory analyses on data we already have (any time, prints only):** R15 (click-label timing and noise floor), R25 (warp-correction labels), R4, R23.
4. **Before step 7 records anything** (step 7 fixes the capture settings its sessions use): R1 (higher capture resolution), R9 (exposure metering on the face).
5. **Inside step 7's pre-registration:** R18 (Fitts across distances, warp vs correction phase), R5 (head share at edge targets), R17 (lag reported without the eye's reaction time).
6. **Step 8, in this order:** R21 (licences and shortlist) -> R29 (runtime check) -> R3 (pretrained model) with R12 (small personal correction) and R2 as the non-network comparator; R13 if the 3D ray needs metric distances.
7. **L-track:** R22 (hierarchical shrinkage) as the design input for L3; R16 (drift) for L2 / L4.
8. **M4 and later:** R27 (talk to users) before the direction decision; then R19 (snap), R26 (dwell) as the direction requires; R28 before any other person is recorded (L6 / M5).

## All actions

| R | What | Why (one line) | Brief | Belongs to | Value | Effort | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | Capture above 1280x720 (e.g. 1920x1080); measure px/deg, landmark noise, fps, latency | Signal scales with pixels per degree; we use a fraction of a 12 MP camera | 02 | before step 7 | high | S-M | proposed |
| R2 | Sub-pixel iris centre (ellipse fit on the eye crop) vs MediaPipe's point | The landmark may under-respond to eye rotation (vertical ~1/3 of expected; horizontal not established, it depends on the assumed seat distance, 02 item 6) | 02, 03 | step 8 comparator | medium | M | proposed |
| R3 | Pretrained appearance-based gaze model with standard normalisation | Changing method family is the next real jump; variants moved error by <= 0.25 deg | 02, 03 | step 8 | high | L | proposed (already planned as step 8) |
| R4 | Upper-lid position relative to the corners as a vertical feature | The lid follows vertical gaze; vertical has half the signal | 01 | exploratory | low-medium | S | proposed |
| R5 | Measure head rotation that accompanies edge targets | Predicts edge undershoot when only iris-in-eye is used | 01 | step 7 | high | S | proposed |
| R6 | Per-person "head mover" parameter | Eye-head coordination is stable per person and varies between people | 01 | step 8 / L3 | medium | M | proposed |
| R7 | Horizontal px/deg per session in the quality report, with a threshold (distance-free form, e.g. divided by eye width) | Dim sessions collapse to near-zero horizontal signal (0.008-0.037 vs 0.07-0.43); the vertical ranges overlap, so the check is horizontal-only; flag before modelling | 02 | 6b/6c follow-up | medium-high | S | proposed |
| R8 | Landmark drift on a still printed face or mannequin | Separates camera/landmark wobble from eye motion | 01 | exploratory | low-medium | S | proposed |
| R9 | Query and use the camera's exposure controls (meter on the face); test backlit | Lighting is the largest measured effect (H2, H3) | 02 | before step 7 | high | S-M | proposed |
| R10 | Front light as a product rule in setup | Cheapest fix for the largest effect | 02 | product / setup check | medium | S | proposed (setup check partly does this) |
| R11 | Distance from the median iris diameter over a calibration | 1 px of iris is ~7% of distance at 720p | 01 | 6c (`distance_cm`) | medium | S | **done in 6c (2026-10-15)**: median iris stored with the assumed field of view (65 deg). Caveat: our sessions' iris size implies 75-77 cm at that FOV (or ~77 deg FOV at 60 cm), so the estimate is unreliable until R13 or a measured seat distance fixes the FOV |
| R12 | Personal model = population model + small (kappa-like) correction, 1-3 dots | Kappa (~5 deg) is the main per-person term | 01, 03 | step 8 / L5 | high | M | proposed |
| R13 | Estimate the camera's focal length once (checkerboard, or one tape-measured seat distance with the iris size); check Center Stage | The 3D ray needs metric geometry; the 65 deg FOV assumed in `conditions.py` is inconsistent with our iris size at 60 cm (**now also needed by 6c / R7**) | 02 | step 8 / M5 | medium | S-M | proposed |
| R14 | Reliability ratio per axis and session; check it predicts the squeeze | Explains the centre squeeze; success metric for R1 / R2 | 04 | 6b report / exploratory | medium | S | proposed |
| R15 | Gaze-to-click distance vs time around clicks from Cursor logs | Label-noise floor for real use; checks the 400-100 ms window | 04 | before L3 / L4 | high | S | proposed |
| R16 | Forgetting factor or Kalman drift rate from data; CUSUM/ADWIN alarms | 0.98 is a guess; sudden vs slow changes need different handling | 04 | L2 / L4 | medium | M | proposed (L2 keeps the states a drift study needs; R16 itself not taken up) |
| R17 | Decompose lag (camera, pipeline, filter, remainder; filter off vs on) instead of subtracting an assumed 150-250 ms | Filter alone is ~90-100 ms on an 8 deg/s ramp with clean input (less with noisy input); the eye's share may be smaller than assumed | 01, 05 | step 7 | medium | S | proposed |
| R18 | Fitts task across distances; warp vs correction phase; break-even distance | MAGIC gains depend on distance | 05 | step 7 | high | M | proposed (step 7 already plans a Fitts task) |
| R19 | Snap to nearest clickable (Accessibility API); offline replay on click logs first | Bubble-cursor style expansion turns coarse gaze into reliable clicks | 05 | M4 | high | M-L | proposed |
| R20 | Can a webcam see the screen's reflection on the cornea? | A glint reference is how infrared trackers reach ~0.5 deg | 02 | parked | unknown | S | parked |
| R21 | Licence check and model shortlist before code | Many gaze datasets and weights are research-only | 03 | step 8 (first) | high | S | proposed |
| R22 | Ridge shrinking toward pooled own weights, strength from between-session variance | The principled middle between H5 (fresh) and pooling | 04 | L3 design | high | M | proposed |
| R23 | Dispersion-based fixation detection (I-DT) as a tested pure function | Cleaner calibration and click windows; per-frame speed is too noisy | 04 | M2 step 6 remainder / L4 | medium | S | proposed |
| R24 | Pre-registrations: two-level CI, session count and weighting, minimum detectable effect | Round 4: 332 clicks but effectively 3 sessions | 04 | process (`preregister`) | high | S | proposed |
| R25 | Warp landing -> click position as a second implicit label stream | EyeO shows implicit labels work; this one is free | 05 | exploratory -> L4 | medium-high | S | proposed |
| R26 | Dwell click with progress ring and undo | Hands-free needs a Midas-touch-safe trigger | 05 | M4 (accessibility) | medium | M | proposed |
| R27 | 5-10 conversations with eye-gaze users or AT professionals | Reliability and recovery may matter more than accuracy | 06 | before M4 decision | high | S | proposed |
| R28 | One-page data and consent design (DPDP, GDPR) | Required before any other person's data | 06 | before L6 / M5 | high (then) | S | proposed |
| R29 | Measure gaze-model latency on the Mac before choosing one | ~15-20 ms per frame budget at 30 fps | 06 | step 8 | medium | S | proposed |
