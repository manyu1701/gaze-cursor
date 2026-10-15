# 01. The eye itself

## Optical vs visual axis (angle kappa)
**Touches:** why calibration is needed; how good a calibration-free tracker can be.
**The science:** the eye's optical axis (through the centre of the pupil and cornea, what a camera can see) and its visual axis (from the fovea to what you look at) differ by roughly 5 degrees, by a different amount and direction in each person. A camera cannot see the fovea, so this offset can only be learned from the person.
**What it means for us:** any tracker without a personal calibration starts with an error of this order unless it learns the offset another way. That sets the expected floor for zero-dot approaches (M3 step 8, calibration-cost curve): several degrees before personalisation. Few-shot methods exist largely to learn this per-person offset.
**What to try:** R12 (treat the personal part of the model as "kappa plus small corrections": 1-3 dots or the first clicks should be enough to learn it once the population model is good).
**Confidence and sources:** high; standard eye-tracking theory. Review: Kar and Corcoran 2017, [arXiv 1708.01817](https://arxiv.org/pdf/1708.01817) (kappa about 5 deg, per-person).

## Fixational eye movements (drift, microsaccades, tremor)
**Touches:** jitter; "jitter is slow drift, not noise" (M2 step 6).
**The science:** even when fixating, the eyes move: tremor of about 1 arcmin, drift of a few arcmin (classically 1.5-4 arcmin at about 4 arcmin/s, but several times faster in natural head-free viewing), and microsaccades usually under 0.5 deg at fewer than 1-2 per second.
**What it means for us:** the real eye is stable to well under 0.5 deg during a fixation. Our within-fixation spread of ~0.8-1.3 deg is therefore mostly **measurement**, not the eye: landmark wobble and slow drift of the landmarks and head (see 02, precision). A tracker cannot do better than the eye's own motion, but we are far from that limit, so better landmarks and head handling, not physiology, set our jitter today.
**What to try:** R8 (measure landmark drift on a still synthetic or printed face to separate camera/landmark wobble from eye motion).
**Confidence and sources:** high. Rucci and Poletti, fixational eye movements review, [PMC5082990](https://pmc.ncbi.nlm.nih.gov/articles/PMC5082990).

## Saccades and smooth pursuit
**Touches:** path-following lag of 190-330 ms in Run; warp timing.
**The science:** a saccade to a new target typically starts about 150-250 ms after the target appears, and smooth pursuit of a moving target lags it and is corrected by catch-up saccades. Saccades themselves are fast (tens of ms).
**What it means for us:** the measured path-following lag (190-330 ms in `live_path` rows) is not yet split into its parts. Known parts: camera frame (~33 ms), pipeline (11-12 ms) and the One-Euro filter, whose 31 / 97 ms are step-response lag50 / lag90; on an 8 deg/s ramp the filter alone probably lags about 90-100 ms (a simulation of `RUN_FILTER` at 30 fps; reproduce before quoting). That leaves roughly 50-190 ms for the eye. Path following is pursuit of a moving target (scored after the first second), whose latency is shorter than the 150-250 ms saccade latency above, so *do not assume the eye is the largest part*. The eye's share cannot be engineered away; it can only be hidden (e.g. warp only on a hotkey, as now). For a warp on keypress, the relevant delay is the filter's.
**What to try:** R17 (decompose the lag in step 7 instead of subtracting an assumed 150-250 ms: run path following with the filter off and on, and report camera + pipeline + filter + remainder separately).
**Confidence and sources:** high; textbook oculomotor physiology. Pursuit and catch-up saccade studies, e.g. [de Brouwer et al.](https://research.dial.uclouvain.be/bitstreams/7d72916b-5cc1-46b6-8613-16c06ec3b88d/download).

## Eye-head coordination
**Touches:** M2 step 11 ("head turns count as looking"); centre compression; head-movement cost (H4, +1.9 deg).
**The science:** shifts under about 20 deg are made almost entirely with the eyes; larger ones increasingly use the head (around 30 deg the head carries about a third). Eyes rarely sit beyond about 30 deg from straight ahead for long. People differ a lot: some are "head movers", some are not, and the pattern is stable per person.
**What it means for us:** the MacBook screen spans only about +-14 deg horizontally and +-9 deg vertically at 60 cm, so most looks are eye-only, but looks to the edges and corners, and anything on a second monitor, will include head rotation. A model that uses only iris-in-eye position will **under-shoot exactly at the edges**, which is the same direction as the centre squeeze. Expect this to be much larger on bigger or extra screens (M5).
**What to try:** R5 (in step 7's head-and-eyes test, measure how much head rotation accompanies edge targets for the author: if it is a few degrees, combining head yaw/pitch with eye-in-head direction is required, which is what step 8's gaze ray does); R6 (per-person head-mover parameter in the personal model).
**Confidence and sources:** high for the physiology; the link to our squeeze is a hypothesis. Summary with citations (Land and Tatler 2009; Freedman 2008; Stahl 1999; Saeb et al. 2011): [arXiv 2602.06164](https://arxiv.org/html/2602.06164v1).

## Eyelid and vertical gaze
**Touches:** the weak vertical axis; blink filtering bug (looking down looked like blinking).
**The science:** the upper eyelid follows vertical eye position closely (it lifts when you look up and drops when you look down), so the visible eye opening and the amount of iris showing change with vertical gaze. Looking down partly covers the iris.
**What it means for us:** vertical gaze is measured through a partly hidden, eyelid-cropped iris, so the iris landmark under-responds vertically. Our own data confirm it: the iris moves about half as many pixels per degree vertically as horizontally (02). This also explains why eyelid openness carries vertical information and why the old blink threshold misfired.
**What to try:** R4 (use eyelid geometry explicitly as a vertical feature: upper-lid height relative to eye corners, normalised, alongside iris y; test on existing sessions, exploratory first. Note: eyelid *openness* alone gave no clear gain (+0.07 deg [-0.02, +0.15], RESULTS.md); the lid's *position* relative to the corners is a different signal, but expect a small effect).
**Confidence and sources:** high for lid-eye coupling (Becker and Fuchs; Fuchs et al. 1992 J Neurophysiol on lid motoneurons during vertical eye movements); the size of its effect on MediaPipe landmarks is measured on our data in 02.

## Iris size
**Touches:** distance estimate (6c `distance_cm`).
**The science:** the horizontal iris diameter is about 11.7 mm +- 0.5 mm across a wide population; MediaPipe used it to estimate distance with about 4-5% mean error.
**What it means for us:** distance-from-iris is sound (+-4% is +-2.5 cm at 60 cm). Our iris is only ~15 px across at 720p, so 1 px of iris-diameter error is ~7% of distance: average over many frames.
**What to try:** R11 (estimate distance from the median iris diameter over a calibration, not per frame).
**Confidence and sources:** high. [Google Research, MediaPipe Iris](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/).
