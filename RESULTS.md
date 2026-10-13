# M2 results: what changed accuracy, and by how much

Webcam gaze tracking on a MacBook Air (built-in camera, 1470 x 956 pt screen, ~60 cm viewing distance),
one person, recorded 2026-10-13. All errors are in **degrees of visual angle** (1 deg is about 1 cm /
~50 px on this screen at 60 cm). Every number below comes from a logged run (`data/results.csv`) or a
RESULT section in `PREREGISTRATIONS.md`; nothing was re-run or re-tuned for this table.

**How to read it**
- **Unit of analysis = target (fixation), never frame.** Frames within one fixation are correlated.
- **Held-out:** "held-out targets" = targets not used to fit the model; "held-out sessions" = a whole
  session (different time / lighting / position) not used to fit it.
- **CI** = 95% bootstrap interval resampling targets (10,000 resamples). **d** = paired difference on
  the same targets (positive = the change helps).
- **Pre-registered** = hypothesis, data rule and success criteria committed to git before the data
  existed; run once, no re-tuning. **Exploratory** = chosen or seen after looking at data: a hypothesis,
  not a result.

## Headline

| Measure | Value | Setting |
| --- | --- | --- |
| Accuracy, held-out targets (offline) | **2.71 mean / 2.37 median / 4.69 p90 deg** | Run's protocol (fresh 9-dot fit -> other 16 targets), 21 front-lit sessions, 336 targets |
| Accuracy, live, dot hidden | **1.9-3.1 deg median** (latest 2.1) | at the calibration spot; 12 targets per test, 5 runs |
| Accuracy, held-out sessions | 6.2-6.3 deg mean | pooled other sessions, mixed lighting (round 1) |
| Cost of a moved head (~10 cm) | **+1.94 deg** [+1.13, +2.65] | pre-registered (H4) |
| Recovered by a 15 s 9-dot recalibration | **+1.29 deg** [+0.80, +1.77] | pre-registered (H5) |
| Jitter (held-out fixations) | 0.77 deg | One-Euro filter |
| Speed | 30 fps; frame -> screen 12 ms; filter lag50 31 / lag90 97 ms | live |

The biggest effects are **conditions, not models**: lighting (+2.4 to +4.1 deg) and head position
(+1.9 deg) dwarf every model variant tried (<= 0.25 deg).

## Every change and its effect

### Pre-registered (confirmatory)

| # | Change tested | Data | Effect d [95% CI] | Verdict |
| --- | --- | --- | --- | --- |
| H1 | 14 eye-shape (blendshape) features + stronger regularisation (`gaze_blend_a100` vs `gaze_a10`), cross-session | 3 new sessions (backlit / dim / front-lit), leave-one-session-out, 75 targets | **+0.12** [-0.43, +0.67] | NOT SUPPORTED |
| H2 | Calibrate in the lighting you use (vs a session in other lighting) | 2 test sessions, 50 targets | **+2.42** [+1.43, +3.41] | SUPPORTED |
| H3 | Face brightness gate (>= 60/255) | 4 sessions, within-session, 36 targets | below-gate sessions **4.08 deg worse** [+2.83, +5.38] | SUPPORTED |
| H4 | Head moved ~10 cm (near / far / left / right / lower) with a fixed calibration | set B, 6 sessions, 150 targets | **+1.94 deg worse** [+1.13, +2.65], 5/5 positions | SUPPORTED |
| H5 | Fresh 9-dot recalibration at the new position vs a model pooled over all other positions (13:12 freeze) | set B, 80 targets | **+1.29** [+0.80, +1.77], 5/5 positions | SUPPORTED |
| H5' | Pooled `gaze_a100` over all calibrations vs today's centre-only model (earlier 05:57 freeze, run second) | set B, 125 targets | **-0.08** [-0.42, +0.26], 2/5 positions | NOT SUPPORTED |

Two freezes of round 3 exist because two sessions worked on diverged branches; both are reported in
the order run (PREREGISTRATIONS.md, "Two freezes of round 3"). They agree: pooling earlier positions does not
fix head movement; recalibrating where you sit does.

### Exploratory, with a pre-written decision rule

| Change / question | Result | Decision |
| --- | --- | --- |
| Smoothing: One-Euro (min_cutoff 0.05 Hz, beta 0.001) vs the old 5-frame moving average | held-out jitter **0.77** [0.54, 1.12] vs 0.83 [0.62, 1.12] deg; lag90 **97 vs 125 ms**; overshoot 0.23 vs 0.39 deg | adopted in Run (jitter CIs overlap; the gain is mostly speed) |
| Is there a real "top predicted too low" bias? | top band +0.49 deg [-0.50, +1.62] | gate NOT met: no specific vertical bias in layout v2 |
| Does the 3D head transform track head moves (toward a geometric gaze ray)? | head effect 0.25 (x) / 0.17 (y) of geometry; transform = eye-corner estimate (r = 1.00); both recover the real moves (L +10.1, Rt -9.3, Lo +6.1 cm) | rule FAILED: geometric route parked; the weak link is the gaze features, not the head estimate |
| Redo rule for the quick calibration | mean fit predicts held-out error (r 0.78) better than worst dot (r 0.66) | Run redoes if mean fit > 3.0 deg |

### Exploratory (hypotheses for later rounds)

| Change | Effect on held-out targets | Note |
| --- | --- | --- |
| Blendshape features inside a fresh 9-dot fit (`gaze_blend_a100`) | **+0.25** [+0.10, +0.40], 13/21 sessions | candidate for a confirmatory round (not frozen) |
| Eyelid openness (`eyes`) | +0.07 [-0.02, +0.15] | no clear gain |
| Less regularisation to undo the centre squeeze (alpha 1 / 0.01) | -0.13 [-0.20, -0.06] / -0.24 | worse |
| Per-dot mean fitting / affine stretch | -0.29 to -0.73 / -0.25 | worse |
| Edge-heavy dot layout instead of today's 9 dots | -0.15 [-0.24, -0.07] | worse |
| Regularisation alone (alpha 100 vs 10, cross-session) | +0.09 [+0.02, +0.16] | real but tiny |
| Few-dot offset / affine correction on a pooled model (set A) | 1 dot -0.72; 3 dots +0.38 [-0.14, +0.90]; 5 dots +0.58 [-0.17, +1.28] | not reliable; superseded by fresh 9-dot (H5) |
| Head-geometry models `head` / `ray` / `geo` (set A, leave-one-position-out) | 7.8 / 7.0 / 17.3 vs 5.37 deg for plain pooling | worse; parked |

## Live behaviour (Run, dot visible or hidden, one person)

| Test | Result |
| --- | --- |
| Accuracy (dot hidden, 12 targets) | 1.9-3.1 deg median at the calibration spot (5 runs) (latest 2.1, 50% within 2.5 deg); bottom third reads 2.8-3.9 deg too high |
| Targets (hold the dot in a box 0.5 s) | 6 deg boxes 67-100% hit, 4 deg boxes 50-100%; first entry ~0.6-0.9 s |
| Path following (8 deg/s target) | 3.0-3.4 deg mean; dot trails by 190-270 ms (eye pursuit + camera + pipeline + filter) |
| Head moved ~20 cm, no recalibration | 6.7-7.4 deg; after R at the new spot 4.0-4.9 deg (off-axis viewing degrades the eye features) |
| Pipeline | 30 fps (camera-limited); frame read -> on screen 11.6-11.9 ms (landmarks 5 ms, draw 6 ms) |

## What did not work, and what we learned

- **Richer features across sessions (H1):** the blendshape gain seen in early sessions did not replicate
  in new sessions with different lighting. Lighting changes swamp feature tweaks.
- **Pooling calibrations across head positions (H5'):** no gain at a moved position.
- **Geometric head models:** worse than plain models. The head position is measured well; the iris
  features respond only ~0.2x as geometry predicts when the head moves, so a better gaze-direction
  estimate (pretrained model, PLAN M2 step 8) is needed before a 3D gaze ray can work.
- **Undoing the centre squeeze:** every de-shrinking variant was worse. The squeeze is the
  bias-variance trade-off of noisy vertical features, not a bug; the fix is a better vertical signal.
- **Heavier smoothing:** most within-fixation "jitter" is slow drift over ~1 s, which a low-pass filter
  cannot remove without becoming too slow.

## Caveats

- **One person, one room, one camera, one day.** Cross-person and cross-camera accuracy is untested (M5).
- Positions were discrete and moderate (~10 cm); set B was recorded in the same sitting as its freeze.
- Several exploratory comparisons were run on the same 21 front-lit sessions; their small effects
  (<= 0.25 deg) need confirmation on new sessions.
- Live tests are short (12 targets) and vary ~1 deg between runs a minute apart.

## What this means for the next milestones

- In good light the tracker is good for **coarse pointing** (regions, windows, big buttons), not for
  pixel-precise clicks: fits the "gaze jump + trackpad finish" use (M3).
- Head movement is the main open problem. Requirement (PLAN M2 step 11): stay accurate as the head
  moves and count head turns as looking, **without user input**. Bridge: click-based self-recalibration
  (M3); proper fix: pretrained gaze direction + tracked head pose (step 8).

**Sources:** `data/results.csv` rows with `confirmatory=1` (modes `losos`, `h2_cross_lighting`,
`h3_within`, `h4_position_cost`, `h5_lopo`, `h5_lopo_earlier_freeze`) and exploratory modes
(`vertical_*`, `compression_explore`, `smoothing_heldout`, `head_transform_check`, `live_*`);
[`PREREGISTRATIONS.md`](PREREGISTRATIONS.md): the protocols and RESULT sections for rounds 1-3, the
vertical-bias diagnosis, smoothing and the head-transform check. The centre-compression exploration and the
live-test runs are summarised here from the author's project notes and the `live_*` / `compression_explore`
rows of `results.csv`.
