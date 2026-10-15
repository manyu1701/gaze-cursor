# 03. Gaze-estimation methods

## The families of methods
**Touches:** where we are and where step 8 goes.
**The science:** (1) **model-based / corneal-reflection**: a 3D eye model fitted with infrared glints (most accurate, needs special light); (2) **feature- or shape-based**: fit points or shapes (iris circle, eye corners) and regress to the screen, which is what we do with MediaPipe landmarks plus ridge; (3) **appearance-based**: a neural network reads the eye or face image and outputs gaze angles, learned from many people. Webcam appearance-based methods are typically 2-4 deg after personal calibration.
**What it means for us:** we are at the ceiling typical for feature-based webcam tracking. The next jump comes from changing family (appearance-based, step 8) or adding a reference (02), not from more regression variants. This matches RESULTS.md: model variants moved error by at most 0.25 deg.
**What to try:** R3 (step 8).
**Confidence and sources:** high. [Kar and Corcoran 2017](https://arxiv.org/pdf/1708.01817).

## What MediaPipe gives us
**Touches:** every feature we use.
**The science:** the Face Landmarker returns 478 points, including 5 per iris, a head transform and 52 blendshapes. The iris model was trained mainly to locate the iris (and estimate depth) accurately in shape, not to measure gaze.
**What it means for us:** our measurement in 02 shows the iris point moves about half as far vertically as horizontally and roughly a third of the geometric expectation vertically (the horizontal shortfall depends on the assumed seat distance and is not established, see 02 item 6), and it stops moving in dim light. Treat MediaPipe as a good face and head tracker and a coarse eye tracker. Use it to find and crop the eyes; let a gaze-specific model or a refined iris fit measure where the eye points.
**What to try:** R2, R3.
**Confidence:** medium-high (our measurement plus the model's stated purpose); [MediaPipe Iris](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/).

## Appearance-based models and data normalisation
**Touches:** step 8; L5 (population prior).
**The science:** modern models (e.g. L2CS-Net: ResNet-50 on face crops, separate yaw and pitch heads with binned classification plus regression) report ~3.9 deg on MPIIGaze and ~10 deg on the much harder Gaze360 without personal calibration. They depend on **data normalisation**: the face image is rotated and scaled into a virtual camera looking straight at the face at a fixed distance, so the network sees a standard view and outputs gaze in a head-independent frame; the result is rotated back. ETH-XGaze adds large head-pose ranges and high-resolution images.
**What it means for us:** step 8 should (a) normalise face or eye crops the standard way, (b) start with a published model as the population level, (c) add our personal correction on top. Expect several degrees before personalisation (kappa, 01) and a meaningful gain after a few samples. Check licences: these datasets and many weights are research-only.
**What to try:** R3, R12, R21 (a licence check and a model shortlist before any code: L2CS-Net, ETH-XGaze baselines; on-device runtime per 06).
**Confidence and sources:** high. [L2CS-Net](https://arxiv.org/pdf/2203.03339); [Zhang, Sugano and Bulling 2018, data normalisation](https://perceptualui.org/publications/zhang18_etra/); [ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze).

## Few-shot personalisation
**Touches:** calibration cost; the three-level loop.
**The science:** FAZE (Park et al. 2019) learns a gaze representation across many people and adapts to a new person with as few as 3 calibration samples (3.18 deg on GazeCapture). The person-specific part is small and mostly the kappa-like offset.
**What it means for us:** the right design for "less calibration" is a strong population model plus a tiny per-person adaptation, which is exactly L5 then L3. Our quick 9 dots and the clicks are ample for that adaptation once the population model is good.
**What to try:** R12, and the calibration-cost curve (PLAN M2 step 9) with 0/1/3/5/9 dots on top of the pretrained model.
**Confidence and sources:** high. [Park et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.html).

## Head pose
**Touches:** head-movement cost; the gaze ray.
**The science:** head pose comes from fitting a 3D face model to landmarks (perspective-n-point); its translation needs the camera's intrinsics (02) to be metric.
**What it means for us:** our head-transform check found head position is measured well (the three real moves of about 10, 9 and 6 cm were recovered as +10.1, -9.3 and +6.1 cm against nominal, untaped values, so the precision is not known). The weak link is eye-in-head direction (iris features respond ~0.2x to head moves as geometry predicts). So better head pose is not the priority; better eye direction is.
**What to try:** none new (covered by R3, R13).
**Confidence:** high (our own pre-written check, RESULTS.md).
