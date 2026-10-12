"""Fullscreen calibration + held-out test recording. Saves a session to data/sessions/.

SPACE starts, ESC aborts. Look at each dot until it moves. Keep your head fairly still.
Frames are never mirrored. Screen size/geometry comes from Quartz (macOS points).

`--check-screen` only shows corner/center markers and the detected size, to verify that the
OpenCV fullscreen window maps 1:1 onto the screen's point coordinates.

`--check-light` shows a calibration-like screen (dark background, one ring) with a live reading of
the face and background brightness, the lighting label, the brightness gate and the round-2 class.
Use it to set up lighting: it measures what a recording will actually see (the calibration screen
is dark, which gives less light than a bright terminal or the checklist screen does).
"""

from __future__ import annotations

import argparse
import textwrap
import time
from collections import deque

import cv2

from gaze_cursor.capture import Camera
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.faces import FPS, QUALITY, crop_box, encode_crop, estimate_mb, faces_path_for
from gaze_cursor.eval.session import SESSIONS_DIR, SessionRecorder, clean_sessions
from gaze_cursor.eval.targets import build_plan
from gaze_cursor.features import extract_features
from gaze_cursor.landmarks import FaceTracker
import cv2
import numpy as np

from gaze_cursor.eval.prereg2 import class_of, lighting_class
from gaze_cursor.posture import (
    LIGHT_MIN_FACE, Reference, build_checklist, lighting_label, lighting_stats, note_mismatch, posture_metrics, summarize_lighting, summarize_recording)
from gaze_cursor.sinks.guide_view import draw_summary, wait_for_key
from gaze_cursor.sinks.overlay import FullscreenCanvas
from gaze_cursor.sinks.posture_view import run_posture_check

SETTLE_S = 0.7   # eyes travel to the new dot; not recorded
RECORD_S = {"v1": 1.5, "v2": 1.0}  # error rises after ~1.2 s on a fixation, so v2 records less


def check_screen(cfg: ScreenConfig) -> None:
    """Draw the notch strip, the safe-area corner markers and the center; ESC/q exits.

    Corner markers sit at the extreme positions ``target_is_safe`` allows, so all four must be
    fully visible (the rounded top corners and the notch must not cut them off).
    """
    with FullscreenCanvas(cfg) as canvas:
        print(f"window origin {canvas.origin}, size {canvas.size}")
        r, ring = cfg.corner_radius_pt, 16
        xs = (ring + 1, cfg.width_px - ring - 1)
        ys = (cfg.top_inset_pt + ring + 1, cfg.height_px - ring - 1)
        while True:
            img = canvas.blank()
            x0, y0 = canvas._px((cfg.notch_x0_pt, 0))
            x1, y1 = canvas._px((cfg.notch_x1_pt, cfg.top_inset_pt))
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)       # notch outline (red)
            cv2.line(img, (0, y1), (img.shape[1], y1), (0, 0, 255), 1)    # menu-bar strip bottom
            for x in xs:
                for y in ys:
                    canvas.dot(img, (x, y), color=(0, 255, 0), r=ring)
            canvas.dot(img, (cfg.width_px / 2, cfg.height_px / 2), color=(0, 255, 0), r=ring)
            canvas.text(img, "Four green rings must be fully visible. ESC to exit.")
            if canvas.show(img) in (27, ord("q")):
                return


def check_light(cfg: ScreenConfig, note: str = "", guided: bool = False) -> bool:
    """Calibration-like screen with a live lighting readout and the front-light guidance.

    ESC or Q returns False. In ``guided`` mode SPACE returns True (continue to the calibration).
    """
    recent: deque[dict[str, float]] = deque(maxlen=15)
    with Camera() as cam, FaceTracker() as tracker, FullscreenCanvas(cfg) as canvas:
        while True:
            frame = cam.read()
            result = tracker.process(frame) if frame is not None else None
            if result is None:
                recent.clear()
            else:
                recent.append(lighting_stats(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), result.points))
            img = canvas.blank()                       # same dark background as the recording screen
            canvas.dot(img, (cfg.width_px / 2, cfg.height_px / 2))
            if recent:
                face = float(np.mean([x["face"] for x in recent]))
                bg = float(np.mean([x["background"] for x in recent]))
                cls = class_of(face, bg)
                stats = {"face": face, "background": bg, "ratio": face / bg,
                         "clipped": float(np.mean([x["clipped"] for x in recent]))}
                item = [i for i in build_checklist(None, None, [], stats, None, note) if i.key == "light"][0]
                lines = [f"face {face:.0f}/255   background {bg:.0f}   ratio {face / bg:.2f}   {lighting_label(face, bg)}",
                         f"brightness gate (60): {'passes' if face >= LIGHT_MIN_FACE else 'BLOCKED (SPACE locked, U overrides)'}"
                         f"   round-2 class: {cls or 'none (would be excluded)'}",
                         "dim 25-50 | backlit 60-97 with ratio < 0.6 | front-lit 100-155"]
                if item.hint:
                    lines.append("-> " + item.hint)
                lines += [("tip: " if k == 0 else "     ") + t for k, t in enumerate(textwrap.wrap(item.tip, 110))] if item.tip else []
            else:
                lines = ["no face detected"]
            for k, line in enumerate(lines):           # small grey text: adds almost no light
                cv2.putText(img, line, (40, canvas.safe_top + 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (150, 150, 150), 1, cv2.LINE_AA)
            footer = "SPACE: continue to calibration     ESC or Q: quit" if guided else "ESC or Q: exit"
            cv2.putText(img, footer, (40, canvas.size[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (150, 150, 150), 1, cv2.LINE_AA)
            key = canvas.show(img)
            if key in (27, ord("q")):
                return False
            if guided and key == 32:
                return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guided", action="store_true",
                    help="(used by Calibrate.command) first set up your lighting on a dark calibration-like "
                         "screen with live guidance, then continue into the calibration with SPACE")
    ap.add_argument("--check-light", action="store_true",
                    help="show the live lighting as a recording would see it (calibration-like dark screen)")
    ap.add_argument("--check-screen", action="store_true")
    ap.add_argument("--layout", choices=("v1", "v2"), default="v2",
                    help="v2: 16 edge-reaching calibration + 9 interior held-out targets, random "
                         "order, 1 s each; v1: the M1 layout (9 + 12, 1.5 s)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for held-out points and order (default: fresh, stored in the session)")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the setup guidance and the position check against the earlier sessions")
    ap.add_argument("--no-images", action="store_true",
                    help="do not save the replayable face crops (default: saved next to the session, "
                         "local only, ~10-15 MB per session)")
    ap.add_argument("--dry-run", action="store_true",
                    help="save to data/dry_runs/ instead of data/sessions/, so a practice run can never "
                         "be mistaken for a real session")
    ap.add_argument("--note", default="", help='condition tag stored in the session, e.g. "evening, lamp"')
    args = ap.parse_args()
    cfg = ScreenConfig.from_system()
    print(f"screen: {cfg}")
    if args.check_screen:
        check_screen(cfg)
        return
    if args.check_light:
        check_light(cfg, args.note)
        return
    if args.guided and not check_light(cfg, args.note, guided=True):
        return
    seed = int(time.time()) % 2**31 if args.seed is None else args.seed
    record_s = RECORD_S[args.layout]
    plan = [(t.phase, t.idx, t.xy) for t in build_plan(cfg, args.layout, seed)]
    unsafe = [p for _, _, p in plan if not cfg.target_is_safe(tuple(p))]
    if unsafe:
        raise SystemExit(f"{len(unsafe)} targets are too close to the edges, notch or rounded corners")
    n_calib = sum(1 for ph, _, _ in plan if ph == 0)
    print(f"layout {args.layout}: {len(plan)} targets ({n_calib} calibration, {len(plan) - n_calib} "
          f"held-out), seed {seed}, ~{len(plan) * (SETTLE_S + record_s):.0f} s")
    rec = SessionRecorder(meta={
        "screen": cfg.to_dict(), "layout": args.layout, "seed": seed, "note": args.note,
        "settle_s": SETTLE_S, "record_s": record_s, "mirrored": False, "blendshape_names": [],
        "n_calib": n_calib, "n_test": len(plan) - n_calib,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    save_images = not args.no_images
    out_dir = SESSIONS_DIR.parent / "dry_runs" if args.dry_run else SESSIONS_DIR
    if save_images:
        lo, hi = estimate_mb(int(len(plan) * record_s * FPS))
        print(f"face images: saved with the session (local only, not uploaded): about {lo:.0f}-{hi:.0f} MB; "
              f"--no-images to skip")
    if args.dry_run:
        print(f"dry run: the session will be saved to {out_dir}, not used as a real session")
    posture_log: list[dict[str, float]] = []
    light_log: list[dict[str, float]] = []   # lighting is captured from the recorded frames themselves
    with Camera() as cam, FaceTracker() as tracker, FullscreenCanvas(cfg) as canvas:
        rec.meta["window"] = [*canvas.origin, *canvas.size]
        hidden = [p for _, _, p in plan if not canvas.visible(p)]
        if hidden:
            raise SystemExit(f"{len(hidden)} targets fall outside the visible window {rec.meta['window']}")
        earlier = [] if args.no_check else clean_sessions()
        ref = Reference.from_sessions(earlier) if earlier else None
        if not args.no_check:
            # Live checklist: each setup step turns green as it is satisfied (first session: no
            # earlier sessions to compare position against, so only light and stillness are checked).
            check = run_posture_check(cam, tracker, canvas, ref,
                                      override_label="calibrate in my current setup", note=args.note)
            if not check.proceed:
                return
            if check.own_setup:  # the user chose to calibrate in their current setup: nothing forced
                ref = check.reference
            rec.meta["posture_check"] = {
                "mode": "own_setup" if check.own_setup else "earlier_sessions" if ref else "no_reference",
                "in_position": check.in_position, "reference_sessions": ref.n_sessions if ref else 0,
                "reference": {m: ref.center[m] for m in ref.center} if ref else None}
        else:
            while True:
                img = canvas.blank()
                canvas.text(img, "Sit ~60 cm away, head still. Press SPACE to start, ESC to abort.")
                key = canvas.show(img)
                cam.read()
                if key in (27, ord("q")):
                    return
                if key == 32:
                    break
        for n, (phase, idx, xy) in enumerate(plan):
            start = time.monotonic()
            while (elapsed := time.monotonic() - start) < SETTLE_S + record_s:
                frame = cam.read()
                img = canvas.blank()
                canvas.dot(img, xy, color=(255, 255, 255) if elapsed < SETTLE_S else (0, 255, 0))
                canvas.text(img, f"{n + 1}/{len(plan)}", y=canvas.safe_top + 26)  # below the notch
                if canvas.show(img) in (27, ord("q")):
                    print("Aborted; nothing saved.")
                    return
                if frame is None or elapsed < SETTLE_S:
                    continue
                result = tracker.process(frame)
                if result is None:
                    rec.add(frame.timestamp_ms, phase, idx, xy, None, None, None)
                    continue
                h, w = frame.image.shape[:2]
                rec.meta["image_size"] = [w, h]
                light_log.append(lighting_stats(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), result.points))
                if ref is not None:
                    posture_log.append(posture_metrics(result.points, result.transform, (w, h)))
                feats = extract_features(result.points, result.transform, (w, h))
                crop = None
                if save_images:
                    box = crop_box(result.points, (w, h))
                    crop = (encode_crop(frame.image, box, QUALITY), box)
                rec.add(frame.timestamp_ms, phase, idx, xy, result.points, result.transform,
                        feats, result.blendshapes, crop)
        rec.meta["blendshape_names"] = list(tracker.blendshape_names)
        summary = summarize_recording(ref, posture_log) if ref is not None else None
        if summary is not None:
            rec.meta["posture_summary"] = summary
        lighting = summarize_lighting(light_log)
        if lighting is not None:
            rec.meta["lighting"] = lighting
        rec.meta["face_images"] = save_images
        path = rec.save(out_dir)
        print(f"Saved {path}")
        if save_images and faces_path_for(path).exists():
            print(f"face images: {faces_path_for(path).name}, {faces_path_for(path).stat().st_size / 1e6:.1f} MB "
                  f"(local only)")
        if lighting is not None:
            print(f"lighting: {lighting['label']} (face {lighting['face']:.0f}/255, background "
                  f"{lighting['background']:.0f}, ratio {lighting['ratio']:.2f}, clipped {lighting['clipped'] * 100:.1f}%)")
        if summary is not None:
            print(f"posture: in position for {summary['fraction_good'] * 100:.0f}% of frames - {summary['verdict']}")
        notes = []
        if lighting is not None:
            cls = lighting_class(rec.meta)
            names = {"F": "front-lit (F)", "B": "backlit (B)", "D": "dim (D)"}
            notes.append("Round-2 class by captured lighting: " + (names[cls] if cls else
                         "none, outside every class range (would be excluded)"))
            mismatch = note_mismatch(args.note, lighting["label"])
            if mismatch:
                notes.append("CHECK YOUR NOTE: " + mismatch)
        for line in notes:
            print(line)
        wait_for_key(canvas, lambda: draw_summary(canvas, summary, path.name, lighting, notes), (32, 27, ord("q")))
    return


if __name__ == "__main__":
    main()
