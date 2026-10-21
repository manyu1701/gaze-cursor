"""Cursor mode (M3): tap Right Command and the real cursor jumps to where you look; the trackpad finishes.

Starts with the 9-dot quick calibration where you sit (same as Run), then gets out of the way: a
transparent status chip in the top-right corner, every app keeps working, clicks go through. The cursor
only ever moves when you tap the hotkey, and not when the tracker cannot see your eyes.

Starts instantly when a calibration from the last 12 h (same screen) exists: it is reused, together with
what the model learned from your clicks. Click learning is on (round 4, H6): every trackpad click nudges
the model towards where you looked. The session is autosaved every 5 min and saved when you quit, close
the window or the process is shut down. --fresh forces the 9-dot calibration; --no-learn turns learning off.
--demo (Demo.command) is for showing a friend: their own 9-dot calibration, the gaze ring visible, and
NOTHING written to disk (no calibration recording, no session log, no saved calibration, no results row).

Hotkeys (work in any app): tap Right Command = jump to gaze; Cmd+Option+G = on / off;
Cmd+Option+R = recalibrate; Cmd+Option+X (or Ctrl+C here) = quit.

The status chip (top-right) is hidden most of the time: it appears for a few seconds when something changes
(start, on/off, a refused jump, a warning), comes back every minute while a warning lasts, and counts down
before the auto-quit. --chip always keeps it on (Demo does), never hides it.

No face for 30 s: low power (the landmarker runs on 1 frame in 10); no face for 5 min: quits and saves
(constants in src/gaze_cursor/idle.py).

Every frame (gaze, head shift) and every hotkey / click is saved to data/quick/cursor_<time>.npz
(positions only, no images) and summarised in data/results.csv, so self-recalibration from clicks can be
built and judged offline. Needs Input Monitoring permission for Terminal (it explains how).

  uv run scripts/cursor_mode.py [--show-dot] [--fresh] [--no-learn] [--demo] [--chip auto|always|never] [--warp-key right_command|right_option]
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import time
from types import SimpleNamespace


from gaze_cursor import model_history, resume
from gaze_cursor.capture import Camera
from gaze_cursor.config import ScreenConfig
from gaze_cursor.cursor_session import CursorSession
from gaze_cursor.eval.configs import REGISTRY
from gaze_cursor.eval.faces import crop_box, faces_path_for
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.features import assemble, extract_features
from gaze_cursor.idle import IdleTimer
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.livetest import head_shift
from gaze_cursor.online import OnlineRidge
from gaze_cursor.posture import Reference, posture_metrics
from gaze_cursor.quick import QUICK_CONFIG
from gaze_cursor.sinks.cursor import (
    DEFAULT_WARP_KEY, PERMISSION_HELP, WARP_KEYS, HotkeyDetector, MacCursor, input_monitoring_granted,
    request_input_monitoring)
from gaze_cursor.sinks.cursor_view import AMBER, GREEN, GREY, ChipTimer, draw_gaze_dot, draw_status
from gaze_cursor.sinks.overlay import ClickThroughOverlay, FullscreenCanvas
from gaze_cursor import backfill
from gaze_cursor.conditions import read_meta
from gaze_cursor.sinks import tags as condition_tags
from gaze_cursor.sinks.quick_view import QUICK_DIR, QuickFit, calibrate_here, set_tags

NOTE = "quick calibration (Cursor)"
AUTOSAVE_S = 300.0
CHIP_SECONDS = 4.0          # in --chip auto, how long the status chip stays after something changed
CHIP_REMIND_S = 60.0        # ... and how often it comes back while a lasting warning is on
LEVEL_COLORS = {"ok": GREEN, "warn": AMBER, "off": GREY}


def calibrate(cam: Camera, tracker: FaceTracker, cfg: ScreenConfig, record: bool = True) -> QuickFit | None:
    """The quick calibration on its own fullscreen canvas (closed again afterwards)."""
    with FullscreenCanvas(cfg) as canvas:
        out = calibrate_here(cam, tracker, canvas, cfg, NOTE, record)
    if isinstance(out, str):
        print("Cursor mode always uses a fresh quick calibration (L is not available here).")
        return None
    return out


def session_conditions(tags: condition_tags.Tags, cfg: ScreenConfig, calib: str,
                       posture: dict | None = None, calibrated_at: float | None = None) -> dict:
    """Conditions block of a Cursor log: the calibration's measured values (camera frame size, iris size,
    lighting) with this run's tags, layout and code version; built without iris size if that file is gone."""
    try:
        calib_meta = read_meta(QUICK_DIR / calib)
    except (OSError, ValueError, KeyError):
        calib_meta = {}
    block = calib_meta.get("conditions") or {}
    dist = block.get("distance") or {}
    cam = block.get("camera") or {}
    size = (cam["width"], cam["height"]) if cam.get("width") and cam.get("height") else None
    iris = dist.get("iris_px")
    lighting = block.get("lighting_measured")
    if size is None and calib_meta.get("image_size"):     # a calibration made before the tags existed
        size = tuple(calib_meta["image_size"])
    if iris is None and calib_meta:                       # ... measure its iris from its own landmarks
        try:
            iris = backfill.median_iris_px(QUICK_DIR / calib)
        except (OSError, ValueError, KeyError):
            iris = None
        iris = None if iris is not None and iris != iris else iris
    lighting = lighting or calib_meta.get("lighting")
    out = condition_tags.conditions_for(tags, cfg, frame_size=size,
                                        iris_px=float("nan") if iris is None else float(iris),
                                        lighting_measured=lighting,
                                        posture=posture or block.get("posture_measured"))
    # A Cursor log's lighting and posture are the CALIBRATION's, which a resumed run may have made hours ago:
    # say so in the source flags and give the age, so they never read as measured during this run.
    for key in ("lighting_tag_source", "posture_note_source"):
        if key in out and out[key] != "typed":
            out[key] = f"{out[key]}@calibration"
    out["posture_measured_in"] = calib
    out["calibration_age_h"] = None if calibrated_at is None else max(0.0, (time.time() - calibrated_at) / 3600)
    out["distance"]["measured_in"] = calib          # these were measured at that calibration, not in this run
    out["lighting_measured_in"] = calib
    return out


def learner(fit: QuickFit, learn: bool):
    """The model Cursor mode runs: the quick fit as an online ridge (identical predictions at the start)
    when learning is on and the fit carries its dots' positions, else the frozen fit."""
    if learn and fit.y is not None:
        return OnlineRidge.from_calibration(fit.x, fit.y, REGISTRY[QUICK_CONFIG].alpha)
    return fit.model


def remember(session: CursorSession, cfg: ScreenConfig, reason: str) -> None:
    """Archive the learned state (L2: every state is kept, with provenance), then write the resume file
    (calibration + learned state) so the next start needs no dots. A full disk must not stop the live loop."""
    if isinstance(session.model, OnlineRidge) and session.ref_center is not None:
        c = session.conditions or {}
        try:
            session.state_id = model_history.archive(
                QUICK_DIR, session.model, session.x_train, session.ref_center, session.calib, cfg,
                session.calibrated_at, reason, parent_id=session.state_id,
                cursor_log=f"cursor_{session.started}.npz", person_id=c.get("person_id"), code=c.get("git_hash"))
            archived = session.state_id
        except Exception as e:    # the live loop (and the cursor log at quit) must survive any archive failure
            archived = None       # session.state_id keeps the last archived state: still the true parent
            print(f"Model history not written: {e}")
        resume.save(QUICK_DIR, session.model, session.x_train, session.ref_center, session.calib, cfg,
                    session.calibrated_at, archived)


def preserve_previous(cfg: ScreenConfig) -> None:
    """Archive the state in ``last_calibration.npz`` before anything can overwrite it (a file written before
    L2, a stale one, one for another screen). The archive is idempotent, so a state already kept costs nothing."""
    saved, screen = resume.read_unchecked(QUICK_DIR)
    if saved is not None:
        model_history.archive(QUICK_DIR, saved.model, saved.x_train, saved.ref_center, saved.calib,
                              cfg if screen is None else ScreenConfig(width_px=screen[0], height_px=screen[1]),
                              saved.calibrated_at, "found on disk at start")


def notify(text: str) -> None:
    """A macOS notification (best effort: it may be switched off for Terminal)."""
    try:
        subprocess.run(["osascript", "-e", f'display notification "{text}" with title "gaze-cursor"'],
                       check=False, timeout=5, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        pass


def ago(hours: float) -> str:
    return f"{hours * 60:.0f} min ago" if hours < 1 else f"{hours:.0f} h ago"


def stop_on_signal(signum, frame) -> None:
    """Closing the Terminal window (SIGHUP) or a kill (SIGTERM) quits cleanly, so the session is saved."""
    raise KeyboardInterrupt


def process_frame(session: CursorSession, frame, result, names: tuple[str, ...], ref: Reference | None) -> None:
    """Features -> session.frame (prediction, smoothing, input range, head shift, log)."""
    x = shift = None
    if frame is not None and result is not None:
        h, w = frame.image.shape[:2]
        if ref is not None:
            shift = head_shift(posture_metrics(result.points, result.transform, (w, h)), ref.center)
        feats = extract_features(result.points, result.transform, (w, h))
        if feats is not None:
            x = assemble(feats, result.blendshapes, names, "gaze")
    t = frame.timestamp_ms if frame is not None else time.monotonic_ns() / 1e6
    session.frame(t, x, shift)
    if session.keep_faces and frame is not None and result is not None:
        session.offer_crop(t, frame.image, crop_box(result.points, (frame.image.shape[1], frame.image.shape[0])),
                           valid=x is not None)


def draw(overlay: ClickThroughOverlay, lines: list[tuple[str, str]], session: CursorSession,
         show_dot: bool) -> None:
    img = overlay.blank()
    draw_status(img, [(t, LEVEL_COLORS[lvl]) for t, lvl in lines], overlay.safe_top)
    if show_dot and session.enabled and session.dot is not None:
        draw_gaze_dot(img, (float(session.dot[0]), float(session.dot[1])))
    overlay.show(img)


def finish(session: CursorSession) -> None:
    path = session.save(QUICK_DIR)
    s = session.summary()
    med = s["click_gaze_deg_median"]
    note = (f"{s['minutes']:.1f} min, face tracked {s['tracked'] * 100:.0f}%; {s['warps']} jumps "
            f"({s['warps_refused']} refused); {s['clicks']} clicks ({s['clicks_learned']} learned from)"
            + (f"; click vs gaze just before it: median {med:.1f} deg (mean {s['click_gaze_deg_mean']:.1f}, "
               f"n {s['click_gaze_n']})" if med is not None else ""))
    append_rows([{"time": now(), "mode": "cursor_session", "confirmatory": 0, "session": session.calib,
                  "config": f"{QUICK_CONFIG} + One-Euro", "n_frames": s["frames"], "n_targets": s["clicks"],
                  "err_deg_mean": s["click_gaze_deg_mean"], "err_deg_median": med, "note": note}])
    faces = faces_path_for(path)
    if session.face_jpegs and faces.exists():
        note += f"; {len(session.face_jpegs)} face crops, {faces.stat().st_size / 1e6:.1f} MB (local only)"
    try:
        print(f"saved {path.name}: {note}")
    except OSError:         # the Terminal window was closed (SIGHUP): the data is already on disk
        pass
    notify(f"Saved: {s['minutes']:.0f} min, {s['clicks']} clicks, {s['clicks_learned']} learned from")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-dot", action="store_true", help="draw a faint ring where the tracker thinks you look")
    ap.add_argument("--fresh", action="store_true", help="always run the 9-dot calibration (do not reuse the last)")
    ap.add_argument("--no-learn", action="store_true", help="keep the model frozen (do not learn from clicks)")
    ap.add_argument("--no-images", action="store_true",
                    help="do not keep the face crops from just before each click (default: saved next to the "
                         "log, local only, at most ~150 MB)")
    ap.add_argument("--demo", action="store_true",
                    help="show a friend: fresh calibration, gaze ring on, nothing is saved to disk")
    ap.add_argument("--chip", choices=("auto", "always", "never"), default=None,
                    help="status chip in the top-right corner: auto (default) shows it for a few seconds when "
                         "something changes, always keeps it on (default for --demo), never hides it")
    ap.add_argument("--warp-key", choices=sorted(WARP_KEYS), default=DEFAULT_WARP_KEY,
                    help="the modifier to tap (alone) to jump the cursor to your gaze")
    condition_tags.add_arguments(ap)
    args = ap.parse_args()
    if not input_monitoring_granted():
        request_input_monitoring()
        print(PERMISSION_HELP)
        return 1
    cfg = ScreenConfig.from_system()
    demo = args.demo
    persist = not demo                      # a demo never writes anything about the face to disk
    tags = None if demo else condition_tags.resolve(args)   # refuses to record without a person id
    if tags is not None:
        set_tags(tags)
    show_dot = args.show_dot or demo
    learn = not args.no_learn
    chip = args.chip or ("always" if demo else "auto")
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, stop_on_signal)
    with Camera() as cam, FaceTracker() as tracker:
        if persist:
            try:
                preserve_previous(cfg)
            except Exception as e:
                print(f"Model history not written: {e}")
        saved, why = (None, "--fresh") if (args.fresh or demo or not learn) else resume.load(QUICK_DIR, cfg)
        names = tuple(tracker.blendshape_names)
        if saved is not None:
            ref_center, resumed = saved.ref_center, ago(saved.age_h())
            session = CursorSession(cfg, saved.model, saved.x_train, ref_center, saved.calib, learn=True,
                                    resumed=resumed, calibrated_at=saved.calibrated_at, state_id=model_history.state_id(saved.model))
            print(f"Reusing your calibration from {resumed} ({saved.model.n_updates} clicks learned so far). "
                  f"Cmd+Option+R recalibrates; --fresh always starts with the dots.")
        else:
            if why not in ("--fresh", "no earlier calibration"):
                print(f"Calibrating again: {why}.")
            fit = calibrate(cam, tracker, cfg) if persist else calibrate(cam, tracker, cfg, record=False)
            if fit is None:
                return 0
            ref_center = Reference.from_current(fit.posture).center
            session = CursorSession(cfg, learner(fit, learn), fit.x, ref_center, fit.path.name, learn=learn)
        if demo:
            session.banner = "DEMO: nothing is saved"
        else:
            session.conditions = session_conditions(tags, cfg, session.calib, session.ref_center,
                                                         session.calibrated_at)
            condition_tags.print_tag_verdict(session.conditions)
            session.keep_faces = not args.no_images
            if saved is None:    # archived only now: the root state of a chain must carry person id and code hash
                remember(session, cfg, "calibration")
        ref = SimpleNamespace(center=ref_center)
        print("Cursor mode on: tap Right Command to jump to your gaze; Cmd+Option+G on/off, "
              "Cmd+Option+R recalibrate, Cmd+Option+X quit.")
        cursor = MacCursor((cfg.width_px, cfg.height_px), HotkeyDetector(args.warp_key))
        try:
            with ClickThroughOverlay(cfg) as overlay:
                last_lines = None
                chip_timer = ChipTimer(chip, CHIP_SECONDS, CHIP_REMIND_S)
                next_save = time.monotonic() + AUTOSAVE_S
                idle = IdleTimer(time.monotonic())
                while True:
                    frame = cam.read()
                    now_s = time.monotonic()
                    # Asleep (no face for 30 s): the landmarker runs on 1 frame in 10, the rest is skipped.
                    result = tracker.process(frame) if frame is not None and idle.process_this_frame(now_s) else None
                    idle.update(now_s, result is not None)
                    if idle.should_quit(now_s):
                        print("No face for 5 min: quitting to save battery (the session is saved).")
                        notify("No face for 5 min: stopped to save battery")
                        break
                    process_frame(session, frame, result, names, ref)
                    action = "none"
                    for event in cursor.poll():
                        action, xy = session.handle(event, time.monotonic_ns() / 1e6)
                        if action == "warp":
                            cursor.warp(xy)
                        if action in ("quit", "recalibrate"):
                            break
                    if action == "quit":
                        break
                    if persist and now_s > next_save:
                        session.save(QUICK_DIR)
                        remember(session, cfg, "autosave")
                        next_save = time.monotonic() + AUTOSAVE_S
                    if action == "recalibrate":
                        new = calibrate(cam, tracker, cfg) if persist else calibrate(cam, tracker, cfg, record=False)
                        overlay.restore_policy()
                        if new is not None:
                            ref = Reference.from_current(new.posture)
                            if persist:
                                remember(session, cfg, "superseded")   # keep what was learned before it is replaced
                            session.recalibrated(learner(new, learn), new.x, ref.center, new.path.name)
                            if persist:
                                remember(session, cfg, "calibration")
                        last_lines = None
                        continue
                    nap = idle.message(now_s)             # low power, then a countdown to the auto-quit
                    extra = [(nap, "warn")] if nap else []
                    key = session.status_key() + tuple(extra)
                    shown = chip_timer.visible(now_s, key, remind=session.lasting_warning())
                    lines = (session.status_lines() + extra) if shown else []
                    if show_dot or lines != last_lines:   # redraw only when something changed
                        draw(overlay, lines, session, show_dot)
                        last_lines = lines
                    else:
                        overlay.pump()
        except KeyboardInterrupt:
            pass
        finally:
            cursor.close()
            if persist:
                remember(session, cfg, "quit")
                finish(session)
            else:
                print("Demo ended: nothing was saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
