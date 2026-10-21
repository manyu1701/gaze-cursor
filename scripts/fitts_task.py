"""Step 7's jump-and-click task: click targets with the trackpad alone or with gaze-jump + trackpad, from the
calibrated seat or a moved one, plus a short eyes-only vs head-turn dot block.

Starts with the 9-dot quick calibration (recorded like any other), then runs the blocks. A target circle
appears; click it. In gaze-jump blocks, tap Right Command to jump the cursor to where you look and finish with
the trackpad; in trackpad blocks the jump is off. Between blocks a seat check says whether you sit as the next
block needs (centre = as calibrated, moved = head moved against the calibration, measured the way round 3 did).
Everything is saved to data/tasks/fitts_<time>.npz (local only): gaze frames, jumps, clicks, the pointer path,
per-trial phase times, the conditions block, the model's state id and face crops before each click
(--no-images skips them). The model is frozen during the task (no click learning).

The protocol (counts, sizes, order) is frozen in PREREGISTRATIONS.md; the defaults here are for development.

  uv run scripts/fitts_task.py [--laps 1] [--amplitudes 600 800] [--widths 70 140] [--no-dots] [--no-seat-gate]
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # cursor_mode.py is a sibling script

import cursor_mode as cm  # noqa: E402

from gaze_cursor import model_history  # noqa: E402
from gaze_cursor.capture import Camera  # noqa: E402
from gaze_cursor.config import ScreenConfig  # noqa: E402
from gaze_cursor.cursor_session import CursorSession  # noqa: E402
from gaze_cursor.eval.configs import REGISTRY  # noqa: E402
from gaze_cursor.eval.results_log import append_rows, now  # noqa: E402
from gaze_cursor.fitts_task import (  # noqa: E402
    DOT_MODES, METHODS, SEATS, FittsTask, dot_schedule, make_blocks, seat_ok)
from gaze_cursor.landmarks import FaceTracker  # noqa: E402
from gaze_cursor.online import OnlineRidge  # noqa: E402
from gaze_cursor.posture import Reference  # noqa: E402
from gaze_cursor.quick import QUICK_CONFIG  # noqa: E402
from gaze_cursor.sinks import tags as condition_tags  # noqa: E402
from gaze_cursor.sinks.cursor import (  # noqa: E402
    PERMISSION_HELP, HotkeyDetector, MacCursor, input_monitoring_granted, request_input_monitoring)
from gaze_cursor.sinks.overlay import FullscreenCanvas  # noqa: E402
from gaze_cursor.sinks.quick_view import QUICK_DIR, set_tags  # noqa: E402

TASKS_DIR = QUICK_DIR.parent / "tasks"
FEEDBACK_MS = 300.0           # green / red flash after a click; not part of any trial time
DOT_SETTLE_MS, DOT_HOLD_MS = 700.0, 1300.0
SEAT_WINDOW = 30              # frames (about a second) the seat check looks at
WHITE, GREEN, RED, GREY = (255, 255, 255), (60, 220, 60), (60, 60, 230), (140, 140, 140)
DOT_HELP = {"eyes": "Keep your head still and move only your eyes to each dot.",
            "head": "Turn your head to face each dot (your eyes stay on it too)."}


class Aborted(Exception):
    """The user pressed q / Esc, or Cmd+Option+X."""


class Runner:
    """Frame loop shared by every phase: camera -> gaze log, input events -> session and task."""

    def __init__(self, cam, tracker, canvas, cursor, session: CursorSession, task: FittsTask, names, ref) -> None:
        self.cam, self.tracker, self.canvas, self.cursor = cam, tracker, canvas, cursor
        self.session, self.task, self.names, self.ref = session, task, names, ref
        self.shifts: list[tuple[float, float, float] | None] = []

    def step(self, img: np.ndarray | None = None) -> tuple[int, list[str]]:
        """One camera frame + one canvas redraw. Returns (key, outcomes of the task's clicks this frame)."""
        frame = self.cam.read()
        result = self.tracker.process(frame) if frame is not None else None
        cm.process_frame(self.session, frame, result, self.names, self.ref)
        self.shifts.append(self.session.shift)
        del self.shifts[:-SEAT_WINDOW]
        outcomes: list[str] = []
        for event in self.cursor.poll():
            t = time.monotonic_ns() / 1e6
            outcome = self.task.on_event(event)
            if outcome:
                outcomes.append(outcome)
            if event.kind in ("move", "toggle", "recalibrate"):   # the path goes to the task's array, not the
                continue                                          # event log; the other two hotkeys do nothing here
            action, xy = self.session.handle(event, t)
            if action == "warp" and xy is not None:
                self.cursor.warp(xy)
            elif action == "quit":
                raise Aborted
        key = self.canvas.show(self.canvas.blank() if img is None else img)
        if key in (27, ord("q")):
            raise Aborted
        return key, outcomes

    def seat_ready(self, seat: str) -> bool:
        return len(self.shifts) >= SEAT_WINDOW and seat_ok(self.shifts, seat)

    def pointer(self) -> tuple[float, float] | None:
        try:
            return self.cursor.pointer()
        except Exception:
            return None


def wait_for_space(run: Runner, lines: list[str], ready=lambda: True, status: str = "") -> bool:
    """Show ``lines`` until SPACE (only once ``ready()``). Returns False if the seat never mattered (unused)."""
    while True:
        img = run.canvas.blank()
        y = run.canvas.size[1] // 2 - 40 * len(lines)
        for line in lines:
            run.canvas.text(img, line, y=y)
            y += 50
        ok = ready()
        if status:
            run.canvas.text(img, status + (": OK, press SPACE" if ok else ": not yet"), y=y + 30)
        else:
            run.canvas.text(img, "SPACE to start, q to quit", y=y + 30)
        key, _ = run.step(img)
        if key == 32 and ok:
            return True


def run_dots(run: Runner, schedule: list[dict], dots_log: list[dict]) -> None:
    """The eyes-only then head-turn dot block: every dot is held DOT_SETTLE + DOT_HOLD ms; the gaze frames in
    between are the labelled samples (the dot's position and the time window are in ``dots_log``)."""
    seen: set[str] = set()
    for item in schedule:
        if item["mode"] not in seen:
            seen.add(item["mode"])
            wait_for_space(run, [f"Dots: {item['mode']}", DOT_HELP[item["mode"]]])
        t0 = time.monotonic_ns() / 1e6
        while (t := time.monotonic_ns() / 1e6) - t0 < DOT_SETTLE_MS + DOT_HOLD_MS:
            img = run.canvas.blank()
            run.canvas.dot(img, item["xy"], color=WHITE if t - t0 < DOT_SETTLE_MS else GREEN)
            run.step(img)
        dots_log.append({"mode": item["mode"], "index": item["index"], "xy": item["xy"], "t_start_ms": t0,
                         "t_end_ms": t, "settle_ms": DOT_SETTLE_MS})


def run_block(run: Runner, number: int, total: int, gate: bool) -> dict:
    """One block of the task: seat check, then targets until the block's trials are done."""
    task, cfg, canvas = run.task, run.session.cfg, run.canvas
    b = task.start_block()
    run.session.enabled = b.method == "gaze_jump"
    seat_text = {"centre": "Sit as you did when you calibrated.",
                 "moved": "Move your head / chair about 10 cm sideways or closer to or away from the screen."}[b.seat]
    how = ("Tap Right Command to jump the cursor to your gaze, then finish with the trackpad."
           if b.method == "gaze_jump" else "Use the trackpad only (the gaze jump is off).")
    wait_for_space(run, [f"Block {number}/{total}: {b.method.replace('_', '-')}, {b.seat} seat", how, seat_text,
                         f"Click each circle ({b.width * cfg.cm_per_px:.1f} cm wide)."],
                   ready=(lambda: run.seat_ready(b.seat)) if gate else (lambda: True),
                   status=f"{b.seat} seat" if gate else "")
    block_start = len(task.trials)
    while not task.block_done:
        trial = task.next_target(time.monotonic_ns() / 1e6, run.pointer(), run.session.shift)
        outcome, flash_until = None, 0.0
        while outcome is None:
            img = canvas.blank()
            canvas.circle(img, trial.center, trial.width / 2, WHITE, 3)
            canvas.text(img, f"block {number}/{total}   target {len(task.trials) - block_start}/{b.n_trials}   q quits",
                        y=canvas.size[1] - 30)
            _, outcomes = run.step(img)
            outcome = outcomes[0] if outcomes else task.update(time.monotonic_ns() / 1e6)
        flash_until = time.monotonic_ns() / 1e6 + FEEDBACK_MS
        while time.monotonic_ns() / 1e6 < flash_until:
            img = canvas.blank()
            canvas.circle(img, trial.center, trial.width / 2, GREEN if outcome == "hit" else RED, -1)
            run.step(img)
    mine = [t.shift_at_onset for t in task.trials[block_start:]]
    return {"index": number - 1, "method": b.method, "seat": b.seat, "amplitude_px": b.amplitude,
            "width_px": b.width, "n_trials": b.n_trials, "first_trial": block_start,
            "seat_ok_during_block": bool(mine) and seat_ok(mine, b.seat)}


def summary_lines(task: FittsTask) -> list[str]:
    return [f"{k}: {v['n']} targets, {v['hit_rate']:.0%} hit, median {v['median_time_ms'] / 1000:.2f} s"
            for k, v in task.summary().items()] or ["No targets finished."]


def save(run: Runner, args, blocks_log: list[dict], dots_log: list[dict], state_id: str | None,
         aborted: bool) -> Path:
    task, session = run.task, run.session
    extra = {"task": {
        "name": "step7_jump_and_click", "seed": args.seed, "aborted": aborted, "model_state_id": state_id,
        "learning": False, "blocks": blocks_log, "dots": dots_log,
        "trials": [t.to_dict() for t in task.trials if t.hit is not None], "summary": task.summary(),
        "ring_points": task.n_ring, "timeout_ms": task.timeout_ms, "feedback_ms": FEEDBACK_MS,
        "dot": {"settle_ms": DOT_SETTLE_MS, "hold_ms": DOT_HOLD_MS, "modes": list(DOT_MODES)},
        "seat_gate": not args.no_seat_gate, "method_notes": {
            "trackpad": "gaze jump off (warp rows log enabled = 0)", "gaze_jump": "tap Right Command, then trackpad"},
        "notes": "Times are camera-clock ms. move rows of 'moves' = [t_ms, x, y, trial index (-1 = no target)]; "
                 "a warp's landing is also a move row. Pointer positions are main-display points, top-left origin.",
        "moves_columns": ["t_ms", "x", "y", "trial"]}}
    return session.save(TASKS_DIR, prefix="fitts", kind="fitts_task", extra_meta=extra,
                        extra_arrays={"moves": task.moves_array()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", type=int, default=1, help="laps of the 9-point ring per block")
    ap.add_argument("--amplitudes", type=float, nargs="+", default=[600.0], help="ring diameters, screen points")
    ap.add_argument("--widths", type=float, nargs="+", default=[70.0, 140.0], help="target diameters, screen points")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--seats", nargs="+", choices=SEATS, default=list(SEATS))
    ap.add_argument("--seed", type=int, default=0, help="block order and dot order")
    ap.add_argument("--no-dots", action="store_true", help="skip the eyes-only / head-turn dot block")
    ap.add_argument("--no-seat-gate", action="store_true", help="development only: do not check the seat")
    ap.add_argument("--no-images", action="store_true", help="do not keep face crops before each click")
    condition_tags.add_arguments(ap)
    args = ap.parse_args(argv)
    if not input_monitoring_granted():
        request_input_monitoring()
        print(PERMISSION_HELP)
        return 1
    cfg = ScreenConfig.from_system()
    tags = condition_tags.resolve(args)               # refuses to record without a person id
    set_tags(tags)
    blocks = make_blocks(cfg, tuple(args.methods), tuple(args.seats), tuple(args.amplitudes), tuple(args.widths),
                         laps=args.laps, seed=args.seed)
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, cm.stop_on_signal)
    with Camera() as cam, FaceTracker() as tracker:
        fit = cm.calibrate(cam, tracker, cfg)
        if fit is None:
            return 0
        if fit.y is None:
            print("The calibration carries no dot positions; cannot start the task.")
            return 1
        ref_center = Reference.from_current(fit.posture).center
        model =OnlineRidge.from_calibration(fit.x, fit.y, REGISTRY[QUICK_CONFIG].alpha)   # frozen: never updated
        session = CursorSession(cfg, model, fit.x, ref_center, fit.path.name, learn=False,
                                keep_faces=not args.no_images)
        session.conditions = cm.session_conditions(tags, cfg, session.calib, ref_center, session.calibrated_at)
        condition_tags.print_tag_verdict(session.conditions)
        c = session.conditions or {}
        try:
            state_id = model_history.archive(QUICK_DIR, model, session.x_train, ref_center, session.calib, cfg,
                                             session.calibrated_at, "calibration", person_id=c.get("person_id"),
                                             code=c.get("git_hash"))
        except Exception as e:                        # a full disk must not stop the task
            state_id = model_history.state_id(model)
            print(f"Model history not written: {e}")
        task = FittsTask(cfg, blocks)
        cursor = MacCursor((cfg.width_px, cfg.height_px), HotkeyDetector(), track_moves=True)
        blocks_log: list[dict] = []
        dots_log: list[dict] = []
        aborted = False
        run: Runner | None = None
        try:
            with FullscreenCanvas(cfg) as canvas:
                canvas.keep_pointer = True
                run = Runner(cam, tracker, canvas, cursor, session, task, tuple(tracker.blendshape_names),
                             SimpleRef(ref_center))
                try:
                    if not args.no_dots:
                        run_dots(run, dot_schedule(cfg, seed=args.seed), dots_log)
                    for i in range(len(blocks)):
                        blocks_log.append(run_block(run, i + 1, len(blocks), not args.no_seat_gate))
                except (Aborted, KeyboardInterrupt):
                    aborted = True
                lines = (["Task " + ("stopped early" if aborted else "finished")] + summary_lines(task)
                         + ["", "SPACE or q to close"])
                while True:
                    img = canvas.blank()
                    y = canvas.size[1] // 2 - 30 * len(lines)
                    for line in lines:
                        canvas.text(img, line, y=y)
                        y += 50
                    if canvas.show(img) in (27, 32, ord("q")):
                        break
        finally:
            cursor.close()
            if run is not None:
                path = save(run, args, blocks_log, dots_log, state_id, aborted)
                s = session.summary()
                append_rows([{"time": now(), "mode": "fitts_task", "confirmatory": 0, "session": session.calib,
                              "config": f"{QUICK_CONFIG} frozen", "n_frames": s["frames"],
                              "n_targets": len([t for t in task.trials if t.hit is not None]),
                              "note": f"{path.name}; " + "; ".join(summary_lines(task))}])
                print(f"saved {path.name} to data/tasks/ (local only) and logged to data/results.csv")
    return 0


class SimpleRef:
    """What ``cm.process_frame`` needs of a reference: the calibration posture's centre."""

    def __init__(self, center: dict) -> None:
        self.center = center


if __name__ == "__main__":
    raise SystemExit(main())
