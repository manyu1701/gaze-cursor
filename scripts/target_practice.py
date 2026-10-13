"""Target practice: a box appears, move the gaze dot into it with your eyes and hold it there.

The box fills while the dot stays inside; after 0.6 s it is a hit and the next box appears elsewhere.
The box shrinks after two hits in a row and grows after a miss (8 s without a hit), so by the end it
sits at the smallest size you can reliably hit. The summary says how big that is, how fast you hit,
how far the dot is from where you look on average, and whether it is systematically shifted.

Every frame is saved as labelled data (you were looking at the box) to data/tasks/ (local only,
gitignored; never used as a calibration), and one row goes to data/results.csv (mode target_task).

  uv run scripts/target_practice.py            # model from your latest calibration (what Run uses)
  uv run scripts/target_practice.py --pool     # all calibrations in the same lighting (under test)
  uv run scripts/target_practice.py --trials 30
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from gaze_cursor.capture import Camera
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.eval.session import SESSIONS_DIR, TEST, SessionRecorder, latest_session, load_session
from gaze_cursor.features import assemble, extract_features
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.live import fit_live_model
from gaze_cursor.posture import Reference, lighting_stats, summarize_lighting
from gaze_cursor.sinks.overlay import FullscreenCanvas
from gaze_cursor.sinks.posture_view import run_posture_check
from gaze_cursor.target_task import TargetTask, TaskConfig, describe_bias

TASKS_DIR = SESSIONS_DIR.parent / "tasks"
SMOOTH_FRAMES = 5          # same moving average as the overlay
FEEDBACK_S = 0.5           # green / red flash between boxes (not counted in the times)
WHITE, GREEN, RED, YELLOW, GREY = (255, 255, 255), (60, 220, 60), (60, 60, 230), (0, 220, 255), (120, 120, 120)


def summary_lines(s: dict) -> list[str]:
    if not s.get("n_trials"):
        return ["No boxes finished."]
    lines = [f"Hits: {s['n_hits']} of {s['n_trials']} ({s['hit_rate']:.0%})"]
    if s["n_hits"]:
        lines.append(f"Median time to hit: {s['median_time_s']:.1f} s (0.6 s of that is the hold)")
    lines += [f"Smallest box you can reliably hit: about {s['box_cm']:.1f} cm ({s['box_deg']:.1f} deg)",
              f"Average distance of the dot from the box centre: {s['error_cm']:.1f} cm ({s['error_deg']:.1f} deg)",
              describe_bias(s["bias_cm"]).capitalize()]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", type=Path, help="calibration to use (default: the latest)")
    ap.add_argument("--pool", action="store_true", help="all calibrations recorded in the same lighting")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-check", action="store_true", help="skip the position and lighting check")
    ap.add_argument("--no-save", action="store_true", help="do not save the frames or log the result")
    args = ap.parse_args()

    session = load_session(args.session or latest_session())
    cfg = ScreenConfig.from_system()
    live = fit_live_model(session, cfg, pool=args.pool)
    print("model: " + live.label)
    names = tuple(session.meta["blendshape_names"])
    task = TargetTask(cfg, TaskConfig(n_trials=args.trials), seed=args.seed)
    rec = SessionRecorder(meta={
        "kind": "target_task", "screen": cfg.to_dict(), "model": live.label, "pool": args.pool,
        "trained_on": [s.path.name for s in live.sessions], "task": vars(task.cfg), "mirrored": False,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "blendshape_names": [],
    })
    light_log: list[dict[str, float]] = []
    recent: deque[np.ndarray] = deque(maxlen=SMOOTH_FRAMES)

    with Camera() as cam, FaceTracker() as tracker, FullscreenCanvas(cfg) as canvas:
        rec.meta["window"] = [*canvas.origin, *canvas.size]
        if not args.no_check:
            check = run_posture_check(cam, tracker, canvas, Reference.from_sessions(live.sessions),
                                      title="Sit as you did when you calibrated",
                                      ref_lighting=session.meta.get("lighting"),
                                      override_label="start anyway (accuracy may drop)")
            if not check.proceed:
                return 0
        while True:                                   # instructions
            img = canvas.blank()
            canvas.text(img, "A box will appear. Look at it: the dot follows your eyes.", y=canvas.size[1] // 2 - 40)
            canvas.text(img, "Keep the dot inside until the box fills. SPACE to start, q to quit.",
                        y=canvas.size[1] // 2 + 10)
            key = canvas.show(img)
            cam.read()
            if key in (27, ord("q")):
                return 0
            if key == 32:
                break

        t0 = time.monotonic()
        task.start(0.0)
        feedback: tuple[float, str] | None = None    # (until, 'hit' | 'miss')
        aborted = False
        n_frames = 0
        while not task.done or feedback is not None:
            t = time.monotonic() - t0
            frame = cam.read()
            result = tracker.process(frame) if frame is not None else None
            dot = None
            if result is not None:
                h, w = frame.image.shape[:2]
                rec.meta["image_size"] = [w, h]
                feats = extract_features(result.points, result.transform, (w, h))
                if feats is not None:
                    recent.append(live.model.predict(assemble(feats, result.blendshapes, names, live.feature_set))[0])
                if len(light_log) < 300 and n_frames % 10 == 0:
                    light_log.append(lighting_stats(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), result.points))
            if recent:
                dot = tuple(np.clip(np.mean(recent, axis=0), [0, 0], [cfg.width_px, cfg.height_px]))
            trial = task.current or task.trials[-1]
            if frame is not None:
                n_frames += 1
                rec.add(frame.timestamp_ms, TEST, trial.index, trial.center,
                        result.points if result else None, result.transform if result else None,
                        feats if result else None, result.blendshapes if result else None)
            if feedback is None and task.current is not None:
                event = task.update(t, dot)
                if event is not None:
                    feedback = (t + FEEDBACK_S, event)
            elif feedback is not None and t >= feedback[0]:
                feedback = None
                if len(task.trials) < task.cfg.n_trials:
                    task.start(t)

            img = canvas.blank()
            trial = task.current or task.trials[-1]
            if feedback is not None:
                canvas.box(img, trial.center, trial.half, GREEN if feedback[1] == "hit" else RED, -1)
            else:
                frac = task.dwell_fraction(t)
                if frac > 0:
                    canvas.box(img, trial.center, trial.half * frac, YELLOW, -1)
                canvas.box(img, trial.center, trial.half, WHITE, 3)
            if dot is not None:
                canvas.marker(img, dot)
            hits = sum(1 for tr in task.trials if tr.hit)
            canvas.text(img, f"box {len(task.trials)}/{task.cfg.n_trials}   hits {hits}   "
                             f"size {2 * trial.half * cfg.cm_per_px:.1f} cm   q quits", y=canvas.size[1] - 30)
            if canvas.show(img) in (27, ord("q")):
                aborted = True
                break

        s = task.summary()
        lines = summary_lines(s) + ["", "SPACE or q to close"]
        while True:
            img = canvas.blank()
            y = canvas.size[1] // 2 - 30 * len(lines)
            for line in ["Target practice" + (" (stopped early)" if aborted else "")] + lines:
                canvas.text(img, line, y=y)
                y += 50
            key = canvas.show(img)
            if key in (27, 32, ord("q")):
                break
        rec.meta["blendshape_names"] = list(tracker.blendshape_names)

    print("\n".join(summary_lines(s)))
    if args.no_save or not s.get("n_trials"):
        return 0
    lighting = summarize_lighting(light_log)
    if lighting is not None:
        rec.meta["lighting"] = lighting
    rec.meta["trials"] = [{"center": tr.center, "half": tr.half, "hit": tr.hit,
                           "time_s": tr.time_to_hit, "mean_offset_pt": tr.mean_offset}
                          for tr in task.trials if tr.hit is not None]
    rec.meta["summary"] = s
    path = rec.save(TASKS_DIR)
    append_rows([{"time": now(), "mode": "target_task", "confirmatory": 0,
                  "config": "pool" if args.pool else "latest", "session": path.name,
                  "n_sessions": len(live.sessions), "n_targets": s["n_trials"],
                  "err_cm_mean": s["error_cm"], "err_deg_mean": s["error_deg"],
                  "note": f"hits {s['n_hits']}/{s['n_trials']}; box {s['box_cm']:.1f} cm; "
                          f"median time {s['median_time_s']:.1f} s; bias cm ({s['bias_cm'][0]:+.1f}, "
                          f"{s['bias_cm'][1]:+.1f}); {live.label}"}])
    print(f"saved {path.name} to data/tasks/ (local only) and logged to data/results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
