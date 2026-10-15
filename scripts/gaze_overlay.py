"""Live gaze dot. By default it starts with a 9-dot quick calibration where you sit (~15 s), then
draws where you're looking; R recalibrates at any time (e.g. after you move). A live panel shows
fps, latency, tracking, jitter and how far your head moved since calibrating.

Live tests: T targets (12 boxes of 6 and 4 deg; hold the dot in each for 0.5 s), A accuracy (the
dot is hidden; look at each of 12 targets for 1.5 s: pure accuracy, no chasing), P path (follow a
gliding target for ~30 s). Each ends with a result screen; every frame is saved to data/quick/
and the summary is logged to data/results.csv, so steadiness, snappiness and accuracy can be
analysed offline.

Usage:
  uv run scripts/gaze_overlay.py                    # quick 9-dot calibration, then the dot
  uv run scripts/gaze_overlay.py --latest           # use the latest full calibration instead
  uv run scripts/gaze_overlay.py session.npz [--model ridge|poly2] [--features gaze|...]
  uv run scripts/gaze_overlay.py --latest --pool    # all calibrations in the latest one's lighting

Keys: SPACE start, L latest full calibration (start screen), R recalibrate, T targets test,
A accuracy test, P path test, ESC ends a test, q quits.
Why quick by default: the pre-registered round-3 test (PREREGISTRATIONS.md) found that moving your head
costs ~2 deg with a fixed calibration and that a fresh 9-dot calibration recovers most of it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gaze_cursor.calibration import MODELS
from gaze_cursor.capture import Camera
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import latest_session, load_session
from gaze_cursor.features import FEATURE_SETS, assemble, extract_features
from gaze_cursor.idle import IdleTimer
from gaze_cursor.filters import RUN_FILTER, make_filter
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.landmarks import FaceTracker
from gaze_cursor.live import fit_live_model
from gaze_cursor.pool import POOL_CONFIG
from gaze_cursor.livetest import (
    ACC_SHOW_MS, ACC_SIZE_DEG, BOX_SIZES_DEG, TIMEOUT_MS, WITHIN_DEG, BoxTrial, LiveStats, PathTarget,
    accuracy_result, head_moved, head_shift, make_boxes, pursuit_summary, summarize_accuracy, summarize_boxes,
    to_deg)
from gaze_cursor.posture import Reference, posture_metrics
from gaze_cursor.quick import QUICK_CONFIG, QUICK_DOTS
from gaze_cursor.sinks.livetest_view import (
    AMBER as P_AMBER, GREEN, GREY as P_GREY, RED, WHITE, draw_box, draw_gaze, draw_panel, draw_path, draw_ring,
    result_screen)
from gaze_cursor.sinks.overlay import FullscreenCanvas
from gaze_cursor.sinks.posture_view import run_posture_check
from gaze_cursor.sinks import tags as condition_tags
from gaze_cursor.sinks.quick_view import QUICK_DIR, QuickFit, calibrate_here, set_tags

OOD_Z = 4.0  # warn when a live feature is this many calibration std-devs from its mean
STAGES = ("camera read", "landmarks", "features + predict + filter", "draw + show")
TEST_KEYS = {ord("t"): "targets", ord("T"): "targets", ord("a"): "accuracy", ord("A"): "accuracy",
             ord("p"): "path", ord("P"): "path"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", type=Path, help="use this full calibration instead of a quick one")
    ap.add_argument("--latest", action="store_true", help="use the latest full calibration instead of a quick one")
    ap.add_argument("--model", choices=MODELS, default="ridge", help="(full calibration only)")
    ap.add_argument("--features", choices=FEATURE_SETS, default="gaze",
                    help="(full calibration only) default 'gaze' (iris only): raw head pose made the dot slam "
                         "to the screen edge as soon as posture changed from calibration")
    ap.add_argument("--pool", action="store_true",
                    help=f"(full calibration only) train {POOL_CONFIG} on every clean calibration recorded in the "
                         "same lighting as the latest (or given) one, instead of that one alone (experimental)")
    ap.add_argument("--no-check", action="store_true",
                    help="(full calibration only) skip the position check and the live posture hints")
    condition_tags.add_arguments(ap)
    args = ap.parse_args()
    cfg = ScreenConfig.from_system()
    quick = args.session is None and not args.latest
    # resolved before the fullscreen canvas hides the terminal: pressing R later records a quick calibration
    # even when the Run started from a saved full calibration
    set_tags(condition_tags.resolve(args))       # refuses to record without a person id

    with Camera() as cam, FaceTracker() as tracker, FullscreenCanvas(cfg) as canvas:
        features, ref = "gaze", None
        if quick:
            out = calibrate_here(cam, tracker, canvas, cfg)
            if out is None:
                return
            if isinstance(out, QuickFit):
                model, x_tr, ref = out.model, out.x, Reference.from_current(out.posture)
            else:
                quick = False
        if not quick:
            try:
                session = load_session(args.session or latest_session())
            except FileNotFoundError:
                print("No full calibration yet: double-click Calibrate.command, or use the quick calibration.")
                return
            if (cfg.width_px, cfg.height_px) != (session.meta["screen"]["width_px"], session.meta["screen"]["height_px"]):
                print("warning: screen size differs from the session's; recalibrate for accurate results")
            live = fit_live_model(session, cfg, pool=args.pool, feature_set=args.features, kind=args.model)
            model, x_tr, features = live.model, live.x_train, live.feature_set
            print("model: " + live.label)
            # The model is only as good as the postures it was trained in: compare against those sessions.
            ref = None if args.no_check else Reference.from_sessions(live.sessions)
            if ref is not None:
                check = run_posture_check(cam, tracker, canvas, ref,
                                          title="Sit as you did when you calibrated this model",
                                          ref_lighting=session.meta.get("lighting"),
                                          override_label="run anyway (accuracy may drop)")
                if not check.proceed:
                    return
        names = tuple(tracker.blendshape_names)
        smooth = make_filter(RUN_FILTER)   # chosen in M2 step 6 (PREREGISTRATIONS.md)
        calib = out.path.name if quick and isinstance(out, QuickFit) else (args.session or latest_session()).name
        dot = raw = None
        mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-9
        worst_z = 0.0
        stats = LiveStats()
        test: LiveTest | None = None
        timing: list[list[float]] = []   # per frame: monotonic ms at the 5 stage boundaries
        idle = IdleTimer(time.monotonic())
        while True:
            t0 = time.monotonic() * 1000
            frame = cam.read()
            t1 = time.monotonic() * 1000
            # Asleep (no face for 30 s): the landmarker runs on 1 frame in 10. After 5 min: quit.
            result = tracker.process(frame) if frame is not None and idle.process_this_frame(t1 / 1000) else None
            idle.update(t1 / 1000, result is not None, busy=test is not None)   # never sleep during a live test
            nap = idle.message(t1 / 1000)
            if idle.should_quit(t1 / 1000):
                print("No face for 5 min: quitting to save battery.")
                break
            t2 = time.monotonic() * 1000
            metrics, x, raw = None, None, None
            if frame is not None and result is not None:
                h, w = frame.image.shape[:2]
                metrics = posture_metrics(result.points, result.transform, (w, h))
                feats = extract_features(result.points, result.transform, (w, h))
                if feats is not None:
                    x = assemble(feats, result.blendshapes, names, features)
                    raw = model.predict(x)[0]
                    dot = smooth.update(frame.timestamp_ms, raw)
                    worst_z = float(np.max(np.abs((x - mu) / sd)))
            shift = head_shift(metrics, ref.center) if metrics is not None and ref is not None else None
            t3 = time.monotonic() * 1000
            if test is not None and frame is not None:
                test.update(frame.timestamp_ms, raw, dot if raw is not None else None, shift, x)
            img = canvas.blank()
            # Panels first, so a target box under a panel is still drawn on top of it.
            draw_panel(canvas, img, live_lines(stats, cfg, shift, worst_z) + ([(nap, P_AMBER)] if nap else []), "left")
            if test is not None:
                draw_panel(canvas, img, test.lines(), "right")
                test.draw(canvas, img, time.monotonic() * 1000)
            if test is None or test.kind != "accuracy":      # the accuracy test hides the dot (no chasing)
                draw_gaze(canvas, img, dot, raw)
            if test is not None:
                canvas.text(img, "ESC: end the test", y=canvas.size[1] - 30)
            else:
                canvas.text(img, "T: targets   A: accuracy (dot hidden)   P: path   R: recalibrate   q: quit",
                            y=canvas.size[1] - 30)
            key = canvas.show(img)
            t4 = time.monotonic() * 1000
            timing.append([t0, t1, t2, t3, t4])
            stats.add(t4, t4 - t1, raw is not None, dot)
            if test is not None and (test.done or key == 27):
                key = finish_test(cam, canvas, test, calib, cfg, aborted=not test.done)
                test = None
                timing.append([np.nan] * 5)
            if key == ord("q") or (key == 27 and test is None):
                break
            if test is None and key in TEST_KEYS:
                test = LiveTest(TEST_KEYS[key], cfg, time.monotonic() * 1000)
            if test is None and key in (ord("r"), ord("R")):
                out = calibrate_here(cam, tracker, canvas, cfg)
                if out is None:
                    break
                if isinstance(out, QuickFit):
                    model, x_tr, calib = out.model, out.x, out.path.name
                    features, ref = "gaze", Reference.from_current(out.posture)
                    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-9
                    smooth.reset(); dot = None; worst_z = 0.0
                timing.append([np.nan] * 5)   # the recalibration is not part of the live loop
    report_latency(np.array(timing))


def live_lines(stats: LiveStats, cfg: ScreenConfig, shift, worst_z: float) -> list[tuple[str, tuple]]:
    """The live metrics panel."""
    lines = [(f"{stats.fps():4.1f} fps    frame -> screen {stats.latency_ms():4.1f} ms", WHITE),
             (f"face tracked {stats.tracked() * 100:3.0f}%    jitter {stats.jitter_deg(cfg):4.2f} deg (last 0.5 s)",
              WHITE if stats.tracked() > 0.9 else P_AMBER)]
    if shift is not None:
        ratio, dx, dy = shift
        moved = head_moved(ratio, dx, dy)
        lines.append((f"head vs calibration: distance x{ratio:.2f}, {abs(dx):.1f} cm {'left' if dx > 0 else 'right'},"
                      f" {abs(dy):.1f} cm {'lower' if dy > 0 else 'higher'}", P_AMBER if moved else P_GREY))
        if moved:
            lines.append(("you moved since calibrating: accuracy drops; press R to recalibrate here", P_AMBER))
    if worst_z > OOD_Z:
        lines.append((f"eye input {worst_z:.0f} sd outside the calibration: press R", P_AMBER))
    lines.append((f"model {QUICK_CONFIG} on {QUICK_DOTS} dots, filter One-Euro", P_GREY))
    return lines


class LiveTest:
    """One running targets or path test; collects every frame for saving and scoring."""

    def __init__(self, kind: str, cfg: ScreenConfig, start_ms: float) -> None:
        self.kind, self.cfg, self.start_ms = kind, cfg, start_ms
        self.rows: list[list[float]] = []    # t, raw x y, dot x y, target x y, trial, head ratio dx dy
        self.feats: list[np.ndarray] = []
        self.done = False
        if kind in ("targets", "accuracy"):
            self.seed = int(start_ms) % 2**31
            sizes = BOX_SIZES_DEG if kind == "targets" else (ACC_SIZE_DEG,)
            self.boxes = make_boxes(cfg, sizes_deg=sizes, seed=self.seed)
            self.trials = [BoxTrial(self.boxes[0], start_ms)]
        else:
            self.path = PathTarget(cfg)

    @property
    def trial(self) -> BoxTrial:
        return self.trials[-1]

    def target_at(self, t_ms: float) -> np.ndarray:
        if self.kind in ("targets", "accuracy"):
            return np.array(self.trial.box.center)
        return self.path.at((t_ms - self.start_ms) / 1000)

    def update(self, t_ms: float, raw, dot, shift, x) -> None:
        if self.done or t_ms < self.start_ms:
            return
        tgt = self.target_at(t_ms)
        nan2 = [np.nan, np.nan]
        self.rows.append([t_ms, *(raw if raw is not None else nan2), *(dot if dot is not None else nan2), *tgt,
                          len(self.trials) - 1 if self.kind != "path" else 0, *(shift or [np.nan] * 3)])
        self.feats.append(x if x is not None else np.full(4, np.nan))
        if self.kind == "accuracy":
            if dot is not None:
                self.trial.samples.append((t_ms, float(dot[0]), float(dot[1])))
            if t_ms - self.trial.start_ms >= ACC_SHOW_MS:
                self.trial.status = "shown"
                if len(self.trials) == len(self.boxes):
                    self.done = True
                else:
                    self.trials.append(BoxTrial(self.boxes[len(self.trials)], t_ms))
        elif self.kind == "targets":
            self.trial.update(t_ms, dot)
            if self.trial.done:
                if len(self.trials) == len(self.boxes):
                    self.done = True
                else:
                    self.trials.append(BoxTrial(self.boxes[len(self.trials)], t_ms))
        elif (t_ms - self.start_ms) / 1000 >= self.path.duration_s:
            self.done = True

    def draw(self, canvas: FullscreenCanvas, img: np.ndarray, now_ms: float) -> None:
        if self.kind == "accuracy":
            draw_ring(canvas, img, self.trial.box.center, (now_ms - self.trial.start_ms) / ACC_SHOW_MS)
        elif self.kind == "targets":
            tr = self.trial
            draw_box(canvas, img, tr.box, tr.dwell_fraction(now_ms), tr.status == "inside")
        else:
            t_s = (now_ms - self.start_ms) / 1000
            close = bool(self.rows) and np.isfinite(self.rows[-1][3]) and \
                to_deg(np.hypot(self.rows[-1][3] - self.rows[-1][5], self.rows[-1][4] - self.rows[-1][6]), self.cfg) <= WITHIN_DEG
            draw_path(canvas, img, self.path, t_s, close)

    def _acc_results(self) -> list[dict]:
        done = self.trials if self.done else self.trials[:-1]
        return [accuracy_result(t.samples, t.box.center, t.start_ms, self.cfg) for t in done]

    def lines(self) -> list[tuple[str, tuple]]:
        if self.kind == "accuracy":
            return [(f"Accuracy test: target {len(self.trials)}/{len(self.boxes)}", WHITE),
                    ("look at the centre of each ring; the dot is hidden on purpose", P_GREY)]
        if self.kind == "targets":
            res = [t.result(self.cfg) for t in self.trials if t.done]
            hits = sum(r["hit"] for r in res)
            out = [(f"Targets test: box {len(self.trials)}/{len(self.boxes)}   hits {hits}/{len(res)}", WHITE),
                   (f"hold the dot in the box for 0.5 s (max {TIMEOUT_MS / 1000:.0f} s)", P_GREY)]
            if res:
                r = res[-1]
                out.append((f"last: {'hit in %.0f ms' % r['hit_ms'] if r['hit'] else 'missed'}, "
                            f"off by {r['error_deg']:.1f} deg", GREEN if r["hit"] else RED))
            return out
        a = np.array(self.rows) if self.rows else np.zeros((0, 11))
        t = (a[:, 0] - self.start_ms) / 1000 if len(a) else np.zeros(0)
        ok = (t >= 1.0) & np.isfinite(a[:, 3]) if len(a) else np.zeros(0, bool)
        err = to_deg(np.hypot(a[ok, 3] - a[ok, 5], a[ok, 4] - a[ok, 6]), self.cfg) if ok.any() else np.zeros(0)
        out = [(f"Path test: {t[-1] if len(t) else 0:4.1f} / {self.path.duration_s:.0f} s   follow the circle", WHITE)]
        if len(err):
            out.append((f"distance now {err[-1]:.1f} deg   mean {err.mean():.1f} deg   within 3 deg "
                        f"{(err <= WITHIN_DEG).mean() * 100:.0f}%", WHITE))
        return out

    def summary(self) -> tuple[str, list[tuple[str, tuple]], dict, list[dict] | None]:
        a = np.array(self.rows) if self.rows else np.zeros((0, 11))
        head = a[:, 8:11] if len(a) else np.zeros((0, 3))
        moved = float(np.mean([head_moved(*h) for h in head if np.all(np.isfinite(h))])) if len(head) else float("nan")
        if self.kind == "accuracy":
            res = self._acc_results()
            s = summarize_accuracy(res)
            if not s.get("n"):
                return "Accuracy test", [("no targets scored", RED)], s, res
            dx, dy = s["offset_deg"]
            lines = [(f"accurate: median error {s['median_deg']:.1f} deg (mean {s['error_deg']:.1f}) over {s['n']} targets;"
                      f" within 2.5 deg on {s['within_2_5'] * 100:.0f}%", WHITE),
                     (f"overall shift {abs(dx):.1f} deg {'right' if dx > 0 else 'left'}, {abs(dy):.1f} deg "
                      f"{'low' if dy > 0 else 'high'}   (vertical by screen third: top {s['dy_top_deg']:+.1f}, "
                      f"middle {s['dy_middle_deg']:+.1f}, bottom {s['dy_bottom_deg']:+.1f}; + = low)", WHITE),
                     (f"steady: jitter {s['jitter_deg']:.2f} deg while looking at a target", WHITE)]
            s["head_moved_frac"] = moved
            if np.isfinite(moved) and moved > 0.1:
                lines.append((f"your head was away from the calibration position {moved * 100:.0f}% of the time", P_AMBER))
            return "Accuracy test (dot hidden)", lines, s, [{**r, "size_deg": ACC_SIZE_DEG, "hit": r["error_deg"] <= 2.5}
                                                            for r in res]
        if self.kind == "targets":
            res = [t.result(self.cfg) for t in self.trials if t.done]
            s = summarize_boxes(res)
            if not s.get("n"):
                return "Targets test", [("no boxes finished", RED)], s, res
            dx, dy = s["offset_deg"]
            sizes = ", ".join(f"{z:g} deg boxes {s.get(f'hit_rate_{z:g}deg', float('nan')) * 100:.0f}%" for z in BOX_SIZES_DEG)
            lines = [(f"hits {s['hit_rate'] * 100:.0f}% of {s['n']} boxes   ({sizes})", WHITE),
                     (f"snappy: first entry {s['median_entry_ms']:.0f} ms, hit {s['median_hit_ms']:.0f} ms (medians)", WHITE),
                     (f"accurate: off by {s['error_deg']:.1f} deg on average; overall shift {abs(dx):.1f} deg "
                      f"{'right' if dx > 0 else 'left'}, {abs(dy):.1f} deg {'low' if dy > 0 else 'high'}", WHITE),
                     (f"steady: jitter {s['jitter_deg']:.2f} deg while holding", WHITE)]
            s["head_moved_frac"] = moved
        else:
            ok = np.isfinite(a[:, 3]) if len(a) else np.zeros(0, bool)
            s = pursuit_summary((a[ok, 0] - self.start_ms) / 1000, a[ok][:, 3:5], self.path, self.cfg) if ok.any() else {"n": 0}
            if s.get("n", 0) < 10:
                return "Path test", [("too few tracked frames", RED)], s, None
            lines = [(f"distance from the target: mean {s['error_deg']:.1f} deg, median {s['median_deg']:.1f} deg", WHITE),
                     (f"within 3 deg {s['within'] * 100:.0f}% of the time", WHITE),
                     (f"snappy: the dot trails the target by about {s['lag_ms']:.0f} ms "
                      f"(error {s['error_at_lag_deg']:.1f} deg after removing the lag)", WHITE)]
            s["head_moved_frac"] = moved
        if np.isfinite(moved) and moved > 0.1:
            lines.append((f"your head was away from the calibration position {moved * 100:.0f}% of the time", P_AMBER))
        return ("Targets test" if self.kind == "targets" else "Path test"), lines, s, \
            ([t.result(self.cfg) for t in self.trials if t.done] if self.kind == "targets" else None)


def finish_test(cam: Camera, canvas: FullscreenCanvas, test: LiveTest, calib: str, cfg: ScreenConfig,
                aborted: bool) -> int:
    """Score, save and log the test, show its result screen; returns the key pressed there."""
    title, lines, summary, boxes = test.summary()
    if aborted:
        title += " (ended early)"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    QUICK_DIR.mkdir(parents=True, exist_ok=True)
    path = QUICK_DIR / f"livetest_{test.kind}_{stamp}.npz"
    meta = {"kind": test.kind, "calibration": calib, "filter": RUN_FILTER, "model": QUICK_CONFIG,
            "screen": cfg.to_dict(), "start_ms": test.start_ms, "aborted": aborted, "summary": summary,
            "boxes": boxes, "seed": getattr(test, "seed", None), "started": stamp,
            "columns": ["t_ms", "raw_x", "raw_y", "dot_x", "dot_y", "target_x", "target_y", "trial",
                        "head_ratio", "head_dx_cm", "head_dy_cm"]}
    np.savez_compressed(path, rows=np.array(test.rows, float), features=np.array(test.feats, float),
                        meta=json.dumps(meta, default=float))
    print(f"{title}: " + "; ".join(t for t, _ in lines) + f"  (saved {path.name})")
    note = "; ".join(t for t, _ in lines)
    append_rows([{"time": now(), "mode": f"live_{test.kind}", "confirmatory": 0, "session": calib,
                  "config": f"{QUICK_CONFIG} + One-Euro", "n_targets": summary.get("n", 0),
                  "err_deg_mean": summary.get("error_deg"), "jitter_deg": summary.get("jitter_deg"),
                  "note": ("ABORTED; " if aborted else "") + note}])
    while True:
        cam.read()                          # keep the camera buffer fresh while the screen waits
        key = canvas.show(result_screen(canvas, title, lines, boxes))
        if key in (32, ord("q"), 27) or key in TEST_KEYS:
            return -1 if key in (32, 27) else key


def report_latency(t: np.ndarray) -> None:
    """Print fps and per-stage times of the live loop, and save them to data/quick/ (record and replay).
    Software only: the camera's own sensor-to-read delay is not measurable here."""
    t = t[~np.isnan(t).any(axis=1)] if len(t) else t
    if len(t) < 10:
        return
    stages = np.diff(t, axis=1)
    loop = np.diff(t[:, 0])
    loop = loop[loop < 1000]
    print(f"live loop: {len(t)} frames, {1000 / np.median(loop):.1f} fps (median frame interval {np.median(loop):.1f} ms)")
    for name, col in zip(STAGES, stages.T):
        print(f"  {name:28s} median {np.median(col):6.1f} ms   p95 {np.percentile(col, 95):6.1f} ms")
    shown = t[:, 4] - t[:, 1]
    print(f"  {'read -> shown':28s} median {np.median(shown):6.1f} ms   p95 {np.percentile(shown, 95):6.1f} ms"
          "   (plus the camera's own delay, not measurable in software)")
    QUICK_DIR.mkdir(parents=True, exist_ok=True)
    path = QUICK_DIR / f"timing_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(path, stage_ms=t, stages=np.array(("start",) + STAGES), filter=str(RUN_FILTER))
    print(f"  saved {path.name}")

if __name__ == "__main__":
    main()
