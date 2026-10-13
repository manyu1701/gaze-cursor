"""M2 step 6: tune the gaze-dot smoothing on spliced fixation streams (offline; rule in PREREGISTRATIONS.md).

Development = clean v2 class-F sessions started before the round-3 freeze; held-out = after it.
Each filter runs over every session's stream (the Run model: fresh gaze_a10 on the 9 quick dots).
Rule (written before tuning): lowest development jitter with median lag50 <= 100 ms and median
lag90 <= 250 ms; snap-on-saccade variants of the 3 best are added; the winner is reported once on
the held-out sessions next to today's moving average (n = 5) and raw.

  uv run scripts/tune_smoothing.py [--no-log]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gaze_cursor.eval.prereg2 import lighting_class
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.eval.runner import common_screen
from gaze_cursor.eval.session import clean_sessions
from gaze_cursor.eval.temporal import fixation_metrics, run_filter, session_stream, summarize
from gaze_cursor.filters import make_filter

FREEZE = "2026-10-13T13:12:00"
LAG50_MAX, LAG90_MAX = 100.0, 250.0
DEBUG = Path(__file__).resolve().parents[1] / "debug"


def name(spec: dict) -> str:
    if spec["kind"] == "raw":
        return "raw"
    if spec["kind"] == "ma":
        return f"moving average n={spec['n']}"
    s = f"one-euro mc={spec['min_cutoff']:g} b={spec['beta']:g}"
    return s + (f" snap {spec['snap_deg']:g} deg" if "snap_deg" in spec else "")


def build(spec: dict, px_per_deg: float):
    spec = dict(spec)
    if "snap_deg" in spec:
        spec["snap_px"] = spec.pop("snap_deg") * px_per_deg
    return make_filter(spec)


def evaluate(spec, streams, screen, px_per_deg) -> dict:
    return summarize([fixation_metrics(s, run_filter(s, build(spec, px_per_deg)), screen) for s in streams])


def plot(results: list[tuple[dict, dict]], chosen: dict, path: Path) -> None:
    w, h = 900, 600
    img = np.full((h, w, 3), 30, np.uint8)
    xs = [r["lag90"] for _, r in results if np.isfinite(r["lag90"])]
    ys = [r["jitter"] for _, r in results]
    xmax, ymax = max(xs + [LAG90_MAX]) * 1.1, max(ys) * 1.1
    def pt(r):
        return int(60 + (w - 90) * min(r["lag90"], xmax) / xmax), int(h - 50 - (h - 90) * r["jitter"] / ymax)
    x0 = pt({"lag90": LAG90_MAX, "jitter": 0})[0]
    cv2.line(img, (x0, 20), (x0, h - 50), (70, 70, 70), 1)
    for spec, r in results:
        c = (90, 200, 90) if spec is chosen else (230, 140, 60) if spec["kind"] == "oneeuro" else (200, 200, 200)
        cv2.circle(img, pt(r), 6 if spec is chosen else 4, c, -1, cv2.LINE_AA)
    cv2.putText(img, "x: median lag90 (ms), y: jitter (deg); grey line = lag90 limit; green = chosen",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-log", dest="log", action="store_false")
    args = ap.parse_args()
    sessions = [s for s in clean_sessions() if s.layout == "v2" and lighting_class(s.meta) == "F"]
    screen = common_screen(sessions)
    px_per_deg = 1.0 / (screen.cm_per_px / (2 * screen.distance_cm * np.tan(np.radians(0.5))))
    dev = [session_stream(s, screen) for s in sessions if str(s.meta.get("started", "")) < FREEZE]
    held = [session_stream(s, screen) for s in sessions if str(s.meta.get("started", "")) >= FREEZE]
    print(f"Smoothing (M2 step 6): {len(dev)} development and {len(held)} held-out sessions; "
          f"{px_per_deg:.0f} px per deg")
    grid = [{"kind": "raw"}] + [{"kind": "ma", "n": n} for n in (3, 5, 9)]
    grid += [{"kind": "oneeuro", "min_cutoff": mc, "beta": b, "d_cutoff": 1.0}
             for mc in (0.05, 0.1, 0.3, 1.0, 3.0) for b in (0.0, 1e-3, 3e-3, 1e-2, 3e-2)]
    results = [(spec, evaluate(spec, dev, screen, px_per_deg)) for spec in grid]
    ok = lambda r: r["lag50"] <= LAG50_MAX and r["lag90"] <= LAG90_MAX
    best3 = sorted([sr for sr in results if ok(sr[1]) and sr[0]["kind"] == "oneeuro"], key=lambda sr: sr[1]["jitter"])[:3]
    for spec, _ in best3:
        for snap in (3.0, 5.0):
            v = {**spec, "snap_deg": snap, "snap_frames": 2}
            results.append((v, evaluate(v, dev, screen, px_per_deg)))
    feasible = [sr for sr in results if ok(sr[1])]
    chosen = min(feasible, key=lambda sr: sr[1]["jitter"])[0]
    print(f"\n{'development':42s} {'jitter deg':>10s} {'95% CI':>15s} {'lag50':>6s} {'lag90':>6s} {'reach90':>8s}"
          f" {'oversh':>7s} {'err':>6s}")
    for spec, r in sorted(results, key=lambda sr: sr[1]["jitter"]):
        mark = " <- chosen" if spec is chosen else ("" if ok(r) else "  (too slow)")
        print(f"{name(spec):42s} {r['jitter']:10.2f} [{r['jitter_ci'][0]:5.2f}, {r['jitter_ci'][1]:5.2f}]"
              f" {r['lag50']:6.0f} {r['lag90']:6.0f} {r['reached90'] * 100:7.0f}% {r['overshoot']:7.2f} {r['err']:6.2f}{mark}")
    print(f"\nRule: lowest jitter with median lag50 <= {LAG50_MAX:.0f} ms and lag90 <= {LAG90_MAX:.0f} ms -> {name(chosen)}")
    print(f"\n{'held-out (reported once)':42s} {'jitter deg':>10s} {'95% CI':>15s} {'lag50':>6s} {'lag90':>6s} {'reach90':>8s}"
          f" {'oversh':>7s} {'err':>6s}")
    rows = []
    for spec in ({"kind": "raw"}, {"kind": "ma", "n": 5}, chosen):
        r = evaluate(spec, held, screen, px_per_deg)
        label = name(spec) + (" (today)" if spec.get("n") == 5 else " (chosen)" if spec is chosen else "")
        print(f"{label:42s} {r['jitter']:10.2f} [{r['jitter_ci'][0]:5.2f}, {r['jitter_ci'][1]:5.2f}]"
              f" {r['lag50']:6.0f} {r['lag90']:6.0f} {r['reached90'] * 100:7.0f}% {r['overshoot']:7.2f} {r['err']:6.2f}")
        rows.append({"time": now(), "mode": "smoothing_heldout", "confirmatory": 0, "config": label,
                     "n_sessions": len(held), "n_targets": r["n_fix"], "jitter_deg": r["jitter"],
                     "err_deg_mean": r["err"], "ci_lo": r["jitter_ci"][0], "ci_hi": r["jitter_ci"][1],
                     "note": f"lag50 {r['lag50']:.0f} ms, lag90 {r['lag90']:.0f} ms, reached90 {r['reached90']:.2f}, "
                             f"overshoot {r['overshoot']:.2f} deg; CI is on jitter"})
    plot(results, chosen, DEBUG / "smoothing_tradeoff.png")
    print("\nplot: debug/smoothing_tradeoff.png")
    print("chosen spec:", {k: v for k, v in chosen.items()})
    if args.log:
        append_rows(rows)
        print(f"logged {len(rows)} held-out rows to data/results.csv (confirmatory=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
