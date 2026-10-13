"""Vertical-bias diagnosis (M2 step 3, EXPLORATORY; the gate is pre-specified in PREREGISTRATIONS.md).

Every clean v2 front-lit (class F) session, two conditions:
  (a) Run today: fresh gaze_a10 on the 9 quick dots -> scored on the other 16 targets;
  (b) full calibration: 16 calibration targets -> 9 held-out.
Per band of screen rows (and columns): signed bias in degrees (vertical: + = predicted too LOW) with
a bootstrap CI over targets, error mean / median / p90, and bias vs within-fixation spread. Writes
one bias-arrow map per condition to debug/ (gitignored) and logs the band rows to results.csv.

  uv run scripts/diagnose_vertical.py [--no-log]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gaze_cursor.eval.configs import REGISTRY
from gaze_cursor.eval.prereg2 import lighting_class
from gaze_cursor.eval.prereg3 import dot_strategy
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.eval.runner import common_screen
from gaze_cursor.eval.session import CALIB, TEST, Session, clean_sessions
from gaze_cursor.eval.stats import TargetStats, concat, per_target
from gaze_cursor.eval.vertical import band_edges, band_report, gate, signed_deg
from gaze_cursor.quick import QUICK_CONFIG, QUICK_DOTS

DEBUG = Path(__file__).resolve().parents[1] / "debug"


def full_calibration(s: Session, i: int, screen) -> TargetStats:
    cfg = REGISTRY[QUICK_CONFIG]
    x_tr, y_tr, _ = s.select(CALIB, cfg.feature_set)
    x_te, y_te, t_te = s.select(TEST, cfg.feature_set)
    m = cfg.make_model(screen).fit(x_tr, y_tr)
    return per_target(m.predict(x_te), y_te, t_te, np.full(len(t_te), i), screen)


def print_bands(title: str, rows: list[dict], axis_name: str) -> None:
    print(f"\n{title}")
    print(f"{'band':13s} {'n':>4s} {f'bias {axis_name} deg':>13s} {'95% CI':>17s} {'err mean':>9s} {'median':>7s}"
          f" {'p90':>6s} {'|bias|':>7s} {'spread':>7s}")
    for r in rows:
        if not r["n"]:
            print(f"{r['band']:13s} {0:4d}")
            continue
        print(f"{r['band']:13s} {r['n']:4d} {r['bias_deg']:+13.2f} [{r['ci'][0]:+6.2f}, {r['ci'][1]:+6.2f}]"
              f" {r['err_mean']:9.2f} {r['err_median']:7.2f} {r['err_p90']:6.2f} {r['abs_bias_deg']:7.2f}"
              f" {r['spread_deg']:7.2f}")


def bias_map(t: TargetStats, screen, path: Path, title: str) -> None:
    """Each target as a dot, with an arrow to its mean prediction; colour = signed vertical bias."""
    s = 0.6
    img = np.full((int(screen.height_px * s), int(screen.width_px * s), 3), 30, np.uint8)
    for y in band_edges(screen, 1):
        cv2.line(img, (0, int(y * s)), (img.shape[1], int(y * s)), (70, 70, 70), 1)
    dy = signed_deg(t.bias_px[:, 1], screen)
    for (x, y), b, d in zip(t.xy, t.bias_px, dy):
        c = (60, 60, 230) if d > 0 else (230, 140, 60)        # red: predicted too low; blue: too high
        k = min(abs(d) / 3.0, 1.0)
        color = tuple(int(70 + (v - 70) * k) for v in c)
        p0 = (int(x * s), int(y * s))
        p1 = (int((x + b[0]) * s), int((y + b[1]) * s))
        cv2.arrowedLine(img, p0, p1, color, 1, cv2.LINE_AA, tipLength=0.15)
        cv2.circle(img, p0, 3, (235, 235, 235), -1, cv2.LINE_AA)
    cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(img, "arrow: target -> mean prediction; red = too low, blue = too high", (10, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-log", dest="log", action="store_false")
    args = ap.parse_args()
    sessions = [s for s in clean_sessions() if s.layout == "v2" and lighting_class(s.meta) == "F"]
    screen = common_screen(sessions)
    cfg = REGISTRY[QUICK_CONFIG]
    print(f"Vertical-bias diagnosis (EXPLORATORY): {len(sessions)} clean v2 front-lit sessions, {cfg.name}")
    conds = {
        f"(a) Run today: fresh on {QUICK_DOTS} quick dots -> other {25 - QUICK_DOTS} targets":
            concat([_relabel(dot_strategy([s], cfg, [], 0, screen, QUICK_DOTS, "fresh"), i)
                    for i, s in enumerate(sessions)]),
        "(b) full calibration: 16 -> 9 held-out":
            concat([full_calibration(s, i, screen) for i, s in enumerate(sessions)]),
    }
    rows_log = []
    for k, (label, t) in enumerate(conds.items()):
        print(f"\n== {label}: {len(t.err_deg)} targets, mean {t.err_deg.mean():.2f}, median "
              f"{np.median(t.err_deg):.2f}, p90 {np.percentile(t.err_deg, 90):.2f} deg ==")
        rows = band_report(t, screen, axis=1)
        print_bands("rows (vertical bias; + = predicted too low)", rows, "y")
        print_bands("columns (horizontal bias; + = predicted too far right)", band_report(t, screen, axis=0), "x")
        if k == 0:
            real, crit = gate(rows)
            print("\nGate (pre-specified, condition (a)):")
            for c, ok in crit.items():
                print(f"  [{'PASS' if ok else 'FAIL'}] {c}")
            print("Vertical bias is " + ("REAL -> step 2 (candidate models)" if real
                                         else "NOT REAL by the gate -> record 'resolved by layout v2'"))
        name = "a_run_quick9" if k == 0 else "b_full16"
        bias_map(t, screen, DEBUG / f"vertical_bias_{name}.png", label)
        print(f"bias map: debug/vertical_bias_{name}.png")
        for r in rows:
            if r["n"]:
                rows_log.append({"time": now(), "mode": f"vertical_{name}", "confirmatory": 0, "config": cfg.name,
                                 "model": cfg.model, "features_set": cfg.feature_set, "alpha": cfg.alpha,
                                 "n_sessions": len(sessions), "n_targets": r["n"], "err_deg_mean": r["err_mean"],
                                 "err_deg_median": r["err_median"], "err_deg_p90": r["err_p90"],
                                 "ci_lo": r["ci"][0], "ci_hi": r["ci"][1],
                                 "note": f"band {r['band']}: signed y bias {r['bias_deg']:+.2f} deg (CI over bias)"})
    if args.log:
        append_rows(rows_log)
        print(f"\nlogged {len(rows_log)} rows to data/results.csv (confirmatory=0)")
    return 0


def _relabel(t: TargetStats, i: int) -> TargetStats:
    """dot_strategy on a single session labels it 0; give each session its own index."""
    return TargetStats(np.full(len(t.session), i), t.target + i * 100_000, *[getattr(t, f) for f in
                       list(TargetStats.__dataclass_fields__)[2:]])


if __name__ == "__main__":
    raise SystemExit(main())
