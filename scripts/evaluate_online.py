"""Round 4: does learning from your clicks beat the frozen quick calibration? (PREREGISTRATIONS.md, "Round 4")

  uv run scripts/evaluate_online.py --develop [cursor_*.npz ...]   # exploratory replay (default: all logs)
  uv run scripts/evaluate_online.py --preregistered                # the confirmatory test: run ONCE
  uv run scripts/evaluate_online.py --status                       # session intake: data rule only, no errors

Each Cursor session is replayed click by click: the frozen calibration model and the click-learning
model both predict the click before the learning model may update on it. Error = distance between the
prediction (from the gaze 400-100 ms before the click) and the click, in degrees.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gaze_cursor.eval import online_eval as oe
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.sinks.quick_view import QUICK_DIR


def describe(rep: oe.SessionReplay) -> str:
    ev = rep.evaluated
    if not ev:
        return f"{rep.path.name}: {rep.minutes:.1f} min, no usable clicks"
    ef = np.array([c.err_frozen_deg for c in ev])
    eo = np.array([c.err_online_deg for c in ev])
    return (f"{rep.path.name}: {rep.minutes:.1f} min, {len(rep.clicks)} labelled clicks, {len(ev)} evaluated, "
            f"{sum(c.updated for c in rep.clicks)} learned from; frozen {ef.mean():.2f} (median {np.median(ef):.2f}), "
            f"learning {eo.mean():.2f} (median {np.median(eo):.2f}) deg; d = {(ef - eo).mean():+.2f}; "
            f"drift alarms {rep.cusum_alarms}; integrity {max(rep.integrity_px.values(), default=0):.2g} px; "
            f"light {rep.lighting}" + (f"; EXCLUDED {rep.excluded_segments}" if rep.excluded_segments else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--develop", nargs="*", type=Path, help="exploratory replay of these logs (default: all)")
    g.add_argument("--preregistered", action="store_true", help="the frozen round-4 test (run once)")
    g.add_argument("--status", action="store_true",
                   help="which logs count for round 4 (minutes, click counts); prints no errors, logs nothing")
    ap.add_argument("--no-log", dest="log", action="store_false")
    args = ap.parse_args()
    logs = sorted(QUICK_DIR.glob("cursor_*.npz"))
    if args.status:
        n = 0
        for p in logs:
            rep = oe.replay(p, QUICK_DIR)
            n += oe.qualifies(rep, oe.FREEZE_AFTER or "9999") is None
            print(oe.status_line(rep, oe.FREEZE_AFTER or "9999"))
        print(f"Round 4: {min(n, oe.N_SESSIONS)} of {oe.N_SESSIONS} qualifying sessions so far.")
        return 0
    if args.develop is not None:
        for p in args.develop or logs:
            print("EXPLORATORY  " + describe(oe.replay(Path(p), QUICK_DIR)))
        return 0
    if oe.FREEZE_AFTER is None:
        print("Refusing: round 4 is not frozen yet (FREEZE_AFTER in eval/online_eval.py).", file=sys.stderr)
        return 2
    reps, why = [], []
    for p in logs:
        rep = oe.replay(p, QUICK_DIR)
        reason = oe.qualifies(rep, oe.FREEZE_AFTER)
        if reason:
            why.append(f"{p.name}: {reason}")
        elif len(reps) < oe.N_SESSIONS:
            reps.append(rep)
    for w in why:
        print("not counted: " + w)
    if len(reps) < oe.N_SESSIONS:
        print(f"Refusing: {len(reps)} of {oe.N_SESSIONS} qualifying sessions so far.", file=sys.stderr)
        return 2
    for r in reps:
        print("  " + describe(r))
    res = oe.decide(reps)
    s = res["stats"]
    print(f"\nd = err(frozen) - err(learning): mean {s['mean']:+.2f} deg over {s['n']} clicks, "
          f"95% CI [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] (sessions then clicks [{s['ci_two_level'][0]:+.2f}, "
          f"{s['ci_two_level'][1]:+.2f}]); per session " + ", ".join(f"{m:+.2f}" for m in s["per_session"]))
    for k, ok in res["criteria"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    verdict = "SUPPORTED" if res["supported"] else "NOT SUPPORTED"
    print(f"H6 (click learning) is {verdict}. No re-tuning on these sessions.")
    if args.log:
        append_rows([{"time": now(), "mode": "h6_click_learning", "confirmatory": 1,
                      "config": "frozen gaze_a10 - OnlineRidge(prior 9, forget 0.98, gate 6)",
                      "n_sessions": len(reps), "n_targets": s["n"], "err_deg_mean": s["mean"],
                      "ci_lo": s["ci"][0], "ci_hi": s["ci"][1],
                      "note": f"preregistered round 4 H6 {verdict}; per session " +
                              ", ".join(f"{m:+.2f}" for m in s["per_session"])}])
        print("logged to data/results.csv (confirmatory=1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
