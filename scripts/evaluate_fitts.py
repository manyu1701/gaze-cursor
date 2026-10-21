"""Round 5 (H7a head cost, H7b gaze-jump gain): analysis of the jump-and-click task files.
(PREREGISTRATIONS.md, "Round 5" and its amendment 1)

  uv run scripts/evaluate_fitts.py --status                  # session intake: the data rule only, no differences
  uv run scripts/evaluate_fitts.py --develop [fitts_*.npz]   # exploratory numbers; refuses post-freeze files
  uv run scripts/evaluate_fitts.py --preregistered-7         # the confirmatory test: run ONCE (always logged)

The confirmatory run uses the first 5 qualifying sessions (data/tasks/fitts_*.npz started after the freeze
commit), in time order, and refuses until there are 5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gaze_cursor.eval import fitts_eval as fe
from gaze_cursor.eval.results_log import append_rows, now

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "data" / "tasks"


def in_worktree() -> bool:
    """True in a linked git worktree (its .git is a file). The intake gate scans only the main checkout's data, so
    the confirmatory run must read the main checkout's files."""
    return (ROOT / ".git").is_file()


def no_verdict(reason: str) -> int:
    """A confirmatory run that cannot give a verdict still counts as the one run: log it, never repeat silently."""
    print(f"No verdict: {reason}", file=sys.stderr)
    append_rows([{"time": now(), "mode": "h7_jump_task", "confirmatory": 1,
                  "config": "frozen gaze_a10 quick calibration, no click learning",
                  "note": f"preregistered round 5: NO VERDICT: {reason}"}])
    print("logged to data/results.csv (confirmatory=1, no verdict)")
    return 2


def files() -> list[Path]:
    return sorted(p for p in TASKS_DIR.glob("fitts_*.npz") if not p.name.endswith(".faces.npz"))


def load_or_none(p: Path) -> fe.FittsFile | None:
    """A file from an older version of the task (the pilot) cannot be analysed: say so and move on."""
    try:
        return fe.load(p)
    except ValueError as e:
        print(f"{p.name}: skipped ({e})")
        return None


def settings_line(f: fe.FittsFile) -> str:
    """What this run was set to (the schedule row and the frozen settings), so a wrong run is caught at intake."""
    t = f.task
    blocks = t["blocks"]
    amps = sorted({(b["layout"], b["amplitude_px"], b["width_px"]) for b in blocks if not b["practice"]})
    return (f"schedule seat-first {t.get('seat_first')} method-first {t.get('method_first')} moved-side "
            f"{t.get('moved_side')}; practice {'yes' if any(b['practice'] for b in blocks) else 'NO'}; learning "
            f"{t.get('learning')}; layouts {amps}")


def describe(s: fe.SessionStats) -> str:
    r = s.reported
    bias = {k: r[k]["bias_cm"] and [round(x, 1) for x in r[k]["bias_cm"]] for k in ("centre", "moved")}
    spread = {k: r[k]["spread_cm"] and [round(x, 1) for x in r[k]["spread_cm"]] for k in ("centre", "moved")}
    return (f"{s.name}: d_a {s.d_a:+.2f} cm, d_b {s.d_b:+.0f} ms, miss {s.miss_pp:+.1f} pp; aim error centre "
            f"{r['centre']['aim_error_cm_mean']:.1f} (median {r['centre']['aim_error_cm_median']:.1f}, p90 "
            f"{r['centre']['aim_error_cm_p90']:.1f}) / moved {r['moved']['aim_error_cm_mean']:.1f} (median "
            f"{r['moved']['aim_error_cm_median']:.1f}, p90 {r['moved']['aim_error_cm_p90']:.1f}) cm; bias (x, y) cm "
            f"{bias['centre']} / {bias['moved']}, spread {spread['centre']} / {spread['moved']}; refused "
            f"{r['centre']['refusal_rate']:.0%} / {r['moved']['refusal_rate']:.0%}; off display "
            f"{r['centre']['share_off_display']:.0%} / {r['moved']['share_off_display']:.0%}; d_a clamped "
            f"{r['d_a_clamped']:+.2f}; fast hits dropped {r['fast_hits_excluded']}; widths for d_b {r['widths_used_b']}; "
            f"dots head-eyes {r['dots_head_minus_eyes_cm']:+.2f} cm")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true", help="which files count (data rule only; no differences)")
    g.add_argument("--develop", nargs="*", type=Path, help="exploratory numbers for these files (pre-freeze only)")
    g.add_argument("--preregistered-7", dest="prereg", action="store_true", help="the frozen round-5 test (run once)")
    ap.add_argument("--no-log", dest="log", action="store_false", help="not allowed with the confirmatory run")
    args = ap.parse_args(argv)
    if args.prereg and not args.log:
        print("Refusing: a confirmatory run is always logged (no --no-log).", file=sys.stderr)
        return 2
    if args.status:
        loaded = [f for f in map(load_or_none, files()) if f is not None]
        sel = fe.select(loaded)
        counted = {f.path.name for f in sel.counted}
        noted = {n.split(":")[0] for n in sel.notes}
        for f in loaded:
            if f.path.name in counted:
                print(f"{f.path.name}: COUNTS; {settings_line(f)}")
            elif fe.started_iso(f) <= fe.FREEZE_AFTER:
                print(f"{f.path.name}: not counted, recorded before the freeze (development data)")
            elif f.path.name not in noted:
                print(f"{f.path.name}: not used (the test set is complete, or the run stopped)")
        for n in sel.notes:
            print(n)
        if sel.stopped:
            print(sel.stopped)
        print(f"Round 5: {len(sel.counted)} of {fe.N_SESSIONS} qualifying sessions so far.")
        return 0
    if args.develop is not None:
        for p in args.develop or files():
            f = load_or_none(Path(p))
            if f is None:
                continue
            if fe.started_iso(f) > fe.FREEZE_AFTER:
                print(f"{f.path.name}: refused, recorded after the freeze (test data stays untouched until the run)")
                continue
            print("EXPLORATORY  " + describe(fe.session_stats(f)))
        return 0
    if in_worktree():
        print("Refusing: run the confirmatory test from the main checkout, where the intake gate looks.",
              file=sys.stderr)
        return 2
    loaded = [f for f in map(load_or_none, files()) if f is not None]
    sel = fe.select(loaded)
    for n in sel.notes:
        print("not counted: " + n)
    if sel.stopped:
        return no_verdict(sel.stopped)
    if len(sel.counted) < fe.N_SESSIONS:
        print(f"Refusing: {len(sel.counted)} of {fe.N_SESSIONS} qualifying sessions so far.", file=sys.stderr)
        return 2
    stats = [fe.session_stats(f) for f in sel.counted]
    for s in stats:
        print("  " + describe(s))
    try:
        res = fe.decide(stats)
    except ValueError as e:
        return no_verdict(str(e))
    for h, label in (("H7a", "d_a = moved - centre aim error, cm"), ("H7b", "d_b = jump - trackpad movement time, ms")):
        r = res[h]
        print(f"\n{h} ({label}): mean {r['mean']:+.2f}; percentile CI [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]; "
              f"t-interval [{r['t_ci'][0]:+.2f}, {r['t_ci'][1]:+.2f}]; per session "
              + ", ".join(f"{m:+.2f}" for m in r["per_session"]))
        if h == "H7b":
            print(f"  miss-rate difference (jump - trackpad): mean {r['miss_pp_mean']:+.1f} pp; per session "
                  + ", ".join(f"{m:+.1f}" for m in r["miss_pp_per_session"]))
        for k, ok in r["criteria"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
        print(f"{h} is {'SUPPORTED' if r['supported'] else 'NOT SUPPORTED'}. No re-tuning on these sessions.")
    n_used = {"H7a": sum(len(a) + len(b) for s in stats for a, b in s.cells_a),      # jump trials in the kept cells
              "H7b": sum(len(a) + len(b) for s in stats for a, b in s.cells_b)}      # usable hit trials, both arms
    rows = []
    for h in ("H7a", "H7b"):
        r = res[h]
        unit = "cm (moved - centre aim error), in err_cm_mean" if h == "H7a" else "ms (jump - trackpad movement time)"
        extra = f"; miss-rate difference {r['miss_pp_mean']:+.1f} pp" if h == "H7b" else ""
        row = {"time": now(), "mode": f"{h.lower()}_jump_task", "confirmatory": 1,
               "config": "frozen gaze_a10 quick calibration, no click learning", "n_sessions": len(stats),
               "n_targets": n_used[h], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1],
               "note": (f"preregistered round 5 {h} {'SUPPORTED' if r['supported'] else 'NOT SUPPORTED'}; mean "
                        f"{r['mean']:+.2f} {unit}; percentile CI [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]; t-interval "
                        f"[{r['t_ci'][0]:+.2f}, {r['t_ci'][1]:+.2f}]{extra}; per session "
                        + ", ".join(f"{m:+.2f}" for m in r["per_session"]))}
        if h == "H7a":
            row["err_cm_mean"] = r["mean"]
        rows.append(row)
    append_rows(rows)
    print("logged to data/results.csv (confirmatory=1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
