"""Evaluation harness over all clean sessions.

Modes: within (calibration targets -> held-out targets of the same session), loto (leave one
target out within a session), losos (leave one session out, pooled training on the others).
Errors are per TARGET (mean over its frames), in degrees of visual angle; CIs are bootstrap 95%
over targets. Every run is logged to data/results.csv with its mode.

  uv run scripts/evaluate.py --configs all
  uv run scripts/evaluate.py --configs gaze_a10 gaze_blend_a100 --modes losos
  uv run scripts/evaluate.py --compare gaze_a10 gaze_blend_a100 --per-target
  uv run scripts/evaluate.py --preregistered     # the pre-registered H1 test (see PROGRESS.md)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gaze_cursor.eval.configs import (
    PREREG_CANDIDATE, PREREG_DIAGNOSTICS, PREREG_REFERENCE, REGISTRY, Config, get_configs)
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.eval.runner import build_datasets, common_screen, evaluate
from gaze_cursor.eval.session import clean_sessions, load_session
from gaze_cursor.eval.splits import MODES
from gaze_cursor.eval.stats import (
    TargetStats, bootstrap_ci, paired_comparison, summarize)

MIN_CRITERION_IMPROVEMENT_DEG = 0.25  # pre-registered: (b)
MIN_SESSIONS_BETTER = 2               # pre-registered: (c)
MIN_VALID_FRACTION = 0.8              # pre-registered objective-failure rule
N_PREREG_SESSIONS = 3


def short(name: str) -> str:
    return name.removeprefix("session_").removesuffix(".npz")


def row_for(cfg: Config, mode: str, t: TargetStats, screen, sessions, args, confirmatory: int,
            note: str = "") -> dict:
    s = summarize(t, screen)
    lo, hi = bootstrap_ci(t.err_deg, args.n_boot, args.seed)
    return {"time": now(), "mode": mode, "confirmatory": confirmatory, "config": cfg.name,
            "model": cfg.model, "features_set": cfg.feature_set, "alpha": cfg.alpha,
            "n_sessions": len(sessions), "ci_lo": lo, "ci_hi": hi, "note": note,
            **{k: v for k, v in s.items()}}


def print_table(mode: str, rows: list[tuple[Config, TargetStats, float, dict]]) -> None:
    print(f"\n== {mode} ==")
    print(f"{'config':18s} {'targets':>7s} {'mean deg [95% CI]':>24s} {'median':>7s} {'p90':>6s} "
          f"{'frame-mean':>10s} {'jitter':>7s}")
    for cfg, _t, frame_mean, r in rows:
        print(f"{cfg.name:18s} {r['n_targets']:7d} "
              f"{r['err_deg_mean']:8.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]".ljust(52)
              + f"{r['err_deg_median']:7.2f} {r['err_deg_p90']:6.2f} {frame_mean:10.2f} "
              f"{r['jitter_deg']:7.2f}")


def print_comparison(a: str, b: str, mode: str, c: dict) -> None:
    print(f"\n-- paired [{mode}]: {a} vs {b}  (d = err({a}) - err({b}); positive = {b} better) --")
    print(f"targets {c['n_targets']}: mean d = {c['mean_diff_deg']:+.3f} deg, 95% CI "
          f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] (targets), "
          f"[{c['cluster_ci_lo']:+.3f}, {c['cluster_ci_hi']:+.3f}] (sessions then targets)")
    print(f"effect size d_z = {c['effect_size_dz']:+.2f};  {b} better on {c['targets_new_better']}/"
          f"{c['n_targets']} targets and {c['sessions_new_better']}/{len(c['sessions'])} sessions "
          f"(per-session d: {', '.join(f'{x:+.2f}' for x in c['per_session_diff_deg'])})")


def print_per_target(cfg: Config, mode: str, t: TargetStats, names: tuple[str, ...]) -> None:
    print(f"\n-- per-target bias / spread: {cfg.name} [{mode}] "
          f"(bias = mean prediction - target, screen points) --")
    print(f"{'session':16s} {'tgt':>4s} {'x':>6s} {'y':>6s} {'bias_x':>7s} {'bias_y':>7s} "
          f"{'spread':>7s} {'err deg':>8s} {'frames':>6s}")
    for i in np.lexsort((t.target, t.session)):
        print(f"{short(names[t.session[i]]):16s} {int(t.target[i]) % 10_000:4d} {t.xy[i, 0]:6.0f} "
              f"{t.xy[i, 1]:6.0f} {t.bias_px[i, 0]:+7.0f} {t.bias_px[i, 1]:+7.0f} "
              f"{t.spread_px[i]:7.0f} {t.err_deg[i]:8.2f} {t.n_frames[i]:6d}")


def run_exploratory(args: argparse.Namespace) -> int:
    sessions = [load_session(Path(p)) for p in args.sessions] if args.sessions else clean_sessions()
    if not sessions:
        print("No clean sessions found.", file=sys.stderr)
        return 1
    names = tuple(s.path.name for s in sessions)
    print(f"{len(sessions)} sessions: {', '.join(short(n) for n in names)}  (exploratory)")
    screen = common_screen(sessions)
    wanted = list(args.configs)
    if args.compare:
        wanted = sorted({*wanted, *args.compare}) if wanted != ["all"] else wanted
    configs = get_configs(wanted)
    datasets = build_datasets(sessions, configs)
    log_rows: list[dict] = []
    for mode in args.modes:
        try:
            results = {c.name: evaluate(datasets, c, mode, screen) for c in configs}
        except ValueError as e:
            print(f"\n== {mode} == skipped: {e}")
            continue
        table = []
        for c in configs:
            t, frame_mean = results[c.name]
            r = row_for(c, mode, t, screen, sessions, args, 0)
            table.append((c, t, frame_mean, r)); log_rows.append(r)
        print_table(mode, table)
        if args.compare:
            a, b = args.compare
            comp = paired_comparison(results[a][0], results[b][0], args.n_boot, args.seed)
            print_comparison(a, b, mode, comp)
            log_rows.append({
                "time": now(), "mode": mode, "confirmatory": 0, "config": f"{a} - {b}",
                "n_sessions": len(sessions), "n_targets": comp["n_targets"],
                "err_deg_mean": comp["mean_diff_deg"], "ci_lo": comp["ci_lo"], "ci_hi": comp["ci_hi"],
                "note": f"paired diff deg; d_z={comp['effect_size_dz']:.2f}; "
                        f"{comp['sessions_new_better']}/{len(comp['sessions'])} sessions favor {b}"})
        if args.per_target:
            c = next(iter(configs)) if not args.compare else REGISTRY[args.compare[1]]
            print_per_target(c, mode, results[c.name][0], names)
    if args.log and log_rows:
        append_rows(log_rows)
        print(f"\nlogged {len(log_rows)} rows to data/results.csv")
    return 0


def run_preregistered(args: argparse.Namespace) -> int:
    """The pre-registered H1 analysis (PROGRESS.md). Confirmatory: runs once, no tuning."""
    clean = clean_sessions()
    v2 = [s for s in clean if s.layout == "v2"]
    failed = [s for s in v2 if s.valid_fraction < MIN_VALID_FRACTION]
    for s in failed:
        print(f"excluded {short(s.path.name)}: only {s.valid_fraction:.0%} valid frames "
              f"(< {MIN_VALID_FRACTION:.0%}, pre-registered objective-failure rule)")
    eligible = [s for s in v2 if s not in failed]
    if len(eligible) < N_PREREG_SESSIONS:
        print(f"Refusing to run: the pre-registered analysis needs {N_PREREG_SESSIONS} clean "
              f"layout-v2 sessions, found {len(eligible)}. Record more with "
              f"`uv run scripts/calibrate.py --note \"...\"`.", file=sys.stderr)
        return 2
    held = eligible[:N_PREREG_SESSIONS]
    for s in eligible[N_PREREG_SESSIONS:]:
        print(f"ignoring extra v2 session {short(s.path.name)} (only the earliest "
              f"{N_PREREG_SESSIONS} are held out; extras stay out of the pool)")
    pool = [s for s in clean if s.layout != "v2"]
    sessions = pool + held
    held_idx = list(range(len(pool), len(sessions)))
    screen = common_screen(sessions)
    names = (PREREG_REFERENCE, PREREG_CANDIDATE, *PREREG_DIAGNOSTICS)
    configs = get_configs(list(names))
    datasets = build_datasets(sessions, configs)
    print(f"Pre-registered H1: {PREREG_CANDIDATE} beats {PREREG_REFERENCE} (leave-one-session-out)")
    print(f"held-out v2: {', '.join(short(s.path.name) for s in held)}; "
          f"training pool also includes {len(pool)} older clean session(s)")
    res = {c.name: evaluate(datasets, c, "losos", screen, held_idx) for c in configs}
    table, log_rows = [], []
    for c in configs:
        t, fm = res[c.name]
        r = row_for(c, "losos", t, screen, held, args, 1, "preregistered")
        table.append((c, t, fm, r)); log_rows.append(r)
    print_table("losos (confirmatory)", table)
    ref, new = res[PREREG_REFERENCE][0], res[PREREG_CANDIDATE][0]
    comp = paired_comparison(ref, new, args.n_boot, args.seed)
    print_comparison(PREREG_REFERENCE, PREREG_CANDIDATE, "losos", comp)
    for d in PREREG_DIAGNOSTICS:  # secondary, not part of the criterion
        print_comparison(PREREG_REFERENCE, d, "losos (diagnostic)",
                         paired_comparison(ref, res[d][0], args.n_boot, args.seed))
    crit = {
        "(a) CI lower bound > 0": comp["ci_lo"] > 0,
        f"(b) mean improvement >= {MIN_CRITERION_IMPROVEMENT_DEG} deg":
            comp["mean_diff_deg"] >= MIN_CRITERION_IMPROVEMENT_DEG,
        f"(c) better in >= {MIN_SESSIONS_BETTER} of {N_PREREG_SESSIONS} sessions":
            comp["sessions_new_better"] >= MIN_SESSIONS_BETTER,
    }
    print("\nSuccess criterion (all three required):")
    for k, ok in crit.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    verdict = "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED"
    print(f"\nH1 is {verdict}. No re-tuning on these sessions.")
    log_rows.append({
        "time": now(), "mode": "losos", "confirmatory": 1,
        "config": f"{PREREG_REFERENCE} - {PREREG_CANDIDATE}", "n_sessions": len(held),
        "n_targets": comp["n_targets"], "err_deg_mean": comp["mean_diff_deg"],
        "ci_lo": comp["ci_lo"], "ci_hi": comp["ci_hi"],
        "note": f"preregistered H1 {verdict}; d_z={comp['effect_size_dz']:.2f}; "
                f"{comp['sessions_new_better']}/{N_PREREG_SESSIONS} sessions"})
    if args.log:
        append_rows(log_rows)
        print("logged to data/results.csv (confirmatory=1)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["all"], help=f"names or 'all': {sorted(REGISTRY)}")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--sessions", nargs="+", help="session .npz paths (default: all clean sessions)")
    ap.add_argument("--compare", nargs=2, metavar=("REF", "NEW"),
                    help="paired comparison; d = err(REF) - err(NEW), positive = NEW better")
    ap.add_argument("--per-target", action="store_true", help="bias/spread table per target")
    ap.add_argument("--preregistered", action="store_true", help="run the pre-registered H1 test")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-log", dest="log", action="store_false", help="do not append to results.csv")
    args = ap.parse_args()
    return run_preregistered(args) if args.preregistered else run_exploratory(args)


if __name__ == "__main__":
    raise SystemExit(main())
