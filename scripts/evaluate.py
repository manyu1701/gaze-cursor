"""Evaluation harness over all clean sessions.

Modes: within (calibration targets -> held-out targets of the same session), loto (leave one
target out within a session), losos (leave one session out, pooled training on the others).
Errors are per TARGET (mean over its frames), in degrees of visual angle; CIs are bootstrap 95%
over targets. Every run is logged to data/results.csv with its mode.

  uv run scripts/evaluate.py --configs all
  uv run scripts/evaluate.py --configs gaze_a10 gaze_blend_a100 --modes losos
  uv run scripts/evaluate.py --compare gaze_a10 gaze_blend_a100 --per-target
  uv run scripts/evaluate.py --preregistered     # the pre-registered H1 test (see PREREGISTRATIONS.md)
  uv run scripts/evaluate.py --preregistered-2   # the pre-registered round-2 tests H2 and H3
  uv run scripts/evaluate.py --positions         # M2 step 3 development: leave-one-position-out on set A
  uv run scripts/evaluate.py --preregistered-3   # round 3 (H4, H5); refuses until the candidate is frozen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gaze_cursor.eval.configs import (
    PREREG_CANDIDATE, PREREG_DIAGNOSTICS, PREREG_REFERENCE, REGISTRY, Config, get_configs)
from gaze_cursor.eval.prereg2 import run_h2, run_h3, select_sessions
from gaze_cursor.eval import prereg3
from gaze_cursor.eval.prereg3 import (
    DISPLACED, PREREG3_REFERENCE, SLOTS_A, describe, fit_predict, leave_one_position_out, run_h4,
    run_h5, select_set_b, set_a)
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.eval.runner import build_datasets, common_screen, evaluate
from gaze_cursor.eval.session import clean_sessions, load_session
from gaze_cursor.eval.splits import MODES
from gaze_cursor.eval.splits import build_dataset, make_splits
from gaze_cursor.eval.stats import (
    TargetStats, bootstrap_ci, concat, paired_comparison, per_target, summarize)

MIN_CRITERION_IMPROVEMENT_DEG = 0.25  # pre-registered: (b)
MIN_SESSIONS_BETTER = 2               # pre-registered: (c)
MIN_VALID_FRACTION = 0.8              # pre-registered objective-failure rule
N_PREREG_SESSIONS = 3


def short(name: str) -> str:
    return name.removeprefix("session_").removesuffix(".npz")


def session_conditions(s) -> str:
    """One line: layout, note, captured lighting and posture check, if the session recorded them."""
    li, pc = s.meta.get("lighting"), s.meta.get("posture_check") or {}
    light = f"lighting {li['label']} (face {li['face']:.0f}, bg {li['background']:.0f})" if li else "lighting not captured"
    pos = f", position {pc['mode']}" if pc.get("mode") else ""
    note = f', note "{s.meta["note"]}"' if s.meta.get("note") else ""
    return f"{short(s.path.name)}: layout {s.layout}, {light}{pos}{note}"


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
    for s in sessions:
        print("  " + session_conditions(s))
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
    """The pre-registered H1 analysis (PREREGISTRATIONS.md). Confirmatory: runs once, no tuning."""
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
    for s in held:
        print("  " + session_conditions(s))
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


def run_preregistered_2(args: argparse.Namespace) -> int:
    """The pre-registered round-2 analysis (PREREGISTRATIONS.md): H2 and H3. Confirmatory: run once."""
    clean = clean_sessions()
    sel = select_sessions(clean)
    for name, why in sel.excluded:
        print(f"excluded {short(name)}: {why}")
    for note in sel.ignored:
        print(f"ignoring {short(note)}")
    if not sel.complete:
        print("Refusing to run: the round-2 analysis still needs:\n  - " + "\n  - ".join(sel.missing)
              + "\nRecord sessions with `uv run scripts/calibrate.py --note \"...\"`; lighting is captured "
                "automatically and decides which class each session counts as.", file=sys.stderr)
        return 2
    cfg = REGISTRY[PREREG_REFERENCE]                       # gaze_a10, the single pre-registered model
    screen = common_screen([sel.f_twin, sel.b_twin, sel.f2, sel.b2, *sel.dim])
    print(f"Pre-registered round 2, model {cfg.name}")
    print("training twins: " + ", ".join(session_conditions(s) for s in (sel.f_twin, sel.b_twin)))
    for tag, s in (("F2", sel.f2), ("B2", sel.b2), ("D1", sel.dim[0]), ("D2", sel.dim[1])):
        print(f"  {tag}: {session_conditions(s)}")
    h2, h3 = run_h2(sel, cfg, screen), run_h3(sel, cfg, screen)
    print(f"\n== H2: calibrate in the lighting you will use ({h2['n_targets']} test targets) ==")
    print(f"error trained on the same-lighting twin {h2['err_same']:.2f} deg; on the other-lighting session "
          f"{h2['err_other']:.2f} deg")
    print(f"d = err(other) - err(same) = {h2['mean_d']:+.2f} deg, 95% CI [{h2['ci'][0]:+.2f}, {h2['ci'][1]:+.2f}]; "
          f"per test session (F2, B2): {', '.join(f'{x:+.2f}' for x in h2['per_session_d'])}")
    print(f"\n== H3: brightness gate ({h3['n_gated']} gated vs {h3['n_below']} below-gate targets) ==")
    print("within-session mean error: " + ", ".join(f"{short(k)} {v:.2f}" for k, v in h3["session_means"].items()))
    print(f"gap = below - gated = {h3['gap']:+.2f} deg, 95% CI [{h3['ci'][0]:+.2f}, {h3['ci'][1]:+.2f}]")
    rows = []
    for name, res in (("H2", h2), ("H3", h3)):
        print(f"\n{name} success criteria (all three required):")
        for k, ok in res["criteria"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
        print(f"{name} is {'SUPPORTED' if res['supported'] else 'NOT SUPPORTED'}. No re-tuning on these sessions.")
    rows.append({"time": now(), "mode": "h2_cross_lighting", "confirmatory": 1, "config": f"{cfg.name}: err(other) - err(same)",
                 "n_sessions": 2, "n_targets": h2["n_targets"], "err_deg_mean": h2["mean_d"],
                 "ci_lo": h2["ci"][0], "ci_hi": h2["ci"][1],
                 "note": f"preregistered round 2 H2 {'SUPPORTED' if h2['supported'] else 'NOT SUPPORTED'}"})
    rows.append({"time": now(), "mode": "h3_within", "confirmatory": 1, "config": f"{cfg.name}: err(below gate) - err(gated)",
                 "n_sessions": 4, "n_targets": h3["n_gated"] + h3["n_below"], "err_deg_mean": h3["gap"],
                 "ci_lo": h3["ci"][0], "ci_hi": h3["ci"][1],
                 "note": f"preregistered round 2 H3 {'SUPPORTED' if h3['supported'] else 'NOT SUPPORTED'}"})
    if args.log:
        append_rows(rows)
        print("\nlogged to data/results.csv (confirmatory=1)")
    return 0


POSITION_CONFIGS = ["gaze_a10", "gaze_a100", "head_a100", "ray_a100", "geo_a1", "geo_a10", "geo_a100"]


def run_positions(args: argparse.Namespace) -> int:
    """M2 step 3 development (EXPLORATORY): leave-one-position-out on set A.

    Every config trains on the other five set-A sessions and is tested on all 25 targets of the
    held-out one (identical pool, paired targets). Baselines: REF trained on C only (what Run does
    today) and each session's own calibration (16 targets -> its 9 held-out; the ceiling)."""
    a = set_a(clean_sessions())
    if not a.complete:
        print("Set A is incomplete on this machine:\n  - " + "\n  - ".join(a.missing), file=sys.stderr)
        return 2
    names = list(args.configs) if args.configs != ["all"] else POSITION_CONFIGS
    configs = get_configs(names)
    sessions = [a.sessions[k] for k in SLOTS_A]
    screen = common_screen(sessions)
    ref_pos = a.positions["C"]
    print("Set A (development, EXPLORATORY): leave-one-position-out, 25 targets per position")
    for k in SLOTS_A:
        print(f"  {k:3s} {session_conditions(a.sessions[k])}; {describe(a.positions[k], None if k == 'C' else ref_pos)}")
    idx = list(range(len(SLOTS_A)))
    res = {c.name: leave_one_position_out(sessions, idx, c, screen) for c in configs}
    ref_cfg = REGISTRY[PREREG3_REFERENCE]
    c_only = [None] + [fit_predict(sessions, ref_cfg, [0], i, screen) for i in idx[1:]]
    ds = build_dataset(sessions, ref_cfg.feature_set)
    own = []
    for sp in make_splits(ds, "within"):
        m = ref_cfg.make_model(screen).fit(ds.x[sp.train], ds.y[sp.train])
        own.append(per_target(m.predict(ds.x[sp.test]), ds.y[sp.test], ds.target[sp.test], ds.session[sp.test], screen))
    head = f"{'mean deg per held-out position':34s}" + "".join(f"{k:>7s}" for k in SLOTS_A) + f"{'all':>7s}  95% CI"
    print("\n" + head)
    rows = []
    def line(label: str, parts: list) -> tuple:
        vals = [p.err_deg.mean() if p is not None else float("nan") for p in parts]
        allv = concat([p for p in parts if p is not None]).err_deg
        lo, hi = bootstrap_ci(allv, args.n_boot, args.seed)
        print(f"{label:34s}" + "".join(f"{v:7.2f}" for v in vals) + f"{allv.mean():7.2f}  [{lo:.2f}, {hi:.2f}]")
        return allv, lo, hi
    for c in configs:
        allv, lo, hi = line(f"{c.name} (pooled, other 5)", res[c.name])
        rows.append({"time": now(), "mode": "positions_A", "confirmatory": 0, "config": c.name, "model": c.model,
                     "features_set": c.feature_set, "alpha": c.alpha, "n_sessions": 6, "n_targets": len(allv),
                     "err_deg_mean": float(allv.mean()), "ci_lo": lo, "ci_hi": hi,
                     "note": "leave-one-position-out on set A; " + ", ".join(
                         f"{k} {p.err_deg.mean():.2f}" for k, p in zip(SLOTS_A, res[c.name]))})
    line(f"{PREREG3_REFERENCE} trained on C only (today)", c_only)
    line(f"{PREREG3_REFERENCE} own calibration (9 tgts)", own)
    ref_name = PREREG3_REFERENCE if PREREG3_REFERENCE in res else configs[0].name
    for c in configs:
        if c.name != ref_name:
            print_comparison(ref_name, c.name, "positions_A",
                             paired_comparison(concat(res[ref_name]), concat(res[c.name]), args.n_boot, args.seed))
    if args.per_target:
        best = min(configs, key=lambda c: concat(res[c.name]).err_deg.mean())
        print_per_target(best, "positions_A", concat(res[best.name]), tuple(s.path.name for s in sessions))
    few_dot_table(sessions, screen, args)
    if args.dots:
        dot_sweep(sessions, screen, args)
    print("\nEXPLORATORY: set A is development data. Choose and freeze ONE candidate before recording set B.")
    if args.log:
        append_rows(rows)
        print(f"logged {len(rows)} rows to data/results.csv (confirmatory=0)")
    return 0


FEW_DOT = [("no correction", 0, "offset"), ("+ 1 dot (offset)", 1, "offset"), ("+ 3 dots (offset)", 3, "offset"),
           ("+ 5 dots (affine)", 5, "affine")]


def few_dot_table(sessions, screen, args) -> None:
    """EXPLORATORY: pooled (other 5) or C-only model + k quick dots of the new session, scored on
    each session's 9 held-out targets; own 16-dot calibration as the ceiling. Paired on targets."""
    print("\n" + "on each session's 9 held-out targets".ljust(44) + "".join(f"{k:>7s}" for k in SLOTS_A)
          + f"{'all':>7s}  95% CI")
    idx = range(len(SLOTS_A))
    out = {}
    def show(label, parts):
        allv = concat(parts)
        lo, hi = bootstrap_ci(allv.err_deg, args.n_boot, args.seed)
        print(f"{label:44s}" + "".join(f"{p.err_deg.mean():7.2f}" for p in parts)
              + f"{allv.err_deg.mean():7.2f}  [{lo:.2f}, {hi:.2f}]")
        out[label] = allv
    for name in ("gaze_a100",):
        cfg = REGISTRY[name]
        for label, k, kind in FEW_DOT:
            show(f"{name} pooled {label}", [prereg3.few_dot(sessions, cfg, [j for j in idx if j != t], t, screen, k, kind)
                                            for t in idx])
    ref = REGISTRY[PREREG3_REFERENCE]
    for label, k, kind in FEW_DOT[:2]:   # today's Run; on C itself it is C's own calibration (no leak)
        show(f"{ref.name} C only {label}", [prereg3.own_calibration(sessions, ref, 0, screen)]
             + [prereg3.few_dot(sessions, ref, [0], t, screen, k, kind) for t in idx[1:]])
    show(f"{ref.name} own 16-dot calibration", [prereg3.own_calibration(sessions, ref, t, screen) for t in idx])
    base = out["gaze_a100 pooled no correction"]
    for label, k, kind in FEW_DOT[1:]:
        print_comparison("pooled", f"pooled {label}", "positions_A, 9 held-out targets",
                         paired_comparison(base, out[f"gaze_a100 pooled {label}"], args.n_boot, args.seed))


DOT_KS = (3, 5, 7, 9)


def dot_sweep(sessions, screen, args) -> None:
    """EXPLORATORY: k quick dots at a new position, k in DOT_KS, scored on all 25 - k other targets
    of each held-out set-A session (identical targets per k, so methods pair). The pooled model
    trains on the other 5 sessions; ``fresh`` uses the k dots alone (is the pool worth anything?)."""
    idx = range(len(SLOTS_A))
    cfg = REGISTRY["gaze_a100"]
    ref = REGISTRY[PREREG3_REFERENCE]
    for k in DOT_KS:
        print(f"\n{k} dots; scored on the other {25 - k} targets".ljust(48) + "".join(f"{s:>7s}" for s in SLOTS_A)
              + f"{'all':>7s}  95% CI")
        out = {}
        variants = [(f"{ref.name} pooled, no dots", ref, "none"), (f"{cfg.name} pooled, no dots", cfg, "none")]
        variants += [(f"{cfg.name} pooled + {m}", cfg, m) for m in ("offset", "affine", "refit")]
        variants += [(f"{c.name} fresh on {k} dots only", c, "fresh") for c in (ref, cfg)]
        for label, c, m in variants:
            if m == "affine" and k < 3:
                continue
            parts = [prereg3.dot_strategy(sessions, c, [j for j in idx if j != t], t, screen, k, m) for t in idx]
            allv = concat(parts)
            lo, hi = bootstrap_ci(allv.err_deg, args.n_boot, args.seed)
            print(f"{label:47s}" + "".join(f"{p.err_deg.mean():7.2f}" for p in parts)
                  + f"{allv.err_deg.mean():7.2f}  [{lo:.2f}, {hi:.2f}]")
            out[label] = allv
        best = min((l for l in out if "pooled +" in l), key=lambda l: out[l].err_deg.mean())
        fresh = min((l for l in out if "fresh" in l), key=lambda l: out[l].err_deg.mean())
        for a, b in ((f"{ref.name} pooled, no dots", best), (fresh, best)):
            print_comparison(a, b, f"positions_A, {k} dots", paired_comparison(out[a], out[b], args.n_boot, args.seed))


def run_preregistered_3(args: argparse.Namespace) -> int:
    """The pre-registered round-3 analysis (PREREGISTRATIONS.md): H4 and H5. Confirmatory: run once."""
    if prereg3.PREREG3_CANDIDATE is None or prereg3.PREREG3_FREEZE_AFTER is None:
        print("Refusing to run: the round-3 candidate is not frozen yet (PREREG3_CANDIDATE in "
              "src/gaze_cursor/eval/prereg3.py). Develop on set A with --positions first.", file=sys.stderr)
        return 2
    clean = clean_sessions()
    a, b = set_a(clean), select_set_b(clean, prereg3.PREREG3_FREEZE_AFTER)
    for name, why in b.excluded:
        print(f"excluded {short(name)}: {why}")
    for note in b.ignored:
        print(f"ignoring {short(note)}")
    if not (a.complete and b.complete):
        print("Refusing to run: still needed:\n  - " + "\n  - ".join(a.missing + b.missing), file=sys.stderr)
        return 2
    ref, cand = REGISTRY[PREREG3_REFERENCE], REGISTRY[prereg3.PREREG3_CANDIDATE]
    screen = common_screen([*a.sessions.values(), *b.sessions.values()])
    n_dots, method = prereg3.PREREG3_DOTS, prereg3.PREREG3_METHOD
    cl = f"{cand.name} {method} on {n_dots} dots"
    print(f"Pre-registered round 3 (H5 amended): reference {ref.name} pooled, no dots; candidate {cl} of the test position")
    for k in ("C", *DISPLACED, "C2"):
        print(f"  B {k:3s} {session_conditions(b.sessions[k])}; {describe(b.positions[k], None if k == 'C' else b.positions['C'])}")
    h4, h5 = run_h4(b, ref, screen), run_h5(a, b, ref, cand, screen)
    print(f"\n== H4: {ref.name} trained on C only, displaced positions vs C2 ==")
    print("mean error per position: " + ", ".join(f"{k} {v:.2f}" for k, v in h4["per_position"].items())
          + f"; C2 {h4['c2']:.2f} deg")
    print(f"gap = displaced - C2 = {h4['gap']:+.2f} deg, 95% CI [{h4['ci'][0]:+.2f}, {h4['ci'][1]:+.2f}]")
    print(f"\n== H5 (amended): {cl} vs {ref.name} pooled (all other sessions), "
          f"scored on the other {25 - n_dots} targets of each displaced position ==")
    print(f"mean error: {ref.name} pooled {h5['err_ref']:.2f}, {cl} {h5['err_cand']:.2f} deg")
    print_comparison(f"{ref.name} pooled", cl, "set B, held-out positions", h5["comparison"])
    print_comparison(f"{ref.name}(C only)", cl, "reported, not a criterion", h5["vs_c_only"])
    print_comparison(f"{ref.name} pooled + refit on the dots", cl, "reported, not a criterion", h5["vs_refit"])
    print(f"reported: own 16-dot calibration on the displaced positions (9 held-out targets each) "
          f"{h5['own_16']:.2f} deg; {cl} on C2 {h5['cand_c2']:.2f} deg")
    rows = []
    for name, res in (("H4", h4), ("H5", h5)):
        print(f"\n{name} success criteria (all three required):")
        for k, ok in res["criteria"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
        print(f"{name} is {'SUPPORTED' if res['supported'] else 'NOT SUPPORTED'}. No re-tuning on these sessions.")
    comp = h5["comparison"]
    rows.append({"time": now(), "mode": "h4_position_cost", "confirmatory": 1,
                 "config": f"{ref.name}(C only): err(displaced) - err(C2)", "n_sessions": 6,
                 "n_targets": h4["n_displaced"] + 25, "err_deg_mean": h4["gap"], "ci_lo": h4["ci"][0],
                 "ci_hi": h4["ci"][1], "note": f"preregistered round 3 H4 {'SUPPORTED' if h4['supported'] else 'NOT SUPPORTED'}"})
    rows.append({"time": now(), "mode": "h5_lopo", "confirmatory": 1, "config": f"{ref.name} pooled - {cl}",
                 "n_sessions": 13, "n_targets": comp["n_targets"], "err_deg_mean": comp["mean_diff_deg"],
                 "ci_lo": comp["ci_lo"], "ci_hi": comp["ci_hi"],
                 "note": f"preregistered round 3 H5 {'SUPPORTED' if h5['supported'] else 'NOT SUPPORTED'}; "
                         f"{comp['sessions_new_better']}/5 sessions"})
    if args.log:
        append_rows(rows)
        print("\nlogged to data/results.csv (confirmatory=1)")
    return 0


def run_preregistered_3_earlier(args: argparse.Namespace) -> int:
    """H5 as frozen earlier on the diverged claude.ai branch (dfa75ff, PREREGISTRATIONS.md "AMENDMENT 1"):
    gaze_a100 pooled vs gaze_a10 on set-B C only. Run once, reported next to the executed protocol."""
    clean = clean_sessions()
    a, b = set_a(clean), select_set_b(clean, prereg3.EARLIER_FREEZE_AFTER)
    if not (a.complete and b.complete):
        print("Refusing to run: still needed:\n  - " + "\n  - ".join(a.missing + b.missing), file=sys.stderr)
        return 2
    ref, cand = REGISTRY[PREREG3_REFERENCE], REGISTRY[prereg3.EARLIER_FREEZE_CANDIDATE]
    screen = common_screen([*a.sessions.values(), *b.sessions.values()])
    print(f"Earlier round-3 freeze (dfa75ff, H5 AMENDMENT 1): {cand.name} pooled vs {ref.name} on set-B C only")
    for k in ("C", *DISPLACED, "C2"):
        print(f"  B {k:3s} {short(b.sessions[k].path.name)}")
    h5 = prereg3.run_h5_earlier_freeze(a, b, ref, cand, screen)
    print(f"mean error: {ref.name} C only {h5['err_ref']:.2f}, {cand.name} pooled {h5['err_cand']:.2f} deg")
    print("per position d: " + ", ".join(f"{k} {v:+.2f}" for k, v in h5["per_position"].items()))
    print_comparison(f"{ref.name}(C only)", f"{cand.name}(pooled)", "held-out set-B positions", h5["comparison"])
    print_comparison(f"{ref.name}(pooled)", f"{cand.name}(pooled)", "reported, not a criterion", h5["vs_same_pool"])
    print(f"{cand.name} on C2 (trained on everything else): {h5['cand_c2']:.2f} deg")
    print("\nH5 (earlier freeze) success criteria (all three required):")
    for k, ok in h5["criteria"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    verdict = "SUPPORTED" if h5["supported"] else "NOT SUPPORTED"
    print(f"H5 (earlier freeze) is {verdict}. No re-tuning on these sessions.")
    comp = h5["comparison"]
    if args.log:
        append_rows([{"time": now(), "mode": "h5_lopo_earlier_freeze", "confirmatory": 1,
                      "config": f"{ref.name}(C only) - {cand.name}(pooled)", "n_sessions": 13,
                      "n_targets": comp["n_targets"], "err_deg_mean": comp["mean_diff_deg"],
                      "ci_lo": comp["ci_lo"], "ci_hi": comp["ci_hi"],
                      "note": f"preregistered round 3 H5, earlier freeze dfa75ff (run after the 13:12 protocol): "
                              f"{verdict}; {comp['sessions_new_better']}/5 sessions"}])
        print("\nlogged to data/results.csv (confirmatory=1)")
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
    ap.add_argument("--preregistered-2", dest="preregistered_2", action="store_true",
                    help="run the pre-registered round-2 tests (H2, H3)")
    ap.add_argument("--positions", action="store_true",
                    help="M2 step 3 development: leave-one-position-out on set A (exploratory)")
    ap.add_argument("--dots", action="store_true",
                    help="with --positions: sweep 3/5/7/9 quick dots, scored on all other targets")
    ap.add_argument("--preregistered-3", dest="preregistered_3", action="store_true",
                    help="run the pre-registered round-3 tests (H4, H5); refuses until frozen")
    ap.add_argument("--preregistered-3-earlier", dest="preregistered_3_earlier", action="store_true",
                    help="run H5 as frozen earlier on the claude.ai branch (dfa75ff): pooled gaze_a100 vs C only")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-log", dest="log", action="store_false", help="do not append to results.csv")
    args = ap.parse_args()
    if args.positions:
        return run_positions(args)
    if args.preregistered_3:
        return run_preregistered_3(args)
    if args.preregistered_3_earlier:
        return run_preregistered_3_earlier(args)
    if args.preregistered_2:
        return run_preregistered_2(args)
    return run_preregistered(args) if args.preregistered else run_exploratory(args)


if __name__ == "__main__":
    raise SystemExit(main())
