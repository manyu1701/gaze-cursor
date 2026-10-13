"""Centre compression of the quick calibration (EXPLORATORY; follows diagnose_vertical.py).

Variants of the 9-dot fit (ridge strength, per-dot mean fitting, an affine stretch on the dots,
richer vertical features: eyelid openness, eye blendshapes) x
two dot sets (today's quick set and an edge-covering set), on every clean v2 front-lit session.
All variants are scored on the SAME targets: those that are dots in neither set (grid 4, 6, 9, 11
and the 9 interior held-out targets), so every comparison is paired.

  uv run scripts/explore_compression.py [--no-log]
"""

from __future__ import annotations

import argparse

import numpy as np

from gaze_cursor.eval.compression import Variant, edge_dot_set, gain, quick_dot_set, score_session
from gaze_cursor.eval.prereg2 import lighting_class
from gaze_cursor.eval.results_log import append_rows, now
from gaze_cursor.eval.runner import common_screen
from gaze_cursor.eval.session import clean_sessions
from gaze_cursor.eval.stats import bootstrap_ci, concat, paired_comparison
from gaze_cursor.eval.vertical import band_report

VARIANTS = [
    Variant("ridge a10 frames (today)", 10),
    Variant("ridge a1 frames", 1), Variant("ridge a0.01 frames", 0.01), Variant("ridge a100 frames", 100),
    Variant("ridge a1 means", 1, "means"), Variant("ridge a0.3 means", 0.3, "means"),
    Variant("ridge a0.01 means", 0.01, "means"),
    Variant("ridge a10 frames + stretch", 10, stretch=True),
    Variant("eyes a10 (+ eyelid openness)", 10, feature_set="eyes"),
    Variant("eyes a100", 100, feature_set="eyes"),
    Variant("gaze_blend a100 (+ blendshapes)", 100, feature_set="gaze_blend"),
    Variant("gaze_blend a1000", 1000, feature_set="gaze_blend"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-log", dest="log", action="store_false")
    args = ap.parse_args()
    sessions = [s for s in clean_sessions() if s.layout == "v2" and lighting_class(s.meta) == "F"]
    screen = common_screen(sessions)
    dot_sets = {"quick": quick_dot_set(screen), "edge": edge_dot_set(screen)}
    scored = np.array(sorted(set(range(25)) - set(dot_sets["quick"]) - set(dot_sets["edge"])))
    print(f"Centre compression (EXPLORATORY): {len(sessions)} sessions; dots quick {dot_sets['quick']}, "
          f"edge {dot_sets['edge']}; scored targets {scored.tolist()} ({len(scored)} per session)")
    res = {}
    for dname, dots in dot_sets.items():
        for v in VARIANTS:
            res[(dname, v.name)] = concat([score_session(s, i, dots, v, scored, screen) for i, s in enumerate(sessions)])
    base = res[("quick", VARIANTS[0].name)]
    print(f"\n{'dots':6s} {'variant':28s} {'mean':>6s} {'median':>7s} {'p90':>6s} {'gain x':>7s} {'gain y':>7s}"
          f" {'top dy':>7s} {'bot dy':>7s}   d vs today [95% CI]       better")
    rows = []
    for (dname, vname), t in res.items():
        gx, gy = gain(t)
        bands = band_report(t, screen, axis=1, n_boot=2000)
        top = bands[0].get("bias_deg", np.nan); bot = bands[-1].get("bias_deg", np.nan)
        if t is base:
            cmp_txt = "(reference)"
        else:
            c = paired_comparison(base, t, 10_000, 0)
            cmp_txt = (f"{c['mean_diff_deg']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]   "
                       f"{c['sessions_new_better']}/{len(sessions)} sess")
        print(f"{dname:6s} {vname:28s} {t.err_deg.mean():6.2f} {np.median(t.err_deg):7.2f}"
              f" {np.percentile(t.err_deg, 90):6.2f} {gx:7.2f} {gy:7.2f} {top:+7.2f} {bot:+7.2f}   {cmp_txt}")
        lo, hi = bootstrap_ci(t.err_deg, 10_000, 0)
        rows.append({"time": now(), "mode": "compression_explore", "confirmatory": 0, "config": f"{dname}: {vname}",
                     "n_sessions": len(sessions), "n_targets": len(t.err_deg), "err_deg_mean": float(t.err_deg.mean()),
                     "err_deg_median": float(np.median(t.err_deg)), "err_deg_p90": float(np.percentile(t.err_deg, 90)),
                     "ci_lo": lo, "ci_hi": hi, "note": f"gain x {gx:.2f}, gain y {gy:.2f}; top dy {top:+.2f}, bottom dy {bot:+.2f}"})
    print("\nd = err(today) - err(variant), positive = variant better; CI over targets (10,000 resamples, seed 0).")
    print("gain = slope of mean prediction on target position (1 = no squeeze). EXPLORATORY: many comparisons.")
    if args.log:
        append_rows(rows)
        print(f"logged {len(rows)} rows to data/results.csv (confirmatory=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
