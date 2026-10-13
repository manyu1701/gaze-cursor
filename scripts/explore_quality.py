"""EXPLORATORY: distributions of the session-quality metrics over the existing clean sessions.

Used once, to write the report thresholds down from data before the report is used on a new
session (PLAN M3 step 6b). Prints a table and percentiles; nothing is logged or claimed.

    uv run scripts/explore_quality.py [--dir data/sessions] [--quick-dir data/quick]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gaze_cursor.eval.quality import all_metrics, refusal
from gaze_cursor.eval.session import SESSIONS_DIR, load_session, session_files

COLS = ("valid_fraction", "thin_targets", "dropped_gap_share", "calib_extent_x", "calib_extent_y",
        "test_outside_hull", "heldout_mean", "heldout_median", "heldout_p90", "loto_mean", "loto_worst",
        "loto_far_share", "bias_deg", "spread_deg", "bias_share", "slope_x", "slope_y",
        "head_shift_cm", "head_rot_deg", "late_minus_early_deg", "face_brightness", "tags_present")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=SESSIONS_DIR)
    ap.add_argument("--quick-dir", type=Path, default=None)
    a = ap.parse_args()
    rows, names = [], []
    paths = list(session_files(a.dir)) + (sorted(a.quick_dir.glob("session_*.npz")) if a.quick_dir else [])
    for p in paths:
        if p.name.endswith(".faces.npz"):
            continue
        s = load_session(p)
        if refusal(s):
            continue
        m = all_metrics(s)
        rows.append(m); names.append(f"{p.parent.name[:1]}:{p.name[8:23]} {s.layout:6} {m['lighting_label'][:6]:6}")
    print(f"{len(rows)} clean sessions")
    print(" " * 33 + " ".join(f"{c[:9]:>9}" for c in COLS))
    for n, m in zip(names, rows):
        print(f"{n:33}" + " ".join(f"{float(m[c]):9.2f}" for c in COLS))
    print("\npercentiles over sessions (nan ignored)")
    print(f"{'':8}" + " ".join(f"{c[:9]:>9}" for c in COLS))
    for q in (0, 10, 50, 90, 100):
        print(f"p{q:<7}" + " ".join(f"{np.nanpercentile([float(m[c]) for m in rows], q):9.2f}" for c in COLS))


if __name__ == "__main__":
    main()
