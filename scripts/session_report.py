"""Session quality report: was this calibration done properly? (M3 step 6b)

Prints one pass / warn / fail line per check with the numbers behind it, saves a PNG of targets and
leave-one-target-out predictions next to the session, and appends a summary row to data/results.csv.
Thresholds are written in PLAN.md (M3 step 6b) and live in ``eval/quality_report.py``.

    uv run scripts/session_report.py data/sessions/session_20261013_133547.npz
    uv run scripts/session_report.py --latest [--no-log] [--out-dir DIR]

Refuses Cursor-mode logs (``cursor_*.npz``) and Cursor-mode calibrations: round 4 was frozen
without this report and stays unreported until the H6 test has run (PREREGISTRATIONS.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from gaze_cursor.eval.quality import FIT_CONFIG, all_metrics, held_out_stats, refusal, screen_of
from gaze_cursor.eval.quality_report import FAIL, PASS, WARN, CheckResult, format_report, grade, overall
from gaze_cursor.eval.results_log import RESULTS, append_rows, now
from gaze_cursor.eval.session import Session, latest_session, load_session
from gaze_cursor.eval.stats import TargetStats

SCALE = 0.6
COLOUR = {PASS: (80, 170, 60), WARN: (30, 160, 230), FAIL: (60, 60, 220)}   # BGR


def refuse_path(path: Path) -> str | None:
    """A reason this file must not be reported on, judged from its name alone."""
    if path.name.startswith("cursor_"):
        return "Cursor-mode log: round-4 data, not reported until H6 has run"
    if path.name.endswith(".faces.npz"):
        return "face-image sidecar, not a session"
    return None


def report_png(s: Session, loto: TargetStats | None, results: list[CheckResult], path: Path) -> None:
    """Targets (calibration blue, test green) with each leave-one-target-out prediction (red line)."""
    cfg = screen_of(s)
    w, h = int(cfg.width_px * SCALE), int(cfg.height_px * SCALE)
    img = np.full((h + 70, w, 3), 245, np.uint8)
    cv2.rectangle(img, (0, 70), (w - 1, h + 69), (160, 160, 160), 1)

    def at(xy: np.ndarray) -> tuple[int, int]:
        return int(xy[0] * SCALE), int(xy[1] * SCALE) + 70

    if loto is not None:
        for xy, bias, err in zip(loto.xy, loto.bias_px, loto.err_deg):
            cv2.line(img, at(xy), at(xy + bias), (60, 60, 220), 2)
            cv2.circle(img, at(xy + bias), 4, (60, 60, 220), -1)
            cv2.putText(img, f"{err:.1f}", (at(xy)[0] + 8, at(xy)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (90, 90, 90), 1, cv2.LINE_AA)
    for phase, colour in ((0, (200, 120, 40)), (1, (60, 160, 60))):
        m = s.phase == phase
        _, first = np.unique(s.target_idx[m], return_index=True)
        for xy in s.target_xy[m][first]:
            cv2.circle(img, at(xy), 7, colour, 2)
    verdict = overall(results)
    cv2.putText(img, f"{s.path.name}   OVERALL {verdict.upper()}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, COLOUR.get(verdict, (0, 0, 0)), 1, cv2.LINE_AA)
    cv2.putText(img, "blue = calibration, green = test, red = leave-one-target-out prediction, number = deg",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    x = 10
    for r in results:
        cv2.putText(img, f"{r.check} {r.level}", (x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOUR.get(r.level, (120, 120, 120)), 1, cv2.LINE_AA)
        x += 130
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def summary_row(s: Session, m: dict, results: list[CheckResult]) -> dict:
    """One results.csv row. The error columns are the calibration -> test mean when the session has a
    test phase, else the leave-one-target-out mean (the note says which)."""
    held = not np.isnan(float(m["heldout_mean"]))
    pre = "heldout" if held else "loto"
    bad = [f"{r.check}:{r.level}" for r in results if r.level in (WARN, FAIL)]
    return {
        "time": now(), "mode": "session_report", "confirmatory": 0, "config": FIT_CONFIG,
        "session": s.path.name, "n_targets": int(m["n_calib"] + m["n_test"]),
        "n_frames": int(s.valid.sum()),
        "err_deg_mean": m[f"{pre}_mean"], "err_deg_median": m[f"{pre}_median"], "err_deg_p90": m[f"{pre}_p90"],
        "jitter_deg": m["spread_deg"],
        "note": f"quality report {overall(results)}; errors are {'calibration->test' if held else 'leave-one-target-out'}"
                + (f"; {' '.join(bad)}" if bad else ""),
    }


def run(path: Path, out_dir: Path | None, results_csv: Path | None) -> int:
    """Report on one session. Returns 0 on success, 2 if the session was refused."""
    why = refuse_path(path)
    if why:
        print(f"REFUSED {path.name}: {why}", file=sys.stderr)
        return 2
    s = load_session(path)
    why = refusal(s)
    if why:
        print(f"REFUSED {path.name}: {why}", file=sys.stderr)
        return 2
    m = all_metrics(s)
    results = grade(m)
    print(format_report(path.name, results))
    _, loto = held_out_stats(s)
    png = (out_dir or path.parent) / f"{path.stem}.report.png"
    report_png(s, loto, results, png)
    print(f"\nPNG: {png}")
    if results_csv is not None:
        append_rows([summary_row(s, m, results)], results_csv)
        print(f"summary row appended to {results_csv}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", type=Path, help="session .npz file(s)")
    ap.add_argument("--latest", action="store_true", help="the newest full calibration in data/sessions")
    ap.add_argument("--out-dir", type=Path, default=None, help="where to save the PNG (default: next to the session)")
    ap.add_argument("--no-log", action="store_true", help="do not append a row to data/results.csv")
    ap.add_argument("--results", type=Path, default=RESULTS, help="results CSV to append to")
    a = ap.parse_args()
    paths = list(a.sessions) + ([latest_session()] if a.latest else [])
    if not paths:
        ap.error("give a session file or --latest")
    codes = [run(p, a.out_dir, None if a.no_log else a.results) for p in paths]
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
