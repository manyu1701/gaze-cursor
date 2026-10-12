"""Offline evaluation: fit on a session's calibration targets, test on its held-out targets.

Usage: uv run scripts/benchmark.py [session.npz] [--model ridge|poly2] [--alpha 1.0]
                                   [--features iris|blend|both]
Appends one row to data/results.csv.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from gaze_cursor.calibration import MODELS, GazeModel
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.metrics import error_metrics, jitter_metrics
from gaze_cursor.features import FEATURE_SETS
from gaze_cursor.eval.session import CALIB, TEST, latest_session, load_session

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", type=Path)
    ap.add_argument("--model", choices=MODELS, default="ridge")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--features", choices=FEATURE_SETS, default="iris",
                    help="iris = iris+pose+openness (baseline); eyes = iris+openness; gaze = iris only; blend = pose+blendshapes; both")
    args = ap.parse_args()

    session = load_session(args.session or latest_session())
    cfg = ScreenConfig(**session.meta["screen"])
    x_tr, y_tr, _ = session.select(CALIB, args.features)
    x_te, y_te, idx_te = session.select(TEST, args.features)
    if len(x_tr) == 0 or len(x_te) == 0:
        raise SystemExit("Session has no valid calibration or test frames.")

    model = GazeModel(args.model, args.alpha).fit(x_tr, y_tr)
    pred = model.predict(x_te)
    metrics = {**error_metrics(pred, y_te, cfg), **jitter_metrics(pred, idx_te, cfg)}
    n_test_all = int((session.phase == TEST).sum())

    print(f"session: {session.path.name}   model: {args.model} (alpha={args.alpha}) features: {args.features}")
    print(f"train frames: {len(x_tr)}   test frames: {len(x_te)} valid of {n_test_all}")
    print(f"error  mean {metrics['err_px_mean']:.0f} px | {metrics['err_cm_mean']:.2f} cm | "
          f"{metrics['err_deg_mean']:.2f} deg   (median {metrics['err_px_median']:.0f} px, "
          f"{metrics['err_deg_median']:.2f} deg)")
    print(f"jitter {metrics['jitter_px']:.1f} px | {metrics['jitter_deg']:.2f} deg")

    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": session.path.name,
           "model": args.model, "features_set": args.features, "alpha": args.alpha, "n_train": len(x_tr), "n_test": len(x_te),
           **{k: round(v, 4) for k, v in metrics.items()}}
    new = not RESULTS.exists()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"appended to {RESULTS}")


if __name__ == "__main__":
    main()
