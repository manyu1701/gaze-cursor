"""Shared writer for data/results.csv (benchmark.py and evaluate.py)."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results.csv"

COLUMNS = (
    "time", "mode", "confirmatory", "config", "session", "model", "features_set", "alpha",
    "n_sessions", "n_targets", "n_frames", "n_train", "n_test",
    "err_px_mean", "err_px_median", "err_px_p90", "err_cm_mean",
    "err_deg_mean", "err_deg_median", "err_deg_p90", "jitter_px", "jitter_deg",
    "ci_lo", "ci_hi", "note",
)
_DEFAULTS_FOR_OLD_ROWS = {"mode": "within", "confirmatory": "0"}


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def append_rows(rows: list[dict], path: Path = RESULTS) -> None:
    """Append rows under the shared header, migrating an older file in place if needed.

    Old rows keep their values; new columns are blank, ``mode`` becomes ``within`` and
    ``confirmatory`` becomes 0. Unknown keys in ``rows`` raise, so the schema cannot drift.
    """
    for r in rows:
        extra = set(r) - set(COLUMNS)
        if extra:
            raise KeyError(f"unknown results column(s): {sorted(extra)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            header = tuple(reader.fieldnames or ())
            if header == COLUMNS:
                with path.open("a", newline="") as fa:
                    w = csv.DictWriter(fa, fieldnames=COLUMNS)
                    for r in rows:
                        w.writerow({c: _fmt(r.get(c)) for c in COLUMNS})
                return
            existing = [{**_DEFAULTS_FOR_OLD_ROWS, **row} for row in reader]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in existing:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in COLUMNS})
    os.replace(tmp, path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
