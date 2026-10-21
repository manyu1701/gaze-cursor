"""Aggregations for the dashboard pages. Pure functions over catalogue rows and a few small files.

Everything here is **descriptive**: counts, sums and medians of numbers the recorders already wrote
(``meta`` and ``meta["summary"]``). Nothing is fitted and no recording's arrays are evaluated, so no model
touches a Cursor log. The click-gaze error is a monitoring signal, not evidence: the model learns from the
same clicks and the warp shapes where clicks land (PLAN L4).
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gaze_cursor.conditions import N_CHECKED
from gaze_cursor.dashboard.catalog import Roots, SessionRow
from gaze_cursor.eval.prereg3 import slot_from_note
from gaze_cursor import model_history
from gaze_cursor.resume import FILE_NAME, MAX_AGE_H

DOCS = ("RESULTS", "PREREGISTRATIONS", "PLAN", "README")
WITHIN = "within-session, not cross-session"
BASIS = {"cursor_session": "monitoring (biased click-gaze)", "session_report": WITHIN, "within": WITHIN,
         "loto": WITHIN, "live_accuracy": WITHIN, "live_targets": WITHIN, "live_path": WITHIN,
         "losos": "cross-session (leave-one-session-out)"}
PAIRED = ("paired difference (A minus B), not an error; CI as logged is click / target level and optimistic, "
          "the two-level CI is in PREREGISTRATIONS.md")


def basis_of(row: dict) -> str:
    """What the error columns of a results.csv row mean (the same columns hold different things per mode)."""
    mode = row.get("mode", "")
    if row.get("confirmatory") in ("1", "True", "true") and mode.startswith("h"):
        return PAIRED
    return BASIS.get(mode, "")
MONITOR_NOTE = ("Monitoring signal, not evidence: the model learns from these same clicks and the jump shapes "
                "where you click. Median over every click in the log, not the gated set of the round-4 test. "
                "Numbers only; no alarm thresholds exist yet (PLAN L4).")


def totals(rows: list[SessionRow]) -> dict:
    """Counts and sums by kind."""
    by_kind: dict[str, dict] = defaultdict(lambda: {"files": 0, "targets": 0, "frames": 0, "minutes": 0.0})
    for r in rows:
        k = by_kind[r.kind]
        k["files"] += 1
        k["targets"] += r.targets
        k["frames"] += r.frames
        k["minutes"] += r.minutes
    cursor = [r for r in rows if r.kind == "cursor"]
    return {
        "files": len(rows),
        "by_kind": {k: v for k, v in sorted(by_kind.items())},
        "targets": sum(r.targets for r in rows if r.kind in ("calibration", "quick", "quick_cursor")),
        "cursor_hours": sum(r.minutes for r in cursor) / 60,
        "clicks": sum(r.clicks for r in cursor),
        "clicks_learned": sum(r.clicks_learned for r in cursor),
        "unreadable": sum(1 for r in rows if r.error),
    }


def tag_completeness(rows: list[SessionRow]) -> dict:
    """How many recordings carry the data-contract condition tags (of ``N_CHECKED`` fields each)."""
    tagged = [r for r in rows if not r.error]
    hist = Counter(r.tags for r in tagged)
    return {
        "n": len(tagged), "n_checked": N_CHECKED,
        "own_block": sum(1 for r in tagged if not r.legacy),
        "backfilled": sum(1 for r in tagged if r.backfilled and r.legacy),
        "legacy_without_tags": sum(1 for r in tagged if r.legacy and not r.backfilled),
        "with_person_id": sum(1 for r in tagged if r.person_id),
        "complete": sum(1 for r in tagged if r.tags == N_CHECKED),
        "histogram": [hist.get(i, 0) for i in range(N_CHECKED + 1)],
        "persons": sorted({r.person_id for r in tagged if r.person_id}),
        "verdicts": dict(Counter(r.tags_verdict for r in tagged)),
    }


def learned_state(roots: Roots, now: float | None = None) -> dict:
    """The current learned state (``last_calibration.npz``, overwritten at every autosave) and the archive of
    every state kept beside it (``model_states/``, L2). Reads small ``meta`` only (no model arrays)."""
    path = roots.quick / FILE_NAME
    states = [m for m in model_history.list_states(roots.quick) if "state_id" in m]
    out: dict = {"exists": False, "max_age_h": MAX_AGE_H, "n_states": len(states),
                 "history": (f"{len(states)} states kept, newest {states[-1]['archived_at_utc']} "
                             f"({states[-1]['reason']}, {states[-1]['n_updates']} updates)" if states else
                             "none yet: states are kept from the next Cursor session on (model_states/)")}
    if not path.exists():
        return out
    try:
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"]))
            n_train = int(len(d["x_train"])) if "x_train" in d.files else 0
        calibrated_at = float(meta["calibrated_at"])
    except (OSError, ValueError, KeyError, EOFError):
        return {**out, "exists": True, "error": "could not be read"}
    age = ((time.time() if now is None else now) - calibrated_at) / 3600
    return {**out, "exists": True, "calibrated_at": calibrated_at, "age_h": age, "reusable": 0 <= age <= MAX_AGE_H,
            "n_updates": int(meta.get("n_updates", 0)), "calib": str(meta.get("calib", "")),
            "forget": meta.get("forget"), "screen": meta.get("screen"), "calibration_frames": n_train}


def per_day(rows: list[SessionRow]) -> list[dict]:
    """One entry per calendar day with recordings, oldest first. Click-to-gaze medians stay per session (with
    their click count and whether learning was on): sessions differ in n and bias, so they are not averaged."""
    days: dict[str, dict] = {}
    for r in rows:
        day = r.started[:10] if r.started else "unknown"
        d = days.setdefault(day, {"day": day, "calibrations": 0, "cursor_sessions": 0, "minutes": 0.0,
                                  "clicks": 0, "clicks_learned": 0, "click_gaze": []})
        if r.kind == "cursor":
            d["cursor_sessions"] += 1
            d["minutes"] += r.minutes
            d["clicks"] += r.clicks
            d["clicks_learned"] += r.clicks_learned
            if r.click_gaze_deg_median is not None:   # one entry per Cursor session, never pooled across them
                d["click_gaze"].append({"median_deg": float(r.click_gaze_deg_median), "n": r.click_gaze_n,
                                        "learn_on": r.learn_on})
        elif r.kind in ("calibration", "quick", "quick_cursor"):
            d["calibrations"] += 1
    return [days[day] for day in sorted(days)]


def _slot(note: str) -> str:
    return slot_from_note(note) or ""


def coverage(rows: list[SessionRow]) -> dict:
    """Counts of full calibrations by measured lighting label and head-position slot (read from the note)."""
    cal = [r for r in rows if r.kind == "calibration"]
    lights = sorted({r.lighting or "unknown" for r in cal})
    slots = sorted({_slot(r.note) or "no slot" for r in cal})
    cells = Counter((r.lighting or "unknown", _slot(r.note) or "no slot") for r in cal)
    return {"lighting": lights, "slots": slots, "n": len(cal),
            "cells": [[cells.get((li, sl), 0) for sl in slots] for li in lights],
            "note": "Lighting is the label measured at recording time; the slot is parsed from the free-text note."}


def health(rows: list[SessionRow], roots: Roots, learned: dict) -> list[dict]:
    """Things worth knowing before using the tool, as {level, text}."""
    out: list[dict] = []
    if not (roots.repo / "models" / "face_landmarker.task").exists():
        out.append({"level": "warn", "text": "Face model missing: run the 'Download face model' tool "
                                             "(uv run scripts/download_model.py)."})
    if not rows:
        out.append({"level": "info", "text": "No recordings found yet. Calibrate or start Cursor mode to create some."})
    tc = tag_completeness(rows)
    if tc["n"] and tc["legacy_without_tags"] == tc["n"]:
        out.append({"level": "info", "text": "No recording has condition tags yet (PLAN step 6c writes them). "
                                             "Every session shows as legacy until then."})
    elif tc["n"] and tc["with_person_id"] < tc["n"]:
        out.append({"level": "info", "text": f"{tc['n'] - tc['with_person_id']} of {tc['n']} recordings have no person id."})
    if learned.get("exists") and learned.get("reusable") is False:
        out.append({"level": "info", "text": f"The last calibration is {learned['age_h']:.0f} h old "
                                             f"(reused only under {MAX_AGE_H:.0f} h): Cursor mode will ask for the 9 dots."})
    if tc["n"] and any(r.error for r in rows):
        out.append({"level": "warn", "text": f"{sum(1 for r in rows if r.error)} file(s) could not be read."})
    return out


def results_summary(path: Path, latest: int = 15) -> dict:
    """data/results.csv grouped by mode, plus the newest rows (as logged; nothing recomputed)."""
    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return {"exists": False, "n": 0, "modes": [], "latest": [], "columns": []}
    modes: dict[str, dict] = {}
    for r in rows:
        m = modes.setdefault(r.get("mode") or "?", {"mode": r.get("mode") or "?", "n": 0, "last": "",
                                                    "confirmatory": 0})
        m["n"] += 1
        m["last"] = max(m["last"], r.get("time", ""))
        m["confirmatory"] += 1 if r.get("confirmatory") in ("1", "True", "true") else 0
    cols = ["time", "mode", "confirmatory", "config", "session", "n_targets", "err_deg_mean", "err_deg_median",
            "jitter_deg", "ci_lo", "ci_hi", "note"]
    newest = sorted(rows, key=lambda r: r.get("time", ""), reverse=True)[:latest]
    latest_rows = [{"basis": basis_of(r), **{c: r.get(c, "") for c in cols}} for r in newest]
    return {"exists": True, "n": len(rows), "modes": sorted(modes.values(), key=lambda m: m["last"], reverse=True),
            "latest": latest_rows, "columns": ["basis", *cols]}


def read_doc(roots: Roots, name: str) -> str | None:
    """A top-level markdown document of the repo, by whitelisted name."""
    if name not in DOCS:
        return None
    try:
        return (roots.repo / f"{name}.md").read_text()
    except OSError:
        return None


def prereg_rounds(text: str) -> list[dict]:
    """Rounds of PREREGISTRATIONS.md with whether a RESULT section follows the pre-registration."""
    rounds: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            current = ({"title": title, "result": False, "frozen": False, "not_supported": []}
                       if title.startswith("Round") else None)
            if current:
                rounds.append(current)
        elif current is not None and "NOT SUPPORTED" in line and len(current["not_supported"]) < 4:
            current["not_supported"].append(line.strip().lstrip("#*- ").strip()[:200])
        if current is not None and line.startswith("###"):
            heading = line.lstrip("#").strip()
            if heading.upper().startswith("RESULT"):
                current["result"] = True
                current.setdefault("result_heading", heading)
            current["frozen"] |= "frozen" in heading.lower() or "pre-registration" in heading.lower()
    for r in rounds:
        r["status"] = "result recorded" if r["result"] else ("frozen, waiting for data" if r["frozen"] else "draft")
        r.setdefault("result_heading", "")
    return rounds


def plan_status(text: str) -> list[str]:
    """The ``**Status`` lines of PLAN.md (one per milestone), as written."""
    return [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("**Status")]
