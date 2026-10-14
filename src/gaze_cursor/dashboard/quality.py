"""Per-session quality report for the dashboard (the 6b report, computed on demand and cached).

Mirrors ``scripts/session_report.py``: full calibrations and Run quick calibrations get the numeric report;
Cursor-mode logs and Cursor-mode calibrations get the tag-only check, because round 4 was frozen without the
report (PREREGISTRATIONS.md) and its classification stays untouched. Nothing is written under ``data/``
(the report PNG is only served if ``session_report.py`` already made it); results are cached outside the repo.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from pathlib import Path

from gaze_cursor.conditions import conditions_block, sidecar_path, read_meta, tags_present, validate, verdict
from gaze_cursor.dashboard.catalog import Roots, path_of
from gaze_cursor.eval import quality as _quality_code
from gaze_cursor.eval.quality import all_metrics, is_cursor_mode, refusal
from gaze_cursor.eval.quality_report import grade, overall
from gaze_cursor.eval.session import load_session

NOTE = ("Descriptive, not evidence: within-session errors (calibration to test targets, or leave-one-target-out), "
        "never cross-session accuracy. Thresholds are the written 6b thresholds; they are not edited here.")


_LOCK = threading.Lock()


def _code_version() -> str:
    """Hash of every source file of the package except the dashboard itself. The grades depend on the metrics,
    the written 6b thresholds, the model configs, features and the tag validator, so any change to them
    invalidates cached reports."""
    root = Path(_quality_code.__file__).resolve().parents[1]
    h = hashlib.sha1()
    for p in sorted(root.rglob("*.py")):
        if "dashboard" not in p.relative_to(root).parts:
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


CODE_VERSION = _code_version()


def _clean(v: object) -> object:
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _tag_only(meta: dict, p: Path, why: str) -> dict:
    issues = validate(conditions_block(meta, p))
    return {"mode": "tags_only", "why": why, "overall": verdict(issues), "tags": tags_present(issues),
            "issues": [{"field": i.field, "severity": i.severity, "message": i.message} for i in issues],
            "checks": [], "note": NOTE}


def compute(p: Path) -> dict:
    """The report for one file. Never raises: a failure is returned as ``{"mode": "error"}``."""
    try:
        meta = read_meta(p)
        if p.name.startswith("cursor_") or meta.get("kind") == "cursor_session":
            return _tag_only(meta, p, "Cursor-mode log: round-4 data, only its condition tags are checked")
        if "(Cursor)" in str(meta.get("note", "")):
            return _tag_only(meta, p, "Cursor-mode calibration: only its condition tags are checked")
        if p.parent.name == "tasks" or p.name.startswith("livetest_"):
            return {"mode": "refused", "why": "live-test and task recordings have no calibration report",
                    "checks": [], "note": NOTE}
        s = load_session(p)
        if is_cursor_mode(s):
            return _tag_only(s.meta, p, "Cursor-mode calibration: only its condition tags are checked")
        why = refusal(s)
        if why:
            return {"mode": "refused", "why": why, "checks": [], "note": NOTE}
        m = all_metrics(s)
        results = grade(m)
        return {"mode": "full", "overall": overall(results), "note": NOTE,
                "checks": [{"check": r.check, "title": r.title, "level": r.level,
                            "lines": [{"name": ln.name, "value": _clean(ln.value), "level": ln.level,
                                       "note": ln.note} for ln in r.lines]} for r in results]}
    except Exception as exc:   # a broken file must not take the dashboard down
        return {"mode": "error", "why": f"{type(exc).__name__}: {exc}", "checks": [], "note": NOTE}


def _stamp(p: Path) -> list:
    """Cache key: the file, its backfilled tag sidecar (read by both report kinds) and the grading code."""
    st = p.stat()
    side = sidecar_path(p)
    return [st.st_mtime_ns, st.st_size, side.stat().st_mtime_ns if side.exists() else 0, CODE_VERSION]


def report_for(roots: Roots, session_id: str) -> dict | None:
    """Cached report for an id, or None if the id is unknown."""
    p = path_of(roots, session_id)
    if p is None:
        return None
    cache_file = roots.cache.with_name("dashboard_quality.json") if roots.cache else None
    cache: dict = {}
    if cache_file:
        try:
            cache = json.loads(cache_file.read_text())
        except (OSError, ValueError):
            cache = {}
    hit = cache.get(session_id)
    if hit and hit.get("stamp") == _stamp(p):
        return hit["report"]
    report = compute(p)
    if cache_file and report.get("mode") != "error":
        with _LOCK:   # concurrent requests must not overwrite each other's entries
            try:
                cache = json.loads(cache_file.read_text())
            except (OSError, ValueError):
                cache = {}
            cache[session_id] = {"stamp": _stamp(p), "report": report}
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_name(cache_file.name + ".tmp")
                tmp.write_text(json.dumps(cache))
                os.replace(tmp, cache_file)
            except OSError:
                pass
    return report


def png_path(roots: Roots, session_id: str) -> Path | None:
    """The report PNG made earlier by ``session_report.py``, if there is one."""
    p = path_of(roots, session_id)
    if p is None:
        return None
    png = p.with_name(p.stem + ".report.png")
    return png if png.is_file() else None
