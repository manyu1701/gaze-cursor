"""Catalogue of everything recorded on disk, read from each file's ``meta`` only (no landmark arrays).

One ``SessionRow`` per recording: calibration sessions, quick calibrations, Cursor-mode logs, live tests and
target-practice tasks. Rows are cached by (path, mtime, size) outside the repo so a long list stays fast.
Pure standard library plus numpy; nothing here writes under ``data/`` and face-image sidecars
(``*.faces.npz``) are never opened.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gaze_cursor.conditions import conditions_block, read_meta, tags_present, validate, verdict

REPO = Path(__file__).resolve().parents[3]
CACHE_VERSION = 2
KINDS = ("calibration", "quick", "quick_cursor", "cursor", "livetest", "task")
PREFIXES = {"session_": "session", "cursor_": "cursor", "livetest_": "livetest"}


@dataclass(frozen=True)
class Roots:
    """Where the data lives. ``from_data_dir`` points the dashboard at any folder with the same layout."""

    sessions: Path
    quick: Path
    tasks: Path
    results: Path
    repo: Path = REPO
    cache: Path | None = None

    @classmethod
    def default(cls) -> Roots:
        return cls.from_data_dir(REPO / "data", cache=Path.home() / ".gaze-cursor" / "dashboard_cache.json")

    @classmethod
    def from_data_dir(cls, data: Path, repo: Path = REPO, cache: Path | None = None) -> Roots:
        data = Path(data)
        return cls(data / "sessions", data / "quick", data / "tasks", data / "results.csv", repo, cache)


@dataclass
class SessionRow:
    """Everything the tables show about one recording. ``id`` is ``<folder>:<file stem>`` and is the only
    handle the web API accepts (never a path)."""

    id: str
    folder: str
    file: str
    kind: str
    started: str
    size_kb: int
    mtime: float
    note: str = ""
    layout: str = ""
    lighting: str = ""            # measured label from the recording ("front-lit", ...), not the tag
    lighting_tag: str = ""        # the condition tag (day / lamp / dark / backlit)
    glasses: str = ""
    person_id: str = ""
    tags: int = 0                 # of N_CHECKED condition fields present
    tags_verdict: str = "warn"
    legacy: bool = True           # no own conditions block (tags absent or backfilled)
    backfilled: bool = False
    issues: list[dict] = field(default_factory=list)
    frames: int = 0
    targets: int = 0
    minutes: float = 0.0
    clicks: int = 0
    clicks_learned: int = 0
    click_gaze_deg_median: float | None = None
    click_gaze_n: int = 0
    learn_on: bool | None = None
    resumed: str = ""
    has_report: bool = False
    error: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("mtime")
        return d


def _kind(folder: str, stem: str, meta: dict) -> str:
    if folder == "tasks":
        return "task"
    if stem.startswith("cursor_"):
        return "cursor"
    if stem.startswith("livetest_"):
        return "livetest"
    if folder == "quick":
        return "quick_cursor" if "(Cursor)" in str(meta.get("note", "")) else "quick"
    return "calibration"


def _file_candidates(roots: Roots) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for folder, root in (("sessions", roots.sessions), ("quick", roots.quick), ("tasks", roots.tasks)):
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.npz")):
            name = p.name
            if name.startswith(".") or name.endswith(".faces.npz") or name.endswith(".tmp.npz"):
                continue
            if folder == "tasks" or name.startswith(tuple(PREFIXES)):
                out.append((folder, p))
    return out


_STAMP = re.compile(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


def _started(meta: dict, stem: str) -> str:
    """ISO start time. Cursor logs store ``YYYYmmdd_HHMMSS`` in meta; sessions store ISO; else the file name."""
    given = str(meta.get("started") or "")
    if "-" in given[:10]:
        return given
    for text in (given, stem):
        m = _STAMP.search(text)
        if m:
            return "{}-{}-{}T{}:{}:{}".format(*m.groups())
    return given


def _frame_stats(p: Path) -> tuple[int, int, float]:
    """(frames, targets, minutes) of a session-format file; zeros if the arrays are not there."""
    try:
        with np.load(p, allow_pickle=False) as d:
            if "t_ms" not in d.files:
                return 0, 0, 0.0
            t = d["t_ms"]
            targets = int(len(np.unique(d["target_idx"]))) if "target_idx" in d.files else 0
            minutes = float((t[-1] - t[0]) / 60000) if len(t) > 1 else 0.0
            return int(len(t)), targets, minutes
    except (OSError, ValueError, EOFError, KeyError):
        return 0, 0, 0.0


def build_row(folder: str, p: Path) -> SessionRow:
    """Read one recording's meta into a row. Never raises: an unreadable file becomes a row with ``error``."""
    st = p.stat()
    row = SessionRow(id=f"{folder}:{p.stem}", folder=folder, file=p.name, kind="task" if folder == "tasks" else "",
                     started="", size_kb=int(st.st_size / 1024), mtime=st.st_mtime)
    try:
        meta = read_meta(p)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        row.error = f"unreadable: {type(exc).__name__}"
        row.kind = row.kind or "calibration"
        return row
    row.kind = _kind(folder, p.stem, meta)
    row.started = _started(meta, p.stem)
    row.note = str(meta.get("note", "") or "")
    row.layout = str(meta.get("layout", "") or "")
    light = meta.get("lighting")
    row.lighting = str(light.get("label", "")) if isinstance(light, dict) else ""
    block = conditions_block(meta, p)
    issues = validate(block)
    row.issues = [{"field": i.field, "severity": i.severity, "message": i.message} for i in issues]
    row.tags, row.tags_verdict = tags_present(issues), verdict(issues)
    row.legacy = not (isinstance(meta.get("conditions"), dict) and bool(meta.get("conditions")))
    row.backfilled = bool(block and block.get("backfilled") is True)
    if block:
        row.person_id = str(block.get("person_id") or "")
        row.lighting_tag = str(block.get("lighting_tag") or "")
        row.glasses = str(block.get("glasses") or "")
    if row.kind == "cursor":
        s = meta.get("summary") if isinstance(meta.get("summary"), dict) else {}
        row.frames, row.minutes = int(s.get("frames") or 0), float(s.get("minutes") or 0.0)
        row.clicks, row.clicks_learned = int(s.get("clicks") or 0), int(s.get("clicks_learned") or 0)
        row.click_gaze_deg_median = s.get("click_gaze_deg_median")
        row.click_gaze_n = int(s.get("click_gaze_n") or 0)
        learning = meta.get("learning") if isinstance(meta.get("learning"), dict) else {}
        row.learn_on = learning.get("on")
        row.resumed = str(learning.get("resumed") or "")
    else:
        row.frames, row.targets, row.minutes = _frame_stats(p)
    row.has_report = p.with_name(p.stem + ".report.png").exists()
    return row


def _load_cache(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text())
        return data["rows"] if data.get("version") == CACHE_VERSION else {}
    except (OSError, ValueError, KeyError, AttributeError):
        return {}


def _save_cache(path: Path | None, rows: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"version": CACHE_VERSION, "rows": rows}))
        os.replace(tmp, path)
    except OSError:
        pass   # a cache that cannot be written only costs speed


def scan(roots: Roots) -> list[SessionRow]:
    """All recordings, newest first. Unchanged files come from the cache (key: mtime and size)."""
    cached = _load_cache(roots.cache)
    fresh: dict[str, dict] = {}
    rows: list[SessionRow] = []
    for folder, p in _file_candidates(roots):
        st = p.stat()
        key = f"{folder}:{p.name}"
        sidecar = p.with_name(p.stem + ".conditions.json")
        stamp = [st.st_mtime_ns, st.st_size, sidecar.stat().st_mtime_ns if sidecar.exists() else 0,
                 p.with_name(p.stem + ".report.png").exists()]
        hit = cached.get(key)
        if hit and hit.get("stamp") == stamp:
            row = SessionRow(**hit["row"])
        else:
            row = build_row(folder, p)
        fresh[key] = {"stamp": stamp, "row": asdict(row)}
        rows.append(row)
    _save_cache(roots.cache, fresh)
    rows.sort(key=lambda r: (r.started or "", r.file), reverse=True)
    return rows


def path_of(roots: Roots, session_id: str) -> Path | None:
    """The file behind an id, found by listing the folders (never by joining user text onto a path)."""
    for folder, p in _file_candidates(roots):
        if f"{folder}:{p.stem}" == session_id:
            return p
    return None
