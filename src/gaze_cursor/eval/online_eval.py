"""Pre-registered round 4 (PREREGISTRATIONS.md, "Round 4: self-recalibration from clicks"): does updating the
gaze model from the user's own clicks (``online.OnlineRidge``) beat the frozen quick calibration?

Replays recorded Cursor sessions (``data/quick/cursor_*.npz``) click by click (prequential: each click is
predicted before the model may learn from it). Every threshold here is the written protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.cursor_session import EVENT_CODES
from gaze_cursor.eval.configs import REGISTRY
from gaze_cursor.eval.session import CALIB, load_session
from gaze_cursor.livetest import px_per_deg
from gaze_cursor.online import ClickResult, Cusum, OnlineRidge, prequential
from gaze_cursor.quick import QUICK_CONFIG

# Frozen by the round-4 freeze commit, BEFORE any evaluation session is recorded.
FREEZE_AFTER: str | None = "2026-10-13T21:00:00"   # local time; only cursor sessions started after it count
N_SESSIONS = 3                           # the first 3 qualifying sessions after the freeze, in time order
MIN_MINUTES = 5.0
MIN_EVAL_CLICKS = 20
MIN_GAIN_DEG = 0.5
MIN_SESSIONS_BETTER = 2
N_BOOT, SEED = 10_000, 0
INTEGRITY_PX = 1.0                       # the rebuilt calibration model must reproduce the logged gaze


@dataclass
class SessionReplay:
    path: Path
    started: str
    minutes: float
    clicks: list[ClickResult] = field(default_factory=list)
    integrity_px: dict[int, float] = field(default_factory=dict)
    excluded_segments: list[str] = field(default_factory=list)
    cusum_alarms: int = 0
    lighting: str | None = None

    @property
    def evaluated(self) -> list[ClickResult]:
        return [c for c in self.clicks if c.evaluated]

    def d(self) -> np.ndarray:
        """Per evaluated click: err(frozen) - err(online), deg (positive = learning helps)."""
        return np.array([c.err_frozen_deg - c.err_online_deg for c in self.evaluated])


def _screen(meta: dict) -> ScreenConfig:
    return ScreenConfig(**{k: v for k, v in meta["screen"].items() if k in ScreenConfig.__dataclass_fields__})


def prior_from_quick(path: Path) -> OnlineRidge:
    """The quick calibration's model, rebuilt from its saved recording (record and replay)."""
    s = load_session(path)
    x, y, _ = s.select(CALIB, "gaze")
    return OnlineRidge.from_calibration(x, y, REGISTRY[QUICK_CONFIG].alpha)


def quick_lighting(path: Path) -> str | None:
    """The calibration's captured lighting as text (reported per session), or None if unavailable."""
    try:
        light = json.loads(str(np.load(path)["meta"])).get("lighting") or {}
    except (OSError, EOFError, KeyError, ValueError):
        return None
    return f"face {light['face']:.0f} / background {light['background']:.0f}" if "face" in light else None


def segments(meta: dict, events: np.ndarray, frames: np.ndarray, quick_dir: Path) -> list[tuple[float, Path]]:
    """(start time, quick calibration file) per segment. New logs store this; for older logs the first
    segment is the quick file saved with the session's own time stamp and the last is ``calibration``,
    split at the recalibrate events (middle segments of older logs cannot be attributed and are dropped)."""
    if meta.get("calibrations"):
        return [(float(t) if t is not None else -np.inf, quick_dir / name) for t, name in meta["calibrations"]]
    recal = sorted(events[events[:, 1] == EVENT_CODES["recalibrate"], 0]) if len(events) else []
    last = quick_dir / meta["calibration"]
    if not recal:
        return [(-np.inf, last)]
    first = quick_dir / f"session_{meta['started']}.npz"
    out = [(-np.inf, first)] if first.exists() and first != last else []
    # The new calibration is active from the first frame after the recalibration gap (no frames are
    # logged while the dots are shown); the frame around the key press itself still used the old model.
    t = frames[:, 0]
    after = np.flatnonzero((t > recal[-1]) & (np.diff(t, prepend=t[0]) > 2000))
    start = float(t[after[0]]) if len(after) else float(recal[-1])
    return out + [(start, last)]


def replay(path: Path, quick_dir: Path) -> SessionReplay:
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    frames, events = d["frames"], d["events"]
    t, feats, raw = frames[:, 0], frames[:, 5:9], frames[:, 1:3]
    cfg = _screen(meta)
    segs = segments(meta, events, frames, quick_dir)
    starts = [s for s, _ in segs]
    seg_of = lambda tt: int(np.searchsorted(starts, tt, side="right") - 1)   # noqa: E731
    rep = SessionReplay(path, meta["started"], float((t[-1] - t[0]) / 60000) if len(t) > 1 else 0.0)
    priors: dict[int, OnlineRidge] = {}
    for k, (start, qpath) in enumerate(segs):
        if not qpath.exists():
            rep.excluded_segments.append(f"segment {k}: {qpath.name} missing")
            continue
        prior = prior_from_quick(qpath)
        end = starts[k + 1] if k + 1 < len(starts) else np.inf
        sel = (t >= start) & (t < end) & ~np.isnan(feats).any(axis=1) & ~np.isnan(raw).any(axis=1)
        diff = float(np.abs(prior.predict(feats[sel]) - raw[sel]).max()) if sel.any() else 0.0
        rep.integrity_px[k] = diff
        if diff > INTEGRITY_PX:
            rep.excluded_segments.append(f"segment {k}: rebuilt model differs from the log by {diff:.1f} px")
            continue
        priors[k] = prior
        rep.lighting = rep.lighting or quick_lighting(qpath)
    clicks = events[events[:, 1] == EVENT_CODES["click"]] if len(events) else events
    rep.clicks = prequential(t, feats, [(c[0], c[2], c[3], seg_of(c[0])) for c in clicks], priors, px_per_deg(cfg))
    cus = Cusum()
    for i, c in enumerate(rep.evaluated):
        cus.update(float(c.err_frozen_deg), i)     # drift diagnostic only (round 4: logged, not used)
    rep.cusum_alarms = len(cus.alarms)
    return rep


def qualifies(rep: SessionReplay, freeze_after: str) -> str | None:
    """Why ``rep`` cannot count, or None if it can."""
    started = f"{rep.started[:4]}-{rep.started[4:6]}-{rep.started[6:8]}T{rep.started[9:11]}:{rep.started[11:13]}:{rep.started[13:15]}"
    if started < freeze_after:
        return "recorded before the freeze"
    if rep.minutes < MIN_MINUTES:
        return f"only {rep.minutes:.1f} min (< {MIN_MINUTES:g})"
    if len(rep.evaluated) < MIN_EVAL_CLICKS:
        return f"only {len(rep.evaluated)} usable clicks (< {MIN_EVAL_CLICKS})"
    return None


def status_line(rep: SessionReplay, freeze_after: str) -> str:
    """Intake view of a log: only what the data rule needs (minutes, click counts, verdict). Deliberately no
    errors and no frozen-vs-learning comparison, so checking a round-4 session does not reveal its result."""
    reason = qualifies(rep, freeze_after)
    return (f"{rep.path.name}: started {rep.started}, {rep.minutes:.1f} min awake, {len(rep.clicks)} labelled "
            f"clicks, {len(rep.evaluated)} usable; segments excluded {len(rep.excluded_segments)}; "
            + ("COUNTS" if reason is None else f"does not count: {reason}"))


def bootstrap_ci(d_by_session: list[np.ndarray], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of per-click d with 95% CIs: resampling clicks, and sessions then clicks."""
    rng = np.random.default_rng(seed)
    allc = np.concatenate(d_by_session)
    flat = rng.choice(allc, size=(n_boot, len(allc)), replace=True).mean(axis=1)
    two = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.integers(0, len(d_by_session), len(d_by_session))
        two[b] = np.concatenate([rng.choice(d_by_session[i], len(d_by_session[i])) for i in picks]).mean()
    return {"mean": float(allc.mean()), "ci": tuple(np.percentile(flat, [2.5, 97.5])),
            "ci_two_level": tuple(np.percentile(two, [2.5, 97.5])), "n": int(len(allc)),
            "per_session": [float(d.mean()) for d in d_by_session]}


def decide(reps: list[SessionReplay]) -> dict:
    stats = bootstrap_ci([r.d() for r in reps])
    crit = {
        "(a) CI lower bound > 0": stats["ci"][0] > 0,
        f"(b) mean gain >= {MIN_GAIN_DEG} deg": stats["mean"] >= MIN_GAIN_DEG,
        f"(c) better in >= {MIN_SESSIONS_BETTER} of {len(reps)} sessions":
            sum(m > 0 for m in stats["per_session"]) >= MIN_SESSIONS_BETTER,
    }
    return {"stats": stats, "criteria": crit, "supported": all(crit.values())}
