"""Round 5 (H7a, H7b): analysis of the jump-and-click task files, exactly as frozen in PREREGISTRATIONS.md.

``load`` reads one ``data/tasks/fitts_<time>.npz``; ``qualifies`` applies the data rule; ``session_stats`` gives
the per-session differences d_a (moved - centre jump error, cm) and d_b (jump - trackpad movement time, ms) and
the reported-not-criteria numbers; ``decide`` runs the two-level bootstrap, the t-interval check and the
criteria. Nothing here is tuned: every constant below is written in the protocol.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats as sps

from gaze_cursor import conditions as cond

# Local time of the freeze commit 45d232a (the laptop's zone then, IST); file names carry local time too.
FREEZE_AFTER = "2026-10-22T04:34:11"
N_SESSIONS = 5
MAX_FAILED = 3                             # failed sessions allowed before the 5th qualifying one (amendment 1)
MAX_PER_DAY = 2                            # sessions recorded per calendar day
MAX_FIT_DEG = 3.0
MIN_TRACKED = 0.80
MIN_JUMPS_PER_SEAT = 10
MIN_CELL_TRIALS = 3                        # H7a: a (layout, width) cell needs this many jump trials in both seats
MIN_MT_MS = 150.0                          # H7b: faster hit trials are anticipations / double polls
N_SCORED_BLOCKS = 16
N_BOOT, SEED = 10_000, 0
EFFECT_A_CM, EFFECT_B_MS, MISS_MARGIN_PP, CONSISTENT = 1.0, -150.0, 5.0, 4
DOT_SETTLE_MS = 700.0

Cell = tuple[np.ndarray, np.ndarray]       # (arm 0 values, arm 1 values); the difference is mean(arm 1) - mean(arm 0)


@dataclass
class FittsFile:
    path: Path
    meta: dict
    frames: np.ndarray
    started: str                           # YYYYMMDD_HHMMSS from the file name

    @property
    def task(self) -> dict:
        return self.meta["task"]

    @property
    def cm_per_px(self) -> float:
        s = self.meta["screen"]
        return s["width_cm"] / s["width_px"]


def load(path: Path) -> FittsFile:
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        frames = np.array(d["frames"])
    if meta.get("kind") != "fitts_task":
        raise ValueError(f"{path.name} is not a fitts_task file")
    f = FittsFile(path, meta, frames, path.stem.removeprefix("fitts_"))
    task = meta.get("task") or {}
    blocks = task.get("blocks") or []
    trials = task.get("trials") or []
    old_format = (not blocks or "practice" not in blocks[0]
                  or (trials and "refused_taps" not in trials[0]))
    if old_format and started_iso(f) <= FREEZE_AFTER:
        raise ValueError(f"{path.name} was written by an older version of the task (a pilot): not analysable")
    meta.setdefault("task", task)
    task.setdefault("blocks", blocks)
    task.setdefault("trials", trials)
    return f


def started_iso(f: FittsFile) -> str:
    s = f.started
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}"


def camera_key(f: FittsFile) -> tuple:
    """Camera name and frame size as recorded in the conditions block (the reference for 'camera unchanged')."""
    cam = (cond.conditions_block(f.meta, f.path) or {}).get("camera") or {}
    return cam.get("name"), cam.get("width"), cam.get("height")


def qualifies(f: FittsFile, freeze_after: str = FREEZE_AFTER, reference_camera: tuple | None = None) -> str | None:
    """None if the session counts, else the reason it does not (the frozen data rule plus amendment 1).
    ``reference_camera`` = ``camera_key`` of the first post-freeze file; a different camera or frame size fails."""
    t = f.task
    if started_iso(f) <= freeze_after:
        return "started before the freeze commit"
    if reference_camera is not None and camera_key(f) != reference_camera:
        return f"camera or frame size {camera_key(f)} differs from the first session's {reference_camera}"
    if t.get("aborted"):
        return "the run was aborted"
    block = cond.conditions_block(f.meta, f.path)
    if (block or {}).get("person_id") != "p01":
        return "person id is not p01"
    verdict = cond.verdict(cond.validate(block))
    if str(verdict).lower() != "pass":
        return f"condition tags {str(verdict).upper()}, not PASS"
    fit = t.get("calibration_fit_deg")
    if fit is None or fit > MAX_FIT_DEG:
        return f"calibration fit {fit} deg (> {MAX_FIT_DEG} or missing)"
    tracked = (f.meta.get("summary") or {}).get("tracked", 0.0)
    if tracked < MIN_TRACKED:
        return f"face tracked in {tracked:.0%} of the task frames (< {MIN_TRACKED:.0%})"
    scored = [b for b in t["blocks"] if not b["practice"]]
    if len(scored) != N_SCORED_BLOCKS:
        return f"{len(scored)} scored blocks finished (need {N_SCORED_BLOCKS}); the run ended early"
    bad = [b["index"] for b in scored if not b["seat_ok_during_block"]]
    if bad:
        return f"blocks {bad} fail the seat rule"
    for seat in ("centre", "moved"):
        n = len(_rows(f, method="gaze_jump", seat=seat, jumped=True))
        if n < MIN_JUMPS_PER_SEAT:
            return f"{n} accepted jumps in the {seat} seat (< {MIN_JUMPS_PER_SEAT})"
    return None


@dataclass
class Selection:
    counted: list[FittsFile]
    notes: list[str]
    stopped: str | None = None            # why the run has no verdict (too many failed sessions), else None


def select(files: list[FittsFile], freeze_after: str = FREEZE_AFTER) -> Selection:
    """The test set: post-freeze files in time order; a 3rd file on one day is skipped (not a failure); a file
    that does not qualify is a failure; a 4th failure before the 5th qualifying session stops the run."""
    post = sorted((f for f in files if started_iso(f) > freeze_after), key=lambda f: f.started)
    sel = Selection([], [])
    if not post:
        return sel
    ref, per_day, failed = camera_key(post[0]), {}, 0
    for f in post:
        day = f.started[:8]
        per_day[day] = per_day.get(day, 0) + 1
        if per_day[day] > MAX_PER_DAY:
            sel.notes.append(f"{f.path.name}: not counted, session no. {per_day[day]} on {day} (at most {MAX_PER_DAY})")
            continue
        if len(sel.counted) >= N_SESSIONS:
            break
        why = qualifies(f, freeze_after, ref)
        if why is None:
            sel.counted.append(f)
            continue
        failed += 1
        sel.notes.append(f"{f.path.name}: FAILED ({why}); failure {failed} of {MAX_FAILED}")
        if failed > MAX_FAILED:
            sel.stopped = (f"{failed} failed sessions before {N_SESSIONS} qualifying ones: no verdict "
                           f"({len(sel.counted)} qualifying so far)")
            break
    return sel


def _rows(f: FittsFile, *, method: str | None = None, seat: str | None = None, layout: str | None = None,
          jumped: bool | None = None) -> list[tuple[dict, dict]]:
    """(trial, block) pairs of analysed trials: scored blocks only, never a block's first target."""
    blocks = f.task["blocks"]
    out = []
    for tr in f.task["trials"]:
        if tr["block"] >= len(blocks):      # an aborted run's unfinished block has no entry
            continue
        b = blocks[tr["block"]]
        if b["practice"] or tr["index"] == b["first_trial"]:
            continue
        if (method and b["method"] != method) or (seat and b["seat"] != seat) or (layout and b["layout"] != layout):
            continue
        if jumped is not None and (tr["warp_target"] is not None) != jumped:
            continue
        out.append((tr, b))
    return out


def jump_error_cm(f: FittsFile, tr: dict, clamped: bool = False) -> float | None:
    """Distance (cm) from the jump's aim point (or, ``clamped``, its landing) to the target centre."""
    p = tr["warp_landing"] if clamped else tr["warp_target"]
    if p is None:
        return None
    return math.hypot(p[0] - tr["center"][0], p[1] - tr["center"][1]) * f.cm_per_px


def cells_a(f: FittsFile, clamped: bool = False) -> list[Cell]:
    """H7a cells (layout, width): (centre-seat errors, moved-seat errors), cm, gaze-jump trials with a jump."""
    cells: list[Cell] = []
    for layout in ("ring", "line"):
        for width in sorted({b["width_px"] for b in f.task["blocks"] if not b["practice"]}):
            arms = []
            for seat in ("centre", "moved"):
                v = [jump_error_cm(f, tr, clamped) for tr, b in _rows(f, method="gaze_jump", seat=seat, layout=layout)
                     if b["width_px"] == width]
                arms.append(np.array([x for x in v if x is not None], float))
            if min(len(arms[0]), len(arms[1])) >= MIN_CELL_TRIALS:
                cells.append((arms[0], arms[1]))
    return cells


def cells_b(f: FittsFile) -> list[Cell]:
    """H7b cells (width): (trackpad times, gaze-jump times), ms, hit trials >= MIN_MT_MS, long layout, centre seat."""
    cells: list[Cell] = []
    for width in sorted({b["width_px"] for b in f.task["blocks"] if not b["practice"]}):
        arms = []
        for method in ("trackpad", "gaze_jump"):
            v = [tr["movement_time_ms"] for tr, b in _rows(f, method=method, seat="centre", layout="line")
                 if b["width_px"] == width and tr["hit"] and (tr["movement_time_ms"] or 0.0) >= MIN_MT_MS]
            arms.append(np.array(v, float))
        if min(len(arms[0]), len(arms[1])) >= 1:
            cells.append((arms[0], arms[1]))
    return cells


def cell_difference(cells: list[Cell]) -> float:
    """Mean over cells of mean(arm 1) - mean(arm 0); NaN if there is no cell."""
    return float(np.mean([c[1].mean() - c[0].mean() for c in cells])) if cells else math.nan


def miss_rate_pp(f: FittsFile) -> float:
    """Miss rate (misses + time-outs) of gaze-jump minus trackpad, percentage points, long layout, centre seat."""
    rate = []
    for method in ("trackpad", "gaze_jump"):
        r = _rows(f, method=method, seat="centre", layout="line")
        rate.append(100.0 * np.mean([not tr["hit"] for tr, _ in r]) if r else math.nan)
    return float(rate[1] - rate[0])


@dataclass
class SessionStats:
    name: str
    d_a: float
    d_b: float
    miss_pp: float
    cells_a: list[Cell] = field(repr=False)
    cells_b: list[Cell] = field(repr=False)
    reported: dict = field(default_factory=dict)


def session_stats(f: FittsFile) -> SessionStats:
    ca, cb = cells_a(f), cells_b(f)
    return SessionStats(f.path.name, cell_difference(ca), cell_difference(cb), miss_rate_pp(f), ca, cb, reported(f))


def reported(f: FittsFile) -> dict:
    """The protocol's 'reported, not criteria' numbers that need no extra model (descriptive)."""
    out: dict = {}
    for seat in ("centre", "moved"):
        jt = _rows(f, method="gaze_jump", seat=seat)
        jumped = [tr for tr, _ in jt if tr["warp_target"] is not None]
        refused = [tr for tr, _ in jt if tr["warp_target"] is None and tr["refused_taps"]]
        err = np.array([jump_error_cm(f, tr) for tr in jumped], float)
        clamped = np.array([jump_error_cm(f, tr, True) for tr in jumped if tr["warp_landing"] is not None], float)
        off = np.array([[(tr["warp_target"][0] - tr["center"][0]) * f.cm_per_px,
                         (tr["warp_target"][1] - tr["center"][1]) * f.cm_per_px] for tr in jumped], float)
        out[seat] = {
            "n_jumps": len(jumped), "refusal_rate": len(refused) / len(jt) if jt else math.nan,
            "aim_error_cm_mean": float(err.mean()) if len(err) else math.nan,
            "aim_error_cm_median": float(np.median(err)) if len(err) else math.nan,
            "aim_error_cm_p90": float(np.percentile(err, 90)) if len(err) else math.nan,
            "clamped_error_cm_mean": float(clamped.mean()) if len(clamped) else math.nan,
            "share_off_display": float(np.mean([_off_display(f, tr["warp_target"]) for tr in jumped])) if jumped else math.nan,
            "bias_cm": off.mean(axis=0).tolist() if len(off) else None,
            "spread_cm": off.std(axis=0).tolist() if len(off) else None}
    out["d_a_clamped"] = cell_difference(cells_a(f, clamped=True))
    hits = [(tr, b) for tr, b in _rows(f, seat="centre", layout="line") if tr["hit"]]
    out["fast_hits_excluded"] = {m: sum(1 for tr, b in hits if b["method"] == m and (tr["movement_time_ms"] or 0) < MIN_MT_MS)
                                 for m in ("trackpad", "gaze_jump")}
    out["widths_used_b"] = len(cells_b(f))
    out["dots_head_minus_eyes_cm"] = dot_difference_cm(f)
    return out


def _off_display(f: FittsFile, p: list[float]) -> bool:
    """The aim point lies outside the display, so the jump was clamped to its edge."""
    s = f.meta["screen"]
    return not (0 <= p[0] <= s["width_px"] - 1 and 0 <= p[1] <= s["height_px"] - 1)


def dot_difference_cm(f: FittsFile) -> float:
    """Dot block: per dot the mean distance (cm) of the raw prediction from the dot, 700 ms after onset to the
    end; head-turn mean minus eyes-only mean."""
    t, raw = f.frames[:, 0], f.frames[:, 1:3]
    by_mode: dict[str, list[float]] = {}
    for d in f.task.get("dots", []):
        sel = (t >= d["t_start_ms"] + DOT_SETTLE_MS) & (t <= d["t_end_ms"]) & ~np.isnan(raw).any(axis=1)
        if sel.sum() == 0:
            continue
        dist = np.hypot(raw[sel, 0] - d["xy"][0], raw[sel, 1] - d["xy"][1]).mean() * f.cm_per_px
        by_mode.setdefault(d["mode"], []).append(float(dist))
    if not {"eyes", "head"} <= set(by_mode):
        return math.nan
    return float(np.mean(by_mode["head"]) - np.mean(by_mode["eyes"]))


def bootstrap_two_level(sessions: list[list[Cell]], n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Bootstrap distribution of the mean over sessions of the per-session difference: resample the sessions
    with replacement, then the trials inside each arm of each cell of each drawn session (independently for a
    session drawn twice). Sessions weigh equally."""
    rng = np.random.default_rng(seed)
    s = len(sessions)

    def one(cells: list[Cell]) -> float:
        diffs = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for a, b in cells]
        return float(np.mean(diffs))

    return np.array([np.mean([one(sessions[i]) for i in rng.integers(0, s, s)]) for _ in range(n_boot)])


def t_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, float)
    half = sps.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return float(v.mean() - half), float(v.mean() + half)


def decide(stats: list[SessionStats], n_boot: int = N_BOOT) -> dict:
    """The frozen criteria for H7a and H7b over the first N_SESSIONS qualifying sessions."""
    bad = [s.name for s in stats if math.isnan(s.d_a) or math.isnan(s.d_b)]
    if bad:
        raise ValueError(f"d_a or d_b is undefined for {bad}: stop and amend the protocol before computing anything")
    da = np.array([s.d_a for s in stats])
    db = np.array([s.d_b for s in stats])
    miss = np.array([s.miss_pp for s in stats])
    ba = bootstrap_two_level([s.cells_a for s in stats], n_boot)
    bb = bootstrap_two_level([s.cells_b for s in stats], n_boot, seed=SEED)
    ci_a, ci_b = tuple(np.percentile(ba, [2.5, 97.5])), tuple(np.percentile(bb, [2.5, 97.5]))
    t_a, t_b = t_interval(da), t_interval(db)
    crit_a = {"percentile CI lower > 0": ci_a[0] > 0, "t-interval lower > 0": t_a[0] > 0,
              f"mean d_a >= {EFFECT_A_CM} cm": da.mean() >= EFFECT_A_CM,
              f"d_a > 0 in >= {CONSISTENT} of {len(da)} sessions": int((da > 0).sum()) >= CONSISTENT}
    crit_b = {"percentile CI upper < 0": ci_b[1] < 0, "t-interval upper < 0": t_b[1] < 0,
              f"mean d_b <= {EFFECT_B_MS:.0f} ms": db.mean() <= EFFECT_B_MS,
              f"d_b < 0 in >= {CONSISTENT} of {len(db)} sessions": int((db < 0).sum()) >= CONSISTENT,
              f"mean miss-rate difference <= +{MISS_MARGIN_PP:.0f} pp": miss.mean() <= MISS_MARGIN_PP}
    return {"H7a": {"per_session": da.tolist(), "mean": float(da.mean()), "ci": ci_a, "t_ci": t_a,
                    "criteria": crit_a, "supported": all(crit_a.values())},
            "H7b": {"per_session": db.tolist(), "mean": float(db.mean()), "ci": ci_b, "t_ci": t_b,
                    "miss_pp_per_session": miss.tolist(), "miss_pp_mean": float(miss.mean()),
                    "criteria": crit_b, "supported": all(crit_b.values())}}
