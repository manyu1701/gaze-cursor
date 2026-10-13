"""Jitter vs lag of smoothing filters, offline from recorded sessions (M2 step 6). Pure functions.

Recorded sessions keep only the fixation frames of each target (the eyes' travel between dots is
not recorded). Splicing the fixations with contiguous time gives a stream of realistic noisy
predictions in which every target change is an instant step at a known time, so the lag a filter
adds can be measured exactly. The selection rule is in PREREGISTRATIONS.md (written before tuning).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.metrics import cm_to_deg, px_to_cm
from gaze_cursor.eval.session import Session
from gaze_cursor.eval.stats import bootstrap_ci
from gaze_cursor.quick import QUICK_DOTS, fit_quick, quick_targets

FRAME_MS = 33.0
STEADY_MS = 300.0      # jitter and steady error are measured from this long after a fixation starts
MIN_JUMP_DEG = 3.0     # lag is only measured on jumps at least this large


@dataclass(frozen=True)
class Stream:
    t_ms: np.ndarray       # (n,) spliced, contiguous time
    raw: np.ndarray        # (n, 2) unfiltered predictions, screen points
    target: np.ndarray     # (n, 2) target of each frame
    fix: np.ndarray        # (n,) fixation index 0..m-1, in order
    start_ms: np.ndarray   # (m,) start time of each fixation (the step time for fixation j > 0)


def splice(t_ms: np.ndarray, pred: np.ndarray, target: np.ndarray, key: np.ndarray) -> Stream:
    """Join the fixations (frames grouped by ``key``) in order of their first frame; each fixation's
    frames keep their own spacing and the next starts FRAME_MS after the previous one's last frame."""
    order = sorted(np.unique(key), key=lambda k: t_ms[key == k].min())
    ts, ps, tg, fx, starts = [], [], [], [], []
    offset = 0.0
    for j, k in enumerate(order):
        m = np.flatnonzero(key == k)
        m = m[np.argsort(t_ms[m])]
        rel = t_ms[m] - t_ms[m[0]]
        starts.append(offset)
        ts.append(offset + rel); ps.append(pred[m]); tg.append(target[m]); fx.append(np.full(len(m), j))
        offset += rel[-1] + FRAME_MS
    return Stream(np.concatenate(ts), np.concatenate(ps), np.concatenate(tg), np.concatenate(fx), np.array(starts))


def session_stream(s: Session, screen: ScreenConfig) -> Stream:
    """The Run model (fresh gaze_a10 on the 9 quick dots) predicting every valid frame of the other
    targets of ``s``, spliced in recorded order."""
    x, y, idx, _ = s.select_all("gaze")
    t = s.t_ms[s.valid].astype(float)
    dots = [q.idx for q in quick_targets(screen)][:QUICK_DOTS]
    on = np.isin(idx, dots)
    model = fit_quick(x[on], y[on], screen)
    return splice(t[~on], model.predict(x[~on]), y[~on], idx[~on])


def run_filter(stream: Stream, filt) -> np.ndarray:
    """Filter the whole stream in time order (one reset at the start, as in Run)."""
    filt.reset()
    return np.array([filt.update(t, p) for t, p in zip(stream.t_ms, stream.raw)])


def _deg(px: np.ndarray | float, screen: ScreenConfig) -> np.ndarray:
    return np.asarray(cm_to_deg(px_to_cm(np.asarray(px, dtype=float), screen), screen))


def fixation_metrics(stream: Stream, out: np.ndarray, screen: ScreenConfig) -> dict[str, np.ndarray]:
    """Per fixation: jitter and steady error (deg). Per qualifying step: lag50, lag90 (ms; NaN if
    never reached within the fixation) and overshoot (deg)."""
    m = len(stream.start_ms)
    jitter, err, lag50, lag90, over = [], [], [], [], []
    steady_raw = []
    for j in range(m):
        f = stream.fix == j
        steady = f & (stream.t_ms >= stream.start_ms[j] + STEADY_MS)
        if steady.sum() < 3:
            steady = f
        steady_raw.append(stream.raw[steady].mean(axis=0))
        jitter.append(float(_deg(np.sqrt(out[steady].var(axis=0).sum()), screen)))
        err.append(float(_deg(np.linalg.norm(out[steady] - stream.target[steady], axis=1), screen).mean()))
    for j in range(1, m):
        a, b = steady_raw[j - 1], steady_raw[j]
        jump = b - a
        size = float(np.linalg.norm(jump))
        if _deg(size, screen) < MIN_JUMP_DEG:
            continue
        f = np.flatnonzero(stream.fix == j)
        prog = (out[f] - a) @ jump / size**2
        dt = stream.t_ms[f] - stream.start_ms[j]
        hit50 = np.flatnonzero(prog >= 0.5); hit90 = np.flatnonzero(prog >= 0.9)
        lag50.append(dt[hit50[0]] if len(hit50) else np.nan)
        lag90.append(dt[hit90[0]] if len(hit90) else np.nan)
        over.append(float(max(prog.max() - 1.0, 0.0) * _deg(size, screen)))
    return {"jitter": np.array(jitter), "err": np.array(err), "lag50": np.array(lag50),
            "lag90": np.array(lag90), "overshoot": np.array(over)}


def summarize(parts: list[dict[str, np.ndarray]], n_boot: int = 2000, seed: int = 0) -> dict:
    """Pool per-fixation metrics over sessions. Lags: median (a never-reached step counts as
    infinitely slow), and the fraction of steps that reached 90%."""
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    lo, hi = bootstrap_ci(cat["jitter"], n_boot, seed)
    l50 = np.where(np.isnan(cat["lag50"]), np.inf, cat["lag50"])
    l90 = np.where(np.isnan(cat["lag90"]), np.inf, cat["lag90"])
    return {"jitter": float(cat["jitter"].mean()), "jitter_ci": (lo, hi), "err": float(cat["err"].mean()),
            "lag50": float(np.median(l50)), "lag90": float(np.median(l90)),
            "reached90": float(np.isfinite(l90).mean()), "overshoot": float(np.median(cat["overshoot"])),
            "n_fix": int(len(cat["jitter"])), "n_steps": int(len(l50))}
