"""Cursor mode's state and decisions, free of OS calls (the script wires camera, overlay and cursor).

Per frame: gaze prediction, One-Euro smoothing, input-range check and head shift, all logged. Per
input event: warp (jump the cursor to the smoothed gaze, unless the gaze is not trustworthy right now),
toggle, recalibrate, quit; every click is logged with its position. ``save`` writes the whole session
(frames + events) to one npz, so the online model can be developed and judged offline (record and
replay); ``summary`` gives the results-log row.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.filters import RUN_FILTER, make_filter
from gaze_cursor.livetest import head_moved, to_deg
from gaze_cursor.sinks.cursor import InputEvent

OOD_Z = 4.0                 # same input-range rule as Run
STALE_MS = 250.0            # no warp if the last good gaze sample is older than this
LABEL_WINDOW_MS = (-400.0, -100.0)   # the fixation that guided a click, before the hand moved
EVENT_CODES = {"warp": 0, "toggle": 1, "quit": 2, "click": 3, "recalibrate": 4}


def window_median(t: np.ndarray, values: np.ndarray, t_event: float,
                  window: tuple[float, float] = LABEL_WINDOW_MS) -> np.ndarray | None:
    """Median of ``values`` (rows, NaN = missing) whose time lies in ``t_event + window``; None if none."""
    sel = (t >= t_event + window[0]) & (t <= t_event + window[1])
    rows = values[sel]
    rows = rows[~np.isnan(rows).any(axis=1)] if rows.ndim == 2 else rows[~np.isnan(rows)]
    return np.median(rows, axis=0) if len(rows) else None


@dataclass
class CursorSession:
    cfg: ScreenConfig
    model: object                       # anything with predict(x) -> (n, 2) points
    x_train: np.ndarray                 # calibration features (input-range check)
    ref_center: dict | None             # posture during calibration (head shift)
    calib: str                          # calibration file name (for replay)
    enabled: bool = True
    dot: np.ndarray | None = None
    raw: np.ndarray | None = None
    worst_z: float = 0.0
    shift: tuple[float, float, float] | None = None
    last_good_ms: float = -np.inf
    message: str = ""
    frames: list[list[float]] = field(default_factory=list)
    events: list[list[float]] = field(default_factory=list)
    started: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))

    def __post_init__(self) -> None:
        self.smooth = make_filter(RUN_FILTER)
        self._set_range(self.x_train)

    def _set_range(self, x: np.ndarray) -> None:
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0) + 1e-9

    def recalibrated(self, model: object, x_train: np.ndarray, ref_center: dict | None, calib: str) -> None:
        self.model, self.x_train, self.ref_center, self.calib = model, x_train, ref_center, calib
        self._set_range(x_train)
        self.smooth.reset()
        self.dot = self.raw = None
        self.worst_z, self.last_good_ms = 0.0, -np.inf

    # --- per frame -----------------------------------------------------------------------------
    def frame(self, t_ms: float, x: np.ndarray | None, shift: tuple[float, float, float] | None) -> None:
        """One camera frame: ``x`` = gaze features (1, n) or None (no face / blink)."""
        self.shift = shift
        self.raw = None
        if x is not None:
            self.raw = np.asarray(self.model.predict(x))[0]
            self.dot = self.smooth.update(t_ms, self.raw)
            self.worst_z = float(np.max(np.abs((x - self.mu) / self.sd)))
            self.last_good_ms = t_ms
        xr = np.full(4, np.nan) if x is None else np.asarray(x, float).ravel()[:4]
        raw = self.raw if self.raw is not None else (np.nan, np.nan)
        dot = self.dot if (self.dot is not None and x is not None) else (np.nan, np.nan)
        sh = shift if shift is not None else (np.nan, np.nan, np.nan)
        self.frames.append([t_ms, *raw, *dot, *xr, float(x is not None), *sh, float(self.enabled), self.worst_z])

    # --- input ---------------------------------------------------------------------------------
    def warp_target(self, now_ms: float) -> tuple[tuple[float, float] | None, str]:
        """Where to warp now, or None with the reason it would not be trustworthy."""
        if self.dot is None or now_ms - self.last_good_ms > STALE_MS:
            return None, "face not tracked"
        if self.worst_z > OOD_Z:
            return None, "eyes outside the calibrated range: recalibrate (Ctrl+Option+R)"
        return (float(self.dot[0]), float(self.dot[1])), ""

    def handle(self, event: InputEvent, now_ms: float) -> tuple[str, tuple[float, float] | None]:
        """Returns the action for the script: ("warp", xy), ("quit", None), ("recalibrate", None) or
        ("none", None). Logs every event."""
        action, xy, ok = "none", None, 0.0
        if event.kind == "toggle":
            self.enabled = not self.enabled
            self.message = "gaze warp ON" if self.enabled else "gaze warp OFF"
        elif event.kind == "quit":
            action = "quit"
        elif event.kind == "recalibrate":
            action = "recalibrate"
        elif event.kind == "warp" and self.enabled:
            xy, reason = self.warp_target(now_ms)
            if xy is None:
                self.message = f"no jump: {reason}"
            else:
                action, ok, self.message = "warp", 1.0, ""
        exy = event.xy if event.xy is not None else (xy if xy is not None else (np.nan, np.nan))
        button = {"left": 1.0, "right": 2.0}.get(event.button or "", 0.0)
        self.events.append([event.t_ms, float(EVENT_CODES[event.kind]), *exy, button, ok, float(self.enabled)])
        return action, xy

    # --- display -------------------------------------------------------------------------------
    def moved(self) -> bool:
        return self.shift is not None and head_moved(*self.shift)

    def status_lines(self) -> list[tuple[str, str]]:
        """(text, level) with level in ok / warn / off."""
        if not self.enabled:
            return [("gaze warp OFF (Ctrl+Option+G)", "off")]
        lines = [("gaze warp ON: tap Right Command", "ok")]
        if self.dot is None or self.raw is None:
            lines.append(("face not tracked", "warn"))
        if self.moved():
            lines.append(("you moved since calibrating (Ctrl+Option+R)", "warn"))
        if self.message:
            lines.append((self.message, "warn" if self.message.startswith("no jump") else "ok"))
        return lines

    # --- record and replay ---------------------------------------------------------------------
    FRAME_COLUMNS = ("t_ms", "raw_x", "raw_y", "dot_x", "dot_y", "f0", "f1", "f2", "f3", "tracked",
                     "head_ratio", "head_dx_cm", "head_dy_cm", "enabled", "worst_z")
    EVENT_COLUMNS = ("t_ms", "kind", "x", "y", "button", "warped", "enabled")

    def click_errors_deg(self) -> np.ndarray:
        """Per click: distance (deg) between the click and the median raw gaze in the label window.
        A live accuracy estimate (you usually look where you click); NaN if no gaze in the window."""
        if not self.frames or not self.events:
            return np.zeros(0)
        f, e = np.array(self.frames, float), np.array(self.events, float)
        clicks = e[e[:, 1] == EVENT_CODES["click"]]
        out = []
        for t, _, x, y, *_ in clicks:
            g = window_median(f[:, 0], f[:, 1:3], t)
            out.append(np.nan if g is None else float(to_deg(np.hypot(g[0] - x, g[1] - y), self.cfg)))
        return np.array(out)

    def summary(self) -> dict:
        e = np.array(self.events, float) if self.events else np.zeros((0, 7))
        warps = e[e[:, 1] == EVENT_CODES["warp"]] if len(e) else e
        err = self.click_errors_deg()
        err = err[~np.isnan(err)]
        f = np.array(self.frames, float) if self.frames else np.zeros((0, len(self.FRAME_COLUMNS)))
        return {"frames": len(f), "minutes": float((f[-1, 0] - f[0, 0]) / 60000) if len(f) > 1 else 0.0,
                "tracked": float(f[:, 9].mean()) if len(f) else 0.0,
                "warps": int(warps[:, 5].sum()) if len(warps) else 0,
                "warps_refused": int((warps[:, 5] == 0).sum()) if len(warps) else 0,
                "clicks": int((e[:, 1] == EVENT_CODES["click"]).sum()) if len(e) else 0,
                "click_gaze_deg_median": float(np.median(err)) if len(err) else None,
                "click_gaze_deg_mean": float(err.mean()) if len(err) else None,
                "click_gaze_n": int(len(err))}

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"cursor_{self.started}.npz"
        meta = {"kind": "cursor_session", "calibration": self.calib, "filter": RUN_FILTER,
                "screen": self.cfg.to_dict(), "started": self.started, "frame_columns": self.FRAME_COLUMNS,
                "event_columns": self.EVENT_COLUMNS, "event_codes": EVENT_CODES,
                "label_window_ms": LABEL_WINDOW_MS, "summary": self.summary()}
        np.savez_compressed(path, frames=np.array(self.frames, float).reshape(-1, len(self.FRAME_COLUMNS)),
                            events=np.array(self.events, float).reshape(-1, len(self.EVENT_COLUMNS)),
                            meta=json.dumps(meta))
        return path
