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
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gaze_cursor.conditions import model_version
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.faces import QUALITY, encode_crop, faces_path_for, save_faces
from gaze_cursor.filters import RUN_FILTER, make_filter
from gaze_cursor.livetest import head_moved, px_per_deg, to_deg
from gaze_cursor.online import GATE_DEG, OnlineRidge, click_label
from gaze_cursor.sinks.cursor import InputEvent

OOD_Z = 4.0                 # same input-range rule as Run
STALE_MS = 250.0            # no warp if the last good gaze sample is older than this
LABEL_WINDOW_MS = (-400.0, -100.0)   # the fixation that guided a click, before the hand moved
EVENT_CODES = {"warp": 0, "toggle": 1, "quit": 2, "click": 3, "recalibrate": 4}
FACE_PER_CLICK = 3          # face crops kept per click, evenly spread over LABEL_WINDOW_MS
FACE_MAX_BYTES = 150_000_000   # stop keeping crops beyond this (a long session); the click log is unaffected


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
    # [frame time from which it is active (None until the next frame), calibration file]: replay needs
    # to know which calibration produced each part of the session.
    calibrations: list[list] = field(default_factory=list)
    learn: bool = False                 # update ``model`` (an OnlineRidge) from the user's clicks
    resumed: str = ""                   # non-empty: the calibration came from an earlier run (its age, in words)
    learned: int = 0                    # clicks the model has learned from since the last (re)calibration
    first_ms: float | None = None
    last_ms: float = 0.0
    banner: str = ""                    # e.g. "DEMO: nothing is saved", shown first on the chip
    calibrated_at: float = field(default_factory=time.time)   # epoch s of the 9 dots (carried over on resume)
    conditions: dict | None = None      # the meta["conditions"] block of the recording (None: demo / tests)
    model_versions: list[str | None] = field(default_factory=list)   # starting-state hash per calibration
    state_id: str | None = None         # model_history id of the last archived state (parent of the next)
    keep_faces: bool = False            # keep face crops from the label window before each click (local only)
    face_jpegs: list[bytes] = field(default_factory=list)
    face_rows: list[list[float]] = field(default_factory=list)   # [event row, frame time, ms before the click]
    face_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    face_bytes: int = 0

    def __post_init__(self) -> None:
        self._recent: deque = deque()   # (t_ms, crop, box) of the last frames, just enough for the window
        self.smooth = make_filter(RUN_FILTER)
        self._set_range(self.x_train)
        self.calibrations.append([None, self.calib])
        self.model_versions.append(self._model_version())

    def _model_version(self) -> str | None:
        """Hash of the starting model state (an OnlineRidge); None for a model that is not one."""
        m = self.model
        if not isinstance(m, OnlineRidge):
            return None
        return model_version(m.mean, m.scale, m.theta, m.P)

    def _set_range(self, x: np.ndarray) -> None:
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0) + 1e-9

    def recalibrated(self, model: object, x_train: np.ndarray, ref_center: dict | None, calib: str) -> None:
        self.model, self.x_train, self.ref_center, self.calib = model, x_train, ref_center, calib
        self._set_range(x_train)
        self.smooth.reset()
        self.dot = self.raw = None
        self.worst_z, self.last_good_ms = 0.0, -np.inf
        self.calibrations.append([None, calib])
        self.model_versions.append(self._model_version())
        self.calibrated_at = time.time()
        self.state_id = None            # a new calibration starts a new chain
        self.learned = 0

    # --- per frame -----------------------------------------------------------------------------
    def frame(self, t_ms: float, x: np.ndarray | None, shift: tuple[float, float, float] | None) -> None:
        """One camera frame: ``x`` = gaze features (1, n) or None (no face / blink)."""
        if self.calibrations[-1][0] is None:
            self.calibrations[-1][0] = float(t_ms)
        self.shift = shift
        self.raw = None
        self.last_ms = float(t_ms)
        if self.first_ms is None:
            self.first_ms = float(t_ms)
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

    def offer_crop(self, t_ms: float, image: np.ndarray, box: tuple[int, int, int, int],
                   valid: bool = True) -> None:
        """Remember this frame's face crop (BGR ``image`` cut at ``box``) for a click that may follow;
        ``valid`` = gaze features were extracted from this frame (not a blink / bad frame).
        Only the last ``-LABEL_WINDOW_MS[0]`` ms are held; nothing is encoded until a click needs it."""
        if not self.keep_faces:
            return
        x0, y0, x1, y1 = box
        self._recent.append((float(t_ms), image[y0:y1, x0:x1].copy(), box, bool(valid)))
        while self._recent and self._recent[0][0] < t_ms + LABEL_WINDOW_MS[0] - 100.0:
            self._recent.popleft()

    def _keep_crops(self, t_click: float, event_row: int) -> None:
        """Encode up to FACE_PER_CLICK crops from the label window before this click."""
        if self.face_bytes >= FACE_MAX_BYTES:
            return
        inside = [c for c in self._recent
                  if t_click + LABEL_WINDOW_MS[0] <= c[0] <= t_click + LABEL_WINDOW_MS[1]]
        if not inside:
            return
        picks = sorted({int(round(i)) for i in np.linspace(0, len(inside) - 1, min(FACE_PER_CLICK, len(inside)))})
        for i in picks:
            t, crop, box, valid = inside[i]
            jpg = encode_crop(crop, (0, 0, crop.shape[1], crop.shape[0]), QUALITY)
            self.face_jpegs.append(jpg)
            self.face_boxes.append(box)
            self.face_rows.append([float(event_row), t, t - t_click, float(valid)])
            self.face_bytes += len(jpg)

    # --- input ---------------------------------------------------------------------------------
    def warp_target(self, now_ms: float) -> tuple[tuple[float, float] | None, str]:
        """Where to warp now, or None with the reason it would not be trustworthy."""
        if self.dot is None or now_ms - self.last_good_ms > STALE_MS:
            return None, "face not tracked"
        if self.worst_z > OOD_Z:
            return None, "eyes outside the calibrated range: recalibrate (Cmd+Option+R)"
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
        if event.kind == "click" and event.xy is not None and self.learn:
            ok = float(self._learn_from_click(event))      # in the log, a click's "warped" column = learned
        exy = event.xy if event.xy is not None else (xy if xy is not None else (np.nan, np.nan))
        button = {"left": 1.0, "right": 2.0}.get(event.button or "", 0.0)
        n_updates = float(getattr(self.model, "n_updates", np.nan))   # model state at this event, for replay
        self.events.append([event.t_ms, float(EVENT_CODES[event.kind]), *exy, button, ok, float(self.enabled),
                            n_updates])
        if event.kind == "click" and event.xy is not None and self.keep_faces:
            self._keep_crops(event.t_ms, len(self.events) - 1)
        return action, xy

    def _learn_from_click(self, event: InputEvent) -> bool:
        """Round 4's rule (H6, supported): the click labels the fixation just before it; the model learns
        from it unless the click is further than GATE_DEG from what the model predicted (a stray click)."""
        if not isinstance(self.model, OnlineRidge) or not self.frames or event.xy is None:
            return False
        recent = np.array(self.frames[-60:], float)
        x = click_label(recent[:, 0], recent[:, 5:9], event.t_ms, LABEL_WINDOW_MS)
        if x is None:
            return False
        target = np.asarray(event.xy, float)
        if np.hypot(*(self.model.predict(x)[0] - target)) / px_per_deg(self.cfg) > GATE_DEG:
            return False
        self.model.update(x, target)
        self.learned += 1
        return True

    # --- display -------------------------------------------------------------------------------
    def elapsed_min(self) -> float:
        """Minutes of use by the camera frame clock (a closed lid does not count)."""
        return 0.0 if self.first_ms is None else (self.last_ms - self.first_ms) / 60000

    def moved(self) -> bool:
        return self.shift is not None and head_moved(*self.shift)

    def status_lines(self) -> list[tuple[str, str]]:
        """(text, level) with level in ok / warn / off."""
        head = [(self.banner, "warn")] if self.banner else []
        if not self.enabled:
            return head + [("gaze warp OFF (Cmd+Option+G)", "off")]
        lines = head + [("gaze warp ON: tap Right Command", "ok")]
        if self.dot is None or self.raw is None:
            lines.append(("face not tracked", "warn"))
        if self.moved():
            lines.append(("you moved since calibrating (Cmd+Option+R)", "warn"))
        if self.message:
            lines.append((self.message, "warn" if self.message.startswith("no jump") else "ok"))
        minutes = int(self.elapsed_min())
        info = f"on {minutes // 60}:{minutes % 60:02d} h" if minutes >= 60 else f"on {minutes} min"
        if self.learn:
            info += f", learned from {self.learned} clicks"
        lines.append((info, "off"))
        if self.resumed and minutes < 1:
            lines.append((f"using your calibration from {self.resumed}", "off"))
        return lines

    def status_key(self) -> tuple[tuple[str, str], ...]:
        """The status lines that matter, without the running 'on N min' / 'using your calibration' info:
        when this changes the chip is worth showing (it is otherwise hidden, see ``--chip``)."""
        return tuple(l for l in self.status_lines() if not l[0].startswith(("on ", "using your")))

    def lasting_warning(self) -> bool:
        """A warning that stays true until you act (moved, face lost), not a one-off message about a jump."""
        return any(lvl == "warn" and text != self.banner and not text.startswith("no jump")
                   for text, lvl in self.status_key())

    # --- record and replay ---------------------------------------------------------------------
    FRAME_COLUMNS = ("t_ms", "raw_x", "raw_y", "dot_x", "dot_y", "f0", "f1", "f2", "f3", "tracked",
                     "head_ratio", "head_dx_cm", "head_dy_cm", "enabled", "worst_z")
    EVENT_COLUMNS = ("t_ms", "kind", "x", "y", "button", "warped", "enabled", "n_updates")

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
        e = np.array(self.events, float) if self.events else np.zeros((0, len(self.EVENT_COLUMNS)))
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
                "click_gaze_n": int(len(err)), "clicks_learned": self.learned}

    def _conditions(self) -> dict | None:
        """The conditions block with the first calibration's starting model hash filled in."""
        if self.conditions is None:
            return None
        return {**self.conditions, "model_version": self.model_versions[0] if self.model_versions else None}

    def save(self, directory: Path) -> Path:
        """Write the whole session (atomically: a crash mid-write keeps the previous autosave)."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"cursor_{self.started}.npz"
        tmp = directory / f".cursor_{self.started}.tmp.npz"
        meta = {"kind": "cursor_session", "calibration": self.calib, "calibrations": self.calibrations,
                "filter": RUN_FILTER,
                "screen": self.cfg.to_dict(), "started": self.started, "frame_columns": self.FRAME_COLUMNS,
                "event_columns": self.EVENT_COLUMNS, "event_codes": EVENT_CODES,
                "label_window_ms": LABEL_WINDOW_MS, "summary": self.summary(),
                "event_notes": {"n_updates": "model update count AFTER this event (a learned click's row includes it)",
                                "click.warped": "1 = the model learned from this click (learning on); warp rows: 1 = jumped"},
                "conditions": self._conditions(), "model_versions": self.model_versions,
                "learning": {"on": self.learn, "resumed": self.resumed, "gate_deg": GATE_DEG,
                             "forget": getattr(self.model, "forget", None)},
                "face_crops": self._face_meta(path)}
        np.savez_compressed(tmp, frames=np.array(self.frames, float).reshape(-1, len(self.FRAME_COLUMNS)),
                            events=np.array(self.events, float).reshape(-1, len(self.EVENT_COLUMNS)),
                            meta=json.dumps(meta))
        tmp.replace(path)
        if self.face_jpegs:
            ftmp = directory / f".cursor_{self.started}.faces.tmp.npz"
            save_faces(ftmp, self.face_jpegs, self.face_boxes, extra={
                "event_row": np.array([r[0] for r in self.face_rows], np.int32),
                "t_ms": np.array([r[1] for r in self.face_rows], float),
                "ms_before_click": np.array([r[2] for r in self.face_rows], float),
                "features_valid": np.array([r[3] for r in self.face_rows], bool)})
            ftmp.replace(faces_path_for(path))
        return path

    def _face_meta(self, path: Path) -> dict | None:
        """Describes the face-crop file saved next to the log (None: crops were not kept)."""
        if not self.keep_faces:
            return None
        return {"file": faces_path_for(path).name if self.face_jpegs else None, "rows": len(self.face_jpegs),
                "per_click_max": FACE_PER_CLICK, "window_ms": LABEL_WINDOW_MS, "bytes": self.face_bytes,
                "capped": self.face_bytes >= FACE_MAX_BYTES,
                "arrays": {"event_row": "row of the click in this log's events",
                           "ms_before_click": "frame time minus click time (inside window_ms)",
                           "features_valid": "gaze features were extracted from this frame (filter blinks with it)"},
                "notes": "Close clicks (double-click) can share a frame: group by event_row and drop repeated t_ms. "
                         "Whether the live model learned from a click is its events row (click.warped); a model that "
                         "learned from a click must not be scored on it."}
