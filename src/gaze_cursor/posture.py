"""Posture check: is the user sitting like they did in the recorded sessions? Pure functions.

The reference is computed from recorded sessions (median and robust spread per metric), so it
updates itself as sessions are added. Lighting is reported but never gates the check: sessions are
recorded in deliberately different lighting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from gaze_cursor.landmarks import head_pose_from_matrix

METRICS = ("pitch", "yaw", "roll", "distance", "x", "y")
LABELS = {"pitch": "head pitch", "yaw": "head yaw", "roll": "head roll",
          "distance": "distance (inter-iris)", "x": "face x in view", "y": "face y in view"}
# Minimum tolerance (one "scale" unit) per metric. HEURISTICS, not measured limits: a metric with
# little spread in the reference sessions would otherwise be impossible to satisfy. ``distance``
# is relative to its reference value, the others are absolute.
FLOORS = {"pitch": 2.5, "yaw": 4.0, "roll": 3.0, "distance": 0.06, "x": 0.05, "y": 0.05}
GOOD_Z, ADJUST_Z = 1.5, 3.0  # |z| <= GOOD_Z: good; <= ADJUST_Z: adjust; beyond: off


def posture_metrics(points: np.ndarray, transform: np.ndarray,
                    image_size: tuple[int, int]) -> dict[str, float]:
    """Head pose (deg, unflipped frame), distance proxy and face center for one frame.

    ``distance`` is the inter-iris distance as a fraction of the frame width (larger = closer).
    ``x``/``y`` are the mean mesh position in the unflipped frame, normalized to [0, 1].
    """
    w, h = image_size
    iod = float(np.linalg.norm((points[468, :2] - points[473, :2]) * (w, h)) / w)
    cx, cy = points[:468, :2].mean(axis=0)
    yaw, pitch, roll = head_pose_from_matrix(transform)
    return {"pitch": pitch, "yaw": yaw, "roll": roll, "distance": iod,
            "x": float(cx), "y": float(cy)}


def lighting_stats(gray: np.ndarray, points: np.ndarray) -> dict[str, float]:
    """Face and background brightness (0-255), their ratio, and the fraction of clipped face pixels."""
    h, w = gray.shape
    xs, ys = points[:468, 0] * w, points[:468, 1] * h
    x0, x1 = int(np.clip(xs.min(), 0, w - 1)), int(np.clip(xs.max(), 1, w))
    y0, y1 = int(np.clip(ys.min(), 0, h - 1)), int(np.clip(ys.max(), 1, h))
    face = gray[y0:y1, x0:x1].astype(float)
    mask = np.ones(gray.shape, bool)
    mask[y0:y1, x0:x1] = False
    bg = float(gray[mask].mean()) if mask.any() else float("nan")
    return {"face": float(face.mean()), "background": bg,
            "ratio": float(face.mean() / bg) if bg else float("nan"),
            "clipped": float((face >= 250).mean())}


def lighting_warning(stats: dict[str, float]) -> str | None:
    """Only extremes are flagged (informational, never blocks)."""
    if stats["face"] < LIGHT_MIN_FACE:
        return "too dark: put a lamp or window in front of you"
    if stats["face"] > 220 or stats["clipped"] > 0.05:
        return "face overexposed"
    return None


def lighting_label(face: float, background: float) -> str:
    """A coarse, human-readable lighting class. Heuristic thresholds, not measured limits."""
    ratio = face / background if background else float("nan")
    if face < 50:
        return "dim"
    if ratio < 0.6:
        return "backlit"
    if ratio >= 0.9 and face >= 100:
        return "front-lit"
    return "mixed"


_NOTE_WORDS = {"dim": "dim", "backlit": "backlit", "backlight": "backlit", "back-lit": "backlit",
               "front-lit": "front-lit", "front lit": "front-lit", "frontlit": "front-lit",
               "front light": "front-lit"}


def note_mismatch(note: str, label: str) -> str | None:
    """A warning if the free-text note names a lighting condition that the captured lighting is not.

    Sessions are classified by the captured lighting, never by the note, so a mislabelled note is a
    recording mistake worth flagging before or right after the recording.
    """
    low = (note or "").lower()
    mentioned = sorted({cls for word, cls in _NOTE_WORDS.items() if word in low})
    if mentioned and label not in mentioned:
        return f"note says {'/'.join(mentioned)} but the lighting is {label}"
    return None


def summarize_lighting(frames: Sequence[dict[str, float]]) -> dict | None:
    """Session-level lighting from per-frame ``lighting_stats`` (JSON-storable), or None."""
    if not frames:
        return None
    face = np.array([f["face"] for f in frames])
    bg = np.array([f["background"] for f in frames])
    clipped = np.array([f["clipped"] for f in frames])
    return {"face": float(face.mean()), "background": float(bg.mean()),
            "ratio": float(face.mean() / bg.mean()), "face_std": float(face.std()),
            "clipped": float(clipped.mean()), "frames": len(frames),
            "label": lighting_label(float(face.mean()), float(bg.mean()))}


@dataclass(frozen=True)
class Reference:
    """Where the user sat in the reference sessions: per-metric center and tolerance scale."""

    center: dict[str, float]
    scale: dict[str, float]
    n_sessions: int = 0
    n_frames: int = 0
    session_names: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_arrays(cls, values: dict[str, np.ndarray], **kw) -> Reference:
        """Median and robust spread (1.4826 * MAD, floored) of per-frame metric arrays."""
        center, scale = {}, {}
        for m in METRICS:
            v = np.asarray(values[m], dtype=float)
            c = float(np.median(v))
            mad = 1.4826 * float(np.median(np.abs(v - c)))
            floor = FLOORS[m] * (abs(c) if m == "distance" else 1.0)
            center[m], scale[m] = c, max(mad, floor)
        return cls(center, scale, **kw)

    @classmethod
    def from_sessions(cls, sessions: Sequence, stride: int = 3) -> Reference:
        """Pool the valid frames of ``sessions`` (every ``stride``-th frame)."""
        cols: dict[str, list[float]] = {m: [] for m in METRICS}
        used = 0
        for s in sessions:
            if s.points is None or s.transform is None:
                continue
            size = tuple(s.meta.get("image_size", (1280, 720)))
            idx = np.flatnonzero(s.valid)[::stride]
            for i in idx:
                for m, v in posture_metrics(s.points[i], s.transform[i], size).items():
                    cols[m].append(v)
            used += 1
        if not cols["pitch"]:
            raise ValueError("no valid frames with landmarks in the reference sessions")
        return cls.from_arrays({m: np.array(v) for m, v in cols.items()}, n_sessions=used,
                               n_frames=len(cols["pitch"]),
                               session_names=tuple(s.path.name for s in sessions))

    @classmethod
    def from_current(cls, frames: Sequence[dict[str, float]]) -> Reference:
        """A reference built from the user's current posture (a short window of recent frames), for
        calibrating in their own setup instead of matching earlier sessions."""
        if len(frames) < STILL_MIN_FRAMES:
            raise ValueError("not enough frames to take the current posture as a reference")
        return cls.from_arrays({m: np.array([f[m] for f in frames]) for m in METRICS},
                               n_sessions=0, n_frames=len(frames), session_names=("current setup",))

    def z(self, metrics: dict[str, float]) -> dict[str, float]:
        return {m: (metrics[m] - self.center[m]) / self.scale[m] for m in METRICS}

    @staticmethod
    def level(z: float) -> str:
        a = abs(z)
        return "good" if a <= GOOD_Z else "adjust" if a <= ADJUST_Z else "off"

    def worst(self, metrics: dict[str, float]) -> tuple[str, str, float]:
        """(metric, level, z) of the metric furthest from the reference."""
        z = self.z(metrics)
        m = max(z, key=lambda k: abs(z[k]))
        return m, self.level(z[m]), z[m]

    def hint(self, metric: str, live: float) -> str:
        """A short instruction for one metric. Directions are only given where unambiguous."""
        d = live - self.center[metric]
        if metric == "distance":
            return "move back a little" if d > 0 else "move closer"
        if metric == "pitch":
            return "adjust screen angle or seat height"
        if metric == "yaw":
            return "face the screen centre"
        if metric == "roll":
            return "level your head"
        return "centre your face in the camera view"


@dataclass
class PostureGate:
    """Debounce: 'in position' after ``hold_s`` of all-good readings; lost after ``release_s`` not good."""

    hold_s: float = 1.0
    release_s: float = 0.5
    in_position: bool = False
    _good_since: float | None = None
    _bad_since: float | None = None

    def update(self, now: float, all_good: bool) -> bool:
        if all_good:
            self._bad_since = None
            if self._good_since is None:
                self._good_since = now
            if not self.in_position and now - self._good_since >= self.hold_s:
                self.in_position = True
        else:
            self._good_since = None
            if self._bad_since is None:
                self._bad_since = now
            if self.in_position and now - self._bad_since >= self.release_s:
                self.in_position = False
        return self.in_position


def summarize_recording(ref: Reference, frames: Sequence[dict[str, float]]) -> dict:
    """How well did the user hold the reference posture during a recording?

    ``frames`` holds one metrics dict per recorded frame. Returns the fraction of frames with
    every metric good, the fraction of frames each metric was not good, the metric that drifted
    most, and a one-line verdict. Purely descriptive: it never changes or filters the data.
    """
    if not frames:
        return {"frames": 0, "fraction_good": 0.0, "verdict": "no face was tracked during the recording",
                "worst_metric": None, "not_good_fraction": {}, "mean_z": {}}
    z = np.array([[ref.z(f)[m] for m in METRICS] for f in frames])
    not_good = np.abs(z) > GOOD_Z
    frac_good = float((~not_good.any(axis=1)).mean())
    per_metric = {m: float(not_good[:, i].mean()) for i, m in enumerate(METRICS)}
    mean_z = {m: float(z[:, i].mean()) for i, m in enumerate(METRICS)}
    worst = max(METRICS, key=lambda m: abs(mean_z[m]) + per_metric[m])
    if frac_good >= 0.9:
        verdict = "good session: you held your position"
    elif frac_good >= 0.7:
        verdict = f"mostly in position; {LABELS[worst]} drifted"
    else:
        verdict = f"you drifted out of position ({LABELS[worst]} most); consider recording again"
    return {"frames": len(frames), "fraction_good": frac_good, "verdict": verdict,
            "worst_metric": worst, "not_good_fraction": per_metric, "mean_z": mean_z}


# --- Live setup checklist -------------------------------------------------------------------
# Head-pose and face-position jitter within one ~1 s fixation in the recorded sessions had p95
# std of about pitch 0.69 deg, yaw 0.48, roll 0.37, distance 0.0008, x 0.0015, y 0.0008. The
# "still" limits are about twice that, so ordinary fixations pass and moving around does not.
STILL_LIMITS = {"pitch": 1.0, "yaw": 0.8, "roll": 0.6, "distance": 0.0016, "x": 0.003, "y": 0.002}
STILL_MIN_FRAMES = 20
# Brightness gate. Three exploratory within-session points: face 37 -> 7.3-8.6 deg, 78 -> 3.0-3.5,
# 124 -> 1.3-1.7. 60 sits between the dim and backlit sessions (geometric mean of 37 and 78 is ~54,
# rounded up) and is the pre-registered H3 threshold (PROGRESS.md); 100 is the recommended level.
# A three-point inference, not a measured optimum. Absolute face brightness only, never the ratio.
LIGHT_MIN_FACE = 60.0
LIGHT_GOOD_FACE = 100.0
BACKLIT_RATIO = 0.6    # face/background below this triggers the (non-blocking) backlight tip


@dataclass(frozen=True)
class CheckItem:
    """One live setup step. ``level``: good / adjust / off (measured), ``unknown`` (nothing to
    compare against; never blocks) or ``wait`` (still measuring; blocks until it resolves)."""

    key: str
    title: str
    howto: str
    level: str
    detail: str
    hint: str = ""
    tip: str = ""                 # non-blocking advice, shown even when the step is green
    value: float | None = None    # a live number for a gauge (face brightness for the light step)

    @property
    def ok(self) -> bool:
        return self.level == "good"


def still_status(window: Sequence[dict[str, float]]) -> tuple[str, str]:
    """(level, detail) for head stillness over a window of recent per-frame metrics."""
    if len(window) < STILL_MIN_FRAMES:
        return "wait", "measuring..."
    over = {m: float(np.std([f[m] for f in window])) / STILL_LIMITS[m] for m in METRICS}
    worst = max(over, key=over.get)
    if over[worst] <= 1.0:
        return "good", "steady"
    return ("adjust" if over[worst] <= 2.0 else "off"), f"moving ({LABELS[worst]})"


LIGHT_CHANGE_TOLERANCE = 0.35  # relative face-brightness change vs the calibration that triggers a hint


def build_checklist(ref: Reference | None, metrics: dict[str, float] | None,
                    window: Sequence[dict[str, float]], lighting: dict[str, float] | None,
                    ref_lighting: dict | None = None, note: str = "") -> list[CheckItem]:
    """The five live setup steps, each with a measured status and, when not good, a hint.

    Chair contact and arm's length cannot be sensed by a webcam: they are checked indirectly
    through distance and stillness, and the on-screen text says so. Without a reference (the very
    first session) the position steps are ``unknown`` and do not block.
    """
    def pos(keys: tuple[str, ...], fmt) -> tuple[str, str, str]:
        if ref is None:
            return "unknown", "no earlier session to compare with", ""
        if metrics is None:
            return "off", "no face detected", "look at the camera"
        z = ref.z(metrics)
        worst = max(keys, key=lambda k: abs(z[k]))
        level = ref.level(z[worst])
        return level, fmt(metrics, ref), ("" if level == "good" else ref.hint(worst, metrics[worst]))

    d_lvl, d_txt, d_hint = pos(("distance",), lambda m, r: f"distance {m['distance'] * 100:.1f}% (usual {r.center['distance'] * 100:.1f}%)")
    p_lvl, p_txt, p_hint = pos(("pitch",), lambda m, r: f"head pitch {m['pitch']:+.1f} deg (usual {r.center['pitch']:+.1f})")
    f_lvl, f_txt, f_hint = pos(("yaw", "roll", "x", "y"), lambda m, r: f"yaw {m['yaw']:+.1f}, roll {m['roll']:+.1f} deg, face {'centred' if abs(m['x'] - r.center['x']) < 2 * r.scale['x'] and abs(m['y'] - r.center['y']) < 2 * r.scale['y'] else 'off-centre'}")
    l_tip = l_value = None
    if lighting is None:
        l_lvl, l_txt, l_hint = "off", "no face detected", "look at the camera"
    else:
        l_value = lighting["face"]
        label = lighting_label(lighting["face"], lighting["background"])
        l_txt = f"face {lighting['face']:.0f}/255, background {lighting['background']:.0f}  ({label})"
        tips = []
        mismatch = note_mismatch(note, label)
        if mismatch:
            tips.append("CHECK YOUR NOTE: " + mismatch)
        if lighting["face"] < LIGHT_GOOD_FACE:
            tips.append(f"more front light helps accuracy (aim for {LIGHT_GOOD_FACE:.0f}+)")
        if lighting["ratio"] < BACKLIT_RATIO:
            tips.append("bright light behind you: close the curtain or face a darker wall")
        l_tip = "; ".join(tips)
        if lighting["face"] < LIGHT_MIN_FACE:
            l_lvl, l_hint = "off", "too dark: put a lamp or window in front of you, aimed at your face"
        elif lighting["face"] > 220 or lighting["clipped"] > 0.05:
            l_lvl, l_hint = "off", "face overexposed: reduce glare or light"
        else:
            l_lvl, l_hint = "good", ""
            if ref_lighting and ref_lighting.get("face"):
                change = (lighting["face"] - ref_lighting["face"]) / ref_lighting["face"]
                if abs(change) > LIGHT_CHANGE_TOLERANCE:
                    l_lvl = "adjust"
                    l_hint = (f"lighting differs from calibration (face was {ref_lighting['face']:.0f}, "
                              f"{ref_lighting.get('label', '')}): match it or recalibrate")
    s_lvl, s_txt = still_status(window)
    return [
        CheckItem("distance", "Sit all the way back, arm's length from the screen",
                  "Back against the chair; stretch out your arm and your fingertips should just reach the screen. "
                  "I can only check your distance, not the chair.", d_lvl, d_txt, d_hint),
        CheckItem("pitch", "Laptop on a stable desk, lid at the usual angle",
                  "Not on your lap. The angle shows up as your head pitch.", p_lvl, p_txt, p_hint),
        CheckItem("facing", "Face the screen squarely",
                  "Head level, shoulders relaxed, chin neither up nor down.", f_lvl, f_txt, f_hint),
        CheckItem("light", "Light on your face",
                  f"Put a lamp or window in front of you, not behind. Face brightness at least "
                  f"{LIGHT_MIN_FACE:.0f}/255 ({LIGHT_GOOD_FACE:.0f}+ is best); avoid glare on glasses.",
                  l_lvl, l_txt, l_hint, l_tip or "", l_value),
        CheckItem("still", "Keep your head still, move only your eyes",
                  "Blink normally; this turns green once you hold still for a moment.", s_lvl, s_txt,
                  "" if s_lvl in ("good", "wait") else "hold still"),
    ]


def checklist_ready(items: Sequence[CheckItem]) -> bool:
    """Ready when every measured step is good (unknown steps never block)."""
    return all(i.level in ("good", "unknown") for i in items) and any(i.level == "good" for i in items)
