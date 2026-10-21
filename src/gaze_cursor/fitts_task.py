"""Step 7's jump-and-click task (a Fitts-style pointing task) and its dot block. Pure logic: no camera, UI or OS.

A target circle appears; the trial ends at the next click (a hit if the click is inside the circle). Targets
visit the points of a ring in the ISO 9241-411 "star" order (each target is on the far side of the ring from
the last), so every move has the ring's amplitude. A block fixes the pointing method (trackpad only, or
gaze-jump + trackpad) and the seat (centre = as calibrated, moved = head moved against the calibration). All
times are monotonic ms on the camera frames' clock. What the task records per trial: the target, the pointer
at onset, the first pointer move, any warp and where it landed, the click and its position; the pointer path
goes to a separate array. Sizes are in screen points; the primary error unit is cm / px (it does not depend on
the viewing distance), degrees are computed later from the per-trial iris distance.

The protocol (counts, amplitudes, widths, block order, thresholds) is frozen in PREREGISTRATIONS.md, not here:
the defaults below are only for development runs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.livetest import head_moved
from gaze_cursor.sinks.cursor import InputEvent

METHODS = ("trackpad", "gaze_jump")
SEATS = ("centre", "moved")
TRIAL_TIMEOUT_MS = 10_000.0


def index_of_difficulty(amplitude: float, width: float) -> float:
    """Shannon form of Fitts' index of difficulty, bits (``amplitude`` and ``width`` in the same unit)."""
    return math.log2(amplitude / width + 1.0)


def star_order(n: int) -> list[int]:
    """Visiting order of ``n`` ring points (n odd): each step jumps to the opposite side of the ring."""
    if n % 2 == 0:
        raise ValueError("a star order needs an odd number of ring points")
    return [(i * (n + 1) // 2) % n for i in range(n)]


def ring_points(cfg: ScreenConfig, amplitude: float, width: float, n: int = 9) -> list[tuple[float, float]]:
    """The ``n`` ring points (diameter ``amplitude``, centred on the usable screen) in star order.
    Raises ``ValueError`` if a target of diameter ``width`` would not be fully visible."""
    cx, cy = cfg.width_px / 2, (cfg.top_inset_pt + cfg.height_px) / 2
    pts = []
    for k in star_order(n):
        a = 2 * math.pi * k / n - math.pi / 2
        p = (cx + amplitude / 2 * math.cos(a), cy + amplitude / 2 * math.sin(a))
        if not cfg.target_is_safe(p, ring_px=width / 2):
            raise ValueError(f"ring amplitude {amplitude:.0f} px does not fit this screen for width {width:.0f} px")
        pts.append(p)
    return pts


def line_points(cfg: ScreenConfig, amplitude: float, width: float) -> list[tuple[float, float]]:
    """Two targets ``amplitude`` apart on a horizontal line through the middle of the usable screen, visited
    alternately (the long-distance layout: a ring cannot be wider than the screen is high)."""
    cy = (cfg.top_inset_pt + cfg.height_px) / 2
    pts = [(cfg.width_px / 2 + s * amplitude / 2, cy) for s in (-1, 1)]
    if not all(cfg.target_is_safe(p, ring_px=width / 2) for p in pts):
        raise ValueError(f"line amplitude {amplitude:.0f} px does not fit this screen for width {width:.0f} px")
    return pts


LAYOUTS = ("ring", "line")


def layout_points(cfg: ScreenConfig, layout: str, amplitude: float, width: float, n: int = 9) -> list[tuple[float, float]]:
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    return ring_points(cfg, amplitude, width, n) if layout == "ring" else line_points(cfg, amplitude, width)


@dataclass(frozen=True)
class Block:
    method: str                       # METHODS
    seat: str                         # SEATS
    amplitude: float                  # ring diameter, points
    width: float                      # target diameter, points
    n_trials: int = 9                 # one lap of the ring
    layout: str = "ring"              # LAYOUTS: "ring" (9 points, star order) or "line" (2 points, alternating)
    practice: bool = False            # unscored warm-up block (analysis leaves it out; it is kept in the file)

    def __post_init__(self) -> None:
        if self.method not in METHODS or self.seat not in SEATS:
            raise ValueError(f"unknown method / seat: {self.method!r} / {self.seat!r}")
        if self.layout not in LAYOUTS:
            raise ValueError(f"unknown layout: {self.layout!r}")


def make_blocks(cfg: ScreenConfig, methods: tuple[str, ...], seats: tuple[str, ...],
                amplitudes: tuple[float, ...], widths: tuple[float, ...], *, laps: int = 1, seed: int = 0,
                n_ring: int = 9, line_amplitudes: tuple[float, ...] = (), practice: bool = False,
                practice_trials: int = 4) -> list[Block]:
    """Every method x seat x amplitude x width, one block each; seat blocks are kept together (the chair is
    moved once per seat; the seat order is shuffled too), and the order of methods and of the (amplitude, width) pairs inside a seat is
    shuffled with ``seed`` so a fixed order cannot masquerade as an effect."""
    rng = np.random.default_rng(seed)
    pairs = [("ring", a, w) for a in amplitudes for w in widths] + [("line", a, w) for a in line_amplitudes for w in widths]
    blocks: list[Block] = []
    for seat in rng.permutation(list(seats)):
        if practice and not blocks:       # one warm-up block per method, in the first seat, easy and short
            lay, a, _ = pairs[0]
            w = max(widths)
            layout_points(cfg, lay, a, w, n_ring)
            blocks += [Block(str(m), str(seat), a, w, practice_trials, lay, True) for m in rng.permutation(list(methods))]
        for method in rng.permutation(list(methods)):
            order = rng.permutation(len(pairs))
            for i in order:
                lay, a, w = pairs[i]
                layout_points(cfg, lay, a, w, n_ring)    # fail early if it does not fit
                blocks.append(Block(str(method), str(seat), a, w, n_ring * laps, lay))
    return blocks


@dataclass
class FittsTrial:
    index: int                        # within the whole run
    block: int
    center: tuple[float, float]
    width: float
    amplitude: float
    t_onset: float                    # requested at next_target; ``mark_shown`` moves it to the first drawn frame
    pointer_at_onset: tuple[float, float] | None
    shift_at_onset: tuple[float, float, float] | None = None   # head shift vs calibration (ratio, dx cm, dy cm)
    t_first_move: float | None = None
    t_requested: float | None = None  # when the target was asked for (before it was drawn)
    t_warp: float | None = None
    t_landing: float | None = None
    warp_landing: tuple[float, float] | None = None
    t_click: float | None = None
    click_xy: tuple[float, float] | None = None
    hit: bool | None = None           # None while running
    timed_out: bool = False

    @property
    def movement_time_ms(self) -> float | None:
        return None if self.t_click is None else self.t_click - self.t_onset

    @property
    def endpoint_error_px(self) -> float | None:
        if self.click_xy is None:
            return None
        return math.hypot(self.click_xy[0] - self.center[0], self.click_xy[1] - self.center[1])

    def to_dict(self) -> dict:
        return {"index": self.index, "block": self.block, "center": self.center, "width_px": self.width,
                "amplitude_px": self.amplitude, "id_bits": index_of_difficulty(self.amplitude, self.width),
                "t_onset_ms": self.t_onset, "pointer_at_onset": self.pointer_at_onset,
                "shift_at_onset": self.shift_at_onset, "t_first_move_ms": self.t_first_move,
                "t_requested_ms": self.t_requested, "t_warp_ms": self.t_warp, "t_landing_ms": self.t_landing, "warp_landing": self.warp_landing, "t_click_ms": self.t_click,
                "click_xy": self.click_xy, "hit": self.hit, "timed_out": self.timed_out,
                "movement_time_ms": self.movement_time_ms, "endpoint_error_px": self.endpoint_error_px}


@dataclass
class FittsTask:
    """One run: a list of blocks, driven by ``start_block`` / ``next_target`` and fed with input events."""
    cfg: ScreenConfig
    blocks: list[Block]
    n_ring: int = 9
    timeout_ms: float = TRIAL_TIMEOUT_MS
    trials: list[FittsTrial] = field(default_factory=list)
    moves: list[tuple[float, float, float, int]] = field(default_factory=list)   # t_ms, x, y, trial index (-1 = none)
    block_index: int = -1
    _ring: list[tuple[float, float]] = field(default_factory=list)
    _in_block: int = 0

    @property
    def block(self) -> Block | None:
        return self.blocks[self.block_index] if 0 <= self.block_index < len(self.blocks) else None

    @property
    def current(self) -> FittsTrial | None:
        return self.trials[-1] if self.trials and self.trials[-1].hit is None else None

    @property
    def block_done(self) -> bool:
        b = self.block
        return b is None or (self._in_block >= b.n_trials and self.current is None)

    @property
    def done(self) -> bool:
        return self.block_index >= len(self.blocks) - 1 and self.block_done

    def start_block(self) -> Block:
        """Advance to the next block (the caller has run its seat check). The ring starts at point 0."""
        if self.block_index + 1 >= len(self.blocks) or not self.block_done:
            raise RuntimeError("no further block, or the current one is not finished")
        self.block_index += 1
        b = self.blocks[self.block_index]
        self._ring = layout_points(self.cfg, b.layout, b.amplitude, b.width, self.n_ring)
        self._in_block = 0
        return b

    def next_target(self, t_ms: float, pointer: tuple[float, float] | None,
                    shift: tuple[float, float, float] | None = None) -> FittsTrial:
        """Show the next target of the block (``pointer`` = where the cursor is now, ``shift`` = head shift)."""
        b = self.block
        if b is None or self._in_block >= b.n_trials or self.current is not None:
            raise RuntimeError("no target to show")
        center = self._ring[self._in_block % len(self._ring)]
        trial = FittsTrial(len(self.trials), self.block_index, center, b.width, b.amplitude, t_ms, pointer, shift)
        self.trials.append(trial)
        self._in_block += 1
        return trial

    def mark_shown(self, trial: FittsTrial, t_ms: float) -> None:
        """The target has been drawn (the first canvas update after ``next_target``): this, not the request, is
        when the person could first see it, so it becomes the onset all times are measured from."""
        if trial.t_requested is None:
            trial.t_requested, trial.t_onset = trial.t_onset, t_ms

    def on_event(self, event: InputEvent) -> str | None:
        """Feed an input event. Returns 'hit' / 'miss' when a click ends the trial, else None."""
        trial = self.current
        if event.kind == "move" and event.xy is not None:
            self.moves.append((event.t_ms, event.xy[0], event.xy[1], trial.index if trial else -1))
            if trial is not None and trial.t_first_move is None:
                trial.t_first_move = event.t_ms
        elif trial is None:
            return None
        elif event.kind == "warp" and trial.t_warp is None:
            trial.t_warp = event.t_ms
        elif event.kind == "landing" and event.xy is not None and trial.t_warp is not None:
            if trial.warp_landing is None:           # the FIRST jump's landing; later jumps are only in the path
                trial.warp_landing, trial.t_landing = event.xy, event.t_ms
            self.moves.append((event.t_ms, event.xy[0], event.xy[1], trial.index))
        elif event.kind == "click" and event.button == "left" and event.xy is not None:
            trial.t_click, trial.click_xy = event.t_ms, event.xy
            trial.hit = (trial.endpoint_error_px or 0.0) <= trial.width / 2
            return "hit" if trial.hit else "miss"
        return None

    def update(self, t_ms: float) -> str | None:
        """Time out a trial that has had no click for ``timeout_ms`` (counted as a miss, flagged)."""
        trial = self.current
        if trial is not None and t_ms - trial.t_onset >= self.timeout_ms:
            trial.hit, trial.timed_out = False, True
            return "miss"
        return None

    def summary(self) -> dict:
        """Per (method, seat), practice blocks left out: trials, hit rate, median movement time of hits. Descriptive only; the
        pre-registered analysis is offline."""
        out: dict[str, dict] = {}
        for tr in self.trials:
            if tr.hit is None:
                continue
            b = self.blocks[tr.block]
            if b.practice:
                continue
            row = out.setdefault(f"{b.method}/{b.seat}", {"n": 0, "hits": 0, "times": []})
            row["n"] += 1
            row["hits"] += int(tr.hit)
            if tr.hit and tr.movement_time_ms is not None:
                row["times"].append(tr.movement_time_ms)
        return {k: {"n": v["n"], "hit_rate": v["hits"] / v["n"],
                    "median_time_ms": float(np.median(v["times"])) if v["times"] else math.nan}
                for k, v in out.items()}

    def moves_array(self) -> np.ndarray:
        return np.array(self.moves, float).reshape(-1, 4)


# --- seat check ---------------------------------------------------------------------------------------

def seat_ok(shifts: Sequence[tuple[float, float, float] | None], seat: str, min_fraction: float = 0.8) -> bool:
    """Is the measured head position what the block's seat says, against the calibration?
    ``centre``: not moved (``head_moved`` false) in at least ``min_fraction`` of the samples; ``moved``:
    moved in at least that fraction. Samples without a face (None) count against both."""
    if seat not in SEATS:
        raise ValueError(seat)
    if not shifts:
        return False
    moved = sum(1 for s in shifts if s is not None and head_moved(*s))
    centred = sum(1 for s in shifts if s is not None and not head_moved(*s))
    return (centred if seat == "centre" else moved) / len(shifts) >= min_fraction


# --- dot block (eyes only vs head turn) ------------------------------------------------------------------

DOT_MODES = ("eyes", "head")


def dot_schedule(cfg: ScreenConfig, n_dots: int = 9, *, seed: int = 0) -> list[dict]:
    """Fixation dots for the short eyes-only / head-turn block: the same ``n_dots`` positions (a 3 x 3 grid
    inset from the edges) shown once in each mode, the mode order and the dot order shuffled with ``seed``. Each item is
    ``{"mode", "index", "xy"}``; the caller times them and logs the frames."""
    side = int(round(math.sqrt(n_dots)))
    if side * side != n_dots:
        raise ValueError("n_dots must be a square number")
    xs = np.linspace(0.12, 0.88, side) * cfg.width_px
    ys = cfg.top_inset_pt + np.linspace(0.15, 0.85, side) * (cfg.height_px - cfg.top_inset_pt)
    pts = [(float(x), float(y)) for y in ys for x in xs]
    rng = np.random.default_rng(seed)
    out = []
    for mode in rng.permutation(list(DOT_MODES)):
        for i in rng.permutation(n_dots):
            out.append({"mode": str(mode), "index": int(i), "xy": pts[i]})
    return out
