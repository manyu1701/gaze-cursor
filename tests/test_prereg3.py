"""Round 3 (head position): a synthetic camera + eyes + screen world, end to end from landmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.configs import REGISTRY
from gaze_cursor.eval.prereg3 import (
    SET_A, Position, PositionSet, leave_one_position_out, position_ok, run_h4, run_h5, select_set_b,
    session_position, set_a, slot_from_note)
from gaze_cursor.eval.session import CALIB, CLEAN_WINDOW, TEST, Session
from gaze_cursor.eval.stats import concat
from gaze_cursor.features import LEFT_EYE, RIGHT_EYE, extract_features
from tests.helpers import NAMES

SCREEN = ScreenConfig()
W, H, F = 1280, 720, 1000.0          # synthetic camera: image size and focal length (px)
IPD, EYE_W, GAIN = 6.3, 3.0, 0.5     # cm, cm, iris offset (eye widths) per unit tan(gaze angle)
CAM_ABOVE_SCREEN_CM = 0.5
FREEZE = "2026-10-20T10:00:00"
HEADS = {"C": (0, 10, 60), "N": (0, 10, 50), "R": (0, 10, 72), "L": (8, 10, 60),
         "Rt": (-8, 10, 60), "Lo": (0, 16, 60), "C2": (0.3, 10.2, 60.5)}


def _project(p: np.ndarray) -> np.ndarray:
    """Camera cm (x image-right, y image-down, z away) -> normalized image coordinates."""
    return np.array([(W / 2 + F * p[0] / p[2]) / W, (H / 2 + F * p[1] / p[2]) / H, 0.0])


def _frame(head: tuple[float, float, float], target_px: np.ndarray, noise: float, rng) -> np.ndarray:
    """Landmarks of a head at ``head`` (eye midpoint, cm) looking at a screen point (px)."""
    e = np.array(head, float)
    # Screen point in camera cm: the camera is in the screen plane, above its top centre; the user's
    # right on screen is the camera's -x (the camera faces the user).
    t = np.array([-(target_px[0] - SCREEN.width_px / 2) * SCREEN.cm_per_px,
                  CAM_ABOVE_SCREEN_CM + target_px[1] * SCREEN.cm_per_px, 0.0])
    tan_a, tan_b = (t[0] - e[0]) / e[2], (t[1] - e[1]) / e[2]
    pts = np.tile(_project(e), (478, 1))                  # face mesh collapsed onto the eye midpoint
    for (a, b, up, lo, iris), side in ((RIGHT_EYE, -1), (LEFT_EYE, 1)):
        c = e + [side * IPD / 2, 0, 0]
        pa, pb = _project(c - [EYE_W / 2, 0, 0]), _project(c + [EYE_W / 2, 0, 0])
        pc = _project(c)
        width = (pb[0] - pa[0]) * W
        pts[a], pts[b] = pa, pb
        pts[up], pts[lo] = pc - [0, 0.3 * width / H, 0], pc + [0, 0.3 * width / H, 0]
        off = GAIN * np.array([tan_a, tan_b]) * width + rng.normal(0, noise, 2)
        pts[iris] = pc + [off[0] / W, off[1] / H, 0]
    return pts.astype(np.float32)


def make_session(slot: str, i: int, started: str = "2026-10-21T10:00:00", note: str | None = None,
                 face: float = 120.0, head=None, noise: float = 0.05, frames: int = 4,
                 sway_cm: float = 0.5) -> Session:
    rng = np.random.default_rng(i)
    head = HEADS[slot] if head is None else head
    targets = rng.uniform([60, 80], [1410, 900], size=(25, 2))
    rows = []
    for k, xy in enumerate(targets):
        for _ in range(frames):
            sway = rng.normal(0, sway_cm, 3) * [1, 1, 2]       # heads never hold perfectly still
            pts = _frame(tuple(np.add(head, sway)), xy, noise, rng)
            rows.append((CALIB if k < 16 else TEST, k, xy, pts, extract_features(pts, np.eye(4), (W, H))))
    n = len(rows)
    meta = {"screen": SCREEN.to_dict(), "window": CLEAN_WINDOW, "layout": "v2", "image_size": [W, H],
            "blendshape_names": list(NAMES), "started": started, "note": f"pos {slot}" if note is None else note,
            "lighting": {"face": face, "background": 75.0, "ratio": face / 75.0,
                         "label": "front-lit" if face >= 100 else "dim"}}
    return Session(path=Path(f"session_syn_{i:03d}.npz"), meta=meta, t_ms=np.arange(n),
                   phase=np.array([r[0] for r in rows], np.int8), target_idx=np.array([r[1] for r in rows], np.int16),
                   target_xy=np.array([r[2] for r in rows], np.float32), features=np.stack([r[4] for r in rows]),
                   blendshapes=np.zeros((n, 52), np.float32), points=np.stack([r[3] for r in rows]),
                   transform=np.tile(np.eye(4), (n, 1, 1)))


def set_b_sessions(heads=HEADS) -> list[Session]:
    return [make_session(k, 100 + j, started=f"2026-10-21T10:{j:02d}:00", head=heads[k])
            for j, k in enumerate(["C", "N", "R", "L", "Rt", "Lo", "C2"])]


def set_a_sessions() -> list[Session]:
    out = []
    for j, k in enumerate(["C", "N", "R", "L", "Rt", "Lo"]):
        s = make_session(k, 200 + j, started="2026-10-13T05:20:00")
        out.append(Session(**{**s.__dict__, "path": Path(SET_A[k])}))
    return out


# --- geometry ------------------------------------------------------------------------------------

def test_head_geometry_recovers_eye_position() -> None:
    for head in [(0, 10, 60), (8, -3, 50), (-5, 12, 75)]:
        f = extract_features(_frame(head, np.array([735, 478]), 0.0, np.random.default_rng(0)), np.eye(4), (W, H))
        assert f[9] == pytest.approx(head[0], abs=0.05) and f[10] == pytest.approx(head[1], abs=0.05)
        assert f[11] == pytest.approx(head[2] * W / F, rel=1e-3)   # depth in units of f / width


def test_session_position_and_rules() -> None:
    c, n, lft = (session_position(make_session(k, j)) for j, k in enumerate(["C", "N", "L"]))
    assert n.relative_to(c)[0] == pytest.approx(1.2, rel=0.01)
    assert lft.relative_to(c)[1] > 0.07                     # user's left = larger x in the unflipped frame
    assert position_ok("N", n, c) and position_ok("L", lft, c)
    assert not position_ok("R", n, c) and not position_ok("Rt", lft, c) and not position_ok("C2", n, c)
    assert position_ok("C2", Position(c.distance * 1.03, c.x + 0.02, c.y - 0.02), c)


@pytest.mark.parametrize("note,slot", [("pos C", "C"), ("C2", "C2"), ("pos Rt", "Rt"), ("R", "R"),
                                       ("pos lo", "Lo"), ("pos L", "L"), ("pos N near", "N"),
                                       ("D2 dim", None), ("current setup", None), ("", None), (None, None)])
def test_slot_from_note(note, slot) -> None:
    assert slot_from_note(note) == slot


# --- the models: the point of step 3 -------------------------------------------------------------

def test_ray_model_survives_head_movement_and_gaze_model_does_not() -> None:
    a = set_a_sessions()
    errs = {name: concat(leave_one_position_out(a, range(6), REGISTRY[name], SCREEN)).err_deg.mean()
            for name in ("gaze_a10", "head_a10", "ray_a1")}
    assert errs["ray_a1"] < 0.5 < errs["gaze_a10"]
    assert errs["ray_a1"] < errs["head_a10"]


# --- selection -----------------------------------------------------------------------------------

def test_set_a_by_name() -> None:
    a = set_a(set_a_sessions())
    assert a.complete and set(a.sessions) == {"C", "N", "R", "L", "Rt", "Lo"}
    assert not set_a(set_a_sessions()[:-1]).complete


def test_set_b_refuses_until_frozen() -> None:
    assert not select_set_b(set_b_sessions(), freeze_after=None).complete


def test_set_b_selection_rules() -> None:
    b = set_b_sessions()
    early = make_session("N", 300, started="2026-10-19T09:00:00")                 # before the freeze
    dim = make_session("L", 301, started="2026-10-21T09:59:00", face=40)          # wrong lighting
    wrong = make_session("R", 302, started="2026-10-21T09:58:00", head=HEADS["N"])  # labelled R, measured near
    extra = make_session("Lo", 303, started="2026-10-21T11:00:00")                # later qualifying Lo
    sel = select_set_b([early, dim, wrong, *b, extra], freeze_after=FREEZE)
    assert sel.complete
    assert sel.sessions["N"] is b[1] and sel.sessions["Lo"] is b[5]
    excluded = dict(sel.excluded)
    assert "not front-lit" in excluded[dim.path.name] and "R rule" in excluded[wrong.path.name]
    assert early.path.name not in excluded and any(extra.path.name in x for x in sel.ignored)


def test_set_b_c2_must_be_back_in_the_centre() -> None:
    heads = {**HEADS, "C2": HEADS["N"]}
    sel = select_set_b(set_b_sessions(heads), freeze_after=FREEZE)
    assert "set B C2" in sel.missing


# --- H4 / H5 detect planted effects and not nulls -------------------------------------------------

def test_h4_detects_position_cost_and_not_a_null() -> None:
    b = select_set_b(set_b_sessions(), freeze_after=FREEZE)
    h4 = run_h4(b, REGISTRY["gaze_a10"], SCREEN)
    assert h4["supported"] and h4["gap"] > 1.0
    # Null: nobody moved (the slots are forced; such sessions would fail the position rules).
    still = PositionSet(sessions=dict(zip(["C", "N", "R", "L", "Rt", "Lo", "C2"],
                                          set_b_sessions({k: HEADS["C"] for k in HEADS}))))
    assert not run_h4(still, REGISTRY["gaze_a10"], SCREEN)["supported"]


def test_h5_detects_a_better_candidate_and_not_a_null() -> None:
    a = set_a(set_a_sessions())
    b = select_set_b(set_b_sessions(), freeze_after=FREEZE)
    h5 = run_h5(a, b, REGISTRY["gaze_a10"], REGISTRY["ray_a1"], SCREEN)
    assert h5["supported"] and h5["err_cand"] < h5["err_ref"]
    null = run_h5(a, b, REGISTRY["gaze_a10"], REGISTRY["gaze_a10"], SCREEN)
    assert not null["supported"] and null["comparison"]["mean_diff_deg"] == pytest.approx(0)


# Positions shaped like the real set A (2026-10-13): N much closer than anything else, every other
# displaced session also lower than C, so each held-out position is an extrapolation.
REAL_LIKE = {"C": (0, 10, 60), "N": (0, 11, 40), "R": (6, 18, 67), "L": (10, 17, 62),
             "Rt": (-10, 19, 66), "Lo": (-2, 22, 70)}


def test_geo_model_extrapolates_where_free_models_do_not() -> None:
    a = [make_session(k, 300 + j, head=REAL_LIKE[k]) for j, k in enumerate(REAL_LIKE)]
    err = {n: [t.err_deg.mean() for t in leave_one_position_out(a, range(6), REGISTRY[n], SCREEN)]
           for n in ("gaze_a10", "head_a10", "geo_a10")}
    assert max(err["geo_a10"]) < 0.5                        # fixed geometry: every position, incl. N
    assert err["head_a10"][1] > 3 * err["geo_a10"][1] + 1   # free head terms extrapolate badly to N
    assert np.mean(err["gaze_a10"]) > 3                     # and the pose-free model cannot move at all


def test_geo_model_needs_the_screen() -> None:
    from gaze_cursor.calibration import GazeModel
    with pytest.raises(ValueError):
        GazeModel("geo")
