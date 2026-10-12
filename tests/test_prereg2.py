from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.configs import REGISTRY
from gaze_cursor.eval.prereg2 import (
    B_TWIN, F_TWIN, PREREG2_AFTER, lighting_class, run_h2, run_h3, select_sessions)
from gaze_cursor.eval.stats import bootstrap_diff_ci
from tests.helpers import fake_session

CFG = REGISTRY["gaze_a10"]
SCREEN = ScreenConfig()
LATER = "2026-10-14T09:00:00"
EARLIER = "2026-10-13T03:30:00"


def _light(face: float, background: float) -> dict:
    ratio = face / background
    label = ("dim" if face < 50 else "backlit" if ratio < 0.6 else "front-lit" if ratio >= 0.9 and face >= 100 else "mixed")
    return {"face": face, "background": background, "ratio": ratio, "label": label}


def _session(i: int, face: float, background: float, started: str = LATER, name: str | None = None,
             gain: float = 1.0, noise: float = 0.0, valid_drop: float = 0.0):
    s = fake_session(i, n_calib=16, n_test=9, frames=5, layout="v2", gain=gain, noise=noise)
    feats = s.features.copy()
    if valid_drop:
        feats[: int(len(feats) * valid_drop)] = np.nan
    meta = {**s.meta, "lighting": _light(face, background), "started": started}
    return replace(s, features=feats, meta=meta, path=Path(name or f"session_new_{i:02d}.npz"))


F_T = lambda **kw: _session(90, 124, 79, EARLIER, F_TWIN, **kw)       # noqa: E731
B_T = lambda **kw: _session(91, 78, 146, EARLIER, B_TWIN, **kw)       # noqa: E731


def test_lighting_class_boundaries() -> None:
    meta = lambda f, b: {"lighting": _light(f, b)}                  # noqa: E731
    # The "front-lit" label itself needs face >= 100 and ratio >= 0.9, so the effective range is 100-155
    # (the written +-25% lower bound of 93 is unreachable: both written conditions must hold).
    assert lighting_class(meta(124, 79)) == "F" and lighting_class(meta(100, 80)) == "F" and lighting_class(meta(155, 100)) == "F"
    assert lighting_class(meta(99, 80)) is None and lighting_class(meta(160, 100)) is None
    assert lighting_class(meta(78, 146)) == "B" and lighting_class(meta(60, 150)) == "B" and lighting_class(meta(97, 200)) == "B"
    assert lighting_class(meta(59, 150)) is None and lighting_class(meta(98, 300)) is None
    assert lighting_class(meta(37, 75)) == "D" and lighting_class(meta(25, 60)) == "D" and lighting_class(meta(49, 90)) == "D"
    assert lighting_class(meta(24, 60)) is None and lighting_class(meta(52, 90)) is None and lighting_class({}) is None


def test_selection_uses_captured_lighting_only_and_the_cutoff() -> None:
    cands = [F_T(), B_T(), _session(1, 130, 80), _session(2, 80, 150), _session(3, 36, 70), _session(4, 40, 80),
             _session(5, 140, 90),                                    # an extra front-lit session: ignored
             _session(6, 130, 80, started=EARLIER),                   # before the pre-registration: never a test session
             _session(7, 70, 70)]                                     # no class: excluded
    sel = select_sessions(cands)
    assert sel.complete and sel.f2.path.name == "session_new_01.npz" and sel.b2.path.name == "session_new_02.npz"
    assert [s.path.name for s in sel.dim] == ["session_new_03.npz", "session_new_04.npz"]
    assert any("session_new_05" in i for i in sel.ignored)
    assert any("session_new_07" in n and "outside every class range" in w for n, w in sel.excluded)
    assert all("session_new_06" not in n for n, _ in sel.excluded)    # silently out: started before the cutoff
    assert PREREG2_AFTER < LATER


def test_incomplete_selection_lists_what_is_missing() -> None:
    sel = select_sessions([F_T(), _session(1, 130, 80), _session(2, 36, 70)])
    assert not sel.complete
    text = " ".join(sel.missing)
    assert "B2" in text and "1 more dim" in text and "training twin" in " ".join(select_sessions([]).missing)


def test_low_valid_fraction_and_missing_lighting_are_excluded_objectively() -> None:
    bad = _session(1, 130, 80, valid_drop=0.3)
    nolight = replace(_session(2, 80, 150), meta={**_session(2, 80, 150).meta, "lighting": None})
    nolight.meta.pop("lighting")
    sel = select_sessions([F_T(), B_T(), bad, nolight])
    reasons = dict(sel.excluded)
    assert "valid frames" in reasons["session_new_01.npz"] and reasons["session_new_02.npz"] == "no captured lighting"
    assert sel.f2 is None and sel.b2 is None


def _selection(f2_gain=1.0, b2_gain=2.0, twin_f_gain=1.0, twin_b_gain=2.0, dim_noise=(0.0, 0.0), b2_noise=0.0):
    cands = [F_T(gain=twin_f_gain), B_T(gain=twin_b_gain),
             _session(1, 130, 80, gain=f2_gain), _session(2, 80, 150, gain=b2_gain, noise=b2_noise),
             _session(3, 36, 70, noise=dim_noise[0]), _session(4, 40, 80, noise=dim_noise[1])]
    sel = select_sessions(cands)
    assert sel.complete
    return sel


def test_h2_detects_a_planted_same_lighting_advantage() -> None:
    res = run_h2(_selection(), CFG, SCREEN)                           # each lighting has its own feature->target map
    assert res["supported"] and res["mean_d"] > 1.0 and res["ci"][0] > 0
    assert all(d > 0 for d in res["per_session_d"]) and res["err_same"] < res["err_other"]
    assert res["n_targets"] == 50


def test_h2_does_not_invent_an_effect_when_lighting_changes_nothing() -> None:
    res = run_h2(_selection(b2_gain=1.0, twin_b_gain=1.0), CFG, SCREEN)
    assert not res["supported"] and abs(res["mean_d"]) < 0.1
    assert not res["criteria"][f"(b) mean d >= 0.5 deg"]


def test_h2_is_deterministic() -> None:
    a, b = run_h2(_selection(), CFG, SCREEN), run_h2(_selection(), CFG, SCREEN)
    assert a["ci"] == b["ci"] and a["mean_d"] == b["mean_d"]


def test_h3_detects_planted_dim_penalty_in_all_four_pairs() -> None:
    res = run_h3(_selection(dim_noise=(0.15, 0.15)), CFG, SCREEN)
    assert res["supported"] and res["gap"] > 1.5 and res["ci"][0] > 0
    assert res["n_gated"] == 18 and res["n_below"] == 18


def test_h3_null_and_the_all_pairs_guard() -> None:
    assert not run_h3(_selection(), CFG, SCREEN)["supported"]                       # no dim penalty
    # one dim session is clean while one gated session is noisy: that dim session beats it, so (c) fails
    mixed = run_h3(_selection(dim_noise=(0.15, 0.0), b2_noise=0.15), CFG, SCREEN)
    assert not mixed["criteria"]["(c) every gated session beats every dim session (4 of 4 pairs)"]
    assert not mixed["supported"]


def test_bootstrap_diff_ci_brackets_a_known_difference_and_is_seeded() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(5.0, 1.0, 40), rng.normal(3.0, 1.0, 40)
    lo, hi = bootstrap_diff_ci(a, b, 4000, seed=1)
    assert lo < 2.0 < hi and lo > 1.0 and (lo, hi) == bootstrap_diff_ci(a, b, 4000, seed=1)
