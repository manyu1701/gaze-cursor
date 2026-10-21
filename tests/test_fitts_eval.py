"""Round 5 analysis (eval/fitts_eval.py, scripts/evaluate_fitts.py) on synthetic task files: a planted effect is
detected, a null is not, and each part of the frozen data rule (and amendment 1) is enforced."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gaze_cursor import conditions as cond
from gaze_cursor.eval import fitts_eval as fe

ROOT = Path(__file__).resolve().parents[1]
CM_PER_PX = 29.0 / 1470
CENTER = (700.0, 450.0)
CAM = {"name": "MacBook Air Camera", "width": 1280, "height": 720}


def synth(path: Path, rng: np.random.Generator, *, effect_cm: float = 0.0, jump_ms: float = 0.0, aborted: bool = False,
          fit_deg: float = 2.3, tracked: float = 0.99, seat_fail: int | None = None, refuse_moved: bool = False,
          person: str = "p01", n_jump_trials: int = 9, jump_miss_every: int = 0, fast_trackpad: bool = False,
          camera: dict | None = None, aim_off_display: bool = False) -> Path:
    """One session: 2 practice + 16 scored blocks of 9 targets; the jump's aim error is 3 cm (+ ``effect_cm`` in the
    moved seat), the jump takes ``jump_ms`` longer (negative: faster) than the trackpad's 900 ms on long targets."""
    session_shift = rng.normal(0, 0.2)
    blocks, trials = [], []
    combos = [(m, s, lay, w) for s in ("centre", "moved") for m in ("trackpad", "gaze_jump")
              for lay in ("ring", "line") for w in (70, 140)]
    plan = [("trackpad", "centre", "ring", 140, True), ("gaze_jump", "centre", "ring", 140, True)]
    plan += [(*c, False) for c in combos]
    for i, (m, s, lay, w, practice) in enumerate(plan):
        first = len(trials)
        blocks.append({"index": i, "method": m, "seat": s, "layout": lay, "practice": practice, "width_px": w,
                       "amplitude_px": 600 if lay == "ring" else 1100, "n_trials": 9, "first_trial": first,
                       "seat_ok_during_block": not (seat_fail is not None and i == seat_fail)})
        for k in range(9):
            tr = {"index": first + k, "block": i, "center": CENTER, "width_px": w, "hit": True, "timed_out": False,
                  "warp_target": None, "warp_landing": None, "refused_taps": [], "t_warp_ms": None}
            mt = 900 + rng.normal(0, 120)
            if m == "gaze_jump":
                if s == "moved" and refuse_moved and k < 3:
                    tr["refused_taps"] = [[0.0, "eyes outside the calibrated range"]]
                elif k < n_jump_trials:
                    err = 3.0 + (effect_cm if s == "moved" else 0.0) + session_shift + rng.normal(0, 0.8)
                    ang = rng.uniform(0, 2 * np.pi)
                    off = abs(err) / CM_PER_PX
                    tr["warp_target"] = [CENTER[0] + off * np.cos(ang), CENTER[1] + off * np.sin(ang)]
                    tr["warp_landing"] = tr["warp_target"]
                    if aim_off_display:
                        tr["warp_target"] = [-50.0, 300.0]
                if lay == "line":
                    mt += jump_ms
                    if jump_miss_every and k % jump_miss_every == 0:
                        tr["hit"] = False
            elif fast_trackpad and lay == "line" and s == "centre" and k in (1, 2):
                mt = 100.0
            tr["movement_time_ms"] = float(mt)
            trials.append(tr)
    meta = {"kind": "fitts_task", "screen": {"width_px": 1470, "height_px": 956, "width_cm": 29.0},
            "summary": {"tracked": tracked}, "conditions": {"person_id": person, "camera": camera or CAM},
            "task": {"aborted": aborted, "calibration_fit_deg": fit_deg, "blocks": blocks, "trials": trials,
                     "dots": [], "seat_first": "centre", "method_first": "trackpad", "moved_side": "left",
                     "learning": False}}
    np.savez(path, meta=json.dumps(meta), frames=np.zeros((2, 15)))
    return path


@pytest.fixture(autouse=True)
def _tag_verdict_pass(monkeypatch):
    monkeypatch.setattr(cond, "conditions_block", lambda meta, npz=None: meta.get("conditions"))
    monkeypatch.setattr(cond, "validate", lambda block: [])
    monkeypatch.setattr(cond, "verdict", lambda issues: "pass")


def load(tmp_path: Path, name: str, rng, **kw) -> fe.FittsFile:
    return fe.load(synth(tmp_path / name, rng, **kw))


def sessions(tmp_path: Path, n: int = 5, **kw) -> list[fe.SessionStats]:
    rng = np.random.default_rng(1)
    return [fe.session_stats(load(tmp_path, f"fitts_2026102{3 + k // 2}_0{k}0000.npz", rng, **kw)) for k in range(n)]


def test_a_planted_head_cost_and_jump_gain_are_detected(tmp_path) -> None:
    res = fe.decide(sessions(tmp_path, effect_cm=3.0, jump_ms=-400.0), n_boot=1000)
    assert res["H7a"]["supported"] and res["H7a"]["mean"] == pytest.approx(3.0, abs=0.6)
    assert res["H7b"]["supported"] and res["H7b"]["mean"] == pytest.approx(-400.0, abs=120)
    assert res["H7a"]["ci"][0] > 0 and res["H7b"]["ci"][1] < 0


def test_a_null_is_not_detected_and_each_criterion_can_fail_alone(tmp_path) -> None:
    res = fe.decide(sessions(tmp_path), n_boot=1000)
    assert not res["H7a"]["supported"] and not res["H7b"]["supported"]
    assert res["H7a"]["ci"][0] < 0 < res["H7a"]["ci"][1]
    small = fe.decide(sessions(tmp_path, effect_cm=0.5, jump_ms=-60.0), n_boot=1000)    # real but below the effect floor
    assert not small["H7a"]["criteria"]["mean d_a >= 1.0 cm"] and not small["H7a"]["supported"]
    assert not small["H7b"]["criteria"]["mean d_b <= -150 ms"] and not small["H7b"]["supported"]
    slower = fe.decide(sessions(tmp_path, jump_ms=+400.0), n_boot=1000)                 # the pilot's direction
    assert not slower["H7b"]["supported"] and slower["H7b"]["mean"] > 0


def test_the_miss_rate_criterion_blocks_a_faster_but_sloppier_jump(tmp_path) -> None:
    sloppy = sessions(tmp_path, effect_cm=3.0, jump_ms=-400.0, jump_miss_every=2)       # 5 of 9 jump targets missed
    assert all(s.miss_pp > 5 for s in sloppy)
    res = fe.decide(sloppy, n_boot=500)
    assert not res["H7b"]["criteria"]["mean miss-rate difference <= +5 pp"] and not res["H7b"]["supported"]
    assert res["H7b"]["criteria"]["percentile CI upper < 0"]                            # speed alone would have passed


def test_practice_and_each_blocks_first_target_are_never_analysed(tmp_path) -> None:
    f = load(tmp_path, "fitts_20261023_010000.npz", np.random.default_rng(0))
    rows = fe._rows(f)
    assert len(rows) == 16 * 8                                  # 16 scored blocks x (9 - the first target)
    assert not any(b["practice"] or tr["index"] == b["first_trial"] for tr, b in rows)


def test_fast_hits_leave_the_time_measure_but_stay_in_the_miss_rate_and_are_counted(tmp_path) -> None:
    f = load(tmp_path, "fitts_20261023_010000.npz", np.random.default_rng(0), fast_trackpad=True)
    cells = fe.cells_b(f)
    assert all((c[0] >= fe.MIN_MT_MS).all() for c in cells)
    assert fe.reported(f)["fast_hits_excluded"]["trackpad"] > 0
    assert fe.miss_rate_pp(f) == 0.0                            # no misses: the fast hits are hits, not dropped


def test_a_cell_with_under_three_jump_trials_is_dropped_for_both_seats(tmp_path) -> None:
    rng = np.random.default_rng(0)
    f = load(tmp_path, "fitts_20261023_010000.npz", rng)
    assert len(fe.cells_a(f)) == 4 and len(fe.cells_b(f)) == 2
    for tr in f.task["trials"]:                                 # strip the jumps from one moved-seat cell
        b = f.task["blocks"][tr["block"]]
        if b["method"] == "gaze_jump" and b["seat"] == "moved" and b["layout"] == "ring" and b["width_px"] == 70:
            tr["warp_target"] = tr["warp_landing"] = None
    assert len(fe.cells_a(f)) == 3


def test_refusals_are_counted_not_dropped_and_off_display_aim_is_flagged(tmp_path) -> None:
    f = load(tmp_path, "fitts_20261023_010000.npz", np.random.default_rng(0), refuse_moved=True)
    r = fe.reported(f)
    assert r["moved"]["refusal_rate"] > 0.2 and r["centre"]["refusal_rate"] == 0 and r["moved"]["share_off_display"] == 0
    g = load(tmp_path, "fitts_20261023_020000.npz", np.random.default_rng(0), aim_off_display=True)
    assert fe.reported(g)["centre"]["share_off_display"] == 1.0


def test_an_aborted_run_with_an_unfinished_block_does_not_crash(tmp_path) -> None:
    f = load(tmp_path, "fitts_20261023_010000.npz", np.random.default_rng(0), aborted=True)
    f.task["trials"].append({**f.task["trials"][-1], "index": 999, "block": 99})        # trial of a block with no entry
    assert len(fe._rows(f)) == 16 * 8


def test_the_data_rule_names_its_reason(tmp_path) -> None:
    rng = np.random.default_rng(0)

    def check(**kw):
        return fe.qualifies(load(tmp_path, "fitts_20261023_010000.npz", rng, **kw))
    assert check() is None
    assert "aborted" in check(aborted=True)
    assert "calibration fit" in check(fit_deg=3.4)
    assert "tracked" in check(tracked=0.7)
    assert "seat rule" in check(seat_fail=5)
    assert "not p01" in check(person="p02")
    assert "accepted jumps" in check(n_jump_trials=2)
    assert "before the freeze" in fe.qualifies(load(tmp_path, "fitts_20261022_041259.npz", rng))
    ref = (CAM["name"], 1280, 720)
    assert fe.qualifies(load(tmp_path, "fitts_20261023_010000.npz", rng), reference_camera=ref) is None
    other = {"name": "External webcam", "width": 1920, "height": 1080}
    assert "differs" in fe.qualifies(load(tmp_path, "fitts_20261023_010000.npz", rng, camera=other), reference_camera=ref)


def test_a_tag_verdict_other_than_pass_disqualifies(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cond, "verdict", lambda issues: "warn")
    why = fe.qualifies(load(tmp_path, "fitts_20261023_010000.npz", np.random.default_rng(0)))
    assert why is not None and "WARN" in why


def test_selection_takes_post_freeze_files_in_time_order_with_a_daily_cap_and_a_failure_cap(tmp_path) -> None:
    rng = np.random.default_rng(0)
    names = ["20261022_010000", "20261023_090000", "20261023_100000", "20261023_110000", "20261024_090000",
             "20261024_100000", "20261025_090000", "20261026_090000"]
    fs = [load(tmp_path, f"fitts_{n}.npz", rng) for n in names]
    sel = fe.select(fs)
    # pre-freeze file ignored; the 3rd file on the 23rd is skipped, not a failure; five qualify
    assert [f.started for f in sel.counted] == ["20261023_090000", "20261023_100000", "20261024_090000",
                                                 "20261024_100000", "20261025_090000"]
    assert any("session no. 3 on 20261023" in n for n in sel.notes) and sel.stopped is None
    bad = [load(tmp_path, f"fitts_2026102{3 + k}_090000.npz", rng, fit_deg=3.5) for k in range(4)]
    stopped = fe.select(bad + [load(tmp_path, "fitts_20261030_090000.npz", rng)])
    assert stopped.stopped is not None and "4 failed" in stopped.stopped and not stopped.counted
    three = fe.select(bad[:3] + [load(tmp_path, f"fitts_2026103{k}_090000.npz", rng) for k in range(5)])
    assert three.stopped is None and len(three.counted) == 5                          # 3 failures are allowed
    odd = {"name": "External webcam", "width": 1920, "height": 1080}
    mixed = fe.select([load(tmp_path, "fitts_20261030_090000.npz", rng),
                       load(tmp_path, "fitts_20261031_090000.npz", rng, camera=odd)])
    assert len(mixed.counted) == 1 and "differs" in mixed.notes[0]
    # a failed first file still sets the camera reference, and failed files count toward the daily cap
    first_bad = load(tmp_path, "fitts_20261101_090000.npz", rng, fit_deg=3.5)
    rest = [load(tmp_path, "fitts_20261101_100000.npz", rng), load(tmp_path, "fitts_20261101_110000.npz", rng)]
    sel2 = fe.select([first_bad] + rest)
    assert [f.started for f in sel2.counted] == ["20261101_100000"] and any("session no. 3" in n for n in sel2.notes)


def test_an_early_aborted_post_freeze_run_is_a_failed_session_not_a_pilot(tmp_path) -> None:
    rng = np.random.default_rng(0)
    path = synth(tmp_path / "fitts_20261023_090000.npz", rng, aborted=True)
    d = json.loads(str(np.load(path)["meta"]))
    d["task"]["blocks"], d["task"]["trials"] = [], []                 # aborted during the dot block: nothing finished
    np.savez(path, meta=json.dumps(d), frames=np.zeros((2, 15)))
    f = fe.load(path)                                                 # loads (only a PRE-freeze old file is a pilot)
    assert "aborted" in fe.qualifies(f)
    sel = fe.select([f] + [load(tmp_path, f"fitts_2026102{4 + k}_090000.npz", rng) for k in range(5)])
    assert len(sel.counted) == 5 and any("FAILED" in n and "aborted" in n for n in sel.notes)
    pilot = tmp_path / "fitts_20261022_041259.npz"                    # old format, before the freeze: not analysable
    np.savez(pilot, meta=json.dumps({"kind": "fitts_task", "task": {"blocks": [{"index": 0}], "trials": []}}),
             frames=np.zeros((2, 15)))
    with pytest.raises(ValueError, match="older version"):
        fe.load(pilot)


def test_one_unusable_width_leaves_d_b_over_the_other(tmp_path) -> None:
    f = load(tmp_path, "fitts_20261023_010000.npz", np.random.default_rng(0))
    for tr in f.task["trials"]:                                       # no usable gaze-jump hit at width 70, long, centre
        b = f.task["blocks"][tr["block"]]
        if b["method"] == "gaze_jump" and b["seat"] == "centre" and b["layout"] == "line" and b["width_px"] == 70:
            tr["hit"] = False
    assert len(fe.cells_b(f)) == 1 and fe.reported(f)["widths_used_b"] == 1
    assert not np.isnan(fe.cell_difference(fe.cells_b(f)))


def test_undefined_differences_stop_the_analysis(tmp_path) -> None:
    ss = sessions(tmp_path)
    ss[0].d_a = float("nan")
    with pytest.raises(ValueError, match="undefined"):
        fe.decide(ss, n_boot=100)


def test_t_interval_is_wider_than_the_percentile_bootstrap_with_five_sessions() -> None:
    v = np.array([1.0, 1.4, 0.6, 1.2, 0.8])
    boot = fe.bootstrap_two_level([[(np.array([0.0, 0.0]), np.array([x, x]))] for x in v], n_boot=2000)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    t_lo, t_hi = fe.t_interval(v)
    assert (t_hi - t_lo) > (hi - lo)                           # why both intervals are required


def _script():
    spec = importlib.util.spec_from_file_location("evaluate_fitts_script", ROOT / "scripts" / "evaluate_fitts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_status_prints_no_differences_develop_refuses_test_data_and_the_confirmatory_run_is_logged_once_ready(
        tmp_path, monkeypatch, capsys) -> None:
    sc = _script()
    monkeypatch.setattr(sc, "TASKS_DIR", tmp_path)
    monkeypatch.setattr(sc, "in_worktree", lambda: False)
    rng = np.random.default_rng(0)
    synth(tmp_path / "fitts_20261022_030000.npz", rng)                         # before the freeze: development data
    for k in range(3):
        synth(tmp_path / f"fitts_2026102{3 + k}_0{k}0000.npz", rng, effect_cm=3.0, jump_ms=-400.0)
    (tmp_path / "fitts_20261023_000000.faces.npz").write_bytes(b"x")          # crops are not logs
    assert sc.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert out.count("COUNTS") == 3 and "3 of 5" in out and "d_a" not in out and "schedule seat-first" in out
    assert "recorded before the freeze (development data)" in out
    assert sc.main(["--develop"]) == 0                    # post-freeze files: refused, no numbers; pre-freeze: numbers
    out = capsys.readouterr().out
    assert out.count("refused, recorded after the freeze") == 3 and out.count("EXPLORATORY") == 1
    assert sc.main(["--preregistered-7"]) == 2                                 # only 3 of 5
    assert "Refusing: 3 of 5" in capsys.readouterr().err
    assert sc.main(["--preregistered-7", "--no-log"]) == 2                     # never an unrecorded confirmatory run
    assert "always logged" in capsys.readouterr().err
    monkeypatch.setattr(sc, "in_worktree", lambda: True)
    assert sc.main(["--preregistered-7"]) == 2                                 # not from a linked worktree
    assert "main checkout" in capsys.readouterr().err
    monkeypatch.setattr(sc, "in_worktree", lambda: False)
    for k in range(3, 5):
        synth(tmp_path / f"fitts_2026102{3 + k}_0{k}0000.npz", rng, effect_cm=3.0, jump_ms=-400.0)
    rows: list = []
    monkeypatch.setattr(sc, "append_rows", rows.extend)
    assert sc.main(["--preregistered-7"]) == 0
    out = capsys.readouterr().out
    assert "H7a is SUPPORTED" in out and "H7b is SUPPORTED" in out
    assert [r["mode"] for r in rows] == ["h7a_jump_task", "h7b_jump_task"] and all(r["confirmatory"] == 1 for r in rows)
    assert rows[0]["err_cm_mean"] == pytest.approx(3.0, abs=0.7) and "t-interval" in rows[1]["note"]
    assert rows[0]["n_targets"] > 0 and rows[0]["n_targets"] != rows[1]["n_targets"]    # each row counts its own trials


def test_a_run_that_cannot_give_a_verdict_is_still_logged_as_the_one_run(tmp_path, monkeypatch, capsys) -> None:
    sc = _script()
    monkeypatch.setattr(sc, "TASKS_DIR", tmp_path)
    monkeypatch.setattr(sc, "in_worktree", lambda: False)
    rng = np.random.default_rng(0)
    for k in range(5):
        synth(tmp_path / f"fitts_2026102{3 + k}_090000.npz", rng)
    rows: list = []
    monkeypatch.setattr(sc, "append_rows", rows.extend)
    monkeypatch.setattr(sc.fe, "decide", lambda stats: (_ for _ in ()).throw(ValueError("d_a undefined")))
    assert sc.main(["--preregistered-7"]) == 2
    assert "NO VERDICT" in rows[0]["note"] and rows[0]["confirmatory"] == 1
