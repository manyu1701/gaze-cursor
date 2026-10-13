"""Grading of session-quality numbers and the report script (M3 step 6b)."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gaze_cursor.eval.quality import all_metrics, held_out_stats
from gaze_cursor.eval.quality_report import (
    FAIL, NA, PASS, T, WARN, band_max, band_min, format_report, grade, overall,
)
from tests.helpers import fake_session

NAN = float("nan")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "session_report", Path(__file__).resolve().parents[1] / "scripts" / "session_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def good() -> dict:
    """Metrics of a clean front-lit full session."""
    return {
        "n_calib": 16.0, "n_test": 9.0, "calib_extent_x": 0.97, "calib_extent_y": 0.92, "test_outside_hull": 0.0,
        "valid_fraction": 1.0, "thin_targets": 0.0, "min_target_frames": 28.0, "dropped_gap_share": 0.0,
        "heldout_mean": 2.5, "heldout_median": 2.3, "heldout_p90": 4.0, "heldout_worst": 5.0, "heldout_far_share": 0.0,
        "loto_mean": 3.0, "loto_median": 2.8, "loto_p90": 5.0, "loto_worst": 6.0, "loto_far_share": 0.0,
        "bias_deg": 2.0, "spread_deg": 1.0, "bias_share": 0.8, "slope_x": 0.95, "slope_y": 0.8,
        "head_shift_cm": 0.4, "head_rot_deg": 1.2, "late_minus_early_deg": 0.1,
        "face_brightness": 120.0, "lighting_ratio": 1.4, "lighting_label": "front-lit",
        "tags_present": 9.0, "tags_expected": 9.0, "tags_verdict": "pass", "tags_issues": "",
    }


def level(results, check: int) -> str:
    return next(r for r in results if r.check == check).level


def test_bands_lower_and_higher_is_better_and_nan_is_not_scored():
    assert [band_max(v, (3.5, 5.0)) for v in (3.5, 4.0, 5.0, 5.1)] == [PASS, WARN, WARN, FAIL]
    assert [band_min(v, (100.0, 60.0)) for v in (100.0, 80.0, 60.0, 59.9)] == [PASS, WARN, WARN, FAIL]
    assert band_max(NAN, (1, 2)) == NA and band_min(NAN, (1, 2)) == NA
    assert band_max(0.10, (0.10, 0.30), strict=True) == WARN   # strict: share < 0.10 passes


def test_good_session_passes_everything():
    r = grade(good())
    assert [level(r, c) for c in range(1, 7)] == [PASS] * 6 and overall(r) == PASS


def test_dim_bad_session_fails_the_expected_checks():
    m = good() | {"heldout_mean": 8.0, "loto_mean": 8.5, "loto_far_share": 0.4, "face_brightness": 30.0,
                  "slope_x": 0.2}
    r = grade(m)
    assert level(r, 3) == FAIL and level(r, 6) == FAIL and level(r, 4) == WARN
    assert overall(r) == FAIL


def test_check_four_never_fails():
    m = good() | {"slope_x": -1.0, "slope_y": -1.0, "spread_deg": 99.0, "bias_deg": 99.0, "bias_share": 1.0}
    assert level(grade(m), 4) == WARN


def test_check_six_follows_the_tag_validator_verdict():
    assert level(grade(good() | {"tags_verdict": "warn", "tags_present": 5.0}), 6) == WARN
    r = grade(good() | {"tags_verdict": "fail", "tags_present": 8.0, "tags_issues": "person_id: person_id missing"})
    assert level(r, 6) == FAIL and overall(r) == FAIL
    assert "person_id missing" in format_report("x.npz", r)


def test_old_metrics_without_a_verdict_still_warn_not_fail():
    assert level(grade({k: v for k, v in good().items() if k != "tags_verdict"} | {"tags_present": 0.0}), 6) == WARN


def test_quick_session_nans_are_not_scored():
    m = good() | {"heldout_mean": NAN, "heldout_median": NAN, "heldout_p90": NAN, "test_outside_hull": NAN,
                  "n_test": 0.0}
    r = grade(m)
    assert level(r, 3) == PASS and level(r, 1) == PASS
    lines = {ln.name: ln.level for c in r for ln in c.lines}
    assert lines["calibration -> test targets, mean deg"] == NA


def test_nine_dot_single_bad_dot_warns_not_fails():
    assert level(grade(good() | {"loto_far_share": 1 / 9}), 3) == WARN


def test_format_report_lists_every_check():
    text = format_report("session_x.npz", grade(good()))
    assert "OVERALL: PASS" in text
    for title in ("Coverage", "Fixation quality", "Held-out fit", "Bias vs noise", "Drift and head", "Conditions"):
        assert title in text


def test_report_on_a_synthetic_session_runs_end_to_end():
    m = all_metrics(fake_session(0, n_calib=10, n_test=5, frames=12))
    r = grade(m)
    assert len(r) == 6
    assert level(r, 2) == PASS            # every frame valid, no thin target
    assert level(r, 6) == WARN            # no condition tags; brightness absent is n/a, not a fail


def test_script_refuses_face_sidecars_by_name_but_cursor_logs_get_the_tag_check(tmp_path, capsys):
    sr = _load_script()
    faces = "session_20261013_133547.faces.npz"
    assert sr.refuse_path(Path(faces)) is not None
    assert sr.run(tmp_path / faces, None, None) == 2             # refused before the file is even opened
    assert "REFUSED" in capsys.readouterr().err
    assert sr.refuse_path(Path("session_20261013_133547.npz")) is None
    assert sr.refuse_path(Path("cursor_20261013_234559.npz")) is None      # tag-only, see below


def test_cursor_log_gets_only_the_tag_check_and_no_row(tmp_path, monkeypatch, capsys):
    sr = _load_script()
    log = tmp_path / "cursor_20261013_234559.npz"
    np.savez(log, meta=json.dumps({"kind": "cursor_session"}), frames=np.zeros((2, 2)))
    monkeypatch.setattr(sr, "load_session", lambda p: pytest.fail("a Cursor log was loaded as a session"))
    monkeypatch.setattr(sr, "all_metrics", lambda s: pytest.fail("metrics computed for Cursor-mode data"))
    csv_path = tmp_path / "r.csv"
    assert sr.run(log, None, csv_path) == 0
    out = capsys.readouterr().out
    assert "Tag-only check" in out and "CONDITION TAGS: WARN" in out
    assert not csv_path.exists() and not (tmp_path / "cursor_20261013_234559.report.png").exists()


def test_script_refuses_cursor_mode_calibration_by_note():
    from gaze_cursor.eval.quality import refusal
    s = fake_session(0)
    assert refusal(replace(s, meta={**s.meta, "note": "quick calibration (Cursor)"})) is not None


def test_png_is_written_and_summary_row_fields(tmp_path):
    import cv2
    sr = _load_script()
    s = fake_session(0, n_calib=10, n_test=5, frames=8)
    m = all_metrics(s)
    r = grade(m)
    _, loto = held_out_stats(s)
    png = tmp_path / "x.report.png"
    sr.report_png(s, loto, r, png)
    img = cv2.imread(str(png))
    assert img is not None and img.shape[1] == int(1470 * sr.SCALE)
    row = sr.summary_row(s, m, r)
    assert row["mode"] == "session_report" and row["confirmatory"] == 0 and row["session"] == s.path.name
    assert row["n_targets"] == 15 and "calibration->test" in row["note"]
    quick = fake_session(0, n_calib=9, n_test=0, frames=8)
    qrow = sr.summary_row(quick, all_metrics(quick), grade(all_metrics(quick)))
    assert "leave-one-target-out" in qrow["note"] and np.isfinite(qrow["err_deg_mean"])


def test_summary_row_appends_to_a_results_csv(tmp_path):
    from gaze_cursor.eval.results_log import append_rows
    sr = _load_script()
    s = fake_session(0, n_calib=10, n_test=5, frames=8)
    m = all_metrics(s)
    csv_path = tmp_path / "results.csv"
    append_rows([sr.summary_row(s, m, grade(m))], csv_path)
    assert "session_report" in csv_path.read_text()


def test_script_needs_a_session_argument(monkeypatch):
    sr = _load_script()
    monkeypatch.setattr("sys.argv", ["session_report.py"])
    with pytest.raises(SystemExit):
        sr.main()


def test_thresholds_are_pinned_to_the_plan():
    # PLAN.md M3 step 6b "6b thresholds". Changing one needs a logged reason there, so this must fail.
    assert T == {
        "extent_x": (0.90, 0.80), "extent_y": (0.85, 0.80), "outside_hull": (0.0, 0.05),
        "valid_fraction": (0.95, 0.90), "thin_targets": (0.0, 1.0), "dropped_gap_share": (0.02, 0.05),
        "heldout_mean": (3.5, 5.0), "loto_mean": (4.5, 6.0), "far_share": (0.10, 0.30),
        "slope_x_warn": 0.85, "slope_y_warn": 0.25, "spread_warn": 2.5,
        "head_shift_cm": (1.0, 2.0), "head_rot_deg": (3.0, 5.0), "face_brightness": (100.0, 60.0),
    }


@pytest.mark.parametrize("key,value,check,expected", [
    ("calib_extent_y", 0.85, 1, PASS), ("calib_extent_y", 0.84, 1, WARN), ("calib_extent_y", 0.79, 1, FAIL),
    ("calib_extent_y", np.float32(0.8), 1, WARN),                      # float32 0.80000001 stays a warn
    ("test_outside_hull", 0.05, 1, WARN), ("test_outside_hull", 0.06, 1, FAIL),
    ("valid_fraction", 0.95, 2, PASS), ("valid_fraction", 0.90, 2, WARN), ("valid_fraction", 0.89, 2, FAIL),
    ("thin_targets", 1.0, 2, WARN), ("thin_targets", 2.0, 2, FAIL),
    ("dropped_gap_share", 0.05, 2, WARN), ("dropped_gap_share", 0.06, 2, FAIL),
    ("heldout_mean", 3.5, 3, PASS), ("heldout_mean", 5.0, 3, WARN), ("heldout_mean", 5.01, 3, FAIL),
    ("loto_mean", 4.5, 3, PASS), ("loto_mean", 6.0, 3, WARN), ("loto_mean", 6.01, 3, FAIL),
    ("loto_far_share", 0.10, 3, WARN), ("loto_far_share", 0.30, 3, FAIL),
    ("head_shift_cm", 1.0, 5, PASS), ("head_shift_cm", 2.0, 5, WARN), ("head_shift_cm", 2.1, 5, FAIL),
    ("head_rot_deg", 3.0, 5, PASS), ("head_rot_deg", 5.0, 5, WARN), ("head_rot_deg", 5.1, 5, FAIL),
    ("face_brightness", 100.0, 6, PASS), ("face_brightness", 60.0, 6, WARN), ("face_brightness", 59.9, 6, FAIL),
])
def test_band_boundaries_through_grade(key, value, check, expected):
    assert level(grade(good() | {key: float(value)}), check) == expected


def test_info_lines_never_change_a_verdict():
    r = grade(good() | {"late_minus_early_deg": 99.0, "bias_deg": 99.0, "bias_share": 1.0})
    assert level(r, 5) == PASS and level(r, 4) == PASS


def test_run_checks_only_tags_of_a_cursor_calibration_before_computing_any_metric(tmp_path, monkeypatch, capsys):
    sr = _load_script()
    s = fake_session(0)
    monkeypatch.setattr(sr, "load_session", lambda p: replace(s, meta={**s.meta, "note": "quick calibration (Cursor)"}))
    monkeypatch.setattr(sr, "all_metrics", lambda s: pytest.fail("metrics computed for Cursor-mode data"))
    assert sr.run(tmp_path / "session_20261014_000000.npz", None, tmp_path / "r.csv") == 0
    assert not (tmp_path / "r.csv").exists() and "Tag-only check" in capsys.readouterr().out
    # a renamed Cursor log is caught by its recorded kind
    monkeypatch.setattr(sr, "load_session", lambda p: replace(s, meta={**s.meta, "kind": "cursor_session"}))
    assert sr.run(tmp_path / "session_renamed.npz", None, None) == 0


def test_run_appends_exactly_one_row_and_no_log_appends_none(tmp_path, monkeypatch):
    sr = _load_script()
    s = fake_session(0, n_calib=10, n_test=5, frames=8)
    monkeypatch.setattr(sr, "load_session", lambda p: s)
    csv_path = tmp_path / "r.csv"
    assert sr.run(tmp_path / "session_x.npz", tmp_path, csv_path) == 0
    assert len(csv_path.read_text().strip().splitlines()) == 2          # header + one row
    assert (tmp_path / "session_x.report.png").exists()
    other = tmp_path / "none.csv"
    assert sr.run(tmp_path / "session_y.npz", tmp_path, None) == 0
    assert not other.exists()


def test_tag_report_reads_a_sidecar_next_to_a_cursor_log(tmp_path, capsys):
    from gaze_cursor.conditions import sidecar_path
    sr = _load_script()
    log = tmp_path / "cursor_20261014_011059.npz"
    np.savez(log, meta=json.dumps({"kind": "cursor_session"}))
    sidecar_path(log).write_text(json.dumps({"person_id": "p01", "backfilled": True}))
    assert sr.run(log, None, None) == 0
    out = capsys.readouterr().out
    assert "CONDITION TAGS: WARN" in out and "person_id" not in out.split("CONDITION TAGS")[1]
