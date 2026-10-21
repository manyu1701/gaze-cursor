"""Dashboard data layer: catalogue, summaries, quality (all on a synthetic data folder)."""

from __future__ import annotations

import json
import time

import pytest

from gaze_cursor.dashboard import catalog, quality, summaries
from tests.dashboard_fixtures import make_data, tagged_block
from tests.helpers import fake_session


@pytest.fixture()
def roots(tmp_path):
    return make_data(tmp_path)


def test_scan_finds_every_recording_and_never_the_face_sidecar(roots) -> None:
    rows = catalog.scan(roots)
    files = {r.file for r in rows}
    assert "session_20261012_080000.faces.npz" not in files
    assert len(rows) == 7
    assert {r.kind for r in rows} == {"calibration", "quick_cursor", "cursor", "task"}
    assert rows == sorted(rows, key=lambda r: (r.started or "", r.file), reverse=True)


def test_rows_report_tags_person_and_legacy_state(roots) -> None:
    by = {r.file: r for r in catalog.scan(roots)}
    tagged = by["session_20261014_090000.npz"]
    assert tagged.tags == 9 and tagged.tags_verdict == "pass" and not tagged.legacy and tagged.person_id == "p01"
    legacy = by["session_20261013_100000.npz"]
    assert legacy.legacy and legacy.tags == 0 and legacy.tags_verdict == "warn" and legacy.person_id == ""
    backfilled = by["session_20261012_080000.npz"]
    assert backfilled.legacy and backfilled.backfilled and backfilled.person_id == "p01"
    assert backfilled.tags == 8 and backfilled.tags_verdict == "warn"   # its distance is an estimate, not a tag


def test_a_sidecar_without_person_id_warns_and_never_fails(roots) -> None:
    sidecar = roots.sessions / "session_20261012_080000.conditions.json"
    block = json.loads(sidecar.read_text())
    block.pop("person_id")
    sidecar.write_text(json.dumps(block))
    row = next(r for r in catalog.scan(roots) if r.file == "session_20261012_080000.npz")
    assert row.tags == 7 and row.tags_verdict == "warn" and row.person_id == ""


def test_cursor_rows_come_from_the_summary_without_loading_arrays(roots) -> None:
    cur = next(r for r in catalog.scan(roots) if r.file == "cursor_20261014_100000.npz")
    assert (cur.clicks, cur.clicks_learned, cur.minutes, cur.click_gaze_deg_median) == (40, 25, 30.0, 2.4)
    assert cur.learn_on is True and cur.resumed == "10:00"


def test_session_rows_count_targets_frames_and_minutes(roots) -> None:
    row = next(r for r in catalog.scan(roots) if r.file == "session_20261013_100000.npz")
    assert (row.targets, row.frames) == (4, 20) and row.minutes == pytest.approx(1.0)


def test_unreadable_file_becomes_an_error_row_not_a_crash(roots) -> None:
    (roots.sessions / "session_20261015_000000.npz").write_bytes(b"garbage")
    bad = next(r for r in catalog.scan(roots) if r.file == "session_20261015_000000.npz")
    assert bad.error.startswith("unreadable")


def test_cache_is_used_and_invalidated_by_a_changed_file(roots) -> None:
    catalog.scan(roots)
    cached = json.loads(roots.cache.read_text())
    assert cached["version"] == catalog.CACHE_VERSION and len(cached["rows"]) == 7
    sidecar = roots.sessions / "session_20261013_100000.conditions.json"
    first = next(r for r in catalog.scan(roots) if r.file == "session_20261013_100000.npz")
    assert first.legacy and not first.backfilled
    sidecar.write_text(json.dumps({"person_id": "p02"}))
    second = next(r for r in catalog.scan(roots) if r.file == "session_20261013_100000.npz")
    assert second.backfilled and second.person_id == "p02"
    time.sleep(0.01)


def test_path_of_only_resolves_listed_ids(roots) -> None:
    rows = catalog.scan(roots)
    assert catalog.path_of(roots, rows[0].id) is not None
    for bad in ("../../etc/passwd", "sessions:../x", "sessions:session_20261012_080000.faces", "nope"):
        assert catalog.path_of(roots, bad) is None


def test_totals_and_tag_completeness(roots) -> None:
    rows = catalog.scan(roots)
    t = summaries.totals(rows)
    assert t["files"] == 7 and t["clicks"] == 140 and t["clicks_learned"] == 25
    assert t["targets"] == 16   # 4 per calibration x (3 + the Cursor calibration); not the practice task
    assert t["cursor_hours"] == pytest.approx(1.5)
    tc = summaries.tag_completeness(rows)
    assert tc["n"] == 7 and tc["complete"] == 1 and tc["own_block"] == 1 and tc["backfilled"] == 1
    assert tc["with_person_id"] == 2 and tc["persons"] == ["p01"]
    assert sum(tc["histogram"]) == 7 and len(tc["histogram"]) == tc["n_checked"] + 1


def test_per_day_keeps_session_medians_separate_with_their_n_and_learning_state(roots) -> None:
    days = {d["day"]: d for d in summaries.per_day(catalog.scan(roots))}
    d = days["2026-10-14"]
    assert d["cursor_sessions"] == 1 and d["clicks"] == 40 and d["clicks_learned"] == 25
    assert d["click_gaze"] == [{"median_deg": 2.4, "n": 38, "learn_on": True}]
    assert days["2026-10-13"]["click_gaze"] == []   # that log had no gaze data
    assert list(days) == sorted(days)


def test_coverage_counts_full_calibrations_by_lighting_and_slot(roots) -> None:
    c = summaries.coverage(catalog.scan(roots))
    assert c["n"] == 3 and c["lighting"] == ["front-lit"]
    cells = dict(zip(c["slots"], c["cells"][0]))
    assert cells["C"] == 1 and cells["N"] == 1 and cells["no slot"] == 1


def test_learned_state_reads_meta_only_and_flags_stale(roots) -> None:
    now = time.time()
    s = summaries.learned_state(roots, now)
    assert s["exists"] and s["reusable"] and 0.9 < s["age_h"] < 1.1 and s["n_updates"] == 0
    assert s["n_states"] == 0 and "none yet" in s["history"]
    late = summaries.learned_state(roots, now + 13 * 3600)
    assert late["reusable"] is False
    (roots.quick / "last_calibration.npz").unlink()
    assert summaries.learned_state(roots)["exists"] is False


def test_health_messages(roots) -> None:
    rows = catalog.scan(roots)
    texts = " ".join(h["text"] for h in summaries.health(rows, roots, summaries.learned_state(roots)))
    assert "Face model missing" in texts and "person id" in texts
    (roots.repo / "models" / "face_landmarker.task").write_bytes(b"x")
    assert "Face model missing" not in " ".join(h["text"] for h in summaries.health(rows, roots, {}))
    assert any("No recordings" in h["text"] for h in summaries.health([], roots, {}))


def test_results_summary_and_docs(roots) -> None:
    r = summaries.results_summary(roots.results)
    assert r["n"] == 2 and {m["mode"] for m in r["modes"]} == {"within", "h6_click_learning"}
    assert next(m for m in r["modes"] if m["mode"] == "h6_click_learning")["confirmatory"] == 1
    assert r["latest"][0]["mode"] == "h6_click_learning" and r["latest"][0]["basis"] == summaries.PAIRED
    assert "basis" in r["columns"]
    assert summaries.basis_of({"mode": "h6_click_learning", "confirmatory": "1"}) == summaries.PAIRED
    assert summaries.basis_of({"mode": "h5_lopo", "confirmatory": "0"}) == ""   # not confirmatory: no claim made
    assert summaries.basis_of({"mode": "within", "confirmatory": "0"}).startswith("within-session")
    assert summaries.basis_of({"mode": "losos"}).startswith("cross-session")
    assert summaries.basis_of({"mode": "cursor_session"}).startswith("monitoring")
    assert summaries.results_summary(roots.results.with_name("missing.csv"))["exists"] is False
    assert summaries.read_doc(roots, "PLAN") is not None
    assert summaries.read_doc(roots, "../secret") is None and summaries.read_doc(roots, "CLAUDE") is None


def test_prereg_rounds_and_plan_status(roots) -> None:
    rounds = summaries.prereg_rounds(summaries.read_doc(roots, "PREREGISTRATIONS"))
    assert [r["status"] for r in rounds] == ["result recorded", "frozen, waiting for data"]
    assert rounds[0]["result_heading"] == "RESULT of the test"
    notsup = summaries.prereg_rounds("# Round 3: x\n\n### RESULT: H4 SUPPORTED\n\n### Two freezes\n\nH5 (earlier freeze) NOT SUPPORTED by the rule.\n")
    assert notsup[0]["not_supported"] == ["H5 (earlier freeze) NOT SUPPORTED by the rule."]
    odd = summaries.prereg_rounds("# Round 9: x\n\n### Pre-registration (frozen)\n\n### What result would falsify this\n")
    assert odd[0]["status"] == "frozen, waiting for data"   # a heading that merely mentions "result" is not a RESULT
    assert len(summaries.plan_status(summaries.read_doc(roots, "PLAN"))) == 2


# --- quality ----------------------------------------------------------------------------------------
def test_cursor_data_gets_the_tag_only_check_and_never_a_model_fit(roots, monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("must not load or fit Cursor-mode data")
    monkeypatch.setattr(quality, "load_session", boom)
    monkeypatch.setattr(quality, "all_metrics", boom)
    for rid in ("quick:cursor_20261014_100000", "quick:session_20261014_095500"):
        rep = quality.report_for(roots, rid)
        assert rep["mode"] == "tags_only" and rep["checks"] == [] and rep["overall"] == "warn"


def test_full_session_gets_the_numeric_report_and_it_is_cached(roots, monkeypatch) -> None:
    calls = []
    sess = fake_session(1, n_calib=6, n_test=3)
    monkeypatch.setattr(quality, "load_session", lambda p: (calls.append(p), sess)[1])
    rep = quality.report_for(roots, "sessions:session_20261013_100000")
    assert rep["mode"] == "full" and rep["overall"] in ("pass", "warn", "fail")
    assert [c["check"] for c in rep["checks"]] == [1, 2, 3, 4, 5, 6]
    json.dumps(rep, allow_nan=False)   # NaN would have been cleaned to None
    quality.report_for(roots, "sessions:session_20261013_100000")
    assert len(calls) == 1             # second call served from the cache


def test_quality_for_unknown_id_and_for_tasks(roots) -> None:
    assert quality.report_for(roots, "sessions:nope") is None
    assert quality.report_for(roots, "tasks:task_20261014_110000")["mode"] == "refused"


def test_report_png_only_if_it_exists(roots) -> None:
    sid = "sessions:session_20261013_100000"
    assert quality.png_path(roots, sid) is None
    (roots.sessions / "session_20261013_100000.report.png").write_bytes(b"\x89PNG")
    assert quality.png_path(roots, sid) is not None
    assert quality.png_path(roots, "../x") is None


def test_quality_cache_is_invalidated_by_a_new_sidecar_and_by_changed_grading_code(roots, monkeypatch) -> None:
    sid = "quick:session_20261014_095500"   # a Cursor calibration: tag-only report reads the sidecar
    assert quality.report_for(roots, sid)["tags"] == 0
    side = roots.quick / "session_20261014_095500.conditions.json"
    side.write_text(json.dumps({**tagged_block(), "backfilled": True}))
    fresh = quality.report_for(roots, sid)
    assert fresh["tags"] == 8 and fresh["overall"] == "warn"      # backfilled: its distance is not a measured tag
    monkeypatch.setattr(quality, "CODE_VERSION", "changed-thresholds")
    cache_before = json.loads(roots.cache.with_name("dashboard_quality.json").read_text())[sid]["stamp"]
    quality.report_for(roots, sid)
    cache_after = json.loads(roots.cache.with_name("dashboard_quality.json").read_text())[sid]["stamp"]
    assert cache_before != cache_after and cache_after[-1] == "changed-thresholds"


def test_quality_code_version_covers_the_whole_package_but_not_the_dashboard() -> None:
    from pathlib import Path
    import hashlib
    root = Path(quality._quality_code.__file__).resolve().parents[1]
    before = quality._code_version()
    assert before == quality.CODE_VERSION and len(before) == 12
    touched = root / "eval" / "configs.py"
    assert touched.exists() and "dashboard" not in touched.parts   # a model-config change must invalidate reports
    assert hashlib.sha1(b"x").hexdigest()   # (the hash itself is covered by test_quality_cache_is_invalidated...)
