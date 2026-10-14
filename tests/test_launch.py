"""Launcher: whitelist, option validation, quoting, one camera tool at a time, job flow, and the server routes."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from gaze_cursor.dashboard.server import App
from gaze_cursor.sinks import launch
from tests.dashboard_fixtures import make_data

REPO = Path(__file__).resolve().parents[1]
H = {"Host": "127.0.0.1", "X-Token": "tok"}


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(launch, "running_camera_processes", lambda: [])
    launch.JOBS.clear()
    launch._LAST_LIVE[0] = 0.0


def resolve(sid):
    return Path("/data/sessions/session_x.npz") if sid == "sessions:session_x" else None


def test_every_tool_script_exists_and_every_flag_is_one_the_script_accepts() -> None:
    for t in launch.TOOLS.values():
        if t.script:
            text = (REPO / "scripts" / f"{t.script}.py").read_text()
            for flag in [*t.args, *(o.flag for o in t.options if o.flag)]:
                assert flag in text, f"{t.key}: {flag} is not in scripts/{t.script}.py"


def test_frozen_tests_and_dev_scripts_are_not_exposed() -> None:
    words = " ".join(w for t in launch.TOOLS.values() for w in (t.script, *t.args, *(o.flag for o in t.options)))
    for banned in ("--preregistered", "--develop", "tune_smoothing", "diagnose", "explore", "evaluate.py"):
        assert banned not in words
    assert launch.TOOLS["evaluate_status"].args == ("--status",)


def test_benchmark_is_not_offered_because_it_pools_frames_and_logs_unnoted_rows() -> None:
    assert "benchmark" not in launch.TOOLS
    assert not any(t.script == "benchmark" for t in launch.TOOLS.values())


def test_build_makes_argv_from_validated_options() -> None:
    _, argv, env = launch.build("cursor", {"fresh": True, "no_learn": False, "warp_key": "right_option"})
    assert argv == ["scripts/cursor_mode.py", "--fresh", "--warp-key", "right_option"] and env == {}
    _, argv, _ = launch.build("demo", {})
    assert argv == ["scripts/cursor_mode.py", "--demo"]
    _, argv, _ = launch.build("calibrate", {"note": "pos N"})
    assert argv == ["scripts/calibrate.py", "--guided", "--note", "pos N"]
    _, argv, _ = launch.build("practice", {"trials": 30, "no_save": True})
    assert argv == ["scripts/target_practice.py", "--trials", "30", "--no-save"]
    assert launch.build("tests", {})[1] == ["pytest", "-q"]


def test_session_report_needs_a_known_recording_and_defaults_to_no_log() -> None:
    _, argv, _ = launch.build("session_report", {"session": "sessions:session_x"}, resolve)
    assert argv == ["scripts/session_report.py", "/data/sessions/session_x.npz", "--no-log"]
    _, argv, _ = launch.build("session_report", {"session": "sessions:session_x", "no_log": False}, resolve)
    assert "--no-log" not in argv
    for bad in ({}, {"session": "sessions:nope"}, {"session": "../../etc/passwd"}, {"session": 5}):
        with pytest.raises(launch.LaunchError):
            launch.build("session_report", bad, resolve)


@pytest.mark.parametrize("key,options", [
    ("rm", {}), (None, {}), ("cursor", {"evil": True}), ("cursor", {"fresh": "yes"}), ("cursor", {"warp_key": "f5"}),
    ("practice", {"trials": 0}), ("practice", {"trials": 10_000}), ("practice", {"trials": True}),
    ("calibrate", {"note": "x; rm -rf ~"}), ("calibrate", {"note": "$(whoami)"}), ("calibrate", {"note": "a" * 500}),
    ("calibrate", {"note": 5}), ("cursor", [1]), ("benchmark", {}), ("practice", {"trials": "20"}),
    ("calibrate", {"person": "bob"}), ("calibrate", {"glasses": "maybe"}), ("cursor", {"person": "p01; id"}),
])
def test_bad_requests_are_refused(key, options) -> None:
    with pytest.raises(launch.LaunchError):
        launch.build(key, options)


def test_tags_become_environment_only_for_tools_that_take_them() -> None:
    opts = {"person": "p02", "lighting_tag": "lamp", "glasses": "no", "posture_note": "upright, 60 cm"}
    assert launch.build("calibrate", opts)[2] == {"GAZE_PERSON_ID": "p02", "GAZE_LIGHTING_TAG": "lamp",
                                                  "GAZE_GLASSES": "no", "GAZE_POSTURE_NOTE": "upright, 60 cm"}
    assert launch.build("demo", {})[2] == {}


def test_command_script_quotes_everything_and_is_valid_bash(tmp_path) -> None:
    text = launch.command_script(Path("/Users/a b/gaze-cursor"), ["scripts/calibrate.py", "--note", "it's pos N"],
                                 {"GAZE_PERSON_ID": "p01", "GAZE_POSTURE_NOTE": "a'b c"})
    assert "cd '/Users/a b/gaze-cursor'" in text and "export GAZE_PERSON_ID=p01" in text
    assert text.startswith("#!/bin/bash") and "uv run scripts/calibrate.py --note 'it'\"'\"'s pos N'" in text
    f = tmp_path / "x.command"
    f.write_text(text)
    assert subprocess.run(["bash", "-n", str(f)]).returncode == 0


def test_launch_live_writes_a_private_command_file_opens_it_and_logs(tmp_path) -> None:
    opened = []
    tool, argv, env = launch.build("cursor", {"fresh": True})
    res = launch.launch_live(tool, argv, env, tmp_path, {"fresh": True}, tmp_path / "home", opened.append)
    assert res["started"] == "cursor" and len(opened) == 1
    f = opened[0]
    assert f.parent == tmp_path / "home" / "launch" and f.stat().st_mode & 0o777 == 0o700
    assert "uv run scripts/cursor_mode.py --fresh" in f.read_text()
    entry = json.loads((tmp_path / "home" / "launch_log.jsonl").read_text().splitlines()[0])
    assert entry["tool"] == "cursor" and entry["args"] == ["--fresh"] and entry["tags"] == []
    assert (tmp_path / "home" / "launch").stat().st_mode & 0o777 == 0o700


def test_a_second_live_launch_within_the_cooldown_is_refused(tmp_path) -> None:
    opened = []
    tool, argv, env = launch.build("demo", {})
    launch.launch_live(tool, argv, env, tmp_path, {}, tmp_path, opened.append)
    with pytest.raises(launch.LaunchError, match="already running"):
        launch.launch_live(tool, argv, env, tmp_path, {}, tmp_path, opened.append)   # double click / second tab
    assert len(opened) == 1


def test_tag_keys_are_only_accepted_for_tools_that_record_sessions() -> None:
    for key in ("viewer", "demo", "positions", "session_report"):
        with pytest.raises(launch.LaunchError, match="unknown option"):
            launch.build(key, {"person": "p01"}, resolve)


def test_whole_number_options_refuse_fractions_and_strings() -> None:
    for bad in (2.5, "20", None):
        with pytest.raises(launch.LaunchError):
            launch.build("practice", {"trials": bad})
    assert launch.build("practice", {"trials": 20})[1][-2:] == ["--trials", "20"]


def test_the_tests_tool_runs_with_ci_set_so_no_window_opens_on_the_real_display() -> None:
    assert launch.build("tests", {})[2] == {"CI": "1"}


def test_only_one_camera_tool_at_a_time_and_macos_only(tmp_path, monkeypatch) -> None:
    tool, argv, env = launch.build("demo", {})
    monkeypatch.setattr(launch, "running_camera_processes", lambda: ["cursor_mode"])
    with pytest.raises(launch.LaunchError, match="already running"):
        launch.launch_live(tool, argv, env, tmp_path, {}, tmp_path, lambda p: None)
    monkeypatch.setattr(launch, "running_camera_processes", lambda: [])
    monkeypatch.setattr(launch.sys, "platform", "linux")
    with pytest.raises(launch.LaunchError, match="macOS"):
        launch.launch_live(tool, argv, env, tmp_path, {}, tmp_path)


def _wait(job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        j = launch.JOBS[job_id]
        if j.status != "running":
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_headless_job_runs_in_the_background_and_only_one_at_a_time(tmp_path) -> None:
    gate = threading.Event()
    tool, argv, env = launch.build("positions", {})
    job = launch.run_headless(tool, argv, env, tmp_path, {}, tmp_path, runner=lambda a: (gate.wait(5) and "output of " + " ".join(a), 0))
    assert job["status"] == "running"
    with pytest.raises(launch.LaunchError, match="still running"):
        launch.run_headless(tool, argv, env, tmp_path, {}, tmp_path, runner=lambda a: ("", 0))
    gate.set()
    done = _wait(job["id"])
    assert done.status == "done" and "scripts/positions.py --all" in done.output


def test_headless_failure_is_reported(tmp_path) -> None:
    tool, argv, env = launch.build("positions", {})
    job = launch.run_headless(tool, argv, env, tmp_path, {}, tmp_path, runner=lambda a: ("boom", 2))
    done = _wait(job["id"])
    assert done.status == "failed" and done.returncode == 2


# --- through the server -------------------------------------------------------------------------------------
@pytest.fixture()
def app(tmp_path):
    opened = []
    a = App(make_data(tmp_path), token="tok", home=tmp_path / "home", opener=opened.append,
            runner=lambda argv: ("ran " + " ".join(argv), 0))
    a.opened = opened
    return a


def post(app, body, headers=None):
    return app.handle("POST", "/api/launch", headers or H, json.dumps(body).encode())


def test_tools_endpoint_lists_specs_without_paths_or_commands(app) -> None:
    d = json.loads(app.handle("GET", "/api/tools", {"Host": "127.0.0.1"}).body)
    keys = {t["key"] for t in d["tools"]}
    assert {"calibrate", "cursor", "demo", "run", "practice", "check_setup", "viewer", "session_report"} <= keys
    assert d["camera_busy"] == [] and d["macos"] is True
    text = json.dumps(d)
    assert "scripts/" not in text and "uv run" not in text


def test_launch_through_the_server_opens_terminal_script(app) -> None:
    r = post(app, {"tool": "demo", "options": {}})
    assert r.status == 200 and json.loads(r.body)["started"] == "demo" and len(app.opened) == 1


def test_launch_needs_the_token_and_refuses_bad_input_and_busy_camera(app, monkeypatch) -> None:
    assert post(app, {"tool": "demo"}, {"Host": "127.0.0.1"}).status == 403
    assert post(app, {"tool": "rm -rf"}).status == 400
    assert post(app, {"tool": "cursor", "options": {"evil": 1}}).status == 400
    assert app.handle("POST", "/api/launch", H, b"not json").status == 400
    assert app.handle("POST", "/api/launch", H, b"[1]").status == 400
    monkeypatch.setattr(launch, "running_camera_processes", lambda: ["gaze_overlay"])
    busy = post(app, {"tool": "cursor"})
    assert busy.status == 409 and "already running" in json.loads(busy.body)["error"]
    assert app.opened == []


def test_session_report_job_flow_through_the_server(app) -> None:
    r = post(app, {"tool": "session_report", "options": {"session": "sessions:session_20261014_090000"}})
    assert r.status == 200
    job = json.loads(r.body)
    done = _wait(job["id"])
    assert done.status == "done" and "--no-log" in done.output and "session_report.py" in done.output and "session_20261014_090000.npz" in done.output
    polled = json.loads(app.handle("GET", f"/api/job/{job['id']}", {"Host": "127.0.0.1"}).body)
    assert polled["status"] == "done"
    assert app.handle("GET", "/api/job/nope", {"Host": "127.0.0.1"}).status == 404
    assert post(app, {"tool": "session_report", "options": {"session": "sessions:../../x"}}).status == 400
