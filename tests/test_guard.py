"""The Claude Code guard hook (.claude/hooks/guard.py) must work from worktrees and leave other repos alone."""

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / ".claude" / "hooks" / "guard.py"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, GUARD)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A main checkout with a PREREGISTRATIONS.md, one worktree, a second unrelated repo, and the hook loaded
    for the main checkout (as Claude Code runs it, CLAUDE_PROJECT_DIR = the main checkout)."""
    main, other = tmp_path / "main", tmp_path / "notes"
    for r in (main, other):
        r.mkdir()
        _git(r, "init", "-q")
    (main / "PREREGISTRATIONS.md").write_text("# Round 4\n- frozen line\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "init")
    wt = main / ".claude" / "worktrees" / "w"
    _git(main, "worktree", "add", "-q", "-b", "w", str(wt))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(main))
    guard = _load("guard_under_test")
    return guard, main, wt, other


def test_the_hook_stays_valid_python_3_9() -> None:
    ast.parse(GUARD.read_text(), feature_version=(3, 9))     # macOS /usr/bin/python3 is 3.9


def test_the_lock_lives_in_the_main_checkout(repo) -> None:
    guard, main, _, _ = repo
    assert guard.MAIN.resolve() == main.resolve()
    assert guard.LOCK.resolve() == (main / ".claude" / "state" / "session.lock").resolve()


@pytest.mark.parametrize("tree", ["main", "wt"])
def test_recorded_data_is_protected_in_the_checkout_and_in_worktrees(repo, tree) -> None:
    guard, main, wt, _ = repo
    base = main if tree == "main" else wt
    hit = {"tool_name": "Write", "tool_input": {"file_path": str(base / "data/quick/cursor_x.npz"), "content": ""}}
    ok = {"tool_name": "Write", "tool_input": {"file_path": str(base / "src/x.py"), "content": ""}}
    assert guard.protect(hit) == 2 and guard.protect(ok) == 0


@pytest.mark.parametrize("tree", ["main", "wt"])
def test_preregistrations_are_append_only_in_the_checkout_and_in_worktrees(repo, tree) -> None:
    guard, main, wt, _ = repo
    f = str((main if tree == "main" else wt) / "PREREGISTRATIONS.md")
    append = {"tool_name": "Edit", "tool_input": {"file_path": f, "old_string": "- frozen line\n",
                                                   "new_string": "- frozen line\n- session 2 logged\n"}}
    change = {"tool_name": "Edit", "tool_input": {"file_path": f, "old_string": "frozen line", "new_string": "edited"}}
    assert guard.protect(append) == 0 and guard.protect(change) == 2


def test_other_repos_are_not_guarded(repo) -> None:
    guard, _, _, other = repo
    (other / "PREREGISTRATIONS.md").write_text("old\n")
    edit = {"tool_name": "Write", "tool_input": {"file_path": str(other / "PREREGISTRATIONS.md"), "content": "new\n"}}
    assert guard.protect(edit) == 0
    commit = {"cwd": str(other), "tool_input": {"command": f"git -C {other} commit -m notes"}}
    assert guard.pre_commit(commit) == 0                      # returns before running any tests


@pytest.mark.parametrize("cmd, expect", [
    ("git commit -m x", "."),
    ("git -C sub commit -m x", "sub"),
    ("cd sub && git add -A && git commit -m 'a | b'", "sub"),
    ("git -c user.name=x commit", "."),
    ("GIT_AUTHOR_NAME=x git commit", "."),
    ("git log --grep commit", None),
    ("git status", None),
    ("echo git commit", None),
])
def test_commit_dir_follows_cd_and_dash_c(tmp_path, cmd, expect) -> None:
    guard = _load("guard_parse")
    got = guard.commit_dir(cmd, tmp_path)
    assert (got is None) if expect is None else (Path(got).resolve() == (tmp_path / expect).resolve())


def test_the_hook_runs_as_a_script(repo) -> None:
    _, main, wt, _ = repo
    data = json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(wt / "models/x.task"), "content": ""}})
    r = subprocess.run(["python3", str(GUARD), "protect"], input=data, capture_output=True, text=True,
                       env={"CLAUDE_PROJECT_DIR": str(main), "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
    assert r.returncode == 2 and "recorded data" in r.stderr


# ---- reviewers run per commit, not per step (REVIEW_RULES) ----

def _scoped_file(base: Path, name: str = "src/gaze_cursor/eval/x.py", text: str = "a = 1\n") -> Path:
    f = base / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return f


def _launch(guard, base: Path, agent: str = "stats-reviewer", tool: str = "Agent") -> None:
    guard.review_record({"tool_name": tool, "cwd": str(base), "tool_input": {"subagent_type": agent}})


def test_no_review_needed_when_nothing_scoped_changed(repo) -> None:
    guard, main, _, _ = repo
    (main / "PLAN.md").write_text("docs only\n")
    _scoped_file(main, "src/gaze_cursor/other.py")                      # not review-scoped
    assert guard.review_gate("git commit -am x", main) is None


def test_a_scoped_change_blocks_until_the_reviewer_was_launched(repo) -> None:
    guard, main, _, _ = repo
    _scoped_file(main)                                                   # untracked counts too
    msg = guard.review_gate("git add -A && git commit -m x", main)
    assert msg is not None and "stats-reviewer" in msg and "src/gaze_cursor/eval/x.py" in msg
    _launch(guard, main)
    assert guard.review_gate("git add -A && git commit -m x", main) is None


def test_the_record_goes_stale_when_a_scoped_file_changes_after_the_review(repo) -> None:
    guard, main, _, _ = repo
    f = _scoped_file(main)
    _launch(guard, main)
    f.write_text("a = 2\n")
    assert guard.review_gate("git commit -am x", main) is not None
    _launch(guard, main)
    assert guard.review_gate("git commit -am x", main) is None
    _scoped_file(main, "scripts/session_report.py")                      # a new scoped file also invalidates it
    assert guard.review_gate("git commit -am x", main) is not None


def test_other_agents_and_tools_do_not_count_as_a_review(repo) -> None:
    guard, main, _, _ = repo
    _scoped_file(main)
    _launch(guard, main, agent="plan-reviewer")
    _launch(guard, main, agent="Explore")
    guard.review_record({"tool_name": "Bash", "cwd": str(main), "tool_input": {"subagent_type": "stats-reviewer"}})
    assert guard.review_gate("git commit -am x", main) is not None
    _launch(guard, main, tool="Task")                                    # the older tool name is accepted
    assert guard.review_gate("git commit -am x", main) is None


def test_skip_review_needs_a_reason_and_is_logged(repo) -> None:
    guard, main, _, _ = repo
    _scoped_file(main)
    assert guard.review_gate('SKIP_REVIEW="" git commit -m x', main) is not None
    assert guard.review_gate("SKIP_REVIEW= git commit -m x", main) is not None
    assert guard.review_gate('SKIP_REVIEW="typo in a docstring" git commit -m x', main) is None
    log = (guard.STATE / "review_skips.log").read_text()
    assert "typo in a docstring" in log and "stats-reviewer" in log


def test_pre_commit_blocks_with_exit_2_before_running_any_test(repo) -> None:
    guard, main, _, _ = repo
    _scoped_file(main)
    data = {"cwd": str(main), "tool_input": {"command": "git add -A && git commit -m x"}}
    assert guard.pre_commit(data) == 2


def test_the_gate_works_in_a_worktree(repo) -> None:
    guard, main, wt, _ = repo
    _scoped_file(wt)
    assert guard.review_gate("git commit -m x", wt) is not None
    _launch(guard, wt)
    assert guard.review_gate("git commit -m x", wt) is None


def test_review_ok_fallback_records_the_state(repo, monkeypatch) -> None:
    guard, main, _, _ = repo
    _scoped_file(main)
    monkeypatch.chdir(main)
    assert guard.review_ok("nobody") == 1
    assert guard.review_ok("stats-reviewer") == 0
    assert guard.review_gate("git commit -m x", main) is None


def test_settings_register_the_review_hook() -> None:
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    post = settings["hooks"]["PostToolUse"]
    assert post[0]["matcher"] == "Agent|Task"
    assert post[0]["hooks"][0]["args"][-1] == "review_record"


# ---- smarter skill gates, context, run-once ledger (guard v2) ----

@pytest.fixture
def g(repo):
    """The repo fixture plus what the real checkout has: a .gitignore for state and recordings, PLAN / RESULTS /
    results.csv committed, and a diary next to the main checkout."""
    guard, main, wt, other = repo
    (main / ".gitignore").write_text(".claude/state/\n.claude/worktrees/\ndata/sessions/*\ndata/quick/*\n")
    (main / "PLAN.md").write_text("# Plan\n**Status (old):** step 1 DONE.\n")
    (main / "RESULTS.md").write_text("| H1 | +0.12 [-0.43, +0.67] |\n")
    (main / "data").mkdir(exist_ok=True)
    (main / "data" / "results.csv").write_text("time,mode,confirmatory,err\nt0,within,0,2.7136\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "docs")
    notes = main.parent / "gaze-cursor-notes"
    notes.mkdir(exist_ok=True)
    (notes / "PROGRESS.md").write_text("# Progress\n\n## Start here\n\n**Next:** record session 2.\n\nolder\n\n## Old\n")
    return guard, main


def _commit_all(main: Path, msg: str = "c") -> None:
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", msg)


def _skill(guard, base: Path, name: str, tool: str = "Skill") -> None:
    guard.skill_record({"tool_name": tool, "cwd": str(base), "tool_input": {"skill": name}})


PREREG = """# Pre-registrations
### Pre-registration: M2 hypothesis H1 (written 2026-10-13)
### Pre-registration log (decisions)
### RESULT of the pre-registered test (run once)
### Pre-registration, round 4: clicks (FROZEN)
### Round-4 recording log
"""


def test_open_preregistrations_are_closed_by_matching_results() -> None:
    guard = _load("guard_prereg")
    assert guard.open_preregistrations(PREREG) == ["Pre-registration, round 4: clicks (FROZEN)"]
    closed = PREREG + "### RESULT, round 4 (H6), run once\n"
    assert guard.open_preregistrations(closed) == []
    assert guard.open_preregistrations((ROOT / "PREREGISTRATIONS.md").read_text()) == []   # the real file: all closed


def test_checkpoint_gate_fires_on_a_new_done_only(g) -> None:
    guard, main = g
    (main / "PLAN.md").write_text("# Plan, reworded\n**Status (old):** step 1 DONE.\n")
    assert guard.skill_gates("git commit -am x", main) is None                # wording: no gate
    (main / "PLAN.md").write_text("# Plan\n**Status (old):** step 1 DONE. Step 2 DONE.\n")
    msg = guard.skill_gates("git commit -am x", main)
    assert msg is not None and "checkpoint" in msg
    _skill(guard, main, "checkpoint")
    assert guard.skill_gates("git commit -am x", main) is None
    _commit_all(main)                                                       # HEAD moved: the record is used up
    (main / "PLAN.md").write_text("# Plan\n**Status (old):** step 1 DONE. Step 2 DONE. Step 3 DONE.\n")
    assert guard.skill_gates("git commit -am x", main) is not None


def test_typed_slash_command_and_plugin_prefix_count_as_the_skill(g) -> None:
    guard, main = g
    (main / "PLAN.md").write_text("**Status:** step 1 DONE, step 2 DONE\n")
    guard.prompt_record({"cwd": str(main), "prompt": "/checkpoint please"})
    assert guard.skill_gates("git commit -am x", main) is None
    _commit_all(main)
    (main / "PLAN.md").write_text("**Status:** step 1 DONE, step 2 DONE, step 3 DONE\n")
    _skill(guard, main, "project:checkpoint")
    assert guard.skill_gates("git commit -am x", main) is None
    guard.prompt_record({"cwd": str(main), "prompt": "no slash here /checkpoint"})   # only a leading command


def test_results_update_gate_needs_the_skill_and_sources(g) -> None:
    guard, main = g
    (main / "RESULTS.md").write_text("| H1 | +0.12 [-0.43, +0.67] | reworded |\n")
    assert guard.skill_gates("git commit -am x", main) is None                # no new number: no gate
    (main / "RESULTS.md").write_text("| H1 | +0.12 [-0.43, +0.67] |\n| run | 2.71 deg |\n")
    msg = guard.skill_gates("git commit -am x", main)
    assert msg is not None and "results-update" in msg and "2.71" in msg
    _skill(guard, main, "results-update")
    assert guard.skill_gates("git commit -am x", main) is None                # 2.71 rounds 2.7136 in results.csv
    (main / "RESULTS.md").write_text("| H1 | +0.12 [-0.43, +0.67] |\n| run | 2.71 deg | 9.99 |\n")
    msg = guard.skill_gates("git commit -am x", main)
    assert msg is not None and "9.99" in msg and "2.71" not in msg.split(":")[-1]
    assert guard.skill_gates('SKIP_GATE="9.99 = 2.71 + 7.28" git commit -am x', main) is None
    log = (guard.STATE / "gate_overrides.log").read_text()
    assert "results-sources" in log and "9.99" in log


def test_number_helpers() -> None:
    guard = _load("guard_numbers")
    assert guard.added_numbers("a 1.5 b 2.0", "a 1.5 b 2.0 c 2.0 d -3.25") == ["2.0", "3.25"]
    assert guard.unsourced(["2.71", "0.64", "1.00"], "x,2.7136\n+0.64 r=1.004") == []
    assert guard.unsourced(["2.72"], "2.7136") == ["2.72"]


def test_new_confirmatory_flag_needs_a_committed_open_protocol(g) -> None:
    guard, main = g
    _scoped_file(main, "scripts/evaluate_x.py", 'ap.add_argument("--preregistered-7")\n')
    msg = guard.skill_gates("git add -A && git commit -m x", main)
    assert msg is not None and "--preregistered-7" in msg and "preregister" in msg
    (main / "scripts" / "evaluate_x.py").unlink()
    (main / "PREREGISTRATIONS.md").write_text(PREREG)
    _commit_all(main, "Pre-register H7")
    _scoped_file(main, "scripts/evaluate_x.py", 'ap.add_argument("--preregistered-7")\n')
    assert guard.skill_gates("git add -A && git commit -m x", main) is None


def test_a_new_protocol_is_committed_alone(g) -> None:
    guard, main = g
    (main / "PREREGISTRATIONS.md").write_text("# Round 4\n- frozen line\n### Pre-registration, round 9 (frozen)\n")
    (main / "PLAN.md").write_text("# Plan\n**Status (old):** step 1 DONE. Round 9 frozen.\n")
    msg = guard.skill_gates("git add -A && git commit -m x", main)
    assert msg is not None and "alone" in msg and "PLAN.md" in msg
    _git(main, "checkout", "--", "PLAN.md")
    assert guard.skill_gates("git add -A && git commit -m x", main) is None


def test_confirmatory_run_needs_intake_and_runs_once(g) -> None:
    guard, main = g
    run = {"cwd": str(main), "tool_input": {"command": "uv run scripts/evaluate_online.py --preregistered"}}
    assert guard.confirmatory_gate(run) == 2                                 # intake never ran
    q = main / "data" / "quick"
    q.mkdir(parents=True)
    (q / "cursor_1.npz").write_bytes(b"x")
    _skill(guard, main, "session-intake")
    (q / "cursor_2.npz").write_bytes(b"x")                                  # recorded after the intake
    (q / "last_calibration.npz").write_bytes(b"x")                          # not a recording
    assert guard.not_intaken() == ["data/quick/cursor_2.npz"]
    assert guard.confirmatory_gate(run) == 2
    _skill(guard, main, "session-intake")
    assert guard.confirmatory_gate(run) == 0                                 # allowed: the pending run is noted
    guard.post_bash(run)                                                     # it logged nothing: not counted
    assert guard.confirmatory_gate(run) == 0
    with open(main / "data" / "results.csv", "a") as fh:
        fh.write("t1,h6,1,0.64\n")
    guard.post_bash(run)
    assert guard.confirmatory_gate(run) == 2                                 # ran once already
    rerun = {"cwd": str(main), "tool_input": {"command": 'CONFIRMATORY_RERUN="author OK: crash" ' + run["tool_input"]["command"]}}
    assert guard.confirmatory_gate(rerun) == 0
    other = {"cwd": str(main), "tool_input": {"command": "uv run scripts/evaluate_online.py --status"}}
    looked = {"cwd": str(main), "tool_input": {"command": "grep -n -- --preregistered scripts/evaluate_online.py"}}
    assert guard.confirmatory_gate(other) == 0 and guard.confirmatory_gate(looked) == 0


def test_docs_only_commits_skip_the_tests(g) -> None:
    guard, main = g
    (main / "README.md").write_text("new\n")
    assert guard._docs_only(main)
    (main / "tests").mkdir()
    (main / "tests" / "test_readme.py").write_text("README.md\n")            # a test reads it: run the suite
    assert not guard._docs_only(main)
    (main / "tests" / "test_readme.py").unlink()
    _scoped_file(main, "src/gaze_cursor/other.py")
    assert not guard._docs_only(main)


def test_live_rows_appended_to_results_need_no_review(g) -> None:
    guard, main = g
    with open(main / "data" / "results.csv", "a") as fh:
        fh.write("t1,live_accuracy,0,2.1\nt2,cursor_session,0,3.0\n")
    assert guard.review_gate("git commit -am x", main) is None
    with open(main / "data" / "results.csv", "a") as fh:
        fh.write("t3,losos,0,6.2\n")                                         # an analysis row: review it
    assert guard.review_gate("git commit -am x", main) is not None


def test_stop_reminds_after_code_committed_in_the_session(g) -> None:
    guard, main = g
    guard.session_start({"session_id": "s1", "cwd": str(main)})
    _scoped_file(main, "src/gaze_cursor/other.py")
    _commit_all(main, "code")
    assert guard.stop({"session_id": "s1", "cwd": str(main)}) == 2
    assert guard.stop({"session_id": "s1", "cwd": str(main)}) == 0          # once per state
    (main / "PLAN.md").write_text("# Plan\n**Status (old):** step 1 DONE. Notes updated.\n")
    _commit_all(main, "plan")
    assert guard.stop({"session_id": "s1", "cwd": str(main)}) == 0
    guard.session_end({"session_id": "s1"})
    assert not (guard.STATE / "start_s1.json").exists()


def test_session_start_gives_context(g, capsys) -> None:
    guard, main = g
    (main / "PREREGISTRATIONS.md").write_text(PREREG)
    (main / "PLAN.md").write_text("# Plan\n**Status (old):** step 1 DONE. Step 2 DONE.\n")
    guard.session_start({"session_id": "s2", "cwd": str(main)})
    out = capsys.readouterr().out
    assert "record session 2" in out and "older" not in out                # newest diary entry only
    assert "round 4: clicks" in out
    assert "checkpoint" in out                                               # a commit now would be blocked
    assert "plan-audit last run: not recorded" in out


def test_editing_eval_code_adds_context_once_while_a_protocol_is_open(g, capsys) -> None:
    guard, main = g
    (main / "PREREGISTRATIONS.md").write_text(PREREG)
    edit = {"session_id": "s3", "tool_name": "Write",
            "tool_input": {"file_path": str(main / "src/gaze_cursor/eval/x.py"), "content": ""}}
    assert guard.protect(edit) == 0
    out = json.loads(capsys.readouterr().out)
    assert "round 4" in out["hookSpecificOutput"]["additionalContext"]
    assert guard.protect(edit) == 0 and capsys.readouterr().out == ""       # once per session


def test_doctor_and_settings_cover_the_new_hooks(g, monkeypatch, capsys) -> None:
    guard, main = g
    monkeypatch.chdir(main)
    assert guard.doctor() == 0
    assert "open pre-registrations" in capsys.readouterr().out
    hooks = json.loads((ROOT / ".claude" / "settings.json").read_text())["hooks"]
    args = {(ev, e.get("matcher", "*")): e["hooks"][0]["args"][-1] for ev, es in hooks.items() for e in es}
    assert args[("PreToolUse", "Bash")] == "pre_bash"
    assert args[("PostToolUse", "Skill")] == "skill_record"
    assert args[("PostToolUse", "Bash")] == "post_bash"
    assert args[("UserPromptSubmit", "*")] == "prompt_record"
