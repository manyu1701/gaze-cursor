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
