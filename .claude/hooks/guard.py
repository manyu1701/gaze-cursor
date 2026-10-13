#!/usr/bin/env python3
"""Project guardrails for Claude Code (gaze-cursor).

One script, several hook events (first argument):
  session_start  - add project status to context; warn if another session holds the lock
  session_end    - release this session's lock
  pre_commit     - before `git commit` in this repo (main checkout or a worktree): run the tests, block on failure
  protect        - before Edit/Write: block edits to recorded data and changes to frozen text
  stop           - at the end of a reply: remind (once per state) to update PLAN/notes after code changes

Reads the hook JSON on stdin. Exit 2 + stderr = block (PreToolUse) / continue (Stop).
Standard library only; must run on macOS's /usr/bin/python3 (3.9) as well as newer Pythons.

Worktrees: background sessions work in `.claude/worktrees/<name>`, so every check works on the git tree the
action targets (the hook input's `cwd`, the edited file's path, `git -C` / `cd` in the command), while the
session lock and the virtualenv live in the main checkout. Commits in other repos (e.g. the private notes
repo) are left alone.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
LOCK_MAX_AGE_S = 12 * 3600

# Recorded data and downloaded models: never edited by hand.
PROTECTED_DIRS = ("data/sessions/", "data/quick/", "data/dry_runs/", "data/excluded/", "data/tasks/", "models/")
# Append-only documents: existing lines may not be changed or deleted.
APPEND_ONLY = ("PREREGISTRATIONS.md",)


def _read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def _git(*args: str, cwd: Path = ROOT) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def _toplevel(d: Path) -> Path | None:
    """The git working tree (main checkout or worktree) containing directory ``d``, if any."""
    out = _git("rev-parse", "--show-toplevel", cwd=d).strip() if d.is_dir() else ""
    return Path(out) if out else None


def _common_dir(d: Path) -> Path | None:
    """The shared .git directory of the repository containing ``d`` (same for all its worktrees)."""
    out = _git("rev-parse", "--git-common-dir", cwd=d).strip() if d.is_dir() else ""
    if not out:
        return None
    p = Path(out)
    return (p if p.is_absolute() else d / p).resolve()


_REPO = _common_dir(ROOT)
MAIN = _REPO.parent if _REPO is not None and _REPO.name == ".git" else ROOT   # the main checkout
STATE = MAIN / ".claude" / "state"
LOCK = STATE / "session.lock"


def _same_repo(d: Path | None) -> bool:
    return d is not None and _REPO is not None and _common_dir(d) == _REPO


def _cwd(data: dict) -> Path:
    c = data.get("cwd")
    return Path(c) if c and Path(c).is_dir() else ROOT


def _rel_in_tree(path: str, cwd: Path) -> tuple[Path | None, str]:
    """(working tree root, path relative to it) for a file path, without resolving symlinks.

    Walks up to the nearest directory holding `.git` (a directory in the main checkout, a file in a
    worktree), so `.claude/worktrees/x/PREREGISTRATIONS.md` is recognised as `PREREGISTRATIONS.md`.
    """
    p = Path(os.path.abspath(os.path.join(str(cwd), path)))
    for parent in p.parents:
        if (parent / ".git").exists():
            return parent, p.relative_to(parent).as_posix()
    return None, p.as_posix()


_SEPARATORS = re.compile(r"&&|\|\||;|\||\n")


def commit_dir(cmd: str, cwd: Path) -> Path | None:
    """Directory a shell command runs `git commit` in, or None if it does not commit.

    Follows `cd <dir>` segments and `git -C <dir>`; global options before the subcommand are skipped.
    """
    d = cwd
    for seg in _SEPARATORS.split(cmd):
        try:
            tok = shlex.split(seg)
        except ValueError:
            tok = seg.split()
        while tok and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok[0]):   # VAR=value prefixes
            tok = tok[1:]
        if not tok:
            continue
        if tok[0] == "cd" and len(tok) > 1:
            d = Path(os.path.expanduser(os.path.join(str(d), tok[1])))
            continue
        if Path(tok[0]).name != "git":
            continue
        g, i = d, 1
        while i < len(tok):
            t = tok[i]
            if t == "-C" and i + 1 < len(tok):
                g = Path(os.path.expanduser(os.path.join(str(g), tok[i + 1])))
                i += 2
            elif t in ("-c", "--git-dir", "--work-tree", "--namespace") and i + 1 < len(tok):
                i += 2
            elif t.startswith("-"):
                i += 1
            else:
                break
        if i < len(tok) and tok[i] == "commit":
            return g
    return None


def session_start(data: dict) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    sid = data.get("session_id", "?")
    notes: list[str] = []
    if LOCK.exists():
        try:
            other = json.loads(LOCK.read_text())
        except Exception:
            other = {}
        age = time.time() - LOCK.stat().st_mtime
        if other.get("session_id") not in (None, sid) and age < LOCK_MAX_AGE_S:
            notes.append(
                f"WARNING: another Claude Code session ({other.get('session_id')}, started "
                f"{other.get('started')}) may be working on this repo. Two sessions caused diverged "
                "branches and competing freezes in round 3. Tell the author before changing anything; "
                "if that session is finished, the author can delete .claude/state/session.lock."
            )
    LOCK.write_text(json.dumps({"session_id": sid, "started": time.strftime("%Y-%m-%d %H:%M:%S")}))

    status = [ln.strip() for ln in (ROOT / "PLAN.md").read_text(errors="ignore").splitlines()
              if ln.strip().startswith("**Status")] if (ROOT / "PLAN.md").exists() else []
    branch = _git("status", "-sb").splitlines()[:1]
    dirty = _git("status", "--porcelain").splitlines()
    log = _git("log", "--oneline", "-5").strip()
    parts = ["Project status (from the session-start hook):"]
    parts += notes
    parts.append("Git: " + (branch[0] if branch else "?") + (f", {len(dirty)} uncommitted change(s)" if dirty else ", clean"))
    parts.append("Recent commits:\n" + log)
    if status:
        parts.append("PLAN.md status lines:\n" + "\n".join(status[-3:]))
    parts.append(
        "Process: work only on the current step; at a checkpoint use the `checkpoint` skill and stop. "
        "New hypothesis -> `preregister` skill. New recording -> `session-intake` skill. "
        "RESULTS.md only via `results-update`. Before committing an analysis, ask the `stats-reviewer` agent."
    )
    print("\n\n".join(parts))
    return 0


def session_end(data: dict) -> int:
    try:
        if LOCK.exists() and json.loads(LOCK.read_text()).get("session_id") == data.get("session_id"):
            LOCK.unlink()
    except Exception:
        pass
    return 0


def pre_commit(data: dict) -> int:
    cmd = (data.get("tool_input") or {}).get("command", "")
    if "commit" not in cmd:
        return 0
    d = commit_dir(cmd, _cwd(data))
    top = _toplevel(d) if d is not None else None
    if top is None or not _same_repo(top):
        return 0                                   # not a commit, or a commit in another repo (notes)
    py = next((p for p in (MAIN / ".venv" / "bin" / "python", top / ".venv" / "bin" / "python") if p.exists()), None)
    if py is None:
        print("pre-commit hook: no .venv found (run `uv sync` in the main checkout); tests NOT run.", file=sys.stderr)
        return 1
    env = dict(os.environ)
    env["CI"] = "1"                                # skip the tests that open windows on the real display
    env["PYTHONPATH"] = os.pathsep.join([str(top / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        r = subprocess.run([str(py), "-m", "pytest", "-q", "-x"], cwd=top, env=env,
                           capture_output=True, text=True, timeout=280)
    except subprocess.TimeoutExpired:
        print("pre-commit hook: test suite timed out (280 s); not blocking.", file=sys.stderr)
        return 1
    out = (r.stdout + r.stderr).strip()
    if "No module named pytest" in out:
        print(f"pre-commit hook: pytest is not installed in {py}; tests NOT run.", file=sys.stderr)
        return 1
    if r.returncode not in (0, 5):  # 5 = no tests collected
        tail = "\n".join(out.splitlines()[-25:])
        print(f"Commit blocked: the test suite failed in {top}. Fix the failures (or tell the author why) "
              f"before committing.\n{tail}", file=sys.stderr)
        return 2
    return 0


def _new_content(tool: str, ti: dict, old: str) -> str | None:
    if tool == "Write":
        return ti.get("content", "")
    if tool == "Edit":
        o, n = ti.get("old_string", ""), ti.get("new_string", "")
        return old.replace(o, n) if ti.get("replace_all") else old.replace(o, n, 1)
    if tool == "MultiEdit":
        text = old
        for e in ti.get("edits", []):
            o, n = e.get("old_string", ""), e.get("new_string", "")
            text = text.replace(o, n) if e.get("replace_all") else text.replace(o, n, 1)
        return text
    return None


def protect(data: dict) -> int:
    tool = data.get("tool_name", "")
    ti = data.get("tool_input") or {}
    path = ti.get("file_path") or ti.get("notebook_path") or ""
    if not path:
        return 0
    top, rel = _rel_in_tree(path, _cwd(data))
    if top is None or not _same_repo(top):
        return 0                                   # outside this repo (e.g. the private notes repo)
    if any(rel.startswith(d) for d in PROTECTED_DIRS):
        print(f"Blocked: {rel} is recorded data or a downloaded model. Recordings are never edited; "
              "exclusions are done by moving files with a logged reason, and only with the author's OK.",
              file=sys.stderr)
        return 2
    if rel in APPEND_ONLY:
        f = top / rel
        if not f.exists():
            return 0
        old = f.read_text(errors="ignore")
        new = _new_content(tool, ti, old)
        if new is None:
            return 0
        removed = [ln for ln in difflib.ndiff(old.splitlines(), new.splitlines()) if ln.startswith("- ")]
        if removed:
            sample = "\n".join(ln[2:][:120] for ln in removed[:3])
            print(f"Blocked: {rel} is append-only (pre-registrations are frozen once written). This edit would change "
                  f"or delete existing lines, e.g.:\n{sample}\nAdd a dated note or amendment below instead; if a fix is "
                  "truly needed, ask the author to make it by hand.", file=sys.stderr)
            return 2
    return 0


def stop(data: dict) -> int:
    if data.get("stop_hook_active"):
        return 0
    top = _toplevel(_cwd(data))
    if top is None or not _same_repo(top):
        return 0
    changed = [ln[3:] for ln in _git("status", "--porcelain", cwd=top).splitlines()]
    code = [p for p in changed if p.startswith(("src/", "scripts/"))]
    docs = [p for p in changed if p in ("PLAN.md", "RESULTS.md", "PREREGISTRATIONS.md", "README.md")]
    if not code or docs:
        return 0
    STATE.mkdir(parents=True, exist_ok=True)
    marker = STATE / "stop_reminded"
    key = hashlib.sha1("\n".join([str(top)] + sorted(changed)).encode()).hexdigest()
    if marker.exists() and marker.read_text() == key:
        return 0
    marker.write_text(key)
    print("Reminder (stop hook): code changed (" + ", ".join(code[:5]) + ") but PLAN.md and the progress notes were "
          "not updated. Update the step's status and the private progress notes, or say briefly why not, before "
          "finishing.", file=sys.stderr)
    return 2


def main() -> int:
    event = sys.argv[1] if len(sys.argv) > 1 else ""
    data = _read_input()
    handlers = {"session_start": session_start, "session_end": session_end, "pre_commit": pre_commit,
                "protect": protect, "stop": stop}
    fn = handlers.get(event)
    return fn(data) if fn else 0


if __name__ == "__main__":
    sys.exit(main())
