#!/usr/bin/env python3
"""Project guardrails for Claude Code (gaze-cursor).

One script, several hook events (first argument):
  session_start  - add project context (diary next action, open pre-registrations, recordings not yet intaken,
                   pending gates, recent overrides, stale audits); warn if another session holds the lock
  session_end    - release this session's lock
  pre_bash       - before a Bash command: the confirmatory-run gate, then `pre_commit`
  pre_commit     - before `git commit` in this repo (main checkout or a worktree): reviewer and skill gates, then the
                   tests (skipped for commits that change only docs no test reads); block on failure
  post_bash      - after a Bash command: record a confirmatory run in the run-once ledger
  protect        - before Edit/Write: block edits to recorded data and changes to frozen text; add context when
                   evaluation code is edited while a pre-registration is open
  stop           - at the end of a reply: remind (once per state) to update PLAN/notes after code changes, including
                   code committed during this session
  review_record  - after the Agent tool launches a reviewer: record which working-tree state it was asked to review
  skill_record   - after the Skill tool runs a skill: record it (for the skill gates)
  prompt_record  - when the author types `/<skill>`: record it the same way
  review-ok NAME - manual fallback for review_record (run in the repo after the reviewer was launched)
  skill-ok NAME  - manual fallback for skill_record (run in the repo after the skill was followed)
  doctor         - print what the guard sees: hooks registered, test runner, records, gates, ledger

Levels: BLOCK only where a slip would damage a result or data (recordings, frozen pre-registrations, failing
tests, unreviewed analysis, unchecked results, a second confirmatory run); REMIND for forgotten process (stop);
INFORM everything else (session-start context, context when evaluation code is edited).

Skill gates (SKILL_GATES): they fire on what changed, not on which file was touched:
  checkpoint      - a commit that adds a DONE to a PLAN.md Status line
  results-update  - a commit that adds numbers to RESULTS.md; every added number must also appear in
                    data/results.csv or PREREGISTRATIONS.md (exactly or as a rounding of a source number)
  preregister     - a commit that adds a new `--preregistered...` flag needs an open protocol already committed;
                    a commit that adds a protocol must contain nothing else
  session-intake  - a `--preregistered...` run needs every recording on disk to have been through intake
A skill record is fresh while HEAD has not moved since the skill ran (the skills themselves edit and commit).
Overrides: `SKIP_REVIEW="reason"` (reviewers), `SKIP_GATE="reason"` (skill gates), `CONFIRMATORY_RERUN="reason"`
(run-once ledger); all logged in .claude/state and shown at session start.

Reads the hook JSON on stdin. Exit 2 + stderr = block (PreToolUse) / continue (Stop). JSON on stdout with exit 0 =
context for Claude without blocking.
Standard library only; must run on macOS's /usr/bin/python3 (3.9) as well as newer Pythons.

Worktrees: background sessions work in `.claude/worktrees/<name>`, so every check works on the git tree the
action targets (the hook input's `cwd`, the edited file's path, `git -C` / `cd` in the command), while the
session lock, the records and the virtualenv live in the main checkout. Commits in other repos (e.g. the private
notes repo) are left alone.
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
LOCK_MAX_AGE_S = 12 * 3600
STALE_DAYS = 7            # plan-audit / ds-strategy older than this are mentioned at session start

# Recorded data and downloaded models: never edited by hand.
PROTECTED_DIRS = ("data/sessions/", "data/quick/", "data/dry_runs/", "data/excluded/", "data/tasks/", "models/")
# Append-only documents: existing lines may not be changed or deleted.
APPEND_ONLY = ("PREREGISTRATIONS.md",)
# Reviewer agent -> path prefixes it must have been run on before they are committed (CLAUDE.md: "ask
# `stats-reviewer` before committing any analysis, evaluation code or results table"). plan-reviewer is per
# step ("before reporting a step as done"), not per commit, so it is not enforced here.
REVIEW_RULES = {
    "stats-reviewer": ("src/gaze_cursor/eval/", "src/gaze_cursor/conditions.py", "scripts/session_report.py",
                       "scripts/evaluate", "scripts/benchmark.py", "scripts/tune_", "scripts/diagnose_",
                       "scripts/explore_", "data/results.csv"),
}
# results.csv rows the app appends during live use; appending only these does not need a stats review.
LIVE_MODES = ("live_accuracy", "live_targets", "live_path", "cursor_session")
# Skills the guard records (gated ones and the ones whose age is reported).
SKILL_GATES = ("checkpoint", "results-update", "preregister", "session-intake")
AUDIT_SKILLS = ("plan-audit", "ds-strategy")
# Editing these while a pre-registration is open adds a one-time context note.
EVAL_CONTEXT = ("src/gaze_cursor/eval/", "src/gaze_cursor/online.py", "scripts/evaluate")
# Where recordings live (main checkout only), and files there that are not recordings.
RECORDING_DIRS = ("data/sessions", "data/quick", "data/tasks")
NOT_RECORDINGS = ("last_calibration.npz",)

SKIP_REVIEW = re.compile(r"""\bSKIP_REVIEW=("[^"]*"|'[^']*'|\S+)""")
_DONE = re.compile(r"\bDONE\b")
_DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
_FLAG = r"--preregistered[A-Za-z0-9_-]*"
_CONFIRM = re.compile(r"(\S*evaluate\w*\.py)\b[^;&|\n]*?(" + _FLAG + ")")


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
NOTES = MAIN.parent / "gaze-cursor-notes" / "PROGRESS.md"                     # the private diary (may be absent)


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


def _head(top: Path) -> str:
    return _git("rev-parse", "HEAD", cwd=top).strip()


def _at_head(top: Path, rel: str) -> str:
    """The committed (HEAD) text of ``rel`` in tree ``top``; empty if it does not exist there."""
    return _git("show", "HEAD:" + rel, cwd=top)


def _work(top: Path, rel: str) -> str:
    f = top / rel
    return f.read_text(errors="ignore") if f.is_file() else ""


def _load_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s or "?")


def _override(cmd: str, var: str, gate: str, files: list) -> bool:
    """True if ``cmd`` carries ``VAR="reason"`` with a non-empty reason; the override is logged."""
    m = re.search(r"\b" + var + r"""=("[^"]*"|'[^']*'|\S+)""", cmd)
    reason = m.group(1).strip("\"'") if m else ""
    if not reason:
        return False
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "gate_overrides.log", "a") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "gate": gate, "var": var,
                             "reason": reason, "files": files}) + "\n")
    return True


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


# ---- what the documents say ----

_HEADING = re.compile(r"^#{2,4}\s+(.*\S)\s*$")
_IDS = re.compile(r"\bround \d+\b|\bH\d+\b", re.I)


def open_preregistrations(text: str) -> list:
    """Protocol headings in PREREGISTRATIONS.md that no RESULT heading has closed yet.

    A protocol heading starts with "Pre-registration" (its "log" sections excluded). A later heading containing
    "RESULT" closes it when they share an id ("round 4", "H6"); a RESULT heading with no id closes the latest
    open protocol before it.
    """
    heads = [m.group(1) for m in map(_HEADING.match, text.splitlines()) if m]
    open_: list = []
    for h in heads:
        ids = {t.lower() for t in _IDS.findall(h)}
        if re.match(r"pre-registration\b", h, re.I) and " log" not in h.lower():
            open_.append((h, ids))
        elif "RESULT" in h and open_:
            hit = [p for p in open_ if ids & p[1]] if ids else open_[-1:]
            for p in hit:
                open_.remove(p)
    return [h for h, _ in open_]


def protocol_headings(text: str) -> list:
    heads = [m.group(1) for m in map(_HEADING.match, text.splitlines()) if m]
    return [h for h in heads if re.match(r"pre-registration\b", h, re.I) and " log" not in h.lower()]


def done_count(plan: str) -> int:
    """How many DONE marks the Status lines of PLAN.md carry."""
    return sum(len(_DONE.findall(ln)) for ln in plan.splitlines() if "Status" in ln)


def added_numbers(old: str, new: str) -> list:
    """Decimal numbers present in ``new`` more often than in ``old`` (signs ignored), in order of appearance."""
    extra = Counter(_DECIMAL.findall(new)) - Counter(_DECIMAL.findall(old))
    seen, out = set(), []
    for n in _DECIMAL.findall(new):
        if extra.get(n) and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def unsourced(numbers: list, sources: str) -> list:
    """Numbers that appear in ``sources`` neither verbatim nor as a rounding of a source number."""
    exact = set(_DECIMAL.findall(sources))
    vals = [float(x) for x in exact]
    out = []
    for n in numbers:
        if n in exact:
            continue
        d, v = len(n.split(".")[1]), float(n)
        if not any(abs(round(s, d) - v) < 1e-9 for s in vals):
            out.append(n)
    return out


def _flags(top: Path, committed: bool) -> set:
    rev = ["HEAD"] if committed else ["--untracked"]
    out = _git("grep", "-h", "-o", "-I", "-E", "-e", _FLAG, *rev, "--", "scripts", "src", cwd=top)
    return {ln.strip() for ln in out.splitlines() if ln.strip()}


def recordings() -> list:
    """Recording files in the main checkout (face-crop sidecars and the resume file excluded)."""
    out = []
    for d in RECORDING_DIRS:
        p = MAIN / d
        if p.is_dir():
            for f in sorted(p.glob("*.npz")):
                if not f.name.endswith(".faces.npz") and f.name not in NOT_RECORDINGS:
                    out.append(f"{d}/{f.name}")
    return out


def not_intaken() -> list | None:
    """Recordings not on disk at the last session-intake run; None if intake never ran under this guard."""
    rec = _load_json(_skill_marker("session-intake"), None)
    if rec is None:
        return None
    seen = set(rec.get("seen", []))
    return [r for r in recordings() if r not in seen]


def _confirmatory_rows(top: Path) -> int:
    try:
        return sum(1 for r in csv.DictReader(io.StringIO(_work(top, "data/results.csv")))
                   if r.get("confirmatory") == "1")
    except Exception:
        return 0


# ---- records ----

def _marker(name: str) -> Path:
    return STATE / ("review_" + name.replace("-", "_") + ".json")


def _skill_marker(name: str) -> Path:
    return STATE / ("skill_" + name.replace("-", "_") + ".json")


def _record_skill(name: str, top: Path, session: str = "") -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    rec = {"head": _head(top), "tree": str(top), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "epoch": time.time(), "session": session}
    if name == "session-intake":
        rec["seen"] = recordings()
    _skill_marker(name).write_text(json.dumps(rec))


def skill_fresh(name: str, top: Path) -> bool:
    """The skill ran in this tree and HEAD has not moved since (its own edits and commit are covered)."""
    rec = _load_json(_skill_marker(name), {})
    return bool(rec) and rec.get("tree") == str(top) and rec.get("head") == _head(top)


def _skill_name(raw: str) -> str:
    return (raw or "").strip().lstrip("/").split(":")[-1].split()[0] if (raw or "").strip() else ""


def skill_record(data: dict) -> int:
    """PostToolUse on the Skill tool: remember that a guarded skill ran, and on which state."""
    ti = data.get("tool_input") or {}
    name = _skill_name(ti.get("skill") or ti.get("name") or ti.get("command") or "")
    top = _toplevel(_cwd(data))
    if name in SKILL_GATES + AUDIT_SKILLS and top is not None and _same_repo(top):
        _record_skill(name, top, data.get("session_id", ""))
    return 0


def prompt_record(data: dict) -> int:
    """UserPromptSubmit: `/checkpoint` typed by the author runs the skill without the Skill tool; record it too."""
    m = re.match(r"\s*/([A-Za-z0-9_:-]+)", data.get("prompt") or "")
    name = _skill_name(m.group(1)) if m else ""
    top = _toplevel(_cwd(data))
    if name in SKILL_GATES + AUDIT_SKILLS and top is not None and _same_repo(top):
        _record_skill(name, top, data.get("session_id", ""))
    return 0


def skill_ok(name: str) -> int:
    """Manual fallback: `guard.py skill-ok checkpoint`, run in the repo after the skill was followed."""
    top = _toplevel(Path.cwd())
    names = SKILL_GATES + AUDIT_SKILLS
    if name not in names or top is None or not _same_repo(top):
        print(f"usage: guard.py skill-ok <{'|'.join(names)}> (run inside the repo)", file=sys.stderr)
        return 1
    _record_skill(name, top)
    return 0


# ---- session start / end ----

def _diary_next() -> str:
    """The newest entry of the diary's "Start here" section (first paragraph), shortened."""
    if not NOTES.is_file():
        return ""
    text = NOTES.read_text(errors="ignore")
    m = re.search(r"^## Start here[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return ""
    paras = [p.strip() for p in m.group(1).split("\n\n") if p.strip()]
    first = paras[0] if paras else ""
    return first if len(first) <= 1500 else first[:1500] + " ..."


def _recent_overrides(days: int = 7) -> list:
    out = []
    cutoff = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - days * 86400))
    for log, key in ((STATE / "review_skips.log", "reviewer"), (STATE / "gate_overrides.log", "gate")):
        if not log.is_file():
            continue
        for ln in log.read_text(errors="ignore").splitlines():
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("time", "") >= cutoff:
                out.append(f"{e.get('time')} {e.get(key)}: {e.get('reason')}")
    return sorted(out)[-5:]


def _first_sentence(msg: str) -> str:
    return msg.split(". ")[0].rstrip(".") + "."


def context_lines(top: Path) -> list:
    """INFORM-level context: what a session should know before it starts."""
    parts = []
    nxt = _diary_next()
    if nxt:
        parts.append("Newest entry of the diary's \"Start here\" (" + str(NOTES) + "):\n" + nxt)
    elif not NOTES.is_file():
        parts.append(f"Diary not found at {NOTES} (expected next to the main checkout).")
    opn = open_preregistrations(_work(top, "PREREGISTRATIONS.md"))
    parts.append("Open pre-registrations (no RESULT yet): " + ("; ".join(h[:120] for h in opn) if opn else "none") + ".")
    pend = not_intaken()
    n_rec = len(recordings())
    if pend is None:
        if n_rec:
            parts.append(f"Recordings: {n_rec} on disk; session-intake has not run under the current guard, so a "
                         "--preregistered run is blocked until it does.")
    elif pend:
        parts.append(f"Recordings not yet through session-intake: {len(pend)} ({', '.join(pend[:5])}"
                     + (" ..." if len(pend) > 5 else "") + ").")
    gates = [g for g in (review_gate("", top), skill_gates("", top)) if g]
    if gates:
        parts.append("A commit right now would be blocked: " + " ".join(_first_sentence(g) for g in gates))
    ov = _recent_overrides()
    if ov:
        parts.append("Gate overrides in the last 7 days:\n" + "\n".join("- " + o for o in ov))
    for name in AUDIT_SKILLS:
        rec = _load_json(_skill_marker(name), {})
        age = (time.time() - rec["epoch"]) / 86400 if rec.get("epoch") else None
        if age is None or age > STALE_DAYS:
            parts.append(f"{name} last run: " + ("not recorded" if age is None else f"{age:.0f} days ago") + ".")
    return parts


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
    start = STATE / f"start_{_safe(sid)}.json"
    if not start.exists():                         # resume / compact keep the original start
        start.write_text(json.dumps({"head": _head(ROOT), "time": time.strftime("%Y-%m-%d %H:%M:%S")}))

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
    try:
        parts += context_lines(ROOT)
    except Exception as e:                         # context is a convenience; never fail the session start
        parts.append(f"(guard context unavailable: {e})")
    parts.append(
        "Process: work only on the current step; at a checkpoint use the `checkpoint` skill and stop. "
        "New hypothesis -> `preregister` skill. New recording -> `session-intake` skill. "
        "RESULTS.md only via `results-update`. Before committing an analysis, ask the `stats-reviewer` agent. "
        "The commit hook enforces these (see .claude/hooks/guard.py)."
    )
    print("\n\n".join(parts))
    return 0


def session_end(data: dict) -> int:
    try:
        if LOCK.exists() and json.loads(LOCK.read_text()).get("session_id") == data.get("session_id"):
            LOCK.unlink()
    except Exception:
        pass
    try:
        (STATE / f"start_{_safe(data.get('session_id', '?'))}.json").unlink()
    except Exception:
        pass
    return 0


# ---- reviewer gate ----

def _live_append_only(top: Path) -> bool:
    """data/results.csv only gained rows the app logs in live use (LIVE_MODES, confirmatory=0)."""
    old, new = _at_head(top, "data/results.csv"), _work(top, "data/results.csv")
    if not old or not new.startswith(old.rstrip("\n")):
        return False
    header = old.splitlines()[0]
    tail = new[len(old.rstrip("\n")):].lstrip("\n")
    rows = list(csv.DictReader(io.StringIO(header + "\n" + tail)))
    return bool(rows) and all(r.get("mode") in LIVE_MODES and r.get("confirmatory") == "0" for r in rows)


def _scoped_changes(top: Path, prefixes: tuple) -> list:
    """Changed (modified, added, deleted or untracked) paths under ``prefixes`` in the working tree ``top``."""
    out = _git("status", "--porcelain", "-uall", cwd=top)
    paths = []
    for ln in out.splitlines():
        p = ln[3:].split(" -> ")[-1].strip().strip('"')
        if p.startswith(prefixes):
            paths.append(p)
    return sorted(set(paths))


def _review_paths(top: Path, prefixes: tuple) -> list:
    paths = _scoped_changes(top, prefixes)
    if "data/results.csv" in paths and _live_append_only(top):
        paths.remove("data/results.csv")
    return paths


def _scope_hash(top: Path, paths: list) -> str:
    """Hash of the working-tree content of ``paths`` (a deleted file hashes as DELETED)."""
    h = hashlib.sha256()
    for p in paths:
        f = top / p
        h.update(p.encode() + b"\0")
        h.update(f.read_bytes() if f.is_file() else b"DELETED")
    return h.hexdigest()


def review_gate(cmd: str, top: Path) -> "str | None":
    """Why this commit must wait for a reviewer, or None. Working-tree based, so `git add && git commit` in one
    command is covered, and a commit of part of the changes still needs a review of all of them (conservative)."""
    for name, prefixes in REVIEW_RULES.items():
        paths = _review_paths(top, prefixes)
        if not paths:
            continue
        digest = _scope_hash(top, paths)
        try:
            rec = json.loads(_marker(name).read_text())
        except Exception:
            rec = {}
        if rec.get("hash") == digest and rec.get("tree") == str(top):
            continue
        m = SKIP_REVIEW.search(cmd)
        reason = m.group(1).strip("\"'") if m else ""
        if reason:
            STATE.mkdir(parents=True, exist_ok=True)
            with open(STATE / "review_skips.log", "a") as fh:
                fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "reviewer": name,
                                     "reason": reason, "files": paths}) + "\n")
            continue
        shown = ", ".join(paths[:5]) + (" ..." if len(paths) > 5 else "")
        return (f"Commit blocked: {name} has not been run on the current state of review-scoped files ({shown}). "
                f"Run the {name} agent on this diff, fix its findings, then commit. The record is made when the agent "
                f"is launched and goes stale if these files change afterwards. Rule: reviewers run per commit, not per "
                f"step. Override (logged in .claude/state/review_skips.log): SKIP_REVIEW=\"reason\" git commit ...")
    return None


def _record_review(name: str, top: Path) -> None:
    paths = _review_paths(top, REVIEW_RULES[name])
    STATE.mkdir(parents=True, exist_ok=True)
    _marker(name).write_text(json.dumps({
        "hash": _scope_hash(top, paths), "tree": str(top), "files": paths,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")}))


def review_record(data: dict) -> int:
    """PostToolUse on the Agent tool: remember which state a reviewer agent was launched on."""
    if data.get("tool_name") not in ("Agent", "Task"):
        return 0
    name = (data.get("tool_input") or {}).get("subagent_type")
    top = _toplevel(_cwd(data))
    if name in REVIEW_RULES and top is not None and _same_repo(top):
        _record_review(name, top)
    return 0


def review_ok(name: str) -> int:
    """Manual fallback: `guard.py review-ok stats-reviewer`, run in the repo after launching the reviewer."""
    top = _toplevel(Path.cwd())
    if name not in REVIEW_RULES or top is None or not _same_repo(top):
        print(f"usage: guard.py review-ok <{'|'.join(REVIEW_RULES)}> (run inside the repo)", file=sys.stderr)
        return 1
    _record_review(name, top)
    return 0


# ---- skill gates ----

_OVERRIDE_HINT = "Override only with a reason the author would accept (logged): SKIP_GATE=\"reason\" git commit ..."


def skill_gates(cmd: str, top: Path) -> "str | None":
    """Why this commit must wait for a skill, or None. Each gate fires on what changed in the working tree."""
    plan_old, plan_new = _at_head(top, "PLAN.md"), _work(top, "PLAN.md")
    if done_count(plan_new) > done_count(plan_old) and not skill_fresh("checkpoint", top):
        if not _override(cmd, "SKIP_GATE", "checkpoint", ["PLAN.md"]):
            return ("Commit blocked: PLAN.md marks a step DONE, but the `checkpoint` skill has not run since the last "
                    "commit. Run it (verify the done-when with evidence, record, then stop and report to the author). "
                    + _OVERRIDE_HINT)

    nums = added_numbers(_at_head(top, "RESULTS.md"), _work(top, "RESULTS.md"))
    if nums:
        if not skill_fresh("results-update", top):
            if not _override(cmd, "SKIP_GATE", "results-update", ["RESULTS.md"]):
                return ("Commit blocked: RESULTS.md gains numbers (" + ", ".join(nums[:6]) + ") but the "
                        "`results-update` skill has not run since the last commit. Run it, then commit. " + _OVERRIDE_HINT)
        missing = unsourced(nums, _work(top, "data/results.csv") + "\n" + _work(top, "PREREGISTRATIONS.md"))
        if missing and not _override(cmd, "SKIP_GATE", "results-sources", missing):
            return ("Commit blocked: these numbers added to RESULTS.md appear in neither data/results.csv nor "
                    "PREREGISTRATIONS.md (verbatim or rounded): " + ", ".join(missing[:10]) + ". Every number must come "
                    "from a logged row or a RESULT section; a derived number (a difference, a mean of logged values) "
                    "needs SKIP_GATE=\"how it is derived\" so the derivation is on record.")

    new_flags = sorted(_flags(top, committed=False) - _flags(top, committed=True))
    if new_flags and not open_preregistrations(_at_head(top, "PREREGISTRATIONS.md")):
        if not _override(cmd, "SKIP_GATE", "preregister", new_flags):
            return ("Commit blocked: new confirmatory entry point(s) " + ", ".join(new_flags) + " but no open "
                    "pre-registration is committed. Use the `preregister` skill: commit the protocol alone first, then "
                    "the guarded code. " + _OVERRIDE_HINT)

    new_protocols = set(protocol_headings(_work(top, "PREREGISTRATIONS.md"))) - \
        set(protocol_headings(_at_head(top, "PREREGISTRATIONS.md")))
    if new_protocols:
        others = [p for p in _scoped_changes(top, ("",)) if p != "PREREGISTRATIONS.md"]
        if others and not _override(cmd, "SKIP_GATE", "preregister-alone", others):
            return ("Commit blocked: a new pre-registration must be committed alone (the `preregister` skill), so the "
                    "git history shows it was frozen before its code and data. Also changed in the working tree: "
                    + ", ".join(others[:5]) + ". Commit PREREGISTRATIONS.md by itself first (finish or stash the other "
                    "changes; the check reads the working tree, as a staged-only view misses `git add X && git commit`). "
                    + _OVERRIDE_HINT)
    return None


def _docs_only(top: Path) -> bool:
    """Every change is a .md file that no test reads (so the tests cannot be affected)."""
    paths = _scoped_changes(top, ("",))
    if not paths or not all(p.endswith(".md") for p in paths):
        return False
    tests = "\n".join(f.read_text(errors="ignore") for f in (top / "tests").glob("*.py")) if (top / "tests").is_dir() else ""
    return not any(Path(p).name in tests for p in paths)


def pre_commit(data: dict) -> int:
    cmd = (data.get("tool_input") or {}).get("command", "")
    if "commit" not in cmd:
        return 0
    d = commit_dir(cmd, _cwd(data))
    top = _toplevel(d) if d is not None else None
    if top is None or not _same_repo(top):
        return 0                                   # not a commit, or a commit in another repo (notes)
    why = review_gate(cmd, top) or skill_gates(cmd, top)
    if why:
        print(why, file=sys.stderr)
        return 2
    if _docs_only(top):
        return 0                                   # docs no test reads: the suite cannot change
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


# ---- confirmatory runs: intake first, run once ----

LEDGER = STATE / "confirmatory_runs.json"
PENDING = STATE / "confirmatory_pending.json"


def confirmatory_gate(data: dict) -> int:
    cmd = (data.get("tool_input") or {}).get("command", "")
    m = _CONFIRM.search(cmd)
    if not m:
        return 0
    top = _toplevel(_cwd(data))
    if top is None or not _same_repo(top):
        return 0
    key = Path(m.group(1)).name + " " + m.group(2)
    ledger = _load_json(LEDGER, {})
    if key in ledger and not _override(cmd, "CONFIRMATORY_RERUN", "run-once", [key]):
        print(f"Blocked: `{key}` already ran as a confirmatory test ({ledger[key].get('time')}). Pre-registered tests "
              "run once. Re-run only with the author's explicit OK: CONFIRMATORY_RERUN=\"reason\" ...", file=sys.stderr)
        return 2
    pend = not_intaken()
    if pend is None or pend:
        if not _override(cmd, "SKIP_GATE", "session-intake", pend or []):
            what = ("session-intake has not run under the current guard" if pend is None else
                    f"{len(pend)} recording(s) have not been through session-intake: " + ", ".join(pend[:5]))
            print(f"Blocked: before a confirmatory run, every recording must be classified: {what}. Run the "
                  "`session-intake` skill first. " + _OVERRIDE_HINT.replace(" git commit", ""), file=sys.stderr)
            return 2
    STATE.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(json.dumps({"key": key, "tree": str(top), "rows": _confirmatory_rows(top),
                                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
    return 0


def post_bash(data: dict) -> int:
    """After a confirmatory run: if it logged confirmatory rows, it counts as the one run."""
    pend = _load_json(PENDING, None)
    if not pend:
        return 0
    m = _CONFIRM.search((data.get("tool_input") or {}).get("command", ""))
    if not m or Path(m.group(1)).name + " " + m.group(2) != pend.get("key"):
        return 0
    rows = _confirmatory_rows(Path(pend["tree"]))
    if rows > pend.get("rows", 0):
        ledger = _load_json(LEDGER, {})
        ledger[pend["key"]] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "rows_added": rows - pend.get("rows", 0)}
        LEDGER.write_text(json.dumps(ledger, indent=1))
    try:
        PENDING.unlink()
    except Exception:
        pass
    return 0


def pre_bash(data: dict) -> int:
    return confirmatory_gate(data) or pre_commit(data)


# ---- edits ----

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


def _eval_context(data: dict, top: Path, rel: str) -> None:
    """Once per session: editing evaluation code while a pre-registration is open adds a factual note."""
    if not rel.startswith(EVAL_CONTEXT):
        return
    opn = open_preregistrations(_work(top, "PREREGISTRATIONS.md"))
    if not opn:
        return
    STATE.mkdir(parents=True, exist_ok=True)
    seen = STATE / f"ctx_{_safe(data.get('session_id', '?'))}"
    if seen.exists():
        return
    seen.write_text(rel)
    note = ("Open pre-registration(s) in PREREGISTRATIONS.md with no RESULT yet: " + "; ".join(h[:120] for h in opn)
            + ". Their data rule reserves the sessions it names: fitting or evaluating on those sessions before the "
            "guarded run would bias the test, and the confirmatory entry point runs once.")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": note}}))


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
    try:
        _eval_context(data, top, rel)
    except Exception:
        pass
    return 0


# ---- stop ----

def stop(data: dict) -> int:
    if data.get("stop_hook_active"):
        return 0
    top = _toplevel(_cwd(data))
    if top is None or not _same_repo(top):
        return 0
    changed = {ln[3:].split(" -> ")[-1].strip().strip('"') for ln in _git("status", "--porcelain", cwd=top).splitlines()}
    start = _load_json(STATE / f"start_{_safe(data.get('session_id', '?'))}.json", {}).get("head")
    if start:                                      # also what was committed during this session
        changed |= {p for p in _git("diff", "--name-only", start, "HEAD", cwd=top).splitlines() if p}
    code = sorted(p for p in changed if p.startswith(("src/", "scripts/")))
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


# ---- doctor ----

def doctor() -> int:
    """Print what the guard sees, so a silent failure (no tests, a hook not registered) is visible."""
    top = _toplevel(Path.cwd()) or ROOT
    print(f"project dir: {ROOT}\nmain checkout: {MAIN}\nthis tree: {top}\nstate: {STATE}")
    settings = _load_json(top / ".claude" / "settings.json", {})
    for event, entries in (settings.get("hooks") or {}).items():
        for e in entries:
            for h in e.get("hooks", []):
                print(f"hook {event} [{e.get('matcher', '*')}] -> {(h.get('args') or ['?'])[-1]}")
    py = next((p for p in (MAIN / ".venv" / "bin" / "python", top / ".venv" / "bin" / "python") if p.exists()), None)
    ok = py is not None and subprocess.run([str(py), "-c", "import pytest"], capture_output=True).returncode == 0
    print("tests before commits: " + (f"yes ({py})" if ok else "NO (no .venv with pytest; run `uv sync`)"))
    print("diary: " + (str(NOTES) if NOTES.is_file() else f"not found at {NOTES}"))
    lock = _load_json(LOCK, {})
    print("lock: " + (f"{lock.get('session_id')} since {lock.get('started')}" if lock else "none"))
    for name in REVIEW_RULES:
        rec = _load_json(_marker(name), {})
        print(f"review {name}: " + (f"recorded {rec.get('time')} on {len(rec.get('files', []))} file(s)" if rec else "none"))
    for name in SKILL_GATES + AUDIT_SKILLS:
        rec = _load_json(_skill_marker(name), {})
        state = ("fresh" if skill_fresh(name, top) else "stale (HEAD moved)") if rec else "never recorded"
        print(f"skill {name}: " + (f"{rec.get('time')}, {state}" if rec else state))
    print("open pre-registrations: " + ("; ".join(open_preregistrations(_work(top, "PREREGISTRATIONS.md"))) or "none"))
    pend = not_intaken()
    print("recordings not intaken: " + ("intake never recorded" if pend is None else str(len(pend))))
    print("confirmatory ledger: " + (", ".join(_load_json(LEDGER, {})) or "empty"))
    for g in (review_gate("", top), skill_gates("", top)):
        if g:
            print("pending: " + _first_sentence(g))
    return 0


def main() -> int:
    event = sys.argv[1] if len(sys.argv) > 1 else ""
    if event == "review-ok":
        return review_ok(sys.argv[2] if len(sys.argv) > 2 else "")
    if event == "skill-ok":
        return skill_ok(sys.argv[2] if len(sys.argv) > 2 else "")
    if event == "doctor":
        return doctor()
    data = _read_input()
    handlers = {"session_start": session_start, "session_end": session_end, "pre_commit": pre_commit,
                "pre_bash": pre_bash, "post_bash": post_bash, "protect": protect, "stop": stop,
                "review_record": review_record, "skill_record": skill_record, "prompt_record": prompt_record}
    fn = handlers.get(event)
    return fn(data) if fn else 0


if __name__ == "__main__":
    sys.exit(main())
