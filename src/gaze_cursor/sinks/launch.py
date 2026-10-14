"""Start the project's tools from the dashboard: a whitelist, validated options, one camera process at a time.

OS-specific (it opens Terminal through macOS ``open``), so it lives in ``sinks``; the dashboard server only calls it.

- **Live tools** (camera, window) are written to a temporary ``.command`` file under ``~/.gaze-cursor/launch/`` and
  opened with ``open`` so they run in Terminal, which already holds the Camera / Input Monitoring permissions,
  exactly like double-clicking ``Cursor.command``. Every argument is checked against the tool's option spec and
  shell-quoted; nothing from the request is ever run as text.
- **Headless tools** run as a subprocess with a timeout; their output is kept for the page to poll.
- Frozen pre-registered tests (``evaluate*.py --preregistered*``, ``--develop``) and dev scripts are not in the whitelist.
- Every launch is appended to ``~/.gaze-cursor/launch_log.jsonl`` (what, options, time; no camera data).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

HOME_DIR = Path.home() / ".gaze-cursor"
CAMERA_SCRIPTS = ("cursor_mode", "gaze_overlay", "calibrate", "target_practice", "check_setup", "viewer")
TAG_ENV = {"person": "GAZE_PERSON_ID", "lighting_tag": "GAZE_LIGHTING_TAG", "glasses": "GAZE_GLASSES",
           "posture_note": "GAZE_POSTURE_NOTE"}
LIVE_COOLDOWN_S = 10.0
_LIVE_LOCK = threading.Lock()
_LAST_LIVE = [0.0]
MAX_OUTPUT = 200_000
HEADLESS_TIMEOUT_S = 900


class LaunchError(ValueError):
    """The request is not allowed (unknown tool, bad option, camera busy). The message is safe to show."""


@dataclass(frozen=True)
class Opt:
    """One allowed option. ``kind``: flag (bool), int, choice, text, session (an id from the catalogue)."""

    name: str
    kind: str
    flag: str = ""                       # the command-line flag, "" for a positional
    choices: tuple[str, ...] = ()
    lo: int = 0
    hi: int = 0
    pattern: str = ""
    default: object = None
    label: str = ""


@dataclass(frozen=True)
class Tool:
    key: str
    title: str
    kind: str                            # "live" | "headless"
    script: str                          # scripts/<script>.py, or "" for the tests tool
    args: tuple[str, ...] = ()           # fixed arguments
    help: str = ""
    options: tuple[Opt, ...] = ()
    note: str = ""                       # shown next to the button (side effects)
    tags: bool = False                   # accepts the condition-tag form (person, lighting, glasses, posture)


def _flag(name: str, flag: str, label: str, default: bool = False) -> Opt:
    return Opt(name, "flag", flag, label=label, default=default)


TOOLS: dict[str, Tool] = {t.key: t for t in (
    Tool("calibrate", "Calibrate", "live", "calibrate", ("--guided",),
         "Lighting guide, setup check, then the 25-dot calibration + held-out test.",
         (Opt("note", "text", "--note", pattern=r"^[\w ,.:;()+/'-]{0,120}$", label="Note (set-B slot, e.g. pos N)"),),
         "Writes data/sessions/session_*.npz", tags=True),
    Tool("cursor", "Cursor mode", "live", "cursor_mode", (),
         "Tap Right Command and the cursor jumps to where you look. Cmd+Option+X quits and saves.",
         (_flag("fresh", "--fresh", "Always calibrate (ignore the saved one)"),
          _flag("no_learn", "--no-learn", "Do not learn from clicks"),
          _flag("show_dot", "--show-dot", "Show a faint ring where it thinks you look"),
          Opt("warp_key", "choice", "--warp-key", ("right_command", "right_option"), label="Jump key",
              default="right_command")),
         "Writes data/quick/cursor_*.npz and last_calibration.npz", tags=True),
    Tool("demo", "Demo (show a friend)", "live", "cursor_mode", ("--demo",),
         "Their own 9-dot calibration, a ring shows where it thinks they look, tap Right Command to jump.",
         (), "Nothing is saved: no recording, no log, your saved calibration is untouched"),
    Tool("run", "Run (live dot + tests)", "live", "gaze_overlay", (),
         "9-dot calibration, then the live gaze dot with T / A / P tests.", (),
         "Saves quick-calibration files and test logs under data/quick/ and logs to results.csv", tags=True),
    Tool("practice", "Target practice", "live", "target_practice", (),
         "Move the gaze dot into a shrinking box with your eyes.",
         (Opt("trials", "int", "--trials", lo=1, hi=200, default=20, label="Trials"),
          _flag("no_save", "--no-save", "Do not save frames or log the result")),
         "Saves data/tasks/*.npz and logs to results.csv unless 'do not save' is ticked"),
    Tool("check_setup", "Posture and lighting check", "live", "check_setup", (),
         "Fullscreen gauge of where you sit compared with your earlier sessions.",
         (Opt("last", "int", "--last", lo=1, hi=50, label="Compare with the last N sessions"),), ""),
    Tool("viewer", "Face-mesh viewer", "live", "viewer", (), "Live face mesh, irises, head pose and FPS (q quits).", (), ""),
    Tool("session_report", "Session report", "headless", "session_report", (),
         "pass / warn / fail per check for one recording (tag-only for Cursor-mode data).",
         (Opt("session", "session", "", label="Recording"),
          _flag("no_log", "--no-log", "Do not append a row to results.csv", True)),
         "Saves <session>.report.png next to the recording, like the command line does"),
    Tool("positions", "Positions list", "headless", "positions", ("--all",), "Lighting and head position of every session.", (), ""),
    Tool("evaluate_status", "Round-4 intake status", "headless", "evaluate_online", ("--status",),
         "Minutes and click counts of Cursor logs and whether they count (no error numbers).", (), ""),
    Tool("download_model", "Download face model", "headless", "download_model", (),
         "Fetches the MediaPipe face model (~4 MB, one time) into models/.", (), "Writes models/face_landmarker.task"),
    Tool("tests", "Run tests", "headless", "", (), "Runs the project test suite (pytest -q).", (), ""),
)}


def tool_specs() -> list[dict]:
    """The tools as the page needs them (no paths, no commands)."""
    return [{"key": t.key, "title": t.title, "kind": t.kind, "help": t.help, "note": t.note, "tags": t.tags,
             "options": [{"name": o.name, "kind": o.kind, "label": o.label, "choices": list(o.choices), "lo": o.lo,
                          "hi": o.hi, "default": o.default} for o in t.options]} for t in TOOLS.values()]


# --- option validation ------------------------------------------------------------------------------
def _validate(tool: Tool, options: dict, resolve_session: Callable[[str], Path | None]) -> list[str]:
    """Turn the request's options into argv words. Unknown options and bad values are refused."""
    argv: list[str] = list(tool.args)
    known = {o.name: o for o in tool.options}
    for name in options:
        if name not in known and not (tool.tags and name in TAG_ENV):
            raise LaunchError(f"unknown option '{name}' for {tool.title}")
    for o in tool.options:
        if o.name not in options:
            if o.kind == "flag" and o.default:
                argv.append(o.flag)
            if o.kind == "session":
                raise LaunchError("choose a recording")
            continue
        v = options[o.name]
        if o.kind == "flag":
            if not isinstance(v, bool):
                raise LaunchError(f"'{o.name}' must be true or false")
            if v:
                argv.append(o.flag)
        elif o.kind == "int":
            if isinstance(v, bool) or not isinstance(v, int) or not o.lo <= v <= o.hi:
                raise LaunchError(f"'{o.name}' must be a whole number from {o.lo} to {o.hi}")
            argv += [o.flag, str(v)]
        elif o.kind == "choice":
            if v not in o.choices:
                raise LaunchError(f"'{o.name}' must be one of {', '.join(o.choices)}")
            argv += [o.flag, str(v)]
        elif o.kind == "text":
            if not isinstance(v, str) or not re.fullmatch(o.pattern, v):
                raise LaunchError(f"'{o.name}' has characters that are not allowed")
            if v.strip():
                argv += [o.flag, v.strip()]
        elif o.kind == "session":
            path = resolve_session(v) if isinstance(v, str) else None
            if path is None:
                raise LaunchError("unknown recording")
            argv.append(str(path))
    return argv


def _tag_env(tool: Tool, options: dict) -> dict[str, str]:
    """Condition-tag form fields as environment variables (read by the recorders once PLAN 6c stage 4 wires them)."""
    out: dict[str, str] = {}
    if not tool.tags:
        return out
    patterns = {"person": r"p\d{2,}", "lighting_tag": r"day|lamp|dark|backlit", "glasses": r"yes|no",
                "posture_note": r"[\w ,.:;()+/'-]{1,120}"}
    for name, env in TAG_ENV.items():
        v = options.get(name)
        if v in (None, ""):
            continue
        if not isinstance(v, str) or not re.fullmatch(patterns[name], v):
            raise LaunchError(f"'{name}' is not a valid value")
        out[env] = v
    return out


def build(key: str, options: dict, resolve_session: Callable[[str], Path | None] = lambda s: None) -> tuple[Tool, list[str], dict[str, str]]:
    """(tool, argv words after ``uv run``, tag environment). Raises ``LaunchError`` for anything not allowed."""
    tool = TOOLS.get(key) if isinstance(key, str) else None
    if tool is None:
        raise LaunchError("unknown tool")
    if not isinstance(options, dict):
        raise LaunchError("options must be an object")
    words = _validate(tool, options, resolve_session)
    env = _tag_env(tool, options)
    argv = ["pytest", "-q"] if key == "tests" else [f"scripts/{tool.script}.py", *words]
    if key == "tests":
        env = {**env, "CI": "1"}   # skip the tests that open windows: a camera tool may be recording
    return tool, argv, env


# --- running -------------------------------------------------------------------------------------------
def running_camera_processes() -> list[str]:
    """Names of camera tools already running (any of them holds the camera)."""
    try:
        out = subprocess.run(["pgrep", "-fl", "scripts/(" + "|".join(CAMERA_SCRIPTS) + ")\\.py"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = {m for line in out.splitlines() for m in CAMERA_SCRIPTS if f"scripts/{m}.py" in line}
    return sorted(found)


def _log(entry: dict, home: Path) -> None:
    try:
        home.mkdir(parents=True, exist_ok=True)
        with (home / "launch_log.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}) + "\n")
    except OSError:
        pass


def command_script(repo: Path, argv: list[str], env: dict[str, str]) -> str:
    """Text of the temporary ``.command`` file: same preamble as the repo's launchers, arguments shell-quoted."""
    exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in sorted(env.items()))
    return ("#!/bin/bash\n"
            f"cd {shlex.quote(str(repo))} || exit 1\n"
            'export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"\n'
            + exports
            + "uv run " + " ".join(shlex.quote(a) for a in argv) + "\n"
            'status=$?\necho\n'
            'if [ $status -ne 0 ]; then echo "Stopped (exit code $status)."; fi\n'
            'read -r -p "Press Enter to close" _\nexit $status\n')


def launch_live(tool: Tool, argv: list[str], env: dict[str, str], repo: Path, options: dict,
                home: Path = HOME_DIR, opener: Callable[[Path], object] | None = None) -> dict:
    """Open a live tool in Terminal. Refuses when another camera tool is running or off macOS."""
    if sys.platform != "darwin" and opener is None:
        raise LaunchError("launching live tools works on macOS only")
    busy = running_camera_processes()
    if busy:
        raise LaunchError(f"{', '.join(busy)} is already running (one camera tool at a time): quit it first")
    with _LIVE_LOCK:   # a double click or a second tab must not start two camera tools before the first shows in pgrep
        wait = LIVE_COOLDOWN_S - (time.time() - _LAST_LIVE[0])
        if wait > 0:
            raise LaunchError(f"a tool was just started: already running, wait {wait:.0f} s")
        _LAST_LIVE[0] = time.time()
    folder = home / "launch"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / f"{tool.key}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.command"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)   # private from the first byte: it holds the tags
    with os.fdopen(fd, "w") as f:
        f.write(command_script(repo, argv, env))
    (opener or (lambda p: subprocess.Popen(["open", str(p)])))(path)
    _log({"tool": tool.key, "kind": "live", "args": argv[1:], "tags": sorted(env)}, home)
    return {"started": tool.key, "window": "Terminal"}


@dataclass
class Job:
    id: str
    tool: str
    started: float
    status: str = "running"            # running | done | failed | timeout
    output: str = ""
    returncode: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_json(self) -> dict:
        return {"id": self.id, "tool": self.tool, "status": self.status, "output": self.output,
                "returncode": self.returncode, "seconds": round(time.time() - self.started, 1)}


JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def run_headless(tool: Tool, argv: list[str], env: dict[str, str], repo: Path, options: dict,
                 home: Path = HOME_DIR, runner: Callable[[list[str]], tuple[str, int]] | None = None) -> dict:
    """Start a headless tool in a background thread; one at a time. Returns the job (poll ``JOBS[id]``)."""
    with _JOBS_LOCK:
        if any(j.status == "running" for j in JOBS.values()):
            raise LaunchError("another tool is still running: wait for it to finish")
        job = Job(uuid.uuid4().hex[:10], tool.key, time.time())
        JOBS[job.id] = job
        for old in sorted(JOBS.values(), key=lambda j: j.started)[:-20]:
            JOBS.pop(old.id, None)
    _log({"tool": tool.key, "kind": "headless", "args": argv[1:], "tags": sorted(env)}, home)

    def work() -> None:
        try:
            if runner is not None:
                job.output, job.returncode = runner(argv)
            else:
                p = subprocess.run(["uv", "run", *argv], cwd=repo, capture_output=True, text=True,
                                   timeout=HEADLESS_TIMEOUT_S, env={**os.environ, **env})
                job.output, job.returncode = (p.stdout + p.stderr)[-MAX_OUTPUT:], p.returncode
            job.status = "done" if job.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            job.status, job.output = "timeout", f"stopped after {HEADLESS_TIMEOUT_S} s"
        except OSError as exc:
            job.status, job.output = "failed", f"could not start: {exc}"

    threading.Thread(target=work, daemon=True).start()
    return job.to_json()
