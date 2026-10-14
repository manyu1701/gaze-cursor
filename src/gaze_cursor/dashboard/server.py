"""The dashboard's local web server (standard library only).

``App.handle`` is the whole API as a pure function of (method, path, headers, body), so it is tested without a
socket; ``serve`` wraps it in a ``ThreadingHTTPServer`` bound to 127.0.0.1 only.

Security model (a page that can start programs and shows personal data must not be reachable from other sites):
bound to loopback, a request whose ``Host`` is not this server is refused (DNS rebinding), no CORS headers, and
``POST`` needs the per-run token that only the served page knows. Recordings are addressed by id, never by path;
face-image sidecars and anything outside the whitelisted file patterns cannot be reached.
"""

from __future__ import annotations

import json
import math
import secrets
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from gaze_cursor.conditions import N_CHECKED
from gaze_cursor.dashboard import catalog, quality, summaries
from gaze_cursor.dashboard.catalog import Roots
from gaze_cursor.sinks import launch

STATIC = Path(__file__).parent / "static"
HIDE_META = {"blendshape_names"}
LEVELS = [
    {"level": "Session", "state": "built", "text": "Click learning adapts to how you sit right now. Round 4: supported by the "
                                                  "frozen rule, low confidence, small gain mostly in long sessions."},
    {"level": "Personal (long term)", "state": "not built", "text": "Needs L2 (versioned model states) and L3; "
                                                                   "raw labelled data is kept in the logs."},
    {"level": "Population", "state": "not built", "text": "Pretrained gaze model, PLAN M3 step 8, then other people (M5)."},
]


@dataclass
class Response:
    status: int
    body: bytes
    content_type: str = "application/json; charset=utf-8"
    headers: dict[str, str] = field(default_factory=dict)


def _scrub(v: object) -> object:
    """NaN / infinity are not JSON: they become null."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _scrub(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_scrub(x) for x in v]
    return v


def _json(obj: object, status: int = 200) -> Response:
    return Response(status, json.dumps(_scrub(obj), allow_nan=False, default=str).encode())


def _err(status: int, message: str) -> Response:
    return _json({"error": message}, status)


class App:
    """Routes. ``port`` is set once the socket is bound; until then any Host that names loopback is accepted."""

    def __init__(self, roots: Roots | None = None, token: str | None = None, port: int = 0,
                 home: Path | None = None, opener: Callable[[Path], object] | None = None,
                 runner: Callable[[list[str]], tuple[str, int]] | None = None) -> None:
        self.roots = roots or Roots.default()
        self.token = token or secrets.token_urlsafe(24)
        self.port = port
        self.home = home or launch.HOME_DIR      # launch log and temporary .command files
        self.opener, self.runner = opener, runner   # injectable so tests start nothing

    # --- request guards -----------------------------------------------------------------------
    def host_ok(self, host: str | None) -> bool:
        if not host:
            return False
        allowed = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"} if self.port else None
        if allowed is not None:
            return host.lower() in allowed
        return host.lower().split(":")[0] in ("127.0.0.1", "localhost")

    # --- routing --------------------------------------------------------------------------------
    def handle(self, method: str, target: str, headers: dict[str, str], body: bytes = b"") -> Response:
        h = {k.lower(): v for k, v in headers.items()}
        if not self.host_ok(h.get("host")):
            return _err(403, "bad host")
        if h.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
            return _err(403, "cross-site request")   # a page on another site must not make this server work
        url = urlsplit(target)
        path = url.path
        if method == "GET":
            return self._get(path, parse_qs(url.query))
        if method == "POST":
            if not secrets.compare_digest(h.get("x-token", "").encode("utf-8", "replace"), self.token.encode()):
                return _err(403, "bad token")
            return self._post(path, body)
        return _err(405, "method not allowed")

    def _rows(self) -> list[catalog.SessionRow]:
        return catalog.scan(self.roots)

    def _get(self, path: str, query: dict[str, list[str]]) -> Response:
        if path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_text().replace("__TOKEN__", self.token)
            return Response(200, html.encode(), "text/html; charset=utf-8")
        if path == "/favicon.ico":
            return Response(204, b"", "image/x-icon")
        if path == "/api/overview":
            return _json(self.overview())
        if path == "/api/sessions":
            return _json({"rows": [r.to_json() for r in self._rows()], "n_checked": N_CHECKED})
        if path.startswith("/api/session/"):
            return self._session(unquote(path[len("/api/session/"):]))
        if path == "/api/tools":
            return _json({"tools": launch.tool_specs(), "camera_busy": launch.running_camera_processes(),
                          "macos": sys.platform == "darwin" or self.opener is not None})
        if path.startswith("/api/job/"):
            job = launch.JOBS.get(unquote(path[len("/api/job/"):]))
            return _err(404, "unknown job") if job is None else _json(job.to_json())
        if path == "/api/learning":
            return _json(self.learning())
        if path == "/api/results":
            return _json(self.results())
        if path.startswith("/api/docs/"):
            text = summaries.read_doc(self.roots, unquote(path[len("/api/docs/"):]))
            if text is None:
                return _err(404, "no such document")
            return Response(200, text.encode(), "text/markdown; charset=utf-8")
        return _err(404, "not found")

    def _post(self, path: str, body: bytes) -> Response:
        if path != "/api/launch":
            return _err(404, "not found")
        try:
            req = json.loads(body or b"{}")
            if not isinstance(req, dict):
                raise ValueError
        except ValueError:
            return _err(400, "bad request body")
        try:
            tool, argv, env = launch.build(req.get("tool"), req.get("options") or {},
                                           lambda sid: catalog.path_of(self.roots, sid))
            if tool.kind == "live":
                result = launch.launch_live(tool, argv, env, self.roots.repo, req.get("options") or {},
                                            self.home, self.opener)
            else:
                result = launch.run_headless(tool, argv, env, self.roots.repo, req.get("options") or {},
                                             self.home, self.runner)
        except launch.LaunchError as exc:
            return _err(409 if "already running" in str(exc) or "still running" in str(exc) else 400, str(exc))
        return _json(result)

    # --- pages ---------------------------------------------------------------------------------
    def overview(self) -> dict:
        rows = self._rows()
        learned = summaries.learned_state(self.roots)
        recent = next((r for r in rows if not r.error), None)
        return {"totals": summaries.totals(rows), "tags": summaries.tag_completeness(rows), "learned": learned,
                "health": summaries.health(rows, self.roots, learned), "latest": recent.to_json() if recent else None}

    def learning(self) -> dict:
        rows = self._rows()
        return {"per_day": summaries.per_day(rows), "learned": summaries.learned_state(self.roots),
                "coverage": summaries.coverage(rows), "levels": LEVELS, "monitor_note": summaries.MONITOR_NOTE,
                "totals": summaries.totals(rows)}

    def results(self) -> dict:
        prereg = summaries.read_doc(self.roots, "PREREGISTRATIONS") or ""
        plan = summaries.read_doc(self.roots, "PLAN") or ""
        return {"results_csv": summaries.results_summary(self.roots.results),
                "rounds": summaries.prereg_rounds(prereg), "plan_status": summaries.plan_status(plan),
                "docs": list(summaries.DOCS)}

    def _session(self, rest: str) -> Response:
        if rest.endswith("/quality"):
            report = quality.report_for(self.roots, rest[: -len("/quality")])
            return _err(404, "unknown session") if report is None else _json(report)
        if rest.endswith("/report.png"):
            png = quality.png_path(self.roots, rest[: -len("/report.png")])
            if png is None:
                return _err(404, "no report image (run the Session report tool to create one)")
            return Response(200, png.read_bytes(), "image/png")
        p = catalog.path_of(self.roots, rest)
        if p is None:
            return _err(404, "unknown session")
        row = catalog.build_row(rest.split(":", 1)[0], p)
        try:
            meta = {k: v for k, v in catalog.read_meta(p).items() if k not in HIDE_META}
        except (OSError, ValueError, KeyError, EOFError):
            meta = {}
        block = catalog.conditions_block(meta, p)
        return _json({"row": row.to_json(), "meta": meta, "conditions": block, "n_checked": N_CHECKED,
                      "monitor_note": summaries.MONITOR_NOTE})


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "gaze-dashboard"

        def _respond(self, method: str) -> None:
            try:
                length = max(0, int(self.headers.get("Content-Length") or 0))
            except ValueError:
                length = -1
            if length < 0:
                self.send_error(400, "bad Content-Length")
                return
            body = self.rfile.read(min(length, 1_000_000)) if length else b""
            r = app.handle(method, self.path, dict(self.headers.items()), body)
            self.send_response(r.status)
            self.send_header("Content-Type", r.content_type)
            self.send_header("Content-Length", str(len(r.body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                             "img-src 'self' data:; frame-ancestors 'none'")
            for k, v in r.headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(r.body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            self._respond("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._respond("POST")

        def log_message(self, *args: object) -> None:   # quiet: no access log in the terminal
            pass

    return Handler


def serve(app: App, port: int = 0) -> ThreadingHTTPServer:
    """Bind to loopback (``port`` 0 = any free port) and return the server; the caller runs ``serve_forever``."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    app.port = httpd.server_address[1]
    return httpd
