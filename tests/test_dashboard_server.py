"""Dashboard API and its security rules, via App.handle (no socket) and once over a real loopback socket."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from gaze_cursor.dashboard.server import App, serve
from tests.dashboard_fixtures import make_data



@pytest.fixture()
def app(tmp_path):
    return App(make_data(tmp_path), token="tok")


def get(app, path, headers=None):
    return app.handle("GET", path, {"Host": "127.0.0.1", **(headers or {})})


def body(r):
    return json.loads(r.body)


def test_overview_has_totals_tags_learned_and_health(app) -> None:
    r = get(app, "/api/overview")
    o = body(r)
    assert r.status == 200 and o["totals"]["files"] == 7 and o["tags"]["n"] == 7
    assert o["learned"]["exists"] and o["latest"]["file"]
    assert any("Face model" in h["text"] for h in o["health"])


def test_sessions_list_excludes_internal_fields(app) -> None:
    rows = body(get(app, "/api/sessions"))["rows"]
    assert len(rows) == 7 and "mtime" not in rows[0] and rows[0]["id"].count(":") == 1


def test_session_detail_includes_meta_without_the_bulky_names(app) -> None:
    d = body(get(app, "/api/session/sessions%3Asession_20261014_090000"))
    assert d["row"]["tags"] == 9 and d["conditions"]["person_id"] == "p01"
    assert "biased" not in d["monitor_note"] and "not evidence" in d["monitor_note"]
    assert "blendshape_names" not in d["meta"]


def test_unknown_session_and_traversal_ids_are_404(app) -> None:
    for sid in ("sessions%3Anope", "..%2F..%2Fetc%2Fpasswd", "sessions%3A..%2Fx", "sessions%3Asession_20261012_080000.faces"):
        assert get(app, f"/api/session/{sid}").status == 404
        assert get(app, f"/api/session/{sid}/quality").status == 404
        assert get(app, f"/api/session/{sid}/report.png").status == 404


def test_quality_endpoint_for_cursor_log_is_tag_only(app) -> None:
    rep = body(get(app, "/api/session/quick%3Acursor_20261014_100000/quality"))
    assert rep["mode"] == "tags_only"


def test_report_png_is_served_when_present(app) -> None:
    (app.roots.sessions / "session_20261013_100000.report.png").write_bytes(b"\x89PNGdata")
    r = get(app, "/api/session/sessions%3Asession_20261013_100000/report.png")
    assert r.status == 200 and r.content_type == "image/png" and r.body.startswith(b"\x89PNG")


def test_learning_and_results_pages(app) -> None:
    lr = body(get(app, "/api/learning"))
    assert len(lr["per_day"]) == 3 and lr["levels"][1]["state"] == "not built" and "not evidence" in lr["monitor_note"]
    assert lr["levels"][0]["state"] == "built" and "low confidence" in lr["levels"][0]["text"]
    assert lr["per_day"][-1]["click_gaze"][0]["n"] == 38
    rs = body(get(app, "/api/results"))
    assert rs["results_csv"]["n"] == 2 and len(rs["rounds"]) == 2 and rs["docs"][0] == "RESULTS"


def test_docs_are_whitelisted(app) -> None:
    assert get(app, "/api/docs/RESULTS").content_type.startswith("text/markdown")
    for name in ("CLAUDE", "..%2FCLAUDE", "pyproject.toml", ""):
        assert get(app, f"/api/docs/{name}").status == 404


def test_index_embeds_the_token_and_nothing_else_is_templated(app) -> None:
    r = get(app, "/")
    text = r.body.decode()
    assert r.status == 200 and "const TOKEN = \"tok\"" in text and "__TOKEN__" not in text


def test_requests_for_another_host_are_refused(app) -> None:
    for host in ("evil.example", "127.0.0.1.evil.example", "", "192.168.1.5"):
        assert app.handle("GET", "/api/overview", {"Host": host}).status == 403
    assert app.handle("GET", "/api/overview", {}).status == 403


def test_bound_port_is_part_of_the_host_check(tmp_path) -> None:
    a = App(make_data(tmp_path), token="t")
    a.port = 5050
    assert a.handle("GET", "/api/overview", {"Host": "127.0.0.1:5050"}).status == 200
    assert a.handle("GET", "/api/overview", {"Host": "localhost:5050"}).status == 200
    assert a.handle("GET", "/api/overview", {"Host": "127.0.0.1:6000"}).status == 403


def test_post_needs_the_token_and_unknown_posts_404(app) -> None:
    assert app.handle("POST", "/api/launch", {"Host": "127.0.0.1"}, b"{}").status == 403
    assert app.handle("POST", "/api/launch", {"Host": "127.0.0.1", "X-Token": "wrong"}, b"{}").status == 403
    assert app.handle("POST", "/api/nothing", {"Host": "127.0.0.1", "x-token": "tok"}, b"{}").status == 404
    assert app.handle("PUT", "/api/overview", {"Host": "127.0.0.1"}).status == 405


def test_cross_site_requests_and_malformed_tokens_are_refused(app) -> None:
    assert get(app, "/api/overview", {"Sec-Fetch-Site": "cross-site"}).status == 403
    assert get(app, "/api/overview", {"Sec-Fetch-Site": "same-site"}).status == 403
    for ok in ("same-origin", "none"):
        assert get(app, "/api/overview", {"Sec-Fetch-Site": ok}).status == 200
    r = app.handle("POST", "/api/launch", {"Host": "127.0.0.1", "X-Token": "tök\u00e9n"}, b"{}")
    assert r.status == 403   # a non-ASCII token is a refusal, not a crash


def test_sessions_and_detail_tell_the_page_how_many_tags_are_checked(app) -> None:
    assert body(get(app, "/api/sessions"))["n_checked"] == 9
    assert body(get(app, "/api/session/sessions%3Asession_20261014_090000"))["n_checked"] == 9


def test_no_cors_headers_are_ever_added(app) -> None:
    assert not any(k.lower().startswith("access-control") for k in get(app, "/api/overview").headers)


def test_real_socket_roundtrip_binds_loopback_only(tmp_path) -> None:
    a = App(make_data(tmp_path), token="t")
    httpd = serve(a, 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        assert httpd.server_address[0] == "127.0.0.1" and a.port == httpd.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/api/overview", timeout=10) as resp:
            assert resp.status == 200
            assert resp.headers["X-Content-Type-Options"] == "nosniff"
            assert "default-src 'self'" in resp.headers["Content-Security-Policy"]
            assert not [k for k in resp.headers if k.lower().startswith("access-control")]
            assert json.loads(resp.read())["totals"]["files"] == 7
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/api/overview", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
