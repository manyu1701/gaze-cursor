from dataclasses import replace
from pathlib import Path

from gaze_cursor.pool import pool_sessions
from tests.helpers import fake_session


def _s(i: int, label: str | None, started: str, layout: str = "v2", window=None):
    s = fake_session(i, layout=layout, window=window)
    meta = {**s.meta, "started": started}
    if label:
        meta["lighting"] = {"label": label, "face": 120.0, "background": 75.0, "ratio": 1.6}
    return replace(s, meta=meta, path=Path(f"session_{i:02d}.npz"))


def test_pool_keeps_same_lighting_clean_v2_sessions() -> None:
    anchor = _s(0, "front-lit", "2026-10-13T05:35:00")
    same = _s(1, "front-lit", "2026-10-13T05:20:00")
    other_light = _s(2, "dim", "2026-10-13T05:10:00")
    no_light = _s(3, None, "2026-10-13T05:00:00")
    v1 = _s(4, "front-lit", "2026-10-13T03:00:00", layout="v1")
    squeezed = _s(5, "front-lit", "2026-10-13T02:50:00", window=[0, 33, 1470, 923])
    pool = pool_sessions([anchor, same, other_light, no_light, v1, squeezed], anchor)
    assert pool == [same, anchor]


def test_anchor_without_lighting_pools_alone() -> None:
    anchor = _s(0, None, "2026-10-13T05:35:00")
    assert pool_sessions([anchor, _s(1, "front-lit", "2026-10-13T05:20:00")], anchor) == [anchor]
