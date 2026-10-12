import numpy as np
import pytest

from gaze_cursor.posture import (
    FLOORS, METRICS, PostureGate, Reference, lighting_stats, lighting_warning, posture_metrics)
from tests.helpers import fake_session
from tests.test_features import _face


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    return m


def test_metrics_from_synthetic_face() -> None:
    pts = _face()
    m = posture_metrics(pts, _rot_y(10), (1000, 1000))
    assert m["yaw"] == pytest.approx(10, abs=1e-6) and m["pitch"] == pytest.approx(0, abs=1e-6)
    assert set(m) == set(METRICS)
    # iris centres are 0.30 apart (0.35 vs 0.65) in a square frame -> 0.30 of the width
    assert m["distance"] == pytest.approx(0.30)


def _ref(pitch=-7.0, spread=0.0) -> Reference:
    rng = np.random.default_rng(0)
    n = 200
    vals = {"pitch": rng.normal(pitch, spread, n) if spread else np.full(n, pitch),
            "yaw": np.zeros(n), "roll": np.zeros(n), "distance": np.full(n, 0.066),
            "x": np.full(n, 0.5), "y": np.full(n, 0.55)}
    return Reference.from_arrays(vals)


def test_reference_center_and_floored_scale() -> None:
    ref = _ref()
    assert ref.center["pitch"] == pytest.approx(-7.0)
    assert ref.scale["pitch"] == FLOORS["pitch"]                  # no spread -> floor
    assert ref.scale["distance"] == pytest.approx(0.066 * FLOORS["distance"])  # relative floor
    wide = _ref(spread=6.0)
    assert wide.scale["pitch"] > 5.0                              # real spread widens tolerance


def test_levels_and_worst_metric() -> None:
    ref = _ref()
    live = {"pitch": -7.0, "yaw": 0.0, "roll": 0.0, "distance": 0.066, "x": 0.5, "y": 0.55}
    assert ref.worst(live)[1] == "good"
    live["pitch"] = -1.2                                          # +5.8 deg = 2.3 units
    m, level, z = ref.worst(live)
    assert (m, level) == ("pitch", "adjust") and z == pytest.approx(5.8 / 2.5)
    live["pitch"] = 10.0
    assert ref.worst(live)[1] == "off"


def test_distance_hint_direction() -> None:
    ref = _ref()
    assert ref.hint("distance", 0.08) == "move back a little"      # larger inter-iris = closer
    assert ref.hint("distance", 0.05) == "move closer"


def test_gate_requires_hold_and_debounces_release() -> None:
    g = PostureGate(hold_s=1.0, release_s=0.5)
    assert not g.update(0.0, True) and not g.update(0.9, True)
    assert g.update(1.0, True)
    assert g.update(1.2, False) and g.update(1.6, False)          # short dropouts are tolerated
    assert not g.update(1.8, False)                               # 0.6 s bad -> lost
    assert not g.update(2.0, True) and g.update(3.0, True)        # must hold again


def test_reference_from_sessions_uses_stored_landmarks() -> None:
    s = fake_session(0)
    n = len(s.phase)
    pts = np.tile(_face(), (n, 1, 1))
    tr = np.tile(_rot_y(0), (n, 1, 1))
    s = type(s)(**{**s.__dict__, "points": pts, "transform": tr,
                   "meta": {**s.meta, "image_size": [1000, 1000]}})
    ref = Reference.from_sessions([s], stride=1)
    assert ref.n_sessions == 1 and ref.n_frames == n
    assert ref.center["distance"] == pytest.approx(0.30) and ref.center["yaw"] == pytest.approx(0, abs=1e-6)


def test_lighting_stats_and_warning() -> None:
    gray = np.full((100, 100), 150, np.uint8)
    pts = np.zeros((478, 3))
    pts[:468, 0] = np.linspace(0.3, 0.7, 468)
    pts[:468, 1] = np.linspace(0.3, 0.7, 468)
    gray[30:70, 30:70] = 60                                       # dark face, bright background
    st = lighting_stats(gray, pts)
    assert st["face"] == pytest.approx(60, abs=1) and st["background"] == pytest.approx(150, abs=1)
    assert st["ratio"] == pytest.approx(0.4, abs=0.05) and lighting_warning(st) is None
    gray[30:70, 30:70] = 20
    assert lighting_warning(lighting_stats(gray, pts)) == "very dark: tracking may fail"


def test_check_screen_keeps_the_notch_strip_clear() -> None:
    from gaze_cursor.config import ScreenConfig
    from gaze_cursor.sinks.overlay import FullscreenCanvas
    from gaze_cursor.sinks.posture_view import draw_check

    cfg = ScreenConfig()                                   # notch strip: top 32 pt, x 646..825
    canvas = FullscreenCanvas(cfg)
    ref = _ref()
    frame = np.full((720, 1280, 3), 120, np.uint8)
    live = {"pitch": -1.2, "yaw": 0.0, "roll": 0.0, "distance": 0.07, "x": 0.5, "y": 0.55}
    for metrics, lit in ((live, True), (None, False)):
        img = draw_check(canvas, frame, metrics, ref, False,
                         {"face": 70.0, "background": 140.0, "ratio": 0.5, "clipped": 0.0} if lit else None,
                         "Sit as you did in the earlier sessions, then press SPACE")
        assert (img[: int(cfg.top_inset_pt) + 2] == canvas.blank()[: int(cfg.top_inset_pt) + 2]).all()


def _frames(n: int, **shift: float) -> list[dict[str, float]]:
    base = {"pitch": -7.0, "yaw": 0.0, "roll": 0.0, "distance": 0.066, "x": 0.5, "y": 0.55}
    return [{**base, **shift} for _ in range(n)]


def test_recording_summary_good_session() -> None:
    from gaze_cursor.posture import summarize_recording

    s = summarize_recording(_ref(), _frames(100))
    assert s["fraction_good"] == 1.0 and s["verdict"].startswith("good session")
    assert all(v == 0.0 for v in s["not_good_fraction"].values())


def test_recording_summary_names_the_metric_that_drifted() -> None:
    from gaze_cursor.posture import summarize_recording

    frames = _frames(60) + _frames(40, pitch=0.0)          # last 40% of frames: pitch +7 deg = 2.8 units
    s = summarize_recording(_ref(), frames)
    assert s["fraction_good"] == pytest.approx(0.6) and s["worst_metric"] == "pitch"
    assert s["not_good_fraction"]["pitch"] == pytest.approx(0.4)
    assert "head pitch" in s["verdict"] and "recording again" in s["verdict"]
    import json
    json.dumps(s)                                            # must be storable in the session meta


def test_recording_summary_without_frames() -> None:
    from gaze_cursor.posture import summarize_recording

    assert summarize_recording(_ref(), [])["frames"] == 0


def test_guide_screens_keep_the_notch_strip_clear() -> None:
    from gaze_cursor.config import ScreenConfig
    from gaze_cursor.posture import summarize_recording
    from gaze_cursor.sinks.guide_view import SETUP_STEPS, draw_instructions, draw_summary
    from gaze_cursor.sinks.overlay import FullscreenCanvas

    cfg = ScreenConfig()
    canvas = FullscreenCanvas(cfg)
    strip = int(cfg.top_inset_pt) + 2
    blank = canvas.blank()[:strip]
    s = summarize_recording(_ref(), _frames(10, pitch=0.0))
    for img in (draw_instructions(canvas), draw_summary(canvas, s, "session_x.npz"),
                draw_summary(canvas, summarize_recording(_ref(), []), "session_x.npz")):
        assert (img[:strip] == blank).all()
    heads = " ".join(h for h, _ in SETUP_STEPS).lower()
    assert "all the way back" in heads and "arm's length" in heads
