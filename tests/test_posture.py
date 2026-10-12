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


def _m(**kw: float) -> dict[str, float]:
    return {**{"pitch": -7.0, "yaw": 0.0, "roll": 0.0, "distance": 0.066, "x": 0.5, "y": 0.55}, **kw}


STILL = [_m() for _ in range(45)]
LIGHT_OK = {"face": 90.0, "background": 140.0, "ratio": 0.64, "clipped": 0.0}


def _levels(items) -> dict[str, str]:
    return {i.key: i.level for i in items}


def test_checklist_all_good_is_ready() -> None:
    from gaze_cursor.posture import build_checklist, checklist_ready

    items = build_checklist(_ref(), _m(), STILL, LIGHT_OK)
    assert set(_levels(items).values()) == {"good"} and checklist_ready(items)


def test_checklist_reacts_when_the_user_corrects_a_step() -> None:
    from gaze_cursor.posture import build_checklist, checklist_ready

    ref = _ref()
    far = build_checklist(ref, _m(distance=0.050), STILL, LIGHT_OK)
    assert _levels(far)["distance"] != "good" and "move closer" in far[0].hint and not checklist_ready(far)
    fixed = build_checklist(ref, _m(distance=0.066), STILL, LIGHT_OK)       # after moving closer
    assert _levels(fixed)["distance"] == "good" and checklist_ready(fixed)
    tilted = build_checklist(ref, _m(pitch=0.0), STILL, LIGHT_OK)
    assert _levels(tilted)["pitch"] != "good" and "screen angle" in tilted[1].hint
    turned = build_checklist(ref, _m(yaw=14.0), STILL, LIGHT_OK)
    assert _levels(turned)["facing"] != "good" and not checklist_ready(turned)


def test_light_step_flags_only_extremes() -> None:
    from gaze_cursor.posture import build_checklist

    dim = {"face": 20.0, "background": 60.0, "ratio": 0.3, "clipped": 0.0}
    assert _levels(build_checklist(_ref(), _m(), STILL, dim))["light"] == "off"
    backlit = {"face": 60.0, "background": 200.0, "ratio": 0.3, "clipped": 0.0}
    assert _levels(build_checklist(_ref(), _m(), STILL, backlit))["light"] == "good"   # never gates on ratio


def test_stillness_waits_then_flags_movement() -> None:
    from gaze_cursor.posture import build_checklist, checklist_ready

    assert _levels(build_checklist(_ref(), _m(), STILL[:5], LIGHT_OK))["still"] == "wait"
    assert not checklist_ready(build_checklist(_ref(), _m(), STILL[:5], LIGHT_OK))   # measuring blocks
    moving = [_m(pitch=-7.0 + (3.0 if i % 2 else -3.0)) for i in range(45)]
    items = build_checklist(_ref(), _m(), moving, LIGHT_OK)
    assert _levels(items)["still"] != "good" and not checklist_ready(items)


def test_no_reference_never_blocks_but_light_and_stillness_still_do() -> None:
    from gaze_cursor.posture import build_checklist, checklist_ready

    items = build_checklist(None, _m(), STILL, LIGHT_OK)
    assert [i.level for i in items[:3]] == ["unknown"] * 3 and checklist_ready(items)
    dark = {"face": 10.0, "background": 20.0, "ratio": 0.5, "clipped": 0.0}
    assert not checklist_ready(build_checklist(None, _m(), STILL, dark))


def test_no_face_is_not_ready() -> None:
    from gaze_cursor.posture import build_checklist, checklist_ready

    items = build_checklist(_ref(), None, [], None)
    assert not checklist_ready(items) and _levels(items)["distance"] == "off"


def test_current_setup_becomes_its_own_reference() -> None:
    from gaze_cursor.posture import build_checklist, checklist_ready

    mine = [_m(pitch=-2.0, yaw=9.0, distance=0.05) for _ in range(45)]      # not like any earlier session
    own = Reference.from_current(mine)
    assert own.center["pitch"] == pytest.approx(-2.0) and own.n_sessions == 0
    assert checklist_ready(build_checklist(own, mine[0], mine, LIGHT_OK))   # my setup is "right" for me
    assert not checklist_ready(build_checklist(_ref(), mine[0], mine, LIGHT_OK))
    with pytest.raises(ValueError):
        Reference.from_current(mine[:3])


def test_checklist_and_summary_screens_keep_the_notch_strip_clear() -> None:
    from gaze_cursor.config import ScreenConfig
    from gaze_cursor.posture import build_checklist, summarize_recording
    from gaze_cursor.sinks.guide_view import draw_summary
    from gaze_cursor.sinks.overlay import FullscreenCanvas
    from gaze_cursor.sinks.posture_view import draw_checklist

    cfg = ScreenConfig()
    canvas = FullscreenCanvas(cfg)
    strip = int(cfg.top_inset_pt) + 2
    blank = canvas.blank()[:strip]
    frame = np.full((720, 1280, 3), 120, np.uint8)
    bad = build_checklist(_ref(), _m(pitch=0.0, distance=0.05), STILL, LIGHT_OK)
    s = summarize_recording(_ref(), _frames(10, pitch=0.0))
    for img in (draw_checklist(canvas, frame, bad, False, True, "Set up your position"),
                draw_checklist(canvas, None, build_checklist(_ref(), None, [], None), False, False, "t"),
                draw_summary(canvas, s, "session_x.npz"),
                draw_summary(canvas, summarize_recording(_ref(), []), "session_x.npz")):
        assert (img[:strip] == blank).all()


def test_lighting_labels_for_the_sessions_measured_so_far() -> None:
    from gaze_cursor.posture import lighting_label

    assert lighting_label(78, 146) == "backlit"      # v2 session 1
    assert lighting_label(37, 75) == "dim"           # dim, screen only
    assert lighting_label(130, 120) == "front-lit"
    assert lighting_label(90, 100) == "mixed"


def test_session_lighting_summary_is_storable() -> None:
    import json
    from gaze_cursor.posture import summarize_lighting

    frames = [{"face": 36.0 + i % 3, "background": 75.0, "ratio": 0.5, "clipped": 0.0} for i in range(30)]
    s = summarize_lighting(frames)
    assert s["label"] == "dim" and s["frames"] == 30 and s["face"] == pytest.approx(37.0, abs=0.1)
    json.dumps(s)
    assert summarize_lighting([]) is None


def test_checklist_flags_lighting_that_differs_from_calibration() -> None:
    from gaze_cursor.posture import build_checklist

    calib = {"face": 37.0, "background": 75.0, "label": "dim"}
    brighter = {"face": 110.0, "background": 140.0, "ratio": 0.8, "clipped": 0.0}
    item = [i for i in build_checklist(_ref(), _m(), STILL, brighter, calib) if i.key == "light"][0]
    assert item.level == "adjust" and "differs from calibration" in item.hint and "dim" in item.hint
    same = {"face": 40.0, "background": 78.0, "ratio": 0.5, "clipped": 0.0}
    assert [i for i in build_checklist(_ref(), _m(), STILL, same, calib) if i.key == "light"][0].level == "good"
    assert [i for i in build_checklist(_ref(), _m(), STILL, brighter, None) if i.key == "light"][0].level == "good"


def test_summary_screen_with_lighting_and_without_a_comparison_keeps_the_notch_clear() -> None:
    from gaze_cursor.config import ScreenConfig
    from gaze_cursor.sinks.guide_view import draw_summary
    from gaze_cursor.sinks.overlay import FullscreenCanvas

    cfg = ScreenConfig()
    canvas = FullscreenCanvas(cfg)
    strip = int(cfg.top_inset_pt) + 2
    img = draw_summary(canvas, None, "session_x.npz", {"label": "dim", "face": 37.0, "background": 75.0})
    assert (img[:strip] == canvas.blank()[:strip]).all() and (img != canvas.blank()).any()


# --- key handling of the live check (fake camera, tracker and canvas) -------------------------

class _FakeCam:
    def read(self):
        from gaze_cursor.capture import Frame
        return Frame(image=np.full((720, 1280, 3), 120, np.uint8), timestamp_ms=0)


class _FakeTracker:
    def __init__(self, yaw: float = 0.0) -> None:
        from gaze_cursor.landmarks import LandmarkResult
        self._res = LandmarkResult(points=_face(), transform=_rot_y(yaw), blendshapes=np.zeros(52, np.float32))

    def process(self, frame):
        return self._res


class _FakeCanvas:
    """Replays scripted key presses and records the footer shown on each frame."""

    def __init__(self, keys) -> None:
        from gaze_cursor.config import ScreenConfig
        from gaze_cursor.sinks.overlay import FullscreenCanvas
        self._real = FullscreenCanvas(ScreenConfig())
        self.size, self.safe_top = self._real.size, self._real.safe_top
        self.keys, self.calls = iter(keys), 0

    def blank(self):
        return self._real.blank()

    def text(self, img, msg, y=None):
        self._real.text(img, msg, y)

    def show(self, img):
        self.calls += 1
        return next(self.keys, 27)           # ESC when the script runs out, so a test can never hang


def _check_setup(monkeypatch, keys, mode="gate", tracker_yaw=0.0):
    import gaze_cursor.sinks.posture_view as pv
    clock = iter(np.arange(0, 1000, 0.1))    # each loop iteration advances the clock by 0.1 s
    monkeypatch.setattr(pv.time, "monotonic", lambda: next(clock))
    mine = posture_metrics(_face(), _rot_y(0.0), (1280, 720))
    ref = Reference.from_arrays({m: np.full(50, mine[m]) for m in METRICS})
    canvas = _FakeCanvas(keys)
    return pv.run_posture_check(_FakeCam(), _FakeTracker(tracker_yaw), canvas, ref, mode=mode), canvas


def test_space_is_ignored_until_every_step_is_green(monkeypatch) -> None:
    keys = [32] * 5 + [-1] * 60 + [32]               # early SPACE presses, later SPACE once ready
    res, canvas = _check_setup(monkeypatch, keys)
    assert res.proceed and res.in_position and not res.own_setup
    assert canvas.calls > 6                           # the early presses did not end the check


def test_space_never_starts_when_not_in_position(monkeypatch) -> None:
    # head turned 25 degrees: the facing step never turns green, so SPACE can never start
    res, canvas = _check_setup(monkeypatch, [-1] * 40 + [32] * 5 + [-1] * 30, tracker_yaw=25.0)
    assert not res.proceed                            # ended only because the script ran out (ESC)


def test_u_overrides_with_the_current_setup_as_reference(monkeypatch) -> None:
    res, _ = _check_setup(monkeypatch, [-1] * 30 + [ord("u")], tracker_yaw=25.0)   # not in position
    assert res.proceed and res.own_setup and not res.in_position
    assert res.reference.center["yaw"] == pytest.approx(25.0, abs=1e-6)


def test_u_is_ignored_before_enough_frames_are_measured(monkeypatch) -> None:
    res, _ = _check_setup(monkeypatch, [ord("u")] * 3 + [27])
    assert not res.proceed and not res.own_setup      # 3 frames is not enough; the ESC ended it


def test_escape_and_q_cancel(monkeypatch) -> None:
    assert not _check_setup(monkeypatch, [-1, 27])[0].proceed
    assert not _check_setup(monkeypatch, [-1, ord("q")])[0].proceed


def test_monitor_mode_space_exits_at_once(monkeypatch) -> None:
    res, canvas = _check_setup(monkeypatch, [-1, -1, 32], mode="monitor", tracker_yaw=25.0)
    assert res.proceed and canvas.calls == 3


def test_footer_text_matches_what_the_keys_do() -> None:
    from gaze_cursor.sinks.posture_view import default_footer

    note, keys = default_footer("gate", False, "use my current setup")
    assert "SPACE unlocks" in note and "SPACE: start" not in keys and "U:" in keys
    note, keys = default_footer("gate", True, "use my current setup")
    assert "SPACE: start" in keys and "continue anyway" not in keys
    assert default_footer("monitor", False, "x") == ("", "SPACE or ESC: exit")
