import json
import math

import numpy as np
import pytest

from gaze_cursor.conditions import (
    DEFAULT_FOV_DEG, FAIL, IRIS_DIAMETER_CM, LEFT_IRIS_RING, PASS, RIGHT_IRIS_RING, SCHEMA_VERSION, WARN,
    build_conditions, estimate_distance_cm, iris_diameter_px, model_version, resolve_person_id, save_person_id,
    validate, verdict,
)

SIZE = (1280, 720)


def _block(**over) -> dict:
    kwargs = dict(
        person_id="p01", camera={"name": "FaceTime HD Camera", "width": 1280, "height": 720, "built_in": True},
        screen_layout={"displays": [{"w_pt": 1470, "h_pt": 956, "w_cm": 30.0, "h_cm": 19.5, "main": True}]},
        assumed_distance_cm=60.0, iris_px=19.6, image_width=1280, lighting_tag="lamp", glasses="no",
        posture_note="upright", git_hash="abc1234",
    )
    kwargs.update(over)
    return build_conditions(**kwargs)


def _face(diameter_px: float) -> np.ndarray:
    """Synthetic landmarks with both iris rings drawn as circles of the given pixel diameter."""
    pts = np.zeros((478, 3), np.float32)
    r = np.array([diameter_px / 2 / SIZE[0], diameter_px / 2 / SIZE[1]])
    for ring, centre in ((RIGHT_IRIS_RING, (0.35, 0.5)), (LEFT_IRIS_RING, (0.65, 0.5))):
        for k, ang in zip(ring, (math.pi, math.pi / 2, 0.0, -math.pi / 2)):
            pts[k, :2] = np.array(centre) + r * np.array([math.cos(ang), math.sin(ang)])
    return pts


def test_iris_diameter_recovers_pixels_on_a_non_square_frame() -> None:
    assert iris_diameter_px(_face(20.0), SIZE) == pytest.approx(20.0, abs=1e-3)


def test_iris_diameter_without_the_ring_is_nan() -> None:
    assert math.isnan(iris_diameter_px(np.zeros((100, 3)), SIZE))
    assert math.isnan(iris_diameter_px(np.zeros((478, 3)), SIZE))


def test_distance_round_trip_and_monotonic() -> None:
    f_px = (SIZE[0] / 2) / math.tan(math.radians(DEFAULT_FOV_DEG) / 2)
    iris_px = f_px * IRIS_DIAMETER_CM / 60.0
    assert estimate_distance_cm(iris_px, SIZE[0]) == pytest.approx(60.0)
    assert estimate_distance_cm(iris_px / 2, SIZE[0]) == pytest.approx(120.0)   # half the size, twice as far
    assert math.isnan(estimate_distance_cm(0.0, SIZE[0]))
    assert math.isnan(estimate_distance_cm(float("nan"), SIZE[0]))
    assert math.isnan(estimate_distance_cm(20.0, SIZE[0], fov_deg=0.0))


def test_model_version_is_stable_and_sensitive() -> None:
    a = [np.arange(3.0), np.ones(3), np.zeros((4, 2)), np.eye(4)]
    assert model_version(*a) == model_version(*[x.copy() for x in a])
    b = [x.copy() for x in a]
    b[2][0, 0] += 1e-9
    assert model_version(*a) != model_version(*b)
    assert len(model_version(*a)) == 16


def test_model_version_depends_on_shape_not_just_bytes() -> None:
    flat, grid = np.zeros(6), np.zeros((2, 3))
    one = model_version(flat, flat, flat, flat)
    two = model_version(grid, grid, grid, grid)
    assert one != two


def test_complete_block_is_clean_and_json_serialisable() -> None:
    block = _block()
    assert block["schema_version"] == SCHEMA_VERSION and block["backfilled"] is False
    assert validate(block) == []
    assert verdict(validate(block)) == PASS
    json.dumps(block)   # plain JSON types only (NaN would still dump, but None must be used for missing)


def test_nan_measurements_are_stored_as_none() -> None:
    block = _block(iris_px=float("nan"))
    assert block["distance"]["iris_px"] is None and block["distance"]["est_cm"] is None
    assert [i.field for i in validate(block)] == ["distance"]


def test_missing_person_id_fails_a_new_session_but_only_warns_when_old_or_backfilled() -> None:
    new = _block(person_id="")
    issues = validate(new)
    assert [(i.field, i.severity) for i in issues] == [("person_id", FAIL)]
    assert verdict(issues) == FAIL
    assert verdict(validate({**new, "backfilled": True})) == WARN
    legacy = {k: v for k, v in new.items() if k != "schema_version"}
    assert verdict(validate(legacy)) == WARN


def test_other_missing_tags_only_warn() -> None:
    block = _block(glasses="unknown", lighting_tag="unknown", posture_note="", git_hash="unknown")
    issues = validate(block)
    assert {i.field for i in issues} == {"glasses", "lighting_tag", "posture_note", "git_hash"}
    assert all(i.severity == WARN for i in issues)
    assert verdict(issues) == WARN


def test_no_block_is_a_legacy_warning_never_a_fail() -> None:
    for missing in (None, {}):
        issues = validate(missing)
        assert verdict(issues) == WARN and issues[0].field == "conditions"


def test_unusual_person_id_and_newer_schema_warn() -> None:
    assert [(i.field, i.severity) for i in validate(_block(person_id="abhimanyu"))] == [("person_id", WARN)]
    newer = {**_block(), "schema_version": SCHEMA_VERSION + 1}
    assert [i.field for i in validate(newer)] == ["schema_version"]


def test_missing_camera_and_layout_are_named() -> None:
    block = _block(camera={"name": "unknown"}, screen_layout={})
    assert {i.field for i in validate(block)} == {"camera", "camera.resolution", "screen_layout"}


def test_person_id_resolution_order_and_persistence(tmp_path) -> None:
    path = tmp_path / "cfg" / "person.json"
    assert resolve_person_id(None, {}, path) is None
    save_person_id(" p07 ", path)
    assert resolve_person_id(None, {}, path) == "p07"
    assert resolve_person_id(None, {"GAZE_PERSON_ID": "p02"}, path) == "p02"        # env beats file
    assert resolve_person_id("p03", {"GAZE_PERSON_ID": "p02"}, path) == "p03"       # flag beats env
    assert resolve_person_id("  ", {"GAZE_PERSON_ID": ""}, path) == "p07"           # blanks are ignored
    path.write_text("not json")
    assert resolve_person_id(None, {}, path) is None


def test_legacy_detection_edge_cases_never_fail() -> None:
    nameless = {k: v for k, v in _block(person_id="").items()}
    assert verdict(validate(nameless)) == FAIL                                   # the control: a real v1 session
    for version in (0, -1, True, "1", 1.0, None):
        assert verdict(validate({**nameless, "schema_version": version})) == WARN, version
    for backfilled in (True, "false", "False", 0, None):
        assert verdict(validate({**nameless, "backfilled": backfilled})) == WARN, backfilled
    no_flag = {k: v for k, v in nameless.items() if k != "backfilled"}
    assert verdict(validate(no_flag)) == WARN                                    # the writers always set it


def test_whitespace_only_person_id_fails_a_new_session() -> None:
    assert verdict(validate(_block(person_id="   "))) == FAIL


def test_sidecar_reading(tmp_path) -> None:
    from gaze_cursor.conditions import conditions_block, sidecar_path
    npz = tmp_path / "session_20261013_133547.npz"
    assert conditions_block({}, npz) is None                                     # no sidecar
    sidecar_path(npz).write_text("{not json")
    assert conditions_block({}, npz) is None                                     # corrupt sidecar
    sidecar_path(npz).write_text(json.dumps({"schema_version": 1, "person_id": ""}))
    block = conditions_block({}, npz)
    assert block["backfilled"] is True and verdict(validate(block)) == WARN       # a sidecar can never fail
    own = _block()
    assert conditions_block({"conditions": own}, npz) == own                     # the recording's own block wins
    assert conditions_block({"conditions": {}}, npz)["backfilled"] is True       # an empty block falls through
    assert conditions_block({"conditions": own}) == own
    assert conditions_block({}, None) is None
