"""Backfilled condition sidecars (L1 stage 5): they can warn, never fail, and recordings are never edited."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gaze_cursor import backfill
from gaze_cursor.conditions import conditions_block, read_meta, validate, verdict


def make_session(path, with_points=True) -> None:
    meta = {"kind": "session", "image_size": [1280, 720],
            "screen": {"width_px": 1470, "height_px": 956, "width_cm": 29.0, "height_cm": 18.9, "distance_cm": 60.0},
            "lighting": {"face": 120.0, "background": 100.0, "ratio": 1.2, "label": "front-lit"}}
    arrays = {"meta": json.dumps(meta)}
    if with_points:
        pts = np.zeros((478, 3), np.float32)
        for ring, cx in (((469, 470, 471, 472), 0.35), ((474, 475, 476, 477), 0.65)):
            for k, (dx, dy) in zip(ring, ((-0.01, 0), (0, -0.01), (0.01, 0), (0, 0.01))):   # iris 0.02 wide
                pts[k] = (cx + dx, 0.5 + dy, 0)
        arrays["points"] = np.stack([pts] * 3)
    np.savez_compressed(path, **arrays)


def test_backfilled_block_warns_but_never_fails_and_leaves_unrecorded_tags_absent(tmp_path) -> None:
    npz = tmp_path / "session_1.npz"
    make_session(npz, with_points=False)
    block = backfill.backfill_block(read_meta(npz), npz, person_id="p01", camera_name="MacBook Air Camera",
                                    glasses="no")
    assert block["backfilled"] is True and block["lighting_tag"] is None and block["git_hash"] is None
    issues = validate(block)
    assert verdict(issues) == "warn" and {i.field for i in issues} >= {"lighting_tag", "git_hash", "posture_note"}
    assert block["glasses"] == "no" and block["lighting_measured"]["label"] == "front-lit"


def test_sidecar_is_written_once_next_to_an_untouched_recording(tmp_path) -> None:
    npz = tmp_path / "session_1.npz"
    make_session(npz, with_points=False)
    before = npz.read_bytes()
    block = backfill.backfill_block(read_meta(npz), npz, person_id="p01", camera_name="cam")
    path = backfill.write_sidecar(npz, block)
    assert path.name == "session_1.conditions.json" and npz.read_bytes() == before
    assert conditions_block(read_meta(npz), npz)["person_id"] == "p01"
    with pytest.raises(FileExistsError):
        backfill.write_sidecar(npz, block)


def test_recordings_skip_face_sidecars_and_temp_saves(tmp_path) -> None:
    for name in ("session_1.npz", "session_1.faces.npz", ".cursor_2.tmp.npz", "cursor_2.npz", "other.npz"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in backfill.recordings(tmp_path)] == ["cursor_2.npz", "session_1.npz"]


def test_iris_size_is_read_from_stored_landmarks_and_a_cursor_log_borrows_its_calibration(tmp_path) -> None:
    calib = tmp_path / "session_1.npz"
    make_session(calib)
    iris = backfill.median_iris_px(calib)
    assert iris > 0 and not np.isnan(iris)
    cursor = tmp_path / "cursor_2.npz"
    np.savez_compressed(cursor, meta=json.dumps({"kind": "cursor_session", "calibration": "session_1.npz",
                                                  "screen": {"width_px": 1470, "height_px": 956}}))
    block = backfill.backfill_block(read_meta(cursor), cursor, person_id="p01", camera_name="cam")
    assert block["distance"]["iris_px"] == pytest.approx(iris)
    assert block["distance"]["measured_in"] == "session_1.npz" and block["lighting_measured_in"] == "session_1.npz"
    assert block["camera"]["width"] == 1280                      # the calibration's frame size, not an assumption
    assert "image_size_assumed" not in block["distance"]


def test_backfill_claims_no_screen_layout_and_marks_an_assumed_image_size(tmp_path) -> None:
    npz = tmp_path / "session_3.npz"
    np.savez_compressed(npz, meta=json.dumps({"kind": "session", "screen": {"width_px": 1470, "height_px": 956}}))
    block = backfill.backfill_block(read_meta(npz), npz, person_id="p01", camera_name="cam")
    assert block["screen_layout"]["displays"] == []              # arrangement was not recorded
    assert block["distance"]["image_size_assumed"] == [1280, 720]
    assert any(i.field == "screen_layout" for i in validate(block))
