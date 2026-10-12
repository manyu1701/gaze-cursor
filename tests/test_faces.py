import os
import stat
from pathlib import Path

import numpy as np
import pytest

from gaze_cursor.eval.faces import (
    FaceCrops, crop_box, encode_crop, estimate_mb, faces_path_for, save_faces)
from gaze_cursor.eval.session import SessionRecorder, load_session, session_files

SIZE = (1280, 720)


def _points(cx: float, cy: float, half: float) -> np.ndarray:
    """478 landmarks spread over a square of side 2*half pixels centred at (cx, cy) (frame pixels)."""
    rng = np.random.default_rng(0)
    pts = np.zeros((478, 3), np.float32)
    xy = rng.uniform(-half, half, (478, 2))
    xy[0], xy[1] = (-half, -half), (half, half)             # make the bbox exactly the square
    pts[:, 0] = (cx + xy[:, 0]) / SIZE[0]
    pts[:, 1] = (cy + xy[:, 1]) / SIZE[1]
    return pts


def test_crop_box_is_a_centred_square_with_margin() -> None:
    x0, y0, x1, y1 = crop_box(_points(640, 360, 100), SIZE)          # face bbox 200 px -> crop 300 px
    assert (x1 - x0, y1 - y0) == (300, 300)
    assert (x0 + x1) / 2 == pytest.approx(640, abs=1) and (y0 + y1) / 2 == pytest.approx(360, abs=1)


def test_crop_box_is_clamped_inside_the_frame() -> None:
    for cx, cy in ((30, 30), (1250, 700), (640, 10)):
        x0, y0, x1, y1 = crop_box(_points(cx, cy, 100), SIZE)
        assert 0 <= x0 < x1 <= SIZE[0] and 0 <= y0 < y1 <= SIZE[1] and (x1 - x0) == (y1 - y0) == 300
    huge = crop_box(_points(640, 360, 600), SIZE)                    # bigger than the frame: shrinks
    assert (huge[2] - huge[0]) <= SIZE[1]


def test_jpeg_round_trip_keeps_size_and_content() -> None:
    img = np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    img[100:400, 200:500] = (40, 120, 200)                             # a flat patch survives JPEG well
    box = (200, 100, 500, 400)
    jpeg = encode_crop(img, box, 90)
    path_img = FaceCrops(np.array([0, len(jpeg)]), np.frombuffer(jpeg, np.uint8), np.array([box], np.int32), 90).image(0)
    assert path_img.shape == (300, 300, 3)                            # native resolution, no resizing
    assert np.abs(path_img[50:250, 50:250].astype(int) - (40, 120, 200)).max() <= 6


def test_sidecar_alignment_empty_rows_and_permissions(tmp_path: Path) -> None:
    a = np.full((720, 1280, 3), 90, np.uint8)
    boxes = [(0, 0, 200, 200), None, (100, 100, 400, 400)]
    jpegs = [encode_crop(a, boxes[0]), None, encode_crop(a, boxes[2])]
    p = save_faces(tmp_path / "s.faces.npz", jpegs, boxes)
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600                    # face data: owner only
    fc = FaceCrops.load(p)
    assert len(fc) == 3 and [fc.has(i) for i in range(3)] == [True, False, True]
    assert fc.image(1) is None and fc.image(0).shape == (200, 200, 3) and fc.image(2).shape == (300, 300, 3)
    assert fc.box[1].tolist() == [-1, -1, -1, -1] and fc.box[2].tolist() == [100, 100, 400, 400]


def test_estimate_matches_the_measured_sizes() -> None:
    lo, hi = estimate_mb(750, side=316, quality=90)
    assert 7 < lo < hi < 16                                          # about 8-13 MB for a v2 session
    assert estimate_mb(1500)[0] == pytest.approx(2 * estimate_mb(750)[0])


def test_recorder_writes_a_row_aligned_sidecar_and_session_loading_ignores_it(tmp_path: Path) -> None:
    img = np.full((720, 1280, 3), 100, np.uint8)
    box = (0, 0, 120, 120)
    rec = SessionRecorder(meta={"screen": {}})
    rec.add(1, 0, 0, (1.0, 2.0), None, None, None)                                # no face, no crop
    rec.add(2, 0, 0, (1.0, 2.0), None, None, None, None, (encode_crop(img, box), box))
    path = rec.save(tmp_path)
    side = faces_path_for(path)
    assert side.name == path.stem + ".faces.npz" and side.exists()
    fc = FaceCrops.load(side)
    assert len(fc) == len(load_session(path, recompute_features=False).phase) == 2
    assert [fc.has(0), fc.has(1)] == [False, True]
    assert session_files(tmp_path) == [path]                                     # the sidecar is not a session


def test_no_sidecar_without_crops(tmp_path: Path) -> None:
    rec = SessionRecorder(meta={"screen": {}})
    rec.add(1, 0, 0, (1.0, 2.0), None, None, None)
    path = rec.save(tmp_path)
    assert not faces_path_for(path).exists()
