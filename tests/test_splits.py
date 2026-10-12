import numpy as np
import pytest

from gaze_cursor.eval.runner import build_datasets, evaluate
from gaze_cursor.eval.configs import Config
from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.session import CALIB, TEST
from gaze_cursor.eval.splits import build_dataset, make_splits
from tests.helpers import fake_session

SESSIONS = [fake_session(i) for i in range(3)]


def _ds(fs: str = "gaze"):
    return build_dataset(SESSIONS, fs)


def _targets(ds, idx):
    return set(ds.target[idx].tolist())


def test_within_trains_on_calibration_only_and_never_shares_targets() -> None:
    ds = _ds()
    splits = make_splits(ds, "within")
    assert len(splits) == 3
    for sp in splits:
        assert set(ds.phase[sp.train]) == {CALIB} and set(ds.phase[sp.test]) == {TEST}
        assert set(ds.session[sp.train]) == set(ds.session[sp.test]) == {sp.session}
        assert not _targets(ds, sp.train) & _targets(ds, sp.test)


def test_loto_holds_out_exactly_one_target_and_keeps_its_frames_together() -> None:
    ds = _ds()
    splits = make_splits(ds, "loto")
    assert len(splits) == 3 * 9
    for sp in splits:
        assert len(_targets(ds, sp.test)) == 1
        assert not _targets(ds, sp.train) & _targets(ds, sp.test)
        assert set(ds.session[sp.train]) == {sp.session}


def test_losos_never_trains_on_the_test_session_and_tests_all_its_targets() -> None:
    ds = _ds()
    for sp in make_splits(ds, "losos"):
        assert sp.session not in set(ds.session[sp.train])
        assert set(ds.session[sp.test]) == {sp.session}
        assert len(_targets(ds, sp.test)) == 9
    only = make_splits(ds, "losos", held_out=[2])
    assert [s.session for s in only] == [2]


def test_unknown_mode() -> None:
    with pytest.raises(ValueError):
        make_splits(_ds(), "random-frames")


def test_feature_sets_align_and_gaze_blend_has_18_columns() -> None:
    a, b = _ds("gaze"), _ds("gaze_blend")
    assert a.x.shape[1] == 4 and b.x.shape[1] == 18
    assert np.array_equal(a.target, b.target) and np.array_equal(a.session, b.session)


def test_harness_recovers_a_noiseless_linear_map_in_every_mode() -> None:
    screen = ScreenConfig()
    cfg = Config("g", "gaze", "ridge", 1e-6)
    ds = build_datasets(SESSIONS, [cfg])
    for mode in ("within", "loto", "losos"):
        stats, _ = evaluate(ds, cfg, mode, screen)
        assert stats.err_px.max() < 1.0, mode
