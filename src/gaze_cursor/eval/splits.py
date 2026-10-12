"""Train/test splits for the three evaluation modes. Pure functions.

The unit of data is a target (fixation), never a frame: every split keeps all frames of a target
on one side, so frames from the same fixation can never leak between train and test.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from gaze_cursor.eval.session import CALIB, TEST, Session

MODES = ("within", "loto", "losos")
_STRIDE = 10_000  # target key = session index * stride + target_idx (unique across sessions)


@dataclass(frozen=True)
class Dataset:
    """All valid frames of several sessions, for one feature set. Rows align across feature sets."""

    x: np.ndarray            # (n, d) model input
    y: np.ndarray            # (n, 2) target positions (screen points)
    session: np.ndarray      # (n,) session index into ``names``
    target: np.ndarray       # (n,) unique target key
    phase: np.ndarray        # (n,) CALIB or TEST
    names: tuple[str, ...]   # session file names

    @property
    def n_sessions(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class Split:
    train: np.ndarray        # row indices
    test: np.ndarray
    session: int             # the session being evaluated
    label: str


def build_dataset(sessions: Sequence[Session], feature_set: str) -> Dataset:
    """Stack the valid frames of ``sessions`` for ``feature_set``."""
    xs, ys, ss, ts, ps = [], [], [], [], []
    for i, s in enumerate(sessions):
        x, y, t, ph = s.select_all(feature_set)
        xs.append(x); ys.append(y); ts.append(i * _STRIDE + t.astype(np.int64))
        ss.append(np.full(len(x), i)); ps.append(ph)
    return Dataset(np.concatenate(xs), np.concatenate(ys), np.concatenate(ss),
                   np.concatenate(ts), np.concatenate(ps), tuple(s.path.name for s in sessions))


def within_session(ds: Dataset) -> Iterator[Split]:
    """Per session: fit on its calibration targets, test on its held-out targets."""
    for s in range(ds.n_sessions):
        in_s = ds.session == s
        train = np.flatnonzero(in_s & (ds.phase == CALIB))
        test = np.flatnonzero(in_s & (ds.phase == TEST))
        if len(train) and len(test):
            yield Split(train, test, s, f"within:{ds.names[s]}")


def leave_one_target_out(ds: Dataset) -> Iterator[Split]:
    """Per session and target: fit on the session's other targets, test on that target."""
    for s in range(ds.n_sessions):
        in_s = ds.session == s
        for t in np.unique(ds.target[in_s]):
            test = np.flatnonzero(in_s & (ds.target == t))
            train = np.flatnonzero(in_s & (ds.target != t))
            if len(train) and len(test):
                yield Split(train, test, s, f"loto:{ds.names[s]}:{int(t) % _STRIDE}")


def leave_one_session_out(ds: Dataset, held_out: Sequence[int] | None = None) -> Iterator[Split]:
    """Per held-out session: pool every target of all OTHER sessions to train, test on all
    targets of the held-out session."""
    for s in (range(ds.n_sessions) if held_out is None else held_out):
        train = np.flatnonzero(ds.session != s)
        test = np.flatnonzero(ds.session == s)
        if len(train) and len(test):
            yield Split(train, test, s, f"losos:{ds.names[s]}")


def make_splits(ds: Dataset, mode: str, held_out: Sequence[int] | None = None) -> list[Split]:
    if mode == "within":
        return list(within_session(ds))
    if mode == "loto":
        return list(leave_one_target_out(ds))
    if mode == "losos":
        return list(leave_one_session_out(ds, held_out))
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
