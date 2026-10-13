"""Which recorded calibrations to pool for the live model. Pure functions.

Round 2 (PREREGISTRATIONS.md) showed that a calibration from other lighting costs ~2.4 deg, so the pool is
restricted to sessions recorded in the same captured lighting label as the anchor (the latest
calibration). Round 3 set A suggested that pooling positions helps; the default switches only if
the pre-registered round-3 test (H5) supports it.
"""

from __future__ import annotations

from collections.abc import Sequence

from gaze_cursor.eval.session import Session

POOL_CONFIG = "gaze_a100"   # the frozen round-3 candidate
MIN_VALID = 0.8


def lighting_label(s: Session) -> str | None:
    li = s.meta.get("lighting")
    return li.get("label") if li else None


def pool_sessions(sessions: Sequence[Session], anchor: Session) -> list[Session]:
    """Clean v2 sessions with >= 80% valid frames and the anchor's lighting label, oldest first.

    The anchor is always included. Sessions without captured lighting are left out (their light
    is unknown), unless the anchor itself has none, in which case only the anchor is used."""
    label = lighting_label(anchor)
    if label is None:
        return [anchor]
    picked = [s for s in sessions
              if s is not anchor and s.is_clean and s.layout == "v2" and s.valid_fraction >= MIN_VALID
              and lighting_label(s) == label]
    return sorted([*picked, anchor], key=lambda s: str(s.meta.get("started", s.path.name)))
