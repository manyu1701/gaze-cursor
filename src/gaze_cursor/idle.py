"""Power saving when nobody is in front of the camera (pure logic, no OS or camera calls).

The expensive part of the live loop is the face landmarker, not the camera read. So the policy is:

* face seen: run every frame (about 30 fps);
* no face for ``SLEEP_AFTER_S`` (30 s): *sleep*, i.e. still read the camera but only run the landmarker on
  every ``SLEEP_EVERY``-th frame (about 3 fps), which is enough to notice you coming back within ~0.3 s;
* no face for ``QUIT_AFTER_S`` (5 min): quit cleanly (the camera light goes off and the session is saved;
  the last calibration is reused for 12 h, so starting again is instant). For the last ``WARN_BEFORE_S``
  (1 min) ``message`` counts down, so someone who is there but not detected (bad light) is warned first.

Any detected face wakes it up immediately and resets both timers; so does ``busy`` (a running live test
must never be starved or cut off).
"""

from __future__ import annotations

SLEEP_AFTER_S = 30.0
QUIT_AFTER_S = 300.0
WARN_BEFORE_S = 60.0        # count down this long before quitting
SLEEP_EVERY = 10            # while asleep, run the landmarker on 1 frame in 10


class IdleTimer:
    """Tracks how long no face has been seen. Call ``process_this_frame`` before the landmarker and
    ``update`` after it."""

    def __init__(self, now_s: float, sleep_after_s: float = SLEEP_AFTER_S, quit_after_s: float = QUIT_AFTER_S,
                 sleep_every: int = SLEEP_EVERY, warn_before_s: float = WARN_BEFORE_S) -> None:
        self.sleep_after_s, self.quit_after_s, self.sleep_every = sleep_after_s, quit_after_s, sleep_every
        self.warn_before_s = warn_before_s
        self.last_face_s = now_s
        self._skipped = 0

    def update(self, now_s: float, face_seen: bool, busy: bool = False) -> None:
        """``busy``: something is running that needs every frame (a live test)."""
        if face_seen or busy:
            self.last_face_s = now_s

    def idle_s(self, now_s: float) -> float:
        return now_s - self.last_face_s

    def sleeping(self, now_s: float) -> bool:
        return self.idle_s(now_s) >= self.sleep_after_s

    def should_quit(self, now_s: float) -> bool:
        return self.idle_s(now_s) >= self.quit_after_s

    def process_this_frame(self, now_s: float) -> bool:
        """True if the landmarker should run on this frame (always when awake, 1 in ``sleep_every`` asleep)."""
        if not self.sleeping(now_s):
            self._skipped = 0
            return True
        self._skipped += 1
        if self._skipped >= self.sleep_every:
            self._skipped = 0
            return True
        return False

    def message(self, now_s: float) -> str | None:
        """A status line while asleep: constant until the last minute, then a countdown to quitting."""
        if not self.sleeping(now_s):
            return None
        left = self.quit_after_s - self.idle_s(now_s)
        if left <= self.warn_before_s:
            return f"no face: closing in {max(0, int(left + 0.999))} s to save battery"
        return "no face: low power until you are back"
