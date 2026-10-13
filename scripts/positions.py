"""List recent sessions with their lighting, measured head position and what they count as.

Position is measured from the stored landmarks (median over the session) relative to the centre
session of the set: set A's named C, or set B's C once the round-3 candidate is frozen. Prints
numbers only; never opens images.

  uv run scripts/positions.py          # the last 12 sessions
  uv run scripts/positions.py --all
"""

from __future__ import annotations

import argparse

from gaze_cursor.eval import prereg3
from gaze_cursor.eval.prereg2 import lighting_class
from gaze_cursor.eval.prereg3 import SET_A, describe, select_set_b, session_position, set_a, slot_from_note
from gaze_cursor.eval.session import load_session, session_files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="every session, not just the last 12")
    args = ap.parse_args()
    sessions = [load_session(p) for p in session_files()]
    if not sessions:
        print("No sessions yet. Double-click Calibrate.command first.")
        return 1
    clean = [s for s in sessions if s.is_clean]
    a = set_a(clean)
    b = select_set_b(clean, prereg3.PREREG3_FREEZE_AFTER)
    role = {s.path.name: f"set A {k}" for k, s in a.sessions.items()}
    role |= {s.path.name: f"set B {k}" for k, s in b.sessions.items()}
    role |= {name: f"excluded: {why}" for name, why in b.excluded}
    named_a = {v: k for k, v in SET_A.items()}
    ref = b.positions.get("C") or a.positions.get("C")
    print(f"position relative to {'set B' if 'C' in b.positions else 'set A'} C"
          + ("" if ref else " (not available on this machine)"))
    for s in sessions if args.all else sessions[-12:]:
        li = s.meta.get("lighting")
        light = f"{li['label']} face {li['face']:.0f}" if li else "lighting not captured"
        try:
            pos = describe(session_position(s), ref) if ref else ""
        except ValueError:
            pos = "no landmarks"
        slot = slot_from_note(s.meta.get("note"))
        status = role.get(s.path.name) or (f"set A {named_a[s.path.name]}" if s.path.name in named_a else "")
        if not status and slot:
            status = "labelled " + slot + ("" if prereg3.PREREG3_FREEZE_AFTER else " (set B opens after the freeze)")
        print(f"{s.path.name.removeprefix('session_').removesuffix('.npz')} | note {s.meta.get('note') or '-'!r:18s} | "
              f"{'clean' if s.is_clean else 'NOT clean'} | valid {s.valid_fraction:4.0%} | {light} "
              f"(class {lighting_class(s.meta) or '-'}) | {pos} | {status}")
    if b.missing and prereg3.PREREG3_FREEZE_AFTER:
        print("set B still needs: " + ", ".join(m.removeprefix("set B ") for m in b.missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
