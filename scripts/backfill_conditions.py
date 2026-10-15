"""Write condition-tag sidecars for recordings made before the tags existed (L1 stage 5).

    uv run scripts/backfill_conditions.py --person p01 --dry-run
    uv run scripts/backfill_conditions.py --person p01 --default-glasses no

For every ``data/sessions|quick/session_*.npz`` and ``cursor_*.npz`` without a ``*.conditions.json`` it writes
one next to the recording (the recording itself is never edited; existing sidecars are never overwritten).
Only what is known is written; lighting tag, posture note and code version stay absent, so those sessions keep
warning on check 6 and none can fail. Run it yourself: only you know the person id and the glasses answer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gaze_cursor import backfill
from gaze_cursor.conditions import PERSON_ID_PATTERN, N_CHECKED, read_meta, sidecar_path, tags_present, validate, verdict

ROOT = Path(__file__).resolve().parents[1]
DIRS = (ROOT / "data" / "sessions", ROOT / "data" / "quick")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True, help="person id of the author's recordings, e.g. p01")
    ap.add_argument("--default-glasses", choices=("yes", "no"), default=None,
                    help="apply this glasses answer to every backfilled recording (only if it is true for all)")
    ap.add_argument("--camera", required=True,
                    help='camera name you used for every recording, e.g. "MacBook Air Camera" (stated, not assumed)')
    ap.add_argument("--dry-run", action="store_true", help="show what would be written, write nothing")
    ap.add_argument("--root", type=Path, nargs="*", default=None, help="directories to scan (default: data/sessions, data/quick)")
    args = ap.parse_args()
    if not PERSON_ID_PATTERN.match(args.person):
        raise SystemExit("person id must look like p01, p02, ...")
    todo = written = 0
    for directory in args.root or DIRS:
        for npz in backfill.recordings(directory):
            if sidecar_path(npz).exists():
                continue
            todo += 1
            try:
                meta = read_meta(npz)
            except (OSError, ValueError, KeyError) as exc:
                print(f"skip {npz.name}: unreadable ({exc})")
                continue
            if meta.get("conditions"):
                continue                      # already tagged by its recorder
            block = backfill.backfill_block(meta, npz, person_id=args.person, camera_name=args.camera,
                                            glasses=args.default_glasses or "unknown")
            issues = validate({**block, "backfilled": True})
            note = f"{verdict(issues)} ({tags_present(issues)}/{N_CHECKED} fields)"
            if args.dry_run:
                print(f"would write {sidecar_path(npz).name}: {note}")
            else:
                backfill.write_sidecar(npz, block)
                written += 1
                print(f"wrote {sidecar_path(npz).name}: {note}")
    print(f"{todo} recording(s) without tags; {written} sidecar(s) written" + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
