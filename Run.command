#!/bin/bash
# Double-click to run the live gaze dot. It starts with a 9-dot quick calibration where you sit
# (~15 s; the round-3 winner), R recalibrates at any time, L on the start screen uses the latest
# full calibration instead (with the posture check against it).
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Install it from https://docs.astral.sh/uv/ and try again."
  read -r -p "Press Enter to close" _; exit 1
fi
echo "gaze-cursor: live gaze dot (quick 9-dot calibration first; R recalibrates, q or ESC quits)"
uv run scripts/gaze_overlay.py "$@"
status=$?
echo
if [ $status -ne 0 ]; then echo "Stopped with an error (exit code $status)."; fi
read -r -p "Press Enter to close" _
exit $status
