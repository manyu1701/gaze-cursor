#!/bin/bash
# Double-click for target practice: boxes appear, move the gaze dot into each one with your eyes.
# Uses your latest calibration; the summary says how small a box you can reliably hit.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Install it from https://docs.astral.sh/uv/ and try again."
  read -r -p "Press Enter to close" _; exit 1
fi
if ! ls data/sessions/session_*.npz 2>/dev/null | grep -qv '\.faces\.npz$'; then
  echo "No calibration found yet. Double-click Calibrate.command first."
  read -r -p "Press Enter to close" _; exit 1
fi
echo "gaze-cursor: target practice (q or ESC quits)"
uv run scripts/target_practice.py "$@"
status=$?
echo
if [ $status -ne 0 ]; then echo "Stopped with an error (exit code $status)."; fi
read -r -p "Press Enter to close" _
exit $status
