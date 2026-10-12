#!/bin/bash
# Double-click to run the live gaze dot with your latest calibration. It first checks that you are
# sitting and lit as you were when you calibrated (SPACE starts when every step is green; U runs anyway).
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
echo "gaze-cursor: live gaze dot (q or ESC quits)"
uv run scripts/gaze_overlay.py "$@"
status=$?
echo
if [ $status -ne 0 ]; then echo "Stopped with an error (exit code $status)."; fi
read -r -p "Press Enter to close" _
exit $status
