#!/bin/bash
# Double-click to calibrate. Steps: (1) set up your lighting on a dark calibration-like screen with
# live guidance (SPACE continues), (2) the live setup checklist, (3) the 25-dot calibration.
# Lighting is captured automatically and saved with the session.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Install it from https://docs.astral.sh/uv/ and try again."
  read -r -p "Press Enter to close" _; exit 1
fi
echo "gaze-cursor: calibration"
read -r -p "Optional note about this session (e.g. 'evening, lamp'; Enter to skip): " NOTE
uv run scripts/calibrate.py --guided --note "$NOTE" "$@"
status=$?
echo
if [ $status -ne 0 ]; then echo "Calibration stopped with an error (exit code $status)."; fi
read -r -p "Press Enter to close" _
exit $status
