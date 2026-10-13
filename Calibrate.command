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
echo "Set B: start the note with the slot: pos C, pos N, pos R, pos L, pos Rt, pos Lo, C2"
while true; do
  read -r -p "Note about this session (e.g. 'pos N' or 'evening, lamp'): " NOTE
  [ -n "${NOTE// /}" ] && break
  read -r -p "No note: this session will NOT count for set B. Record without a note? [y/N] " ok
  [ "$ok" = "y" ] || [ "$ok" = "Y" ] && break
done
uv run scripts/calibrate.py --guided --note "$NOTE" "$@"
status=$?
echo
if [ $status -ne 0 ]; then echo "Calibration stopped with an error (exit code $status)."; fi
read -r -p "Press Enter to close" _
exit $status
