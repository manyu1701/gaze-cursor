#!/bin/bash
# Double-click to show a friend: they sit in front of the laptop, do the 9-dot calibration (~15 s), then
# tap Right Command and the cursor jumps to where THEY look (a faint ring shows what the tracker thinks).
# NOTHING is saved: no calibration recording, no session log, no results row, and your own saved
# calibration is left alone. Cmd+Option+R recalibrates (e.g. for the next friend), Cmd+Option+X quits and
# closes this window. Expect ~2-5 degrees of accuracy (a few cm), best with a lamp in front of the face.
# First time: macOS needs Input Monitoring for Terminal (the script explains where).
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Install it from https://docs.astral.sh/uv/ and try again."
  read -r -p "Press Enter to close" _; exit 1
fi
echo "gaze-cursor DEMO (nothing is saved): calibrate, then tap Right Command to jump; Cmd+Option+X quits"
uv run scripts/cursor_mode.py --demo "$@"
status=$?
echo
if [ $status -ne 0 ]; then
  echo "Stopped (exit code $status)."
  read -r -p "Press Enter to close" _
  exit $status
fi
tty_name="$(tty)"
(sleep 1; osascript -e "tell application \"Terminal\" to close (every window whose tty of tabs contains \"$tty_name\")" >/dev/null 2>&1) &
exit 0
