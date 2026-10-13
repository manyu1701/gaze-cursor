#!/bin/bash
# Double-click for Cursor mode: tap Right Command and the real cursor jumps to where you look; the
# trackpad finishes. Reuses your last calibration (under 12 h old) and learns from your clicks; the
# first time of the day it asks for a 9-dot calibration (~15 s). Hotkeys work in any app:
# Cmd+Option+G on/off, Cmd+Option+R recalibrate, Cmd+Option+X quit (saves everything and closes this
# window by itself). Closing this window also saves. Extra options: --fresh (always calibrate), --no-learn.
# First time: macOS needs Input Monitoring for Terminal (the script explains where).
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Install it from https://docs.astral.sh/uv/ and try again."
  read -r -p "Press Enter to close" _; exit 1
fi
echo "gaze-cursor: Cursor mode (tap Right Command to jump; Cmd+Option+X quits and saves)"
uv run scripts/cursor_mode.py "$@"
status=$?
echo
if [ $status -ne 0 ]; then
  echo "Stopped (exit code $status)."
  read -r -p "Press Enter to close" _
  exit $status
fi
# Clean quit: everything is saved, so close this Terminal window without waiting for Enter.
tty_name="$(tty)"
(sleep 1; osascript -e "tell application \"Terminal\" to close (every window whose tty of tabs contains \"$tty_name\")" >/dev/null 2>&1) &
exit 0
