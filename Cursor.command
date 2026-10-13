#!/bin/bash
# Double-click for Cursor mode: tap Right Command and the real cursor jumps to where you look; the
# trackpad finishes. Starts with a 9-dot quick calibration where you sit (~15 s). Hotkeys work in any
# app: Cmd+Option+G on/off, Cmd+Option+R recalibrate, Cmd+Option+X quit.
# First time: macOS needs Input Monitoring for Terminal (the script explains where).
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Install it from https://docs.astral.sh/uv/ and try again."
  read -r -p "Press Enter to close" _; exit 1
fi
echo "gaze-cursor: Cursor mode (tap Right Command to jump; Cmd+Option+X quits)"
uv run scripts/cursor_mode.py "$@"
status=$?
echo
if [ $status -ne 0 ]; then echo "Stopped (exit code $status)."; fi
read -r -p "Press Enter to close" _
exit $status
