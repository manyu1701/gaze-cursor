#!/bin/bash
# Double-click to open the dashboard in your browser: launch the tools and see your sessions, what has been
# logged and what the model has learned. It runs only on this Mac (127.0.0.1); close this window or press
# Ctrl+C to stop it.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Install it from https://docs.astral.sh/uv/ and try again."
  read -r -p "Press Enter to close" _; exit 1
fi
echo "gaze-cursor: dashboard"
uv run scripts/dashboard.py "$@"
status=$?
echo
if [ $status -ne 0 ]; then echo "The dashboard stopped with an error (exit code $status)."; fi
read -r -p "Press Enter to close" _
exit $status
