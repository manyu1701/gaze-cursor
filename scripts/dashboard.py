"""Local dashboard: launch the tools and see what has been recorded and learned (opens in your browser).

    uv run scripts/dashboard.py                  # serve on a free port and open the browser
    uv run scripts/dashboard.py --no-open        # print the address only
    uv run scripts/dashboard.py --data-dir DIR   # read another folder with sessions/ quick/ tasks/ results.csv

The server listens on 127.0.0.1 only, shows recordings by id (never face images) and writes nothing under
data/. Press Ctrl+C in this window to stop it.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from gaze_cursor.dashboard.catalog import Roots
from gaze_cursor.dashboard.server import App, serve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=0, help="port to listen on (default: any free port)")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    ap.add_argument("--data-dir", type=Path, default=None, help="folder with sessions/, quick/, tasks/, results.csv")
    args = ap.parse_args()
    roots = Roots.default() if args.data_dir is None else Roots.from_data_dir(args.data_dir)
    app = App(roots)
    httpd = serve(app, args.port)
    url = f"http://127.0.0.1:{app.port}/"
    print(f"gaze-cursor dashboard: {url}\nCtrl+C stops it. Nothing leaves this computer.")
    if not args.no_open:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
