"""Download the MediaPipe Face Landmarker model into models/ (idempotent)."""

import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEST = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def main() -> None:
    if DEST.exists():
        print(f"Already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.rename(DEST)
    print(f"Downloaded {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
