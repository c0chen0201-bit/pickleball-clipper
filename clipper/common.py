"""Small helpers shared by every stage: running FFmpeg, the work folder, JSON files."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

# Every stage reads and writes files in one "work folder" per game, so any
# stage can be rerun on its own.
VIDEO = "video.mp4"      # normalized copy of the game
AUDIO = "audio.wav"      # mono 16 kHz audio used for hit detection
PROXY = "proxy.mp4"      # small 480p copy for later vision steps
INFO = "info.json"       # duration and source path
HITS = "hits.json"
HITS_CHECK = "hits_check.mp4"
MOTION = "motion.json"   # how much the picture moves, 10 times a second
RALLIES = "rallies.json"

SAMPLE_RATE = 16_000


class ClipperError(Exception):
    """A problem we can explain to the user in plain words."""


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise ClipperError(
                f"Could not find '{tool}'. Install FFmpeg first "
                "(Mac: 'brew install ffmpeg', Windows: 'winget install ffmpeg')."
            )


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ClipperError(f"FFmpeg failed:\n{result.stderr.strip()}")


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ClipperError(f"Could not read video '{path}':\n{result.stderr.strip()}")
    return float(result.stdout.strip())


def default_work_dir(video: Path) -> Path:
    """game.mp4 -> game_work/ next to it."""
    return video.with_name(video.stem + "_work")


def need(work: Path, name: str, made_by: str) -> Path:
    path = work / name
    if not path.exists():
        raise ClipperError(f"Missing {path}. Run 'clip {made_by}' first.")
    return path


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def timecode(seconds: float) -> str:
    """75.5 -> '01:15.50'; 3725 -> '1:02:05.00'."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 1:
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"
    return f"{int(m):02d}:{s:05.2f}"


def parse_time(text: str) -> float:
    """Accepts '75.5', '1:15.5' or '0:01:15.5'."""
    total = 0.0
    for part in text.strip().split(":"):
        total = total * 60 + float(part)
    return total
