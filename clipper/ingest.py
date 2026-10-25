"""Step 1: make clean working copies of a game recording."""

from __future__ import annotations

from pathlib import Path

from . import common as c


def ingest(source: Path, work: Path) -> Path:
    c.require_ffmpeg()
    if not source.exists():
        raise c.ClipperError(f"Video not found: {source}")
    work.mkdir(parents=True, exist_ok=True)

    # Normalized copy: at most 1080 tall, constant 30 fps. Phones record at a
    # variable frame rate, which makes cuts drift; a constant rate fixes that.
    c.run_ffmpeg([
        "-i", str(source),
        "-vf", "scale=-2:'min(1080,ih)',fps=30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        str(work / c.VIDEO),
    ])
    # Mono 16 kHz audio is all the hit detector needs.
    c.run_ffmpeg([
        "-i", str(source), "-vn", "-ac", "1", "-ar", str(c.SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(work / c.AUDIO),
    ])
    # Small, silent 480p copy at 10 fps for the player/ball tracking step later.
    c.run_ffmpeg([
        "-i", str(source), "-an",
        "-vf", "scale=-2:480,fps=10",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        str(work / c.PROXY),
    ])

    duration = c.probe_duration(work / c.VIDEO)
    c.write_json(work / c.INFO, {"source": str(source.resolve()), "duration": duration})
    return work
