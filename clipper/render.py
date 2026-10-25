"""Steps 4 and 5: cut rallies out of the game and join them into videos."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import common as c
from . import score as scoring
from . import vertical
from .hits import load_audio

ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]


def cut(video: Path, start: float, end: float, out: Path) -> None:
    # Re-encoding (instead of copying) makes the cut land on the exact frame.
    c.run_ffmpeg(["-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
                  *ENCODE, "-avoid_negative_ts", "make_zero", str(out)])


def join(parts: list[Path], out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{p.resolve()}'\n")
        listing = Path(f.name)
    try:
        c.run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                      "-movflags", "+faststart", str(out)])
    finally:
        listing.unlink(missing_ok=True)


def load_rallies(work: Path) -> list[dict]:
    rallies = c.read_json(c.need(work, c.RALLIES, "rallies"))["rallies"]
    if not rallies:
        raise c.ClipperError("No rallies were found, so there is nothing to export. "
                             "Check hits_check.mp4 to see whether hits are being heard.")
    return rallies


def export_full(work: Path, out_dir: Path) -> Path:
    """Every rally, in order, with the dead time between them removed."""
    video = c.need(work, c.VIDEO, "ingest")
    rallies = load_rallies(work)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for r in rallies:
            part = Path(tmp) / f"rally_{r['id']:03d}.mp4"
            cut(video, r["start"], r["end"], part)
            parts.append(part)
        out = out_dir / "full_game.mp4"
        join(parts, out)
    return out


def export_highlights(work: Path, out_dir: Path, top: int = 5) -> tuple[Path, list[dict]]:
    """The best `top` rallies as separate clips, plus one reel (best first)."""
    video = c.need(work, c.VIDEO, "ingest")
    audio: np.ndarray = load_audio(c.need(work, c.AUDIO, "ingest"))
    ranked = scoring.rank(load_rallies(work), audio)
    best = ranked[:top]
    out_dir.mkdir(parents=True, exist_ok=True)
    c.write_json(out_dir / "ranking.json", ranked)

    clips = []
    for place, r in enumerate(best, 1):
        clip = out_dir / f"highlight_{place}_rally{r['id']:03d}.mp4"
        cut(video, r["start"], r["end"], clip)
        clips.append(clip)
    reel = out_dir / "highlight_reel.mp4"
    join(clips, reel)
    return reel, best


def export_vertical(work: Path, out_dir: Path, top: int = 5) -> tuple[Path, list[dict]]:
    """The best `top` rallies as tall 9:16 clips that follow the ball, plus one reel."""
    audio: np.ndarray = load_audio(c.need(work, c.AUDIO, "ingest"))
    best = scoring.rank(load_rallies(work), audio)[:top]
    clips = vertical.export(work, out_dir, best)
    reel = out_dir / "vertical_reel.mp4"
    join(clips, reel)
    return reel, best
