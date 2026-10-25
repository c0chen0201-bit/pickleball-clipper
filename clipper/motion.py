"""Step 2b: measure how much the picture moves, moment by moment.

Sound alone struggles indoors: the room echoes and other courts are playing,
so paddle pops are barely louder than the background. The picture is a second
witness. During a point all four players move quickly; between points they
walk, chat or fetch balls, which moves far less. We don't track anyone, we
just measure how different each frame is from the one before it, using the
small proxy copy made by 'clip ingest'.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from . import common as c

FPS = 10            # the proxy copy is 10 frames per second
SIZE = (96, 54)     # tiny frames are plenty to see "a lot" vs "a little" movement
SMOOTH = 1.0        # seconds of averaging, so a single twitch doesn't count


Region = tuple[float, float, float, float]  # left, top, right, bottom as fractions of the frame


def parse_region(text: str) -> Region:
    """'0.2,0.3,0.8,0.75' -> (0.2, 0.3, 0.8, 0.75)."""
    try:
        parts = tuple(float(x) for x in text.split(","))
    except ValueError:
        parts = ()
    if len(parts) != 4 or not (0 <= parts[0] < parts[2] <= 1 and 0 <= parts[1] < parts[3] <= 1):
        raise c.ClipperError("--court needs four numbers between 0 and 1: left,top,right,bottom, "
                             "for example 0.2,0.3,0.8,0.75")
    return parts


def frame_differences(proxy: Path, region: Region | None = None) -> np.ndarray:
    """Average pixel change between each frame and the previous one, FPS per second.

    With a region, only that part of the frame is watched. Pointing it at the
    court keeps someone walking right past the phone from counting as play.
    """
    filters = []
    if region is not None:
        l, t, r, b = region
        filters.append(f"crop=iw*{r - l:.4f}:ih*{b - t:.4f}:iw*{l:.4f}:ih*{t:.4f}")
    filters.append(f"scale={SIZE[0]}:{SIZE[1]}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(proxy),
           "-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise c.ClipperError(f"FFmpeg failed:\n{result.stderr.decode().strip()}")
    per_frame = SIZE[0] * SIZE[1]
    n = len(result.stdout) // per_frame
    frames = np.frombuffer(result.stdout[: n * per_frame], dtype=np.uint8).reshape(n, per_frame)
    frames = frames.astype(np.float32)
    diff = np.abs(np.diff(frames, axis=0)).mean(axis=1)
    return np.concatenate([[0.0], diff]).astype(np.float32)


def smooth(values: np.ndarray, seconds: float = SMOOTH) -> np.ndarray:
    k = max(1, int(round(seconds * FPS)))
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def score(diff: np.ndarray) -> tuple[np.ndarray, dict]:
    """Turn raw differences into 0 (as still as this game gets) .. 1 (as busy as it gets).

    The game is its own yardstick: the 20th percentile is "idle", the 90th is
    "busy". A fixed camera on a tripod and a shaky hand-held phone then get
    the same treatment. If the picture barely changes at all (a test pattern,
    or a lens cap), the motion signal is marked unusable.
    """
    s = smooth(diff)
    idle = float(np.percentile(s, 20)) if len(s) else 0.0
    busy = float(np.percentile(s, 90)) if len(s) else 0.0
    usable = busy - idle >= 0.5
    scaled = np.clip((s - idle) / (busy - idle), 0, 1) if usable else np.zeros_like(s)
    return scaled.astype(np.float32), {"idle": round(idle, 3), "busy": round(busy, 3), "usable": usable}


def detect(work: Path, region: Region | None = None) -> dict:
    proxy = c.need(work, c.PROXY, "ingest")
    diff = frame_differences(proxy, region)
    scaled, stats = score(diff)
    data = {"fps": FPS, "smooth_seconds": SMOOTH, "region": region, **stats,
            "score": [round(float(v), 3) for v in scaled]}
    c.write_json(work / c.MOTION, data)
    return data


def load(work: Path) -> np.ndarray | None:
    """The motion score per frame, or None if 'clip motion' was not run or saw nothing."""
    path = work / c.MOTION
    if not path.exists():
        return None
    data = c.read_json(path)
    if not data.get("usable"):
        return None
    return np.asarray(data["score"], dtype=np.float32)


def level(scores: np.ndarray, start: float, end: float, how: str = "mean") -> float:
    """Motion between two times: the average, or the quietest moment ('min')."""
    a = max(0, int(start * FPS))
    b = min(len(scores), int(np.ceil(end * FPS)))
    if b <= a:
        return 0.0
    part = scores[a:b]
    return float(part.mean() if how == "mean" else part.min())


def busy_stretches(scores: np.ndarray, threshold: float, min_seconds: float) -> list[tuple[float, float]]:
    """Times when the picture stays busier than `threshold` for at least `min_seconds`."""
    active = scores >= threshold
    out: list[tuple[float, float]] = []
    start = None
    for i, on in enumerate(np.concatenate([active, [False]])):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if (i - start) / FPS >= min_seconds:
                out.append((start / FPS, i / FPS))
            start = None
    return out
