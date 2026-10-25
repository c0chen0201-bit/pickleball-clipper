"""Step 7: vertical (9:16) clips that follow the ball, for phones.

A tall window slides left and right over the wide recording. It aims at the
ball when the ball is seen, holds still through short gaps, and drifts back
to the middle of the court when the ball is lost for longer. The path is
smoothed so the picture pans like a camera operator, not a twitch.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import ball as ball_mod
from . import common as c

SMOOTH = 0.7      # seconds of averaging on the aim
MAX_PAN = 18.0    # pixels per frame the window may move (at 1080p)
HOLD = 1.0        # seconds to keep aiming at the last known ball before drifting home


def crop_size(width: int, height: int) -> tuple[int, int]:
    """The biggest 9:16 window that fits, with even sides for the video encoder."""
    w = min(width, int(height * 9 / 16))
    return w - w % 2, height - height % 2


def aim_path(ball: dict | None, start: float, end: float, width: int, crop_w: int) -> np.ndarray:
    """Centre x of the window for every frame from start to end."""
    fps = ball["fps"] if ball else 30.0
    f0, f1 = int(start * fps), int(end * fps)
    n = max(1, f1 - f0)
    home = width / 2
    x = np.full(n, np.nan)
    if ball is not None:
        for f, (bx, _, _) in ball_mod.positions_by_frame(ball).items():
            if f0 <= f < f1:
                x[f - f0] = bx
    known = np.flatnonzero(~np.isnan(x))
    if len(known) == 0:
        return np.full(n, home)
    # Bridge short gaps by interpolation; hold, then drift home, over long ones.
    filled = np.interp(np.arange(n), known, x[known])
    hold = int(HOLD * fps)
    gap_start = None
    for i in range(n):
        if not np.isnan(x[i]):
            gap_start = None
            continue
        if gap_start is None:
            gap_start = i
        away = i - gap_start
        if away > hold:
            t = min(1.0, (away - hold) / (3 * fps))
            filled[i] = filled[i] * (1 - t) + home * t
    before, after = known[0], known[-1]
    filled[:before] = filled[before]
    if after + hold < n:
        for i in range(after + hold, n):
            t = min(1.0, (i - after - hold) / (3 * fps))
            filled[i] = filled[after] * (1 - t) + home * t
    k = max(1, int(SMOOTH * fps))
    smooth = np.convolve(np.pad(filled, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    # A camera operator can only pan so fast.
    out = np.empty(n)
    out[0] = smooth[0]
    for i in range(1, n):
        step = np.clip(smooth[i] - out[i - 1], -MAX_PAN, MAX_PAN)
        out[i] = out[i - 1] + step
    return np.clip(out, crop_w / 2, width - crop_w / 2)


def cut_vertical(video: Path, start: float, end: float, path: np.ndarray, out: Path) -> None:
    ball_mod.require_cv2()
    cv2 = ball_mod.cv2
    cap, fps, (w, h), _ = ball_mod.open_video(video)
    cw, ch = crop_size(w, h)
    silent = out.with_name(out.stem + "_silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    n = int((end - start) * fps)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        cx = path[min(i, len(path) - 1)]
        left = int(round(cx - cw / 2))
        left = max(0, min(w - cw, left))
        writer.write(frame[:ch, left: left + cw])
    cap.release()
    writer.release()
    try:
        c.run_ffmpeg(["-i", str(silent), "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
                      "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                      "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
                      "-movflags", "+faststart", str(out)])
    finally:
        silent.unlink(missing_ok=True)


def export(work: Path, out_dir: Path, rallies: list[dict], prefix: str = "vertical") -> list[Path]:
    video = c.need(work, c.VIDEO, "ingest")
    ball = ball_mod.load(work)
    info = c.read_json(c.need(work, c.INFO, "ingest"))
    cap, fps, (w, h), _ = ball_mod.open_video(video)
    cap.release()
    cw, _ = crop_size(w, h)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for place, r in enumerate(rallies, 1):
        path = aim_path(ball, r["start"], min(r["end"], info["duration"]), w, cw)
        clip = out_dir / f"{prefix}_{place}_rally{r['id']:03d}.mp4"
        cut_vertical(video, r["start"], r["end"], path, clip)
        clips.append(clip)
    return clips
