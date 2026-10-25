"""Step 6: find the ball in every frame and follow it.

Two ways to spot the ball:

- **No training** (the default): the ball is a small, fast, round thing. We
  compare each frame with the one before and the one after; whatever changed
  in both, is tiny and roughly round is a ball candidate. Players' arms and
  bodies are big, shadows are slow, so most of them drop out.
- **A trained model** (`--model ball_model.pt`): a YOLO detector taught what
  this club's ball looks like, made by 'clip label' and 'clip train-ball'.

Either way the candidates are linked frame to frame into *tracks*, since a
real ball moves smoothly and a speck of noise does not. From the tracks we
read off events: the ball bounced, or the ball left the picture. A point is
over after the second bounce or when the ball leaves.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import common as c

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

WIDTH = 640  # frames are shrunk to this width for the no-training detector


@dataclass
class BallSettings:
    diff_threshold: float = 28.0  # how much a pixel must change (0-255) to count as moving
    min_area: int = 2             # ball size in pixels at WIDTH, smallest...
    max_area: int = 140           # ...and largest
    link_radius: float = 45.0     # how far (pixels at WIDTH) the ball may move between frames
    max_coast: int = 4            # frames a track survives without seeing the ball
    min_track: int = 6            # shorter tracks are noise
    min_travel: float = 25.0      # a track that never goes anywhere is noise (pixels)
    model_conf: float = 0.25      # trained model: lowest confidence to accept


def require_cv2() -> None:
    if cv2 is None:
        raise c.ClipperError("The ball step needs OpenCV. Run: pip install opencv-python-headless")


# ----------------------------------------------------------------- reading frames

def open_video(path: Path):
    require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise c.ClipperError(f"Could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, (w, h), n


def small_gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = WIDTH / w
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA).astype(np.float32)


# ------------------------------------------------------------ finding candidates

def moving_blobs(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray,
                 s: BallSettings = BallSettings()) -> list[tuple[float, float, float]]:
    """Small round things that moved: (x, y, size) at WIDTH scale."""
    # A pixel that differs from BOTH neighbours is where the ball is now;
    # where it was last frame only differs from one of them.
    d = np.minimum(np.abs(cur - prev), np.abs(cur - nxt))
    mask = (d > s.diff_threshold).astype(np.uint8)
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (s.min_area <= area <= s.max_area):
            continue
        if max(w, h) > 24 or min(w, h) == 0 or max(w, h) / min(w, h) > 2.5:
            continue
        out.append((float(centers[i][0]), float(centers[i][1]), float(max(w, h))))
    return merge_close(out)


def merge_close(blobs: list[tuple[float, float, float]], within: float = 10.0) -> list[tuple[float, float, float]]:
    """A fast ball can show up as two fragments; join blobs that nearly touch."""
    blobs = sorted(blobs, key=lambda b: -b[2])
    merged: list[tuple[float, float, float]] = []
    for x, y, size in blobs:
        for j, (mx, my, ms) in enumerate(merged):
            if np.hypot(x - mx, y - my) <= within:
                merged[j] = ((mx + x) / 2, (my + y) / 2, max(ms, size))
                break
        else:
            merged.append((x, y, size))
    return merged


def model_blobs(model, frame: np.ndarray, conf: float) -> list[tuple[float, float, float]]:
    """Ball detections from a trained YOLO model, scaled to WIDTH."""
    scale = WIDTH / frame.shape[1]
    out = []
    for r in model.predict(frame, conf=conf, verbose=False, imgsz=1280):
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(((x1 + x2) / 2 * scale, (y1 + y2) / 2 * scale, max(x2 - x1, y2 - y1) * scale))
    return out


# ------------------------------------------------------------------ tracking

class Track:
    def __init__(self, frame: int, x: float, y: float, size: float):
        self.points: list[tuple[int, float, float, float]] = [(frame, x, y, size)]
        self.misses = 0

    @property
    def last(self):
        return self.points[-1]

    def velocity(self) -> tuple[float, float]:
        if len(self.points) < 2:
            return 0.0, 0.0
        (f0, x0, y0, _), (f1, x1, y1, _) = self.points[-2], self.points[-1]
        dt = max(1, f1 - f0)
        return (x1 - x0) / dt, (y1 - y0) / dt

    def predict(self, frame: int) -> tuple[float, float]:
        f, x, y, _ = self.last
        vx, vy = self.velocity()
        dt = frame - f
        return x + vx * dt, y + vy * dt

    def travel(self) -> float:
        xs = [p[1] for p in self.points]
        ys = [p[2] for p in self.points]
        return float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))


def link(frames: list[list[tuple[float, float, float]]], s: BallSettings = BallSettings()) -> list[Track]:
    """Join per-frame candidates into tracks: nearest candidate to each track's predicted spot."""
    active: list[Track] = []
    done: list[Track] = []
    for frame, cands in enumerate(frames):
        free = list(range(len(cands)))
        pairs = []
        for ti, t in enumerate(active):
            px, py = t.predict(frame)
            reach = s.link_radius * (1 + 0.5 * t.misses)
            for ci in free:
                d = np.hypot(cands[ci][0] - px, cands[ci][1] - py)
                if d <= reach:
                    pairs.append((d, ti, ci))
        pairs.sort()
        used_t, used_c = set(), set()
        for d, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            x, y, size = cands[ci]
            active[ti].points.append((frame, x, y, size))
            active[ti].misses = 0
        for ci in free:
            if ci not in used_c:
                active.append(Track(frame, *cands[ci]))
        still = []
        for ti, t in enumerate(active):
            if ti not in used_t and t.points[-1][0] != frame:
                t.misses += 1
            if t.misses > s.max_coast:
                done.append(t)
            else:
                still.append(t)
        active = still
    done += active
    return [t for t in done if len(t.points) >= s.min_track and t.travel() >= s.min_travel]


# ------------------------------------------------------------------- events

def events_from_track(t: Track, fps: float, size: tuple[int, int], reach: int = 4) -> list[dict]:
    """Bounces (ball reaches its lowest point and comes back up) and exits."""
    out = []
    pts = t.points
    ys = np.array([p[2] for p in pts])
    for i in range(1, len(pts) - 1):
        lo, hi = max(0, i - reach), min(len(pts), i + reach + 1)
        # Lowest point on screen in its neighbourhood, and clearly lower than both sides.
        if ys[i] < ys[lo:hi].max() or ys[i] - ys[lo:i].min() < 3 or ys[i] - ys[i + 1:hi].min() < 3:
            continue
        f0, _, y0, _ = pts[lo]
        f1, x1, y1, _ = pts[i]
        f2, _, y2, _ = pts[hi - 1]
        down = (y1 - y0) / max(1, f1 - f0)
        up = (y2 - y1) / max(1, f2 - f1)
        if down > 0.8 and up < -0.5:
            out.append({"time": round(f1 / fps, 3), "type": "bounce", "x": round(x1, 1), "y": round(y1, 1)})
    f, x, y, _ = pts[-1]
    w, h = size
    edge = 0.03
    vx, vy = t.velocity()
    leaving = (x < w * edge and vx < 0) or (x > w * (1 - edge) and vx > 0) or \
              (y < h * edge and vy < 0) or (y > h * (1 - edge) and vy > 0)
    if leaving:
        out.append({"time": round(f / fps, 3), "type": "off_screen", "x": round(x, 1), "y": round(y, 1)})
    return out


# -------------------------------------------------------------------- running

def track_video(video: Path, s: BallSettings = BallSettings(), model_path: Path | None = None,
                progress=None) -> dict:
    cap, fps, (w, h), n = open_video(video)
    model = None
    if model_path is not None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise c.ClipperError("Using a trained model needs Ultralytics. Run: pip install ultralytics")
        model = YOLO(str(model_path))

    per_frame: list[list[tuple[float, float, float]]] = []
    prev = cur = None
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if model is not None:
            per_frame.append(model_blobs(model, frame, s.model_conf))
        else:
            nxt = small_gray(frame)
            if prev is not None and cur is not None:
                per_frame.append(moving_blobs(prev, cur, nxt, s))
            elif cur is not None:
                per_frame.append([])
            prev, cur = cur, nxt
        frame_i += 1
        if progress and frame_i % 300 == 0:
            progress(frame_i, n)
    cap.release()
    if model is None:
        per_frame.append([])  # the last frame has no "next", so no candidates
    while len(per_frame) < frame_i:
        per_frame.append([])

    tracks = link(per_frame, s)
    tracks.sort(key=lambda t: t.points[0][0])
    scale = w / WIDTH
    small = (WIDTH, int(round(h / scale)))
    events = []
    for t in tracks:
        events += events_from_track(t, fps, small)
    events.sort(key=lambda e: e["time"])
    for e in events:
        e["x"] = round(e["x"] * scale, 1)
        e["y"] = round(e["y"] * scale, 1)
    return {
        "fps": fps, "width": w, "height": h, "frames": frame_i,
        "settings": asdict(s), "model": str(model_path) if model_path else None,
        "tracks": [{
            "start": round(t.points[0][0] / fps, 3), "end": round(t.points[-1][0] / fps, 3),
            "points": [[f, round(x * scale, 1), round(y * scale, 1), round(sz * scale, 1)]
                       for f, x, y, sz in t.points],
        } for t in tracks],
        "events": events,
    }


def detect(work: Path, s: BallSettings = BallSettings(), model_path: Path | None = None,
           check: bool = True, progress=None) -> dict:
    video = c.need(work, c.VIDEO, "ingest")
    data = track_video(video, s, model_path, progress)
    c.write_json(work / c.BALL, data)
    if check:
        write_check(work, data)
    return data


def load(work: Path) -> dict | None:
    path = work / c.BALL
    return c.read_json(path) if path.exists() else None


def positions_by_frame(data: dict) -> dict[int, tuple[float, float, float]]:
    """Frame -> (x, y, size), preferring the longest track where tracks overlap."""
    out: dict[int, tuple[float, float, float]] = {}
    for t in sorted(data["tracks"], key=lambda t: len(t["points"])):
        for f, x, y, sz in t["points"]:
            out[f] = (x, y, sz)
    return out


def write_check(work: Path, data: dict, trail: int = 12) -> Path:
    """A 640-wide copy of the game with a circle on the ball and its recent path."""
    require_cv2()
    video = c.need(work, c.VIDEO, "ingest")
    cap, fps, (w, h), _ = open_video(video)
    scale = WIDTH / w
    size = (WIDTH, int(round(h * scale)))
    pos = positions_by_frame(data)
    silent = work / "ball_check_silent.mp4"
    out = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    recent: list[tuple[float, float]] = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if f in pos:
            x, y, sz = pos[f]
            recent.append((x * scale, y * scale))
            cv2.circle(frame, (int(x * scale), int(y * scale)), max(6, int(sz * scale)), (0, 255, 255), 2)
        else:
            recent.append(None)
        recent = recent[-trail:]
        pts = [p for p in recent if p is not None]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 200, 255), 1)
        out.write(frame)
        f += 1
    cap.release()
    out.release()
    dest = work / c.BALL_CHECK
    try:
        c.run_ffmpeg(["-i", str(silent), "-i", str(video), "-map", "0:v", "-map", "1:a?",
                      "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-shortest", str(dest)])
    finally:
        silent.unlink(missing_ok=True)
    return dest


# ------------------------------------------------------- ending a point by the ball

def point_end(data: dict, last_hit: float, max_tail: float) -> float | None:
    """When the point ended after the last hit heard: the second bounce, or the ball leaving.

    Returns None if the ball data has nothing to say in that window.
    """
    bounces = 0
    for e in data["events"]:
        t = e["time"]
        if t < last_hit - 0.1:
            continue
        if t > last_hit + max_tail:
            break
        if e["type"] == "off_screen":
            return t
        bounces += 1
        if bounces >= 2:
            return t
    return None
