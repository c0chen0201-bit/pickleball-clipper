"""Step 6b: collect ball examples to teach a model, and let a person fix them.

'clip label' takes frames from a game, marks where the no-training tracker
saw the ball, and writes them in the YOLO format (one image, one small text
file with the box). It also writes review.html: open it in a browser, and on
each frame click the ball to move the mark, or press X if there is no ball.
At the end, "Save fixes" downloads fixes.json, and 'clip label-fix' writes
the corrected marks back. The same folder can hold frames from many games.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from . import ball as ball_mod
from . import common as c

BOX_MIN = 22      # pixels: a box this big always covers the ball at 1080p
BOX_MAX = 60      # a blurred ball near the camera streaks, but the ball itself is never bigger
BOX_SCALE = 1.6   # box side relative to the blob the tracker saw


def choose_frames(data: dict, n: int, negatives: float = 0.2, min_track: int = 12,
                  seed: int = 0) -> tuple[list[tuple[int, tuple[float, float, float]]], list[int]]:
    """Frames with a confident ball mark, and frames that should have no ball."""
    rng = random.Random(seed)
    good = [t for t in data["tracks"] if len(t["points"]) >= min_track]
    marks = {}
    for t in good:
        for f, x, y, sz in t["points"]:
            marks[f] = (x, y, sz)
    n_pos = int(round(n * (1 - negatives)))
    pos_frames = sorted(marks)
    if len(pos_frames) > n_pos:
        pos_frames = [pos_frames[i] for i in np.linspace(0, len(pos_frames) - 1, n_pos).astype(int)]
    positives = [(f, marks[f]) for f in pos_frames]

    fps = data["fps"]
    busy = np.zeros(data["frames"], dtype=bool)
    for t in data["tracks"]:
        a, b = int(t["start"] * fps) - int(fps), int(t["end"] * fps) + int(fps)
        busy[max(0, a): b] = True
    quiet = [f for f in range(data["frames"]) if not busy[f]]
    rng.shuffle(quiet)
    return positives, sorted(quiet[: n - len(positives)])


def grab_frames(video: Path, frames: list[int]):
    """Yields (frame_number, image) for the frames asked for, in order."""
    cap, fps, size, n = ball_mod.open_video(video)
    want = sorted(set(frames))
    for f in want:
        cap.set(ball_mod.cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            yield f, img
    cap.release()


def box_for(x: float, y: float, size: float, w: int, h: int) -> tuple[float, float, float, float]:
    side = min(BOX_MAX, max(BOX_MIN, size * BOX_SCALE))
    return x / w, y / h, side / w, side / h


def export(work: Path, out: Path, n: int = 300, seed: int = 0) -> dict:
    data = ball_mod.load(work)
    if data is None:
        raise c.ClipperError(f"No ball data in {work}. Run 'clip ball' first.")
    video = c.need(work, c.VIDEO, "ingest")
    prefix = work.name.removesuffix("_work")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    positives, negatives = choose_frames(data, n, seed=seed)
    marks = dict(positives)
    w, h = data["width"], data["height"]
    written = []
    for f, img in grab_frames(video, [f for f, _ in positives] + negatives):
        name = f"{prefix}_{f:06d}"
        ball_mod.cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [ball_mod.cv2.IMWRITE_JPEG_QUALITY, 92])
        label = out / "labels" / f"{name}.txt"
        if f in marks:
            cx, cy, bw, bh = box_for(*marks[f], w, h)
            label.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            written.append({"name": name, "x": round(marks[f][0], 1), "y": round(marks[f][1], 1), "has_ball": True})
        else:
            label.write_text("")
            written.append({"name": name, "x": None, "y": None, "has_ball": False})
    index = out / "index.json"
    existing = json.loads(index.read_text()) if index.exists() else []
    existing = [e for e in existing if not e["name"].startswith(prefix + "_")] + written
    index.write_text(json.dumps(existing, indent=1))
    write_review_page(out, existing, w, h)
    return {"positives": len(positives), "negatives": len(written) - len(positives), "total": len(existing)}


def apply_fixes(out: Path, fixes: Path) -> int:
    """Write the marks a person corrected in review.html back into the label files."""
    changes = json.loads(fixes.read_text())
    index = out / "index.json"
    entries = {e["name"]: e for e in json.loads(index.read_text())}
    n = 0
    for name, fix in changes.items():
        if name not in entries:
            continue
        label = out / "labels" / f"{name}.txt"
        e = entries[name]
        if fix.get("has_ball"):
            img = ball_mod.cv2.imread(str(out / "images" / f"{name}.jpg"))
            h, w = img.shape[:2]
            cx, cy, bw, bh = box_for(fix["x"], fix["y"], BOX_MIN / BOX_SCALE, w, h)
            label.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            e.update({"x": fix["x"], "y": fix["y"], "has_ball": True})
        else:
            label.write_text("")
            e.update({"x": None, "y": None, "has_ball": False})
        e["checked"] = True
        n += 1
    index.write_text(json.dumps(list(entries.values()), indent=1))
    return n


def write_review_page(out: Path, entries: list[dict], w: int, h: int) -> Path:
    page = out / "review.html"
    page.write_text(REVIEW_HTML.replace("__ENTRIES__", json.dumps(entries)))
    return page


REVIEW_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Check the ball marks</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
 #top{padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 #wrap{position:relative;display:inline-block;cursor:crosshair}
 img{display:block;max-width:100vw;max-height:85vh}
 #mark{position:absolute;width:28px;height:28px;border:2px solid #ff0;border-radius:50%;
       transform:translate(-50%,-50%);pointer-events:none;box-shadow:0 0 0 2px #000}
 button{padding:6px 12px;font-size:14px}
 kbd{background:#333;padding:2px 6px;border-radius:4px}
</style></head><body>
<div id="top">
 <b id="count"></b>
 <span>Click the ball to move the mark. <kbd>X</kbd> = no ball here, <kbd>→</kbd>/<kbd>Space</kbd> next, <kbd>←</kbd> back, <kbd>U</kbd> undo this frame.</span>
 <button onclick="save()">Save fixes</button>
 <span id="status"></span>
</div>
<div id="wrap"><img id="img"><div id="mark"></div></div>
<script>
const entries = __ENTRIES__;
let i = 0;
const fixes = JSON.parse(localStorage.getItem('ballfixes') || '{}');
const img = document.getElementById('img'), mark = document.getElementById('mark');
function current(){ const e = entries[i]; return fixes[e.name] || {x:e.x, y:e.y, has_ball:e.has_ball}; }
function show(){
  const e = entries[i]; img.src = 'images/' + e.name + '.jpg';
  document.getElementById('count').textContent = (i+1) + ' / ' + entries.length + '  ' + e.name;
  const n = Object.keys(fixes).length;
  document.getElementById('status').textContent = n ? n + ' changed (saved in this browser until you press Save fixes)' : '';
  place();
}
function place(){
  const cur = current();
  if (!cur.has_ball || cur.x == null){ mark.style.display = 'none'; return; }
  const sx = img.clientWidth / img.naturalWidth, sy = img.clientHeight / img.naturalHeight;
  mark.style.display = 'block'; mark.style.left = (cur.x * sx) + 'px'; mark.style.top = (cur.y * sy) + 'px';
}
img.onload = place;
img.onclick = ev => {
  const r = img.getBoundingClientRect();
  const x = (ev.clientX - r.left) * img.naturalWidth / img.clientWidth;
  const y = (ev.clientY - r.top) * img.naturalHeight / img.clientHeight;
  fixes[entries[i].name] = {x: Math.round(x*10)/10, y: Math.round(y*10)/10, has_ball: true};
  localStorage.setItem('ballfixes', JSON.stringify(fixes)); show();
};
document.onkeydown = ev => {
  if (ev.key === 'ArrowRight' || ev.key === ' ') { i = Math.min(entries.length-1, i+1); show(); ev.preventDefault(); }
  else if (ev.key === 'ArrowLeft') { i = Math.max(0, i-1); show(); }
  else if (ev.key.toLowerCase() === 'x') { fixes[entries[i].name] = {x:null, y:null, has_ball:false};
    localStorage.setItem('ballfixes', JSON.stringify(fixes)); i = Math.min(entries.length-1, i+1); show(); }
  else if (ev.key.toLowerCase() === 'u') { delete fixes[entries[i].name];
    localStorage.setItem('ballfixes', JSON.stringify(fixes)); show(); }
};
function save(){
  const blob = new Blob([JSON.stringify(fixes, null, 1)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'fixes.json'; a.click();
}
show();
</script></body></html>
"""
