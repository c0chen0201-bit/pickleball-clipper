"""Step 6c: teach a small detector what this club's ball looks like.

Starts from YOLO (a free detector already trained on everyday objects) and
fine-tunes it on the frames collected by 'clip label'. Needs the 'ai' extras:
pip install -e '.[ai]'. On a Mac with an Apple chip, the GPU is used.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from . import common as c


def split(labels_dir: Path, val_share: float = 0.15, seed: int = 0) -> tuple[Path, Path]:
    """Write train.txt / val.txt listing the images; YOLO finds each label next to it."""
    images = sorted((labels_dir / "images").glob("*.jpg"))
    if len(images) < 20:
        raise c.ClipperError(f"Only {len(images)} labelled frames in {labels_dir}; collect more with 'clip label'.")
    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * val_share))
    val, train = images[:n_val], images[n_val:]
    (labels_dir / "train.txt").write_text("\n".join(str(p.resolve()) for p in sorted(train)) + "\n")
    (labels_dir / "val.txt").write_text("\n".join(str(p.resolve()) for p in sorted(val)) + "\n")
    data = labels_dir / "data.yaml"
    data.write_text(
        f"path: {labels_dir.resolve()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: ball\n"
    )
    return data, labels_dir


def pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def train(labels_dir: Path, out: Path, epochs: int = 40, imgsz: int = 1280, batch: int = 8,
          base: str = "yolov8n.pt", device: str | None = None) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise c.ClipperError("Training needs Ultralytics and PyTorch. Run: pip install -e '.[ai]'")
    data, _ = split(labels_dir)
    device = device or pick_device()
    model = YOLO(base)
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                project=str(labels_dir / "runs"), name="ball", exist_ok=True,
                # The ball is tiny, so don't shrink it away: mild scaling, no mosaic late on.
                scale=0.3, mosaic=1.0, close_mosaic=10, patience=15, verbose=False)
    best = labels_dir / "runs" / "ball" / "weights" / "best.pt"
    if not best.exists():
        raise c.ClipperError("Training finished but no weights were written; see the runs/ folder.")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    return out
