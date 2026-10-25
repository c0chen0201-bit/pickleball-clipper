"""Step 3b: measure how well rally detection matches rallies you wrote down by hand.

The labels file is a CSV with a header row and two columns, start and end,
in seconds or as minutes:seconds:

    start,end
    0:12.5,0:21.0
    0:38,0:47.2
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import common as c


def read_labels(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"start", "end"} <= set(rows[0]):
        raise c.ClipperError(f"{path} needs a header row with 'start' and 'end' columns.")
    return [(c.parse_time(r["start"]), c.parse_time(r["end"])) for r in rows if r["start"].strip()]


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def compare(detected: list[tuple[float, float]], labels: list[tuple[float, float]]) -> dict:
    # A labeled rally counts as found when a detected rally covers at least half of it.
    found = [lab for lab in labels
             if any(overlap(lab, d) >= 0.5 * (lab[1] - lab[0]) for d in detected)]
    false_alarms = [d for d in detected if not any(overlap(d, lab) > 0 for lab in labels)]
    return {
        "labeled": len(labels),
        "detected": len(detected),
        "found": len(found),
        "missed": [lab for lab in labels if lab not in found],
        "false_alarms": false_alarms,
        "recall": len(found) / len(labels) if labels else 0.0,
    }


def evaluate(work: Path, labels_path: Path) -> dict:
    rallies = c.read_json(c.need(work, c.RALLIES, "rallies"))["rallies"]
    return compare([(r["start"], r["end"]) for r in rallies], read_labels(labels_path))
