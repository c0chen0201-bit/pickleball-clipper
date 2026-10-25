"""Step 3: group hits into rallies. Everything between rallies is dead time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from . import common as c


@dataclass
class RallySettings:
    max_gap: float = 2.5    # a longer silence than this ends the rally (seconds)
    min_hits: int = 3       # serve, return, third shot
    pad_before: float = 1.0  # show the server getting ready
    pad_after: float = 1.5   # show the point ending


def group_hits(times: list[float], duration: float, settings: RallySettings = RallySettings()) -> list[dict]:
    runs: list[list[float]] = []
    for t in sorted(times):
        if runs and t - runs[-1][-1] <= settings.max_gap:
            runs[-1].append(t)
        else:
            runs.append([t])

    rallies: list[dict] = []
    for run in runs:
        if len(run) < settings.min_hits:
            continue
        start = max(0.0, run[0] - settings.pad_before)
        end = min(duration, run[-1] + settings.pad_after)
        if rallies and start <= rallies[-1]["end"]:
            # Padding made two rallies touch; join them so no frame plays twice.
            prev = rallies[-1]
            prev["end"] = end
            prev["hits"] += run
            continue
        rallies.append({"start": start, "end": end, "hits": run})

    for i, r in enumerate(rallies, 1):
        r["id"] = i
        r["start"] = round(r["start"], 3)
        r["end"] = round(r["end"], 3)
        r["hit_count"] = len(r["hits"])
    return [{k: r[k] for k in ("id", "start", "end", "hit_count", "hits")} for r in rallies]


def find(work: Path, settings: RallySettings = RallySettings()) -> list[dict]:
    hits = c.read_json(c.need(work, c.HITS, "hits"))["hits"]
    duration = c.read_json(c.need(work, c.INFO, "ingest"))["duration"]
    rallies = group_hits([h["time"] for h in hits], duration, settings)
    c.write_json(work / c.RALLIES, {"settings": asdict(settings), "duration": duration, "rallies": rallies})
    return rallies
