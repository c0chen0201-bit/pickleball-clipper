"""Step 3: group hits into rallies. Everything between rallies is dead time.

With only the sound, a rally is a run of paddle hits close together. With
--use-motion (after 'clip motion'), the picture helps in five ways:

- a run of hits during a still picture (a towel break near a noisy court) is
  dropped;
- two runs with busy play between them (hits the microphone missed) are
  joined into one rally instead of being split in two;
- a short run of only two hits counts if the players were clearly playing;
- a busy stretch with no hits heard at all can be kept as a rally
  (off by default: turn on with --motion-only);
- a point plays out: after the last hit heard, the rally keeps going while
  the players are still moving, and ends once they settle (the ball has
  bounced twice or left the court), instead of a fixed second and a half.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import common as c
from . import motion as motion_mod


@dataclass
class RallySettings:
    max_gap: float = 2.5       # a longer silence than this ends the rally (seconds)
    min_hits: int = 3          # serve, return, third shot
    pad_before: float = 1.0    # show the server getting ready
    pad_after: float = 1.5     # show the point ending
    # The motion settings only matter when motion.json exists. Scores run
    # from 0 (the stillest the picture gets in this game) to 1 (the busiest).
    still_below: float = 0.25   # a run of hits with less average motion than this is not a rally
    join_gap: float = 10.0      # join two runs this close if the picture stays busy between them
    join_above: float = 0.35    # ...meaning the motion never drops below this in the gap
    motion_only: bool = False   # also keep busy stretches with no hits at all
    busy_above: float = 0.6     # ...if the motion stays above this
    busy_seconds: float = 4.0   # ...for at least this long
    settle_below: float = 0.3   # after the last hit, the point is over once motion drops below this
    settle_seconds: float = 0.7 # ...and stays there this long
    max_tail: float = 8.0       # never play out more than this after the last hit


def _runs(times: list[float], max_gap: float) -> list[list[float]]:
    runs: list[list[float]] = []
    for t in sorted(times):
        if runs and t - runs[-1][-1] <= max_gap:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


def play_out(motion: np.ndarray, last_hit: float, settings: RallySettings) -> float:
    """When the point ended: the first moment after the last hit that the players settle."""
    fps = motion_mod.FPS
    a = int(last_hit * fps)
    b = min(len(motion), int((last_hit + settings.max_tail) * fps))
    need = max(1, int(round(settings.settle_seconds * fps)))
    quiet = 0
    for i in range(a, b):
        quiet = quiet + 1 if motion[i] < settings.settle_below else 0
        if quiet >= need:
            return (i - need + 1) / fps
    return b / fps


def group_hits(times: list[float], duration: float, settings: RallySettings = RallySettings(),
               motion: np.ndarray | None = None) -> list[dict]:
    runs = _runs(times, settings.max_gap)

    if motion is not None:
        # Keep a run if it has enough hits, or if two hits came with busy play.
        kept = []
        for run in runs:
            busy = motion_mod.level(motion, run[0], run[-1] + settings.pad_after)
            if busy < settings.still_below:
                continue
            if len(run) >= settings.min_hits or (len(run) >= 2 and busy >= settings.busy_above):
                kept.append(run)
        runs = kept
        # Join runs whose gap is short and never goes quiet.
        joined: list[list[float]] = []
        for run in runs:
            if joined and run[0] - joined[-1][-1] <= settings.join_gap and \
                    motion_mod.level(motion, joined[-1][-1], run[0], "min") >= settings.join_above:
                joined[-1] += run
            else:
                joined.append(run)
        runs = joined
    else:
        runs = [run for run in runs if len(run) >= settings.min_hits]

    def end_of(run: list[float]) -> float:
        last = run[-1] if motion is None else play_out(motion, run[-1], settings)
        return min(duration, last + settings.pad_after)

    spans = [(max(0.0, run[0] - settings.pad_before), end_of(run), run) for run in runs]

    if motion is not None and settings.motion_only:
        for s, e in motion_mod.busy_stretches(motion, settings.busy_above, settings.busy_seconds):
            if not any(s < e2 and e > s2 for s2, e2, _ in spans):
                spans.append((max(0.0, s - settings.pad_before), min(duration, e + settings.pad_after), []))
        spans.sort()

    rallies: list[dict] = []
    for start, end, run in spans:
        if rallies and start <= rallies[-1]["end"]:
            # Padding made two rallies touch; join them so no frame plays twice.
            prev = rallies[-1]
            prev["end"] = max(prev["end"], end)
            prev["hits"] += run
            continue
        rallies.append({"start": start, "end": end, "hits": list(run)})

    for i, r in enumerate(rallies, 1):
        r["id"] = i
        r["start"] = round(r["start"], 3)
        r["end"] = round(r["end"], 3)
        r["hit_count"] = len(r["hits"])
    return [{k: r[k] for k in ("id", "start", "end", "hit_count", "hits")} for r in rallies]


def find(work: Path, settings: RallySettings = RallySettings(), use_motion: bool = False) -> list[dict]:
    hits = c.read_json(c.need(work, c.HITS, "hits"))["hits"]
    duration = c.read_json(c.need(work, c.INFO, "ingest"))["duration"]
    motion = None
    if use_motion:
        motion = motion_mod.load(work)
        if motion is None:
            raise c.ClipperError("No usable movement data. Run 'clip motion' first.")
    rallies = group_hits([h["time"] for h in hits], duration, settings, motion)
    c.write_json(work / c.RALLIES, {"settings": asdict(settings), "used_motion": motion is not None,
                                    "duration": duration, "rallies": rallies})
    return rallies
