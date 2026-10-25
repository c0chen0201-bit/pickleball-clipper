"""Step 5: rank rallies so the best ones go in the highlight reel.

Simple rules for now, from the audio alone:
  - more hits and a longer rally are better,
  - fast exchanges near the end (hands battles at the kitchen) are better,
  - a burst of noise right after the last hit (cheering, the "crowd pop") is better.
"""

from __future__ import annotations

import numpy as np

from . import common as c

WEIGHTS = {"hit_count": 0.35, "duration": 0.15, "end_speed": 0.25, "crowd_pop": 0.25}


def loudness_db(audio: np.ndarray, start: float, end: float) -> float:
    a = audio[int(start * c.SAMPLE_RATE): int(end * c.SAMPLE_RATE)]
    if len(a) == 0:
        return -100.0
    return float(10 * np.log10(np.mean(a.astype(np.float64) ** 2) + 1e-10))


def features(rally: dict, audio: np.ndarray, typical_db: float) -> dict:
    hits = rally["hits"]
    last_gaps = np.diff(hits[-5:])
    end_speed = float(1.0 / np.mean(last_gaps)) if len(last_gaps) else 0.0
    after = loudness_db(audio, hits[-1] + 0.15, hits[-1] + 2.5)
    return {
        "hit_count": len(hits),
        "duration": hits[-1] - hits[0],
        "end_speed": end_speed,            # hits per second over the last few shots
        "crowd_pop": after - typical_db,   # dB louder than usual after the point
    }


def rank(rallies: list[dict], audio: np.ndarray) -> list[dict]:
    """Returns the rallies with a 0-100 'score', best first."""
    if not rallies:
        return []
    typical = float(np.median([
        loudness_db(audio, t, t + 1) for t in np.arange(0, len(audio) / c.SAMPLE_RATE - 1, 1.0)
    ] or [-100.0]))
    feats = [features(r, audio, typical) for r in rallies]

    # Compare each rally with the others in the same game (0 = lowest, 1 = highest),
    # so the rules work whether the recording is loud or quiet.
    n = len(rallies)
    scores = np.zeros(n)
    for name, weight in WEIGHTS.items():
        values = np.array([f[name] for f in feats])
        order = values.argsort().argsort()
        scores += weight * (order / (n - 1) if n > 1 else np.ones(n))

    ranked = []
    for r, f, s in zip(rallies, feats, scores):
        ranked.append({**r, "score": round(float(s) * 100, 1),
                       "features": {k: round(float(v), 2) for k, v in f.items()}})
    ranked.sort(key=lambda r: -r["score"])
    return ranked
