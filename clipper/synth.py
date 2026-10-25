"""Makes a fake "game" video for testing, since we don't have a real recording yet.

The picture is FFmpeg's test pattern with a running clock, so you can see
which parts of the game ended up in each clip. The sound has paddle pops
grouped into rallies, plus things that should NOT count as hits: background
noise, ball bounces, people talking between points, cheering, and the odd
pop from a neighboring court. It also writes the true rally times as a
labels CSV, so 'clip eval' can score the detector.
"""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np

from . import common as c

SR = 44_100


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x))


def _pop(rng, amp: float) -> np.ndarray:
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    click = _bandpass(rng.standard_normal(n), 900, 5000) * np.exp(-t / 0.006)
    ring = np.sin(2 * np.pi * rng.uniform(1300, 2200) * t) * np.exp(-t / 0.012)
    s = click / (np.abs(click).max() + 1e-9) + 0.6 * ring
    return amp * s / np.abs(s).max()


def _bounce(rng, amp: float) -> np.ndarray:
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * rng.uniform(150, 260) * t) * np.exp(-t / 0.01)


def _syllable(rng, amp: float) -> np.ndarray:
    n = int(rng.uniform(0.12, 0.3) * SR)
    t = np.arange(n) / SR
    f0 = rng.uniform(110, 240)
    voice = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 16))
    return amp * voice * np.hanning(n) / 3


def _cheer(rng, amp: float) -> np.ndarray:
    n = int(rng.uniform(1.0, 2.0) * SR)
    env = np.minimum(1, np.arange(n) / (0.15 * SR)) * np.linspace(1, 0, n) ** 0.5
    return amp * _bandpass(rng.standard_normal(n), 300, 4000) * env


def _add(track: np.ndarray, sound: np.ndarray, at: float) -> None:
    s = int(at * SR)
    e = min(len(track), s + len(sound))
    if s < len(track):
        track[s:e] += sound[: e - s]


def make_game(minutes: float = 3.0, seed: int = 7):
    """Returns (audio at 44.1 kHz, true rallies as (start, end) seconds, true hit times)."""
    rng = np.random.default_rng(seed)
    duration = minutes * 60
    track = np.zeros(int(duration * SR))
    # Background: low rumble (wind, traffic) plus quieter hiss.
    track += 0.02 * _bandpass(rng.standard_normal(len(track)), 20, 400)
    track += 0.004 * rng.standard_normal(len(track))

    truth = []
    hit_times = []
    t = rng.uniform(4, 8)
    while True:
        n_hits = int(rng.integers(3, 15))
        gaps = rng.uniform(0.7, 1.5, n_hits - 1)
        if rng.random() < 0.4 and n_hits >= 6:  # fast hands at the kitchen to finish
            gaps[-4:] = rng.uniform(0.33, 0.5, len(gaps[-4:]))
        times = t + np.concatenate([[0], np.cumsum(gaps)])
        if times[-1] + 4 > duration:
            break
        for i, ht in enumerate(times):
            far_side = i % 2 == 1
            _add(track, _pop(rng, rng.uniform(0.15, 0.3) if far_side else rng.uniform(0.35, 0.7)), ht)
            if i < len(times) - 1:  # ball bounces before most shots
                _add(track, _bounce(rng, 0.2), ht + gaps[i] * rng.uniform(0.4, 0.7))
        if rng.random() < 0.3:
            _add(track, _cheer(rng, 0.12), times[-1] + 0.4)
        hit_times += [round(float(x), 3) for x in times]
        truth.append((round(times[0] - 0.5, 2), round(times[-1] + 1.0, 2)))

        dead = rng.uniform(7, 18)
        for _ in range(int(rng.integers(2, 7))):  # chatting between points
            _add(track, _syllable(rng, 0.15), times[-1] + rng.uniform(2, dead - 1))
        if rng.random() < 0.3:  # stray pop from the next court
            _add(track, _pop(rng, 0.12), times[-1] + rng.uniform(3, dead - 1))
        t = times[-1] + dead
    return np.clip(track, -1, 1), truth, hit_times


def write_sample(out: Path, minutes: float = 3.0, seed: int = 7) -> Path:
    c.require_ffmpeg()
    audio, truth, _ = make_game(minutes, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = out.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    try:
        c.run_ffmpeg([
            "-f", "lavfi", "-i", f"testsrc=size=1280x720:rate=30:duration={minutes * 60}",
            "-i", str(wav), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out),
        ])
    finally:
        wav.unlink(missing_ok=True)
    labels = out.with_name(out.stem + "_labels.csv")
    with labels.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end"])
        writer.writerows(truth)
    return labels
