"""Step 2: find paddle hits by listening for the "pop".

A paddle hitting a plastic ball makes a click that is loud between about
1 and 4 kHz, starts within a few milliseconds and is gone within ~50 ms.
Voices, footsteps and wind are lower, slower or longer, so we look for
sounds in that band that jump up sharply and die away fast.
"""

from __future__ import annotations

import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import common as c

FRAME = 512          # 32 ms analysis window at 16 kHz
HOP = 80             # 5 ms between windows, so hit times are precise
BAND = (1000, 4000)  # where the pop is loudest, in Hz


@dataclass
class HitSettings:
    rise_db: float = 9.0        # how sharply the band must jump within ~30 ms
    above_bg_db: float = 12.0   # how far above the surrounding noise the peak must be
    decay_db: float = 6.0       # how much it must fade within 60 ms (pops are short)
    min_gap: float = 0.25       # two real hits can't be closer than this (seconds)


def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != c.SAMPLE_RATE:
            raise c.ClipperError(f"{path} is not mono 16-bit {c.SAMPLE_RATE} Hz; rerun 'clip ingest'.")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def band_energy_db(audio: np.ndarray, sr: int = c.SAMPLE_RATE) -> np.ndarray:
    """Loudness in the pop band, one value every HOP samples, in dB."""
    freqs = np.fft.rfftfreq(FRAME, 1 / sr)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    window = np.hanning(FRAME).astype(np.float32)
    n_frames = max(0, 1 + (len(audio) - FRAME) // HOP)
    out = np.empty(n_frames, dtype=np.float32)
    chunk = 20_000  # frames per batch, keeps memory small on hour-long games
    for start in range(0, n_frames, chunk):
        stop = min(n_frames, start + chunk)
        idx = np.arange(start, stop)[:, None] * HOP + np.arange(FRAME)[None, :]
        spec = np.abs(np.fft.rfft(audio[idx] * window, axis=1)) ** 2
        out[start:stop] = spec[:, band].sum(axis=1)
    return 10 * np.log10(out + 1e-10)


def background_db(energy: np.ndarray, seconds: float = 2.0) -> np.ndarray:
    """Typical noise level around each moment (the median of nearby blocks)."""
    per_sec = c.SAMPLE_RATE / HOP
    block = max(1, int(per_sec * 0.5))
    n_blocks = int(np.ceil(len(energy) / block))
    padded = np.pad(energy, (0, n_blocks * block - len(energy)), mode="edge")
    block_med = np.median(padded.reshape(n_blocks, block), axis=1)
    reach = max(1, int(seconds / 0.5 / 2))
    smooth = np.array([
        np.median(block_med[max(0, i - reach): i + reach + 1]) for i in range(n_blocks)
    ])
    return np.repeat(smooth, block)[: len(energy)]


def find_hits(audio: np.ndarray, settings: HitSettings = HitSettings()) -> list[dict]:
    energy = band_energy_db(audio)
    if len(energy) < 30:
        return []
    bg = background_db(energy)
    per_sec = c.SAMPLE_RATE / HOP
    look_back = int(0.03 * per_sec)
    look_ahead = int(0.06 * per_sec)

    # Quietest point in the 30 ms before each frame: a pop jumps up from there.
    prev = np.lib.stride_tricks.sliding_window_view(
        np.pad(energy, (look_back, 0), mode="edge"), look_back)[:-1].min(axis=1)
    rise = energy - prev
    later = np.pad(energy, (0, look_ahead), mode="edge")[look_ahead:]

    is_peak = np.zeros_like(energy, dtype=bool)
    is_peak[1:-1] = (energy[1:-1] >= energy[:-2]) & (energy[1:-1] > energy[2:])
    candidate = (
        is_peak
        & (rise >= settings.rise_db)
        & (energy - bg >= settings.above_bg_db)
        & (energy - later >= settings.decay_db)
    )

    # Keep the strongest hit when two are closer than min_gap.
    idx = np.flatnonzero(candidate)
    strength = (energy - bg)[idx]
    keep: list[int] = []
    min_frames = settings.min_gap * per_sec
    for i in idx[np.argsort(-strength)]:
        if all(abs(i - k) >= min_frames for k in keep):
            keep.append(i)
    keep.sort()
    # The frame's centre is the best guess for when the sound happened.
    return [
        {"time": round((i * HOP + FRAME / 2) / c.SAMPLE_RATE, 3),
         "strength_db": round(float(energy[i] - bg[i]), 1)}
        for i in keep
    ]


def detect(work: Path, settings: HitSettings = HitSettings(), beep_check: bool = True) -> list[dict]:
    audio = load_audio(c.need(work, c.AUDIO, "ingest"))
    hits = find_hits(audio, settings)
    c.write_json(work / c.HITS, {"settings": asdict(settings), "hits": hits})
    if beep_check:
        write_beep_check(work, [h["time"] for h in hits], len(audio) / c.SAMPLE_RATE)
    return hits


def write_beep_check(work: Path, times: list[float], duration: float) -> None:
    """Copy of the game with a short beep on every detected hit, to check by ear."""
    sr = c.SAMPLE_RATE
    track = np.zeros(int(duration * sr) + sr, dtype=np.float32)
    t = np.arange(int(0.08 * sr)) / sr
    beep = (0.3 * np.sin(2 * np.pi * 660 * t) * np.hanning(len(t))).astype(np.float32)
    for time in times:
        s = int(time * sr)
        track[s: s + len(beep)] += beep[: len(track) - s]
    beeps = work / "beeps.wav"
    with wave.open(str(beeps), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
    try:
        c.run_ffmpeg([
            "-i", str(c.need(work, c.VIDEO, "ingest")), "-i", str(beeps),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
            str(work / c.HITS_CHECK),
        ])
    finally:
        beeps.unlink(missing_ok=True)
