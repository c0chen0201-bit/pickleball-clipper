import shutil

import numpy as np
import pytest

from clipper import common, evaluate, hits, motion, rallies, score, synth
from clipper.cli import main


def to_16k(audio):
    t_out = np.arange(0, len(audio) / synth.SR, 1 / common.SAMPLE_RATE)
    return np.interp(t_out, np.arange(len(audio)) / synth.SR, audio).astype(np.float32)


def test_parse_time_and_timecode():
    assert common.parse_time("75.5") == 75.5
    assert common.parse_time("1:15.5") == 75.5
    assert common.parse_time("1:00:00") == 3600
    assert common.timecode(75.5) == "01:15.50"


def test_group_hits_makes_padded_rallies_and_drops_short_runs():
    times = [10, 11, 12, 13,   30, 31,   50, 51.5, 53]
    found = rallies.group_hits(times, duration=60)
    assert [(r["start"], r["end"], r["hit_count"]) for r in found] == [(9, 14.5, 4), (49, 54.5, 3)]


def test_group_hits_joins_rallies_whose_padding_overlaps():
    settings = rallies.RallySettings(max_gap=2.0)
    found = rallies.group_hits([1, 2, 3, 5.2, 6.2, 7.2], duration=20, settings=settings)
    assert len(found) == 1 and found[0]["hit_count"] == 6


def test_compare_counts_found_missed_and_false_alarms():
    result = evaluate.compare(detected=[(0, 10), (50, 55)], labels=[(1, 9), (20, 30)])
    assert result["found"] == 1
    assert result["missed"] == [(20, 30)]
    assert result["false_alarms"] == [(50, 55)]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_finds_every_rally_in_a_fake_game(seed):
    audio, truth, true_hits = synth.make_game(minutes=1.5, seed=seed)
    found = [h["time"] for h in hits.find_hits(to_16k(audio))]
    heard = sum(any(abs(f - t) < 0.05 for f in found) for t in true_hits)
    assert heard >= 0.95 * len(true_hits)
    detected = rallies.group_hits(found, len(audio) / synth.SR)
    result = evaluate.compare([(r["start"], r["end"]) for r in detected], truth)
    assert result["recall"] == 1.0
    assert result["false_alarms"] == []


def test_rank_puts_long_fast_rally_first():
    audio = np.zeros(60 * common.SAMPLE_RATE, dtype=np.float32)
    short = {"id": 1, "start": 0, "end": 5, "hit_count": 3, "hits": [1, 2, 3]}
    long_fast = {"id": 2, "start": 10, "end": 25, "hit_count": 10,
                 "hits": [11, 12, 13, 14, 15, 16, 16.4, 16.8, 17.2, 17.6]}
    ranked = score.rank([short, long_fast], audio)
    assert ranked[0]["id"] == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs FFmpeg")
def test_whole_pipeline_on_a_fake_game(tmp_path):
    video = tmp_path / "game.mp4"
    assert main(["sample", str(video), "--minutes", "0.75"]) == 0
    assert main(["run", str(video), "--top", "2"]) == 0
    clips = tmp_path / "game_clips"
    full = common.probe_duration(clips / "full_game.mp4")
    assert 0 < full < 45
    assert (clips / "highlight_reel.mp4").exists()
    assert len(list(clips.glob("highlight_*_rally*.mp4"))) == 2
    labels = tmp_path / "game_labels.csv"
    assert main(["eval", str(tmp_path / "game_work"), "--labels", str(labels)]) == 0


def _motion(duration, busy_spans, fps=10):
    scores = np.full(int(duration * fps), 0.1, dtype=np.float32)
    for s, e in busy_spans:
        scores[int(s * fps): int(e * fps)] = 0.9
    return scores


def test_motion_drops_hits_during_a_still_picture():
    times = [10, 11, 12, 13,   30, 31, 32]
    found = rallies.group_hits(times, 60, motion=_motion(60, [(9, 15)]))
    assert [r["start"] for r in found] == [9]


def test_motion_joins_split_rallies_and_keeps_two_hit_runs():
    times = [10, 11, 12,   18, 19, 20,   40, 41]
    found = rallies.group_hits(times, 60, motion=_motion(60, [(9, 22), (39, 43)]))
    assert [(r["start"], r["end"], r["hit_count"]) for r in found] == [(9, 21.5, 6), (39, 42.5, 2)]


def test_motion_only_stretches_are_kept_when_asked():
    settings = rallies.RallySettings(motion_only=True)
    motion = _motion(60, [(9, 15), (30, 40)])
    found = rallies.group_hits([10, 11, 12], 60, settings, motion)
    assert [(r["start"], r["end"], r["hit_count"]) for r in found] == [(9, 13.5, 3), (29, 41.5, 0)]
    assert len(rallies.group_hits([10, 11, 12], 60, motion=motion)) == 1


def test_motion_score_ignores_a_still_picture():
    scaled, stats = motion.score(np.full(600, 0.2, dtype=np.float32))
    assert not stats["usable"] and scaled.max() == 0
    diff = np.where(np.arange(600) % 100 < 50, 8.0, 0.5).astype(np.float32)
    scaled, stats = motion.score(diff)
    assert stats["usable"] and scaled[25] > 0.9 and scaled[75] < 0.1
