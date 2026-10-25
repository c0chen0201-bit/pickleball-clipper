import json
import shutil

import numpy as np
import pytest

from clipper import ball, common, evaluate, hits, label, motion, rallies, score, synth, train
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
    assert [(r["start"], r["end"], r["hit_count"]) for r in found] == [(9, 16, 4), (49, 56, 3)]


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
    assert [(r["start"], r["end"], r["hit_count"]) for r in found] == [(9, 25, 6), (39, 46, 2)]


def test_point_plays_out_until_the_players_settle():
    motion = _motion(60, [(9, 16)])       # play goes on 3 s after the last hit at 13
    found = rallies.group_hits([10, 11, 12, 13], 60, motion=motion)
    assert found[0]["end"] == 19           # settled at 16, plus pad_after
    motion = _motion(60, [(9, 30)])       # never settles: capped by max_tail
    found = rallies.group_hits([10, 11, 12, 13], 60, motion=motion)
    assert found[0]["end"] == 24


def test_motion_only_stretches_are_kept_when_asked():
    settings = rallies.RallySettings(motion_only=True)
    motion = _motion(60, [(9, 15), (30, 40)])
    found = rallies.group_hits([10, 11, 12], 60, settings, motion)
    assert [(r["start"], r["end"], r["hit_count"]) for r in found] == [(9, 18, 3), (29, 43, 0)]
    assert len(rallies.group_hits([10, 11, 12], 60, motion=motion)) == 1


def test_motion_score_ignores_a_still_picture():
    scaled, stats = motion.score(np.full(600, 0.2, dtype=np.float32))
    assert not stats["usable"] and scaled.max() == 0
    diff = np.where(np.arange(600) % 100 < 50, 8.0, 0.5).astype(np.float32)
    scaled, stats = motion.score(diff)
    assert stats["usable"] and scaled[25] > 0.9 and scaled[75] < 0.1


def _ball_video(path, frames=180, fps=30):
    """A ball flying across a noisy background, bouncing once, then leaving the picture."""
    import cv2
    w, h = 1280, 720
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    bg = rng.integers(60, 120, (h, w, 3), dtype=np.uint8)
    x, y, vx, vy = 100.0, 200.0, 9.0, -6.0
    truth, bounces = [], []
    for f in range(frames):
        frame = bg.copy()
        px = 400 + 2 * f  # a slow big "player"
        cv2.rectangle(frame, (px, 300), (px + 80, 600), (30, 30, 200), -1)
        if f < 150:
            vy += 0.45
            x += vx
            y += vy
            if y > 650:
                y, vy = 650, -vy * 0.7
                bounces.append(f / fps)
            cv2.circle(frame, (int(x), int(y)), 9, (60, 230, 230), -1)
            truth.append((f, x, y))
        out.write(frame)
    out.release()
    return truth, bounces


def test_ball_tracker_follows_a_flying_ball(tmp_path):
    truth, bounces = _ball_video(tmp_path / "ball.mp4")
    data = ball.track_video(tmp_path / "ball.mp4")
    pos = ball.positions_by_frame(data)
    close = sum(f in pos and abs(pos[f][0] - x) < 12 and abs(pos[f][1] - y) < 12 for f, x, y in truth)
    assert close >= 0.8 * len(truth)
    found = [e["time"] for e in data["events"] if e["type"] == "bounce"]
    assert all(any(abs(b - t) < 0.1 for t in found) for b in bounces)
    assert any(e["type"] == "off_screen" for e in data["events"])
    # The point ends at the second bounce (or at the exit if there is only one).
    end = ball.point_end(data, last_hit=0.0, max_tail=8)
    assert end is not None and abs(end - bounces[1]) < 0.1


def test_rallies_end_at_the_second_bounce():
    data = {"events": [{"time": 14.0, "type": "bounce"}, {"time": 15.2, "type": "bounce"},
                       {"time": 30.0, "type": "off_screen"}]}
    found = rallies.group_hits([10, 11, 12, 13], 60, ball=data)
    assert found[0]["end"] == 15.7
    found = rallies.group_hits([25, 26, 27, 28], 60, ball=data)
    assert found[0]["end"] == 30.5
    found = rallies.group_hits([40, 41, 42], 60, ball=data)   # nothing seen: fixed tail
    assert found[0]["end"] == 45


def test_label_export_and_fix_roundtrip(tmp_path):
    work = tmp_path / "g_work"
    work.mkdir()
    _ball_video(work / common.VIDEO)
    data = ball.track_video(work / common.VIDEO)
    common.write_json(work / common.BALL, data)
    out = tmp_path / "labels"
    result = label.export(work, out, n=20)
    assert result["positives"] >= 10 and (out / "review.html").exists()
    names = sorted(p.stem for p in (out / "images").glob("*.jpg"))
    assert len(names) == result["total"]
    with_ball = [n for n in names if (out / "labels" / f"{n}.txt").read_text().strip()]
    fixes = tmp_path / "fixes.json"
    fixes.write_text(json.dumps({with_ball[0]: {"has_ball": False}, names[-1]: {"x": 100, "y": 50, "has_ball": True}}))
    assert label.apply_fixes(out, fixes) == 2
    assert (out / "labels" / f"{with_ball[0]}.txt").read_text() == ""
    assert (out / "labels" / f"{names[-1]}.txt").read_text().startswith("0 0.078125 0.069444")
    data_yaml, _ = train.split(out)
    assert "names" in data_yaml.read_text() and (out / "train.txt").exists()
