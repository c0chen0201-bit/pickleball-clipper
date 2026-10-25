"""The 'clip' command. Run 'clip --help' to see every step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import common as c
from . import evaluate as ev
from . import hits as hit_mod
from . import ingest as ingest_mod
from . import motion as motion_mod
from . import rallies as rally_mod
from . import render, synth


def work_for(path: str) -> Path:
    """Accepts either the game video or its work folder."""
    p = Path(path)
    return p if p.is_dir() else c.default_work_dir(p)


def cmd_sample(a):
    labels = synth.write_sample(Path(a.out), a.minutes, a.seed)
    print(f"Made fake game {a.out} ({a.minutes:g} min)")
    print(f"True rally times are in {labels}")


def cmd_ingest(a):
    work = ingest_mod.ingest(Path(a.video), Path(a.work) if a.work else c.default_work_dir(Path(a.video)))
    info = c.read_json(work / c.INFO)
    print(f"Ingested {a.video} ({c.timecode(info['duration'])}) into {work}/")


def cmd_hits(a):
    work = work_for(a.game)
    settings = hit_mod.HitSettings(rise_db=a.rise_db, above_bg_db=a.above_bg_db, min_gap=a.min_gap)
    found = hit_mod.detect(work, settings, beep_check=not a.no_check)
    print(f"Heard {len(found)} paddle hits -> {work / c.HITS}")
    if not a.no_check:
        print(f"Listen to {work / c.HITS_CHECK}: it beeps on every hit it found.")


def cmd_motion(a):
    work = work_for(a.game)
    data = motion_mod.detect(work)
    if data["usable"]:
        busy = sum(v >= rally_mod.RallySettings.busy_above for v in data["score"]) / data["fps"]
        print(f"Measured movement -> {work / c.MOTION}. The picture is busy for about "
              f"{c.timecode(busy)} of the game.")
    else:
        print("The picture barely changes, so movement will be ignored.")


def cmd_rallies(a):
    work = work_for(a.game)
    settings = rally_mod.RallySettings(max_gap=a.max_gap, min_hits=a.min_hits,
                                       still_below=a.still_below, join_gap=a.join_gap,
                                       join_above=a.join_above, motion_only=a.motion_only)
    rallies = rally_mod.find(work, settings, use_motion=not a.no_motion)
    duration = c.read_json(work / c.INFO)["duration"]
    if c.read_json(work / c.RALLIES)["used_motion"]:
        print("Using both the sound and the picture.")
    else:
        print("Using the sound only (run 'clip motion' first to use the picture too).")
    kept = sum(r["end"] - r["start"] for r in rallies)
    for r in rallies:
        print(f"  rally {r['id']:3d}  {c.timecode(r['start'])} - {c.timecode(r['end'])}  {r['hit_count']} hits")
    print(f"Found {len(rallies)} rallies. Keeping {c.timecode(kept)} of {c.timecode(duration)} "
          f"({100 * kept / duration:.0f}%) -> {work / c.RALLIES}")


def cmd_eval(a):
    result = ev.evaluate(work_for(a.game), Path(a.labels))
    print(f"Found {result['found']} of {result['labeled']} labeled rallies ({100 * result['recall']:.0f}%).")
    print(f"False alarms (detected but not a real rally): {len(result['false_alarms'])}")
    for s, e in result["missed"]:
        print(f"  missed     {c.timecode(s)} - {c.timecode(e)}")
    for s, e in result["false_alarms"]:
        print(f"  false alarm {c.timecode(s)} - {c.timecode(e)}")


def cmd_export(a):
    work = work_for(a.game)
    out = Path(a.out) if a.out else work.with_name(work.name.removesuffix("_work") + "_clips")
    if a.style == "full":
        path = render.export_full(work, out)
        print(f"Full Game video -> {path}")
    else:
        reel, best = render.export_highlights(work, out, a.top)
        for place, r in enumerate(best, 1):
            print(f"  #{place}  rally {r['id']:3d}  {c.timecode(r['start'])}  "
                  f"{r['hit_count']} hits  score {r['score']}")
        print(f"Highlight reel -> {reel} (plus each clip on its own in {out}/)")


def cmd_run(a):
    video = Path(a.video)
    work = c.default_work_dir(video)
    if not (work / c.INFO).exists():
        cmd_ingest(argparse.Namespace(video=a.video, work=None))
    if not (work / c.HITS).exists():
        print("Listening for hits...")
        hit_mod.detect(work, beep_check=False)
    if not (work / c.MOTION).exists():
        print("Measuring movement...")
        motion_mod.detect(work)
    rally_mod.find(work)
    common = {"game": str(work), "out": a.out, "top": a.top}
    cmd_export(argparse.Namespace(style="full", **common))
    cmd_export(argparse.Namespace(style="highlights", **common))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clip", description="Turn a recorded pickleball game into clips.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="make a fake game video (with true rally times) for testing")
    s.add_argument("out", nargs="?", default="samples/fake_game.mp4")
    s.add_argument("--minutes", type=float, default=3.0)
    s.add_argument("--seed", type=int, default=7, help="change for a different fake game")
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("ingest", help="step 1: make working copies of a game video")
    s.add_argument("video")
    s.add_argument("--work", help="where to put working files (default: <video>_work/)")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("hits", help="step 2: find paddle hits from the sound")
    s.add_argument("game", help="the game video or its work folder")
    s.add_argument("--rise-db", type=float, default=hit_mod.HitSettings.rise_db,
                   help="lower = more sensitive (default %(default)s)")
    s.add_argument("--above-bg-db", type=float, default=hit_mod.HitSettings.above_bg_db,
                   help="lower = hears quieter hits, but more false hits (default %(default)s)")
    s.add_argument("--min-gap", type=float, default=hit_mod.HitSettings.min_gap)
    s.add_argument("--no-check", action="store_true", help="skip the beep check video")
    s.set_defaults(func=cmd_hits)

    s = sub.add_parser("motion", help="step 2b: measure how much the picture moves")
    s.add_argument("game", help="the game video or its work folder")
    s.set_defaults(func=cmd_motion)

    s = sub.add_parser("rallies", help="step 3: group hits into rallies")
    s.add_argument("game")
    s.add_argument("--max-gap", type=float, default=rally_mod.RallySettings.max_gap,
                   help="seconds of quiet that end a rally (default %(default)s)")
    s.add_argument("--min-hits", type=int, default=rally_mod.RallySettings.min_hits)
    s.add_argument("--no-motion", action="store_true", help="ignore the picture, use the sound only")
    s.add_argument("--still-below", type=float, default=rally_mod.RallySettings.still_below,
                   help="drop hits made while the picture is stiller than this, 0-1 (default %(default)s)")
    s.add_argument("--join-gap", type=float, default=rally_mod.RallySettings.join_gap,
                   help="join two rallies this close in seconds if play stays busy (default %(default)s)")
    s.add_argument("--join-above", type=float, default=rally_mod.RallySettings.join_above,
                   help="...busy meaning movement never drops below this, 0-1 (default %(default)s)")
    s.add_argument("--motion-only", action="store_true",
                   help="also keep busy stretches of play where no hits were heard")
    s.set_defaults(func=cmd_rallies)

    s = sub.add_parser("eval", help="compare found rallies with a CSV of real rally times")
    s.add_argument("game")
    s.add_argument("--labels", required=True)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("export", help="steps 4-5: write the Full Game video or a highlight reel")
    s.add_argument("game")
    s.add_argument("--style", choices=["full", "highlights"], default="full")
    s.add_argument("--top", type=int, default=5, help="how many rallies in the highlight reel")
    s.add_argument("--out", help="output folder (default: <video>_clips/)")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("run", help="do every step on one video")
    s.add_argument("video")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--out")
    s.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except c.ClipperError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
