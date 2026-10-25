# pickleball-clipper

Turns a recorded pickleball game into two videos:

- **Full Game**: every rally back to back, with the walking, ball-fetching and waiting cut out.
- **Highlight Reel**: the 5 best rallies, plus each one as its own clip.

It finds rallies by *listening*: every paddle hit makes a sharp "pop", so a
run of pops a second or so apart is a rally, and long quiet stretches are dead
time. It also *watches* how much the picture moves, since players move much
more during a point than between points. No AI model is needed for this part.
Vertical (9:16) clips that follow the ball come later.

## Setting it up (once)

You need two things on your computer:

1. **Python 3.10 or newer.** Check with `python3 --version`. Get it from
   [python.org](https://www.python.org/downloads/) if you don't have it.
2. **FFmpeg**, the free tool that cuts and joins video.
   - Mac: install [Homebrew](https://brew.sh), then run `brew install ffmpeg`
   - Windows: run `winget install ffmpeg` in PowerShell

Then download this project and install it. In a terminal:

```
git clone https://github.com/c0chen0201-bit/pickleball-clipper.git
cd pickleball-clipper
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -e .
```

That gives you a new command called `clip`. (Run the `source` line again each
time you open a new terminal.)

## Using it

The one-step way, on a game you recorded:

```
clip run my_game.mp4
```

You'll get a folder called `my_game_clips/` with `full_game.mp4`,
`highlight_reel.mp4`, and the top 5 rallies as separate files.

### Trying it without a real game

```
clip sample                      # makes samples/fake_game.mp4 (3 minutes, fake pops)
clip run samples/fake_game.mp4
```

The fake game's picture is a test pattern with a clock in it, so you can see
which moments were kept.

### Step by step (to see and tune what's happening)

Each step saves its result in a work folder (`my_game_work/`), so you can
rerun any step on its own.

| Step | Command | What you get |
|---|---|---|
| 1 | `clip ingest my_game.mp4` | Clean copies of the video and its sound |
| 2 | `clip hits my_game.mp4` | `hits.json`, plus `hits_check.mp4`, which **beeps on every hit it heard**. Watch it to check by ear. |
| 2b | `clip motion my_game.mp4` | `motion.json`: how much the picture moves, moment by moment |
| 3 | `clip rallies my_game.mp4` | A list of rallies and how much of the game is kept. Add `--use-motion` after step 2b to use the picture too. |
| 3b | `clip eval my_game.mp4 --labels my_labels.csv` | How many of *your* rallies it found, and any false alarms |
| 4 | `clip export my_game.mp4` | `full_game.mp4` |
| 5 | `clip export my_game.mp4 --style highlights --top 5` | The highlight reel and the 5 clips |

**If it misses hits**, try `clip hits my_game.mp4 --above-bg-db 9` (lower =
more sensitive). **If it hears hits that aren't there**, raise it to 15.
**If rallies get split in two**, try `clip rallies my_game.mp4 --max-gap 3.5`.

**Indoors, or near other courts**, the sound alone is not enough: your own
hits are barely louder than the room. Try `clip motion` and then
`clip rallies --use-motion`. The picture then settles what the sound can't:
hits during a still picture (a break) are dropped, two runs of hits with busy
play between them are joined into one rally, each point plays out until the
players settle, and `--motion-only` also keeps busy stretches of play where
no hit was heard at all.

This only works if the camera mostly sees the court. If people walk right
past the phone between points, that counts as a lot of movement, so tell it
where the court is in the frame: `clip motion my_game.mp4 --court 0.2,0.3,0.8,0.75`
(left, top, right, bottom, as fractions of the picture).

### Following the ball (step 6)

`clip ball my_game.mp4` finds the ball in every frame with no training at all
(it looks for a small, round, fast-moving thing) and writes `ball_check.mp4`
with a circle following it. Then `clip rallies my_game.mp4 --use-ball` ends
each point exactly when the ball bounces for the second time or leaves the
picture.

To make it much more accurate, teach a small AI model what *your* ball looks
like. Install the AI tools once with `pip install -e '.[ai]'` (about 1 GB),
then:

```
clip label my_game.mp4 --out ball_labels --frames 300   # repeat for several games
clip train-ball ball_labels --out ball_model.pt          # a few hours on a laptop
clip ball my_game.mp4 --model ball_model.pt
```

`clip label` marks the ball automatically. The marks are not perfect, so it
also writes `ball_labels/review.html`: open it in a browser, click the ball
where a mark is wrong, press X on frames with no ball, and press "Save
fixes". Then `clip label-fix ball_labels fixes.json` writes your corrections
in. An hour of fixing makes a big difference to the model.

### Writing down your own rally times

To measure accuracy on a real game, make a spreadsheet with a `start` and
`end` column (seconds, or minutes:seconds), one row per rally, and save it as
CSV:

```
start,end
0:12.5,0:21.0
0:38,0:47.2
```

## How the highlights are ranked

For now it's simple rules, using sound only: more hits, longer rallies, fast
exchanges at the end (hands battles at the kitchen), and how loud it gets
right after the point (cheering). Each rally's score and the reasons are saved
in `ranking.json`.

## For developers

```
pip install -e '.[dev]'
pytest
```

Code lives in `clipper/`: `ingest.py`, `hits.py`, `rallies.py`, `score.py`,
`render.py`, `evaluate.py`, and `synth.py` (the fake game maker).
