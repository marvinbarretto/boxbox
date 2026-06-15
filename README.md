# f1-experiments

A local web app for exploring Formula 1 race data — telemetry, car positions,
team radio (with transcripts), and race analysis. Built on **FastF1** + F1's raw
live-timing feed + local **Whisper** + **OpenRouter** for LLM bits. No paid F1
data, no cloud — everything runs on your machine and caches to disk.

## Run it

```bash
cd ~/development/f1-experiments
.venv/bin/python app.py          # serves http://127.0.0.1:5050
```

First load of any session downloads telemetry from FastF1 (~10–20s), then it's
cached under the `*_cache/` dirs and is instant thereafter.

## Pages

| Route | What it does |
|-------|--------------|
| `/` | **Team radio** — browse + locally transcribe driver radio clips |
| `/track` | **Track playback** — all cars animated round the circuit, scrub + speed, **with radio firing on the timeline** (auto-pause → play clip → show transcript) |
| `/telemetry` | Speed / throttle / brake / gear / DRS traces + track coloured by speed; two-driver compare |
| `/racestory` | Position-by-lap "spaghetti" chart, tyre stints, race-control timeline, **+ a team-radio lane** |
| `/ghost` | Two drivers' fastest laps raced as ghosts with a live time delta |
| `/overtakes` | Detected on-track passes; click one to replay it on a mini track-map |
| `/radio-ai` | Whisper transcripts + LLM cleanup/classification + a race "radio mood" |

There's also a CLI: `team_radio.py` (list / download / transcribe radio clips).

## Architecture

One Flask app (`app.py`) + one module per feature. Each feature module owns:
- a **core function** that loads a FastF1 session, transforms the data, and
  **caches JSON to disk** (so the UI never recomputes), and
- a Flask **Blueprint** with its `/api/...` route(s).

Each page is a single self-contained `static/*.html` (hand-written CSS, no
Tailwind; hand-rolled canvas/SVG, no chart libraries).

```
app.py            Flask server: registers blueprints + serves the pages
f1radio.py        radio clip listing (raw timing feed) + Whisper transcription
playback.py       car positions resampled onto a shared SessionTime grid  -> /api/playback
telemetry.py      per-lap telemetry channels + speed-coloured track        -> /api/telemetry
racestory.py      positions/stints/pits/race-control                       -> /api/racestory
ghost.py          two laps aligned on a lap-relative time grid + delta     -> /api/ghost
overtakes.py      lap-resolution overtake detection                        -> /api/overtakes
radio_ai.py       Whisper + OpenRouter cleanup/classification + mood        -> /api/radio-ai
radio_timed.py    maps each clip's UTC -> SessionTime + lap                 -> /api/radio-timed
static/*.html     one page per feature
*_cache/          on-demand disk caches (gitignored, safe to delete)
```

### Key data facts
- **Track geometry / positions** come from FastF1; coordinates are rotated by the
  circuit's `rotation` and Y-flipped for screen space (`playback._rotate_flip`).
- **Radio clips** are *not* in FastF1's processed API — they live in F1's raw
  `TeamRadio.jsonStream` (mp3 URLs), fetched in `f1radio.py`.
- **Radio timing**: clips carry only a UTC timestamp; `radio_timed.py` maps it to
  the session clock via `sessionTime = clip_utc - t0_date` (same origin as the
  playback grid). Validated: 48/49 in-window clips land on the right car's frame.

## Configuration

`radio-ai` needs an OpenRouter key. It's read from the environment (loaded from a
gitignored **`.env`** at startup):

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Without it, `radio-ai` degrades gracefully to raw Whisper transcripts (no LLM
tags/mood) and shows a banner. The LLM model defaults to `anthropic/claude-haiku-4.5`
(a valid OpenRouter slug — the dated Anthropic API id is **not**). Override per
request with `?model=`.

> The `.env` holds a live secret and is gitignored. If it's ever exposed, rotate
> the key on the OpenRouter dashboard.

## Known limitations
- **2026 radio coverage**: F1 cut team-radio publishing sharply in 2026 — recent
  sessions may return few/zero clips. Historical (2023→) is fine.
- **Overtakes** are lap-resolution: one car gaining several places spawns several
  rows; the replay window centres near lap-end, not the exact pass; pit/SC/lapped
  shuffles aren't fully separated (pit swaps are flagged + hidden by default).
- **Radio timing** is feed-accurate, not frame-perfect — a clip may pulse a second
  or two off the real moment.
- **Ghost duel**: same-team pairs share a colour (still labelled).
- **First load** of a new session downloads telemetry before caching.
- Everything is a **local dev server** (Flask debug) — not for network exposure.

## Stack
Python 3.14, Flask, FastF1, faster-whisper (+ ctranslate2), matplotlib (data
validation), Playwright (headless verification). All in `.venv/`.
