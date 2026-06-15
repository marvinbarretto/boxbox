"""
Pull F1 team-radio clips for a session.

Why this is a two-source script:
  - FastF1's *processed* Session API drops team radio entirely (no `.team_radio`).
    It's great for laps/telemetry/race-control but won't help here.
  - The radio lives only in F1's raw "live timing" static feed. FastF1 still
    knows the path to that feed via `session.api_path`, so we use FastF1 just to
    resolve the session + map racing numbers -> driver names, then go straight
    to the raw `TeamRadio.jsonStream` for the actual clips.

Dry-run by default (lists clips). Pass --live to download the mp3s.

Usage:
  python team_radio.py 2024 Monaco R
  python team_radio.py 2024 Monaco R --driver VER
  python team_radio.py 2024 Monaco R --live
"""
import argparse
import json
import re
import sys
from pathlib import Path

import fastf1
import requests

LIVETIMING_BASE = "https://livetiming.formula1.com/static"
# F1's static host 403s without a browser-ish UA.
HEADERS = {"User-Agent": "Mozilla/5.0"}


def parse_jsonstream(text):
    """F1 .jsonStream files are one record per line: a timestamp prefix
    (HH:MM:SS.mmm) immediately followed by a JSON blob. Not a JSON array.
    Analogous to JS: text.split('\\n').map(line => JSON.parse(line.slice(12)))."""
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip the leading "00:07:18.967" timestamp before the JSON starts
        m = re.match(r"^(\d{2}:\d{2}:\d{2}\.\d+)(.*)$", line)
        if not m:
            continue
        try:
            records.append(json.loads(m.group(2)))
        except json.JSONDecodeError:
            continue
    return records


def extract_captures(records):
    """Each record has a "Captures" field whose shape is inconsistent: sometimes
    a list, sometimes a dict keyed by index. Normalise both to a flat list."""
    out = []
    for rec in records:
        caps = rec.get("Captures")
        if isinstance(caps, list):
            out.extend(caps)
        elif isinstance(caps, dict):
            out.extend(caps.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("event", help="GP name or round number, e.g. Monaco")
    ap.add_argument("session", help="R, Q, FP1, Sprint, etc.")
    ap.add_argument("--driver", help="filter to a driver code, e.g. VER")
    ap.add_argument("--live", action="store_true", help="actually download mp3s")
    ap.add_argument("--transcribe", action="store_true",
                    help="run each clip through local Whisper (implies download)")
    ap.add_argument("--model", default="base.en",
                    help="faster-whisper model: tiny.en/base.en/small.en/medium.en")
    ap.add_argument("--out", default="radio_clips", help="download dir")
    args = ap.parse_args()

    fastf1.Cache.enable_cache(str(Path.home() / "Library/Caches/fastf1"))
    session = fastf1.get_session(args.year, args.event, args.session)
    # load() pulls the processed data; we only need it for the driver->name map
    # and to confirm api_path resolves. laps=False keeps it light.
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    feed_path = session.api_path  # e.g. /static/2024/.../2024-05-26_Race/
    radio_url = LIVETIMING_BASE + feed_path.replace("/static", "", 1) + "TeamRadio.jsonStream"

    r = requests.get(radio_url, headers=HEADERS)
    if r.status_code != 200:
        print(f"No radio feed (HTTP {r.status_code}) at {radio_url}", file=sys.stderr)
        sys.exit(1)

    captures = extract_captures(parse_jsonstream(r.text))

    # Map racing number -> driver code using FastF1's results table.
    res = session.results
    num_to_code = dict(zip(res["DriverNumber"].astype(str), res["Abbreviation"]))

    rows = []
    for c in captures:
        num = str(c.get("RacingNumber"))
        code = num_to_code.get(num, f"#{num}")
        if args.driver and code.upper() != args.driver.upper():
            continue
        clip_url = LIVETIMING_BASE + feed_path.replace("/static", "", 1) + c["Path"]
        rows.append((c.get("Utc", ""), code, clip_url))

    rows.sort()
    print(f"\n{len(rows)} radio clips for {args.year} {args.event} {args.session}"
          + (f" (driver {args.driver})" if args.driver else ""))
    for utc, code, url in rows:
        print(f"  {utc:30s} {code:5s} {url}")

    # Transcribing requires the files locally, so --transcribe implies download.
    if not (args.live or args.transcribe):
        print("\n(dry-run — pass --live to download, or --transcribe to download + transcribe)")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    saved = []  # (utc, code, path) in feed order, for transcription
    for utc, code, url in rows:
        name = f"{code}_{url.rsplit('/', 1)[-1]}"
        dest = out_dir / name
        if not dest.exists():
            audio = requests.get(url, headers=HEADERS)
            dest.write_bytes(audio.content)
            print(f"  saved {dest} ({len(audio.content)} bytes)")
        saved.append((utc, code, dest))
    print(f"\nDownloaded to {out_dir}/")

    if not args.transcribe:
        return

    # Lazy import — Whisper deps are heavy, only load them when actually asked.
    from faster_whisper import WhisperModel

    # int8 on CPU is the fast/light combo; fine for short, clean radio clips.
    print(f"\nLoading Whisper model '{args.model}' (first run downloads it)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    print("\n--- TRANSCRIPT ---")
    for utc, code, path in saved:
        # vad_filter trims dead air/static that otherwise produces hallucinated text.
        segments, _ = model.transcribe(str(path), vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        print(f"  {utc:30s} {code:5s} {text or '(no speech detected)'}")


if __name__ == "__main__":
    main()
