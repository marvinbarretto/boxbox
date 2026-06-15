"""
Shared core for fetching + transcribing F1 team radio.

Two-source design (see team_radio.py header for the full why):
  FastF1   -> resolve session, map racing number -> driver code
  raw feed -> the actual mp3 clips (FastF1's processed API drops radio)
  Whisper  -> local transcription

Both the CLI (team_radio.py) and the web app (app.py) import from here so the
logic lives in exactly one place.
"""
import io
import json
import re
from functools import lru_cache
from pathlib import Path

import fastf1
import requests

LIVETIMING_BASE = "https://livetiming.formula1.com/static"
# F1's static host 403s without a browser-ish UA.
HEADERS = {"User-Agent": "Mozilla/5.0"}

fastf1.Cache.enable_cache(str(Path.home() / "Library/Caches/fastf1"))


def parse_jsonstream(text):
    """F1 .jsonStream files are one record per line: a timestamp prefix
    (HH:MM:SS.mmm) immediately followed by a JSON blob. Not a JSON array."""
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d{2}:\d{2}:\d{2}\.\d+)(.*)$", line)
        if not m:
            continue
        try:
            records.append(json.loads(m.group(2)))
        except json.JSONDecodeError:
            continue
    return records


def extract_captures(records):
    """"Captures" is sometimes a list, sometimes an index-keyed dict. Flatten both."""
    out = []
    for rec in records:
        caps = rec.get("Captures")
        if isinstance(caps, list):
            out.extend(caps)
        elif isinstance(caps, dict):
            out.extend(caps.values())
    return out


def get_clips(year, event, session, driver=None):
    """Return [{utc, code, url}] of radio clips for a session, sorted by time."""
    s = fastf1.get_session(year, event, session)
    # We only need the driver list / api_path, so load light.
    s.load(laps=False, telemetry=False, weather=False, messages=False)

    feed_root = LIVETIMING_BASE + s.api_path.replace("/static", "", 1)
    r = requests.get(feed_root + "TeamRadio.jsonStream", headers=HEADERS)
    if r.status_code != 200:
        return []  # no radio published for this session (common in 2026)

    captures = extract_captures(parse_jsonstream(r.text))
    res = s.results
    num_to_code = dict(zip(res["DriverNumber"].astype(str), res["Abbreviation"]))

    rows = []
    for c in captures:
        code = num_to_code.get(str(c.get("RacingNumber")), f"#{c.get('RacingNumber')}")
        if driver and code.upper() != driver.upper():
            continue
        rows.append({"utc": c.get("Utc", ""), "code": code, "url": feed_root + c["Path"]})
    rows.sort(key=lambda x: x["utc"])
    return rows


@lru_cache(maxsize=4)
def _get_model(name):
    """Cache loaded Whisper models so the web server pays the load cost once."""
    from faster_whisper import WhisperModel
    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe_url(url, model="base.en"):
    """Download a clip and transcribe it in-memory. Returns the text."""
    audio = requests.get(url, headers=HEADERS).content
    m = _get_model(model)
    # vad_filter trims dead air/static that otherwise produces hallucinated text.
    segments, _ = m.transcribe(io.BytesIO(audio), vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()
