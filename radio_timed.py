"""
Feature #1 (Wave 2): place team-radio clips on the session clock.

Each radio clip carries only a UTC capture timestamp. To drop it onto the track
playback (which runs on FastF1 SessionTime) and the race-story lap axis, we map:

    sessionTime = clip_utc - t0_date        # same origin as pos_data SessionTime
    lap         = the driver's lap whose [LapStartTime, Time] contains sessionTime

Validated empirically: for Brazil 2024 R, 48/49 in-window clips land on a frame
where that driver's car actually has a position — i.e. the alignment is real,
not coincidental. Pre-green / post-flag clips get lap=None (correct) and are kept
so the UI can show them in a side log rather than silently dropping ~30%.

`t0_date` requires telemetry to be loaded, but the track page has already built
the playback cache for the same session, so the FastF1 telemetry is warm.
"""
import json
import warnings
from pathlib import Path

import pandas as pd
import fastf1

import f1radio

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(Path.home() / "Library/Caches/fastf1"))

CACHE_DIR = Path(__file__).parent / "radio_timed_cache"
CACHE_DIR.mkdir(exist_ok=True)

try:
    from flask import Blueprint, jsonify, request
    bp = Blueprint("radio_timed", __name__)
except Exception:  # flask always present in this app; guard keeps module importable
    bp = None


def get_timed_clips(year, event, session):
    key = CACHE_DIR / f"{year}_{event}_{session}.json".replace(" ", "-").replace("/", "-")
    if key.exists():
        return json.loads(key.read_text())

    s = fastf1.get_session(year, event, session)
    # telemetry=True so t0_date (the SessionTime origin) is available.
    s.load(laps=True, telemetry=True, weather=False, messages=False)
    t0 = pd.Timestamp(s.t0_date)  # tz-naive UTC

    # Pre-index each driver's laps as (start_sec, end_sec, lap_number) for mapping.
    laps_by_drv = {}
    for code, dl in s.laps.groupby("Driver"):
        rows = []
        for _, L in dl.iterrows():
            a, b = L["LapStartTime"], L["Time"]
            if pd.notna(a) and pd.notna(b):
                rows.append((a.total_seconds(), b.total_seconds(), int(L["LapNumber"])))
        laps_by_drv[code] = rows

    def lap_of(code, sec):
        for a, b, n in laps_by_drv.get(code, []):
            if a <= sec <= b:
                return n
        return None

    clips = f1radio.get_clips(year, event, session)
    out = []
    for c in clips:
        sec = (pd.Timestamp(c["utc"]).tz_convert(None) - t0).total_seconds()
        out.append({
            "utc": c["utc"],
            "code": c["code"],
            "url": c["url"],
            "t": round(sec, 1),          # SessionTime seconds (same origin as playback)
            "lap": lap_of(c["code"], sec),
        })
    out.sort(key=lambda x: x["t"])
    payload = {"event": event, "year": year, "session": session, "clips": out}
    key.write_text(json.dumps(payload))
    return payload


if bp is not None:
    @bp.route("/api/radio-timed")
    def api_radio_timed():
        try:
            return jsonify(get_timed_clips(
                int(request.args["year"]),
                request.args["event"],
                request.args["session"],
            ))
        except Exception as e:
            return jsonify({"error": str(e)}), 400
