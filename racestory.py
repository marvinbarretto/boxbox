"""
Core for the "race story" timeline feature.

Produces one JSON payload per session, cached to disk, that tells the story of a
race in three layers:
  - position-by-lap for every driver (the classic "spaghetti" chart data)
  - tyre stints per driver (compound + lap range), with pit-stop laps
  - a filtered race-control timeline (flags, safety cars, time penalties)

Mirrors playback.py's pattern: enable the FastF1 cache, load the session,
transform to plain JSON-able dicts, cache to disk under a sibling cache dir, and
expose a Flask Blueprint so app.py can mount it. We also keep playback.py's
NaN-handling discipline: pandas NaN/NaT must never reach json.dumps (it would
emit bare `NaN`, which is invalid JSON and makes the browser's .json() throw).
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1
from flask import Blueprint, jsonify, request

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(Path.home() / "Library/Caches/fastf1"))

CACHE_DIR = Path(__file__).parent / "racestory_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Matches "10 SECOND TIME PENALTY FOR CAR 50 (BEA)" — the only penalty rows we
# keep. NOTED / UNDER INVESTIGATION / WILL BE INVESTIGATED rows are the *same*
# incident logged repeatedly, so they'd flood the timeline; we drop them.
_PENALTY_RE = re.compile(r"(\d+)\s*SECOND\s+TIME\s+PENALTY\s+FOR\s+CAR\s+(\d+)\s*\(([A-Z]{3})\)", re.I)


def _key(year, event, session):
    return CACHE_DIR / f"{year}_{event}_{session}".replace(" ", "-").replace("/", "-")


def _driver_meta(s):
    """{DriverNumber(str) -> {code, color}} from results.
    Team colour is a bare hex in results (e.g. '3671C6'); prefix '#', fall back
    to grey. DriverNumber is str-keyed to match how laps reports it."""
    code, color = {}, {}
    for _, d in s.results.iterrows():
        num = str(d["DriverNumber"])
        code[num] = d["Abbreviation"]
        c = d.get("TeamColor")
        color[num] = "#" + c if isinstance(c, str) and c else "#888"
    return code, color


def _positions(s, code, color):
    """Per-driver position-by-lap: [{driver, color, positions:[{lap,pos}]}].
    Position is float in laps (NaN once a driver retires), so we skip NaN rows
    and int-cast the rest. Drivers are ordered by their best (lowest) position
    so the legend reads roughly top-to-bottom of the final order."""
    out = []
    for num, g in s.laps.groupby("DriverNumber"):
        num = str(num)
        c = code.get(num)
        if not c:
            continue
        pts = []
        for lap, pos in zip(g["LapNumber"], g["Position"]):
            if pd.isna(pos):
                continue
            pts.append({"lap": int(lap), "pos": int(pos)})
        if not pts:
            continue
        pts.sort(key=lambda p: p["lap"])
        out.append({
            "driver": c,
            "color": color.get(num, "#888"),
            "positions": pts,
            # sort key: best position reached (front-runners first)
            "_best": min(p["pos"] for p in pts),
        })
    out.sort(key=lambda d: d["_best"])
    for d in out:
        del d["_best"]
    return out


def _stints(s, code, color, total_laps):
    """Per-driver tyre stints + pit laps.
    A stint is a contiguous run on one compound (FastF1's Stint column already
    numbers these). We collapse each (driver, Stint) group to its compound and
    lap range. Pit laps come from PitInTime being non-null on a lap row (the lap
    on which the car came into the pits)."""
    out = []
    for num, g in s.laps.groupby("DriverNumber"):
        num = str(num)
        c = code.get(num)
        if not c:
            continue
        g = g.sort_values("LapNumber")
        stints = []
        for _, sg in g.groupby("Stint"):
            comp = sg["Compound"].dropna()
            if comp.empty:
                continue
            laps = sg["LapNumber"]
            stints.append({
                "compound": str(comp.iloc[0]).upper(),
                "startLap": int(laps.min()),
                "endLap": int(laps.max()),
            })
        if not stints:
            continue
        stints.sort(key=lambda x: x["startLap"])
        # Pit-in laps: a non-null PitInTime marks the lap the car pitted on.
        pits = [int(l) for l, t in zip(g["LapNumber"], g["PitInTime"]) if pd.notna(t)]
        out.append({
            "driver": c,
            "color": color.get(num, "#888"),
            "stints": stints,
            "pits": sorted(set(pits)),
            "_best": stints[0]["startLap"],
        })
    # Keep the same driver order as the position chart (final order-ish) by
    # best position; reuse results Position for a stable, meaningful order.
    final_pos = {}
    for _, d in s.results.iterrows():
        p = d.get("Position")
        final_pos[d["Abbreviation"]] = int(p) if pd.notna(p) else 999
    out.sort(key=lambda d: final_pos.get(d["driver"], 999))
    for d in out:
        del d["_best"]
    return out


def _events(s, total_laps):
    """Filtered race-control timeline: [{lap, type, label}].
    types: 'red' | 'green' | 'chequered' | 'sc' | 'vsc' | 'penalty' | 'yellow'.
    We aggressively filter the ~100 raw messages down to the ones worth showing:
      - Safety Car / VSC deploy + end (the real caution anchors)
      - RED / GREEN (start) / CHEQUERED flags
      - actual time penalties (not the NOTED/INVESTIGATION duplicates)
      - sector yellows collapsed to at most one marker per lap
    """
    rcm = s.race_control_messages
    events = []
    yellow_laps = set()  # collapse the flood of sector yellows to 1/lap

    for _, r in rcm.iterrows():
        lap = r.get("Lap")
        if pd.isna(lap):
            continue
        lap = int(lap)
        cat = r.get("Category")
        flag = r.get("Flag")
        msg = str(r.get("Message", ""))
        status = r.get("Status")

        if cat == "SafetyCar":
            # Status is DEPLOYED / ENDING / IN THIS LAP. VSC vs full SC from msg.
            is_vsc = "VIRTUAL" in msg.upper()
            st = str(status).title() if pd.notna(status) else ""
            label = ("VSC " if is_vsc else "SC ") + st
            events.append({"lap": lap, "type": "vsc" if is_vsc else "sc",
                           "label": label.strip()})
            continue

        if flag == "RED":
            events.append({"lap": lap, "type": "red", "label": "Red flag"})
            continue
        if flag == "CHEQUERED":
            events.append({"lap": lap, "type": "chequered", "label": "Chequered"})
            continue
        if flag == "GREEN" and "PIT EXIT" in msg.upper():
            # GREEN LIGHT - PIT EXIT OPEN marks the race start / restart green.
            events.append({"lap": lap, "type": "green", "label": "Green"})
            continue

        m = _PENALTY_RE.search(msg)
        if m:
            secs, _carnum, code = m.group(1), m.group(2), m.group(3)
            events.append({"lap": lap, "type": "penalty",
                           "label": f"{secs}s penalty {code}"})
            continue

        if flag in ("YELLOW", "DOUBLE YELLOW"):
            yellow_laps.add(lap)

    for lap in sorted(yellow_laps):
        events.append({"lap": lap, "type": "yellow", "label": "Yellow"})

    # De-dup identical (lap, type, label) and keep timeline in lap order.
    seen, uniq = set(), []
    for e in sorted(events, key=lambda e: (e["lap"], e["type"])):
        k = (e["lap"], e["type"], e["label"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)
    return uniq


def build(year, event, session):
    """Build (or load cached) the race-story payload for a session."""
    f = _key(year, event, session).with_suffix(".story.json")
    if f.exists():
        out = json.loads(f.read_text())
        out["cached"] = True
        return out

    s = fastf1.get_session(year, event, session)
    # laps for position/stints/pits, messages for race control. No telemetry.
    s.load(laps=True, messages=True, telemetry=False, weather=False)

    code, color = _driver_meta(s)
    total_laps = int(s.total_laps) if s.total_laps else int(s.laps["LapNumber"].max())

    out = {
        "event": str(event),
        "year": int(year),
        "session": str(session),
        "total_laps": total_laps,
        "positions": _positions(s, code, color),
        "stints": _stints(s, code, color, total_laps),
        "events": _events(s, total_laps),
    }

    # json.dumps with allow_nan default would emit invalid `NaN`; we've already
    # stripped NaN above, but keep the guard explicit so a future column change
    # fails loudly rather than producing un-parseable JSON.
    f.write_text(json.dumps(out, allow_nan=False))
    out["cached"] = False
    return out


# --- Flask Blueprint (app.py mounts this) ---------------------------------
bp = Blueprint("racestory", __name__)


@bp.route("/api/racestory")
def api_racestory():
    try:
        out = build(
            int(request.args["year"]),
            request.args["event"],
            request.args["session"],
        )
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import sys
    import time
    y, e, ss = (sys.argv[1:] + ["2024", "Brazil", "R"])[:3]
    t = time.perf_counter()
    out = build(int(y), e, ss)
    dt = time.perf_counter() - t
    pf = _key(int(y), e, ss).with_suffix(".story.json")
    print(f"built in {dt:.1f}s | cached={out['cached']} | total_laps={out['total_laps']}")
    print(f"drivers(pos)={len(out['positions'])} stints-rows={len(out['stints'])} events={len(out['events'])}")
    print(f"events: {[ (e['lap'], e['type'], e['label']) for e in out['events'] ]}")
    print(f"story JSON size: {pf.stat().st_size/1024:.0f} KB")
