"""
Core for the "overtake & battle detector + highlight reel" feature (Feature 5).

Detects on-track overtakes across a race at LAP RESOLUTION and emits them as a
list the frontend turns into a clickable highlight reel. Clicking an overtake
replays a few seconds of the (separately-served) /api/playback position data
centred on the pass.

Detection (lap-resolution v1)
-----------------------------
For each lap N>=2 we compare every driver's running Position against lap N-1.
An overtake event = driver A is AHEAD of driver B on lap N but was BEHIND on
lap N-1 (so A's relative position improved past B). We emit one event per
(A, B) pair passed, so a car gaining several places on one lap yields several
rows — each is a real *relative* pass.

Known limitations (be honest — this is lap-resolution, not GPS-proximity):
  - Time used is lap N's `Time`, i.e. when A *crosses the line* completing lap N.
    The actual pass happened somewhere earlier on that lap, so the reel window is
    only approximately centred on the move.
  - Position swaps caused by PIT STOPS (one car pits, another sails by) are NOT
    real on-track passes. We can't perfectly separate them at lap resolution, so
    we FLAG (don't drop) any event where either driver had a non-null
    PitInTime/PitOutTime on lap N or N-1, and let the UI hide them.
  - Lapped traffic and Safety-Car restart shuffles can also masquerade as passes;
    those are left in (a green-flag filter would need track-status correlation).

Mirrors playback.py / racestory.py: enable the FastF1 cache, load the session,
transform to plain JSON-able dicts, cache to disk, expose a Flask Blueprint.
NaN/NaT must never reach json.dumps (it would emit invalid bare `NaN`), so we
strip them while building.
"""
import json
import warnings
from pathlib import Path

import pandas as pd
import fastf1
from flask import Blueprint, jsonify, request

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(Path.home() / "Library/Caches/fastf1"))

CACHE_DIR = Path(__file__).parent / "overtakes_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _key(year, event, session):
    return CACHE_DIR / f"{year}_{event}_{session}".replace(" ", "-").replace("/", "-")


def _driver_meta(s):
    """{driver code -> '#hex' team colour}. results carries a bare hex (e.g.
    '3671C6'); prefix '#', fall back to grey. Keyed by Abbreviation because the
    laps DataFrame reports drivers by their 3-letter code in the Driver column."""
    color = {}
    for _, d in s.results.iterrows():
        c = d.get("TeamColor")
        color[d["Abbreviation"]] = "#" + c if isinstance(c, str) and c else "#888"
    return color


def _lap_order(laps, lap_num):
    """Snapshot of one lap: {driverCode -> {pos, tsec, pit}}.
    pos = running Position (int). tsec = session time at lap completion (seconds,
    None if missing). pit = True if this driver pitted in/out on this lap.
    Drivers with a NaN Position (retired before completing the lap) are skipped."""
    g = laps[laps["LapNumber"] == lap_num]
    out = {}
    for _, r in g.iterrows():
        pos = r["Position"]
        if pd.isna(pos):
            continue
        t = r["Time"]
        out[r["Driver"]] = {
            "pos": int(pos),
            "tsec": t.total_seconds() if pd.notna(t) else None,
            # PitInTime / PitOutTime are non-null only on the in-/out-lap.
            "pit": pd.notna(r["PitInTime"]) or pd.notna(r["PitOutTime"]),
        }
    return out


def _detect(laps, color):
    """Walk laps in order, diff each lap's order against the previous, emit one
    event per driver A that moved ahead of a driver B it was behind on. Sorted by
    lap then session time."""
    max_lap = int(laps["LapNumber"].max())
    events = []
    prev = _lap_order(laps, 1)
    for ln in range(2, max_lap + 1):
        cur = _lap_order(laps, ln)
        # Only drivers present on BOTH laps can be compared (a car appearing or
        # disappearing isn't an on-track pass).
        common = list(set(prev) & set(cur))
        for a in common:
            for b in common:
                if a == b:
                    continue
                # A ahead now (lower pos number) but was behind on the prior lap.
                if cur[a]["pos"] < cur[b]["pos"] and prev[a]["pos"] > prev[b]["pos"]:
                    pit = (cur[a]["pit"] or cur[b]["pit"]
                           or prev[a]["pit"] or prev[b]["pit"])
                    events.append({
                        "lap": ln,
                        # Lap-completion time of the GAINER as the reel anchor.
                        # Approximate: the move happened earlier on the lap.
                        "sessionTime": (round(cur[a]["tsec"], 1)
                                        if cur[a]["tsec"] is not None else None),
                        "gained": {"code": a, "color": color.get(a, "#888")},
                        "lost": {"code": b, "color": color.get(b, "#888")},
                        "posAfter": cur[a]["pos"],
                        "pit": bool(pit),
                    })
        prev = cur

    # Stable order for the reel: by lap, then by session time within the lap.
    events.sort(key=lambda e: (e["lap"], e["sessionTime"] if e["sessionTime"] is not None else 0))
    return events


def build(year, event, session):
    """Build (or load cached) the overtake payload for a session."""
    f = _key(year, event, session).with_suffix(".overtakes.json")
    if f.exists():
        out = json.loads(f.read_text())
        out["cached"] = True
        return out

    s = fastf1.get_session(year, event, session)
    # Only need laps (Position/Time/pit columns) — no telemetry, no messages.
    s.load(laps=True, telemetry=False, weather=False, messages=False)

    color = _driver_meta(s)
    laps = s.laps
    total_laps = int(s.total_laps) if s.total_laps else int(laps["LapNumber"].max())

    out = {
        "event": str(event),
        "year": int(year),
        "session": str(session),
        "total_laps": total_laps,
        "overtakes": _detect(laps, color),
    }

    # allow_nan=False makes a stray NaN fail loudly rather than emit invalid JSON.
    f.write_text(json.dumps(out, allow_nan=False))
    out["cached"] = False
    return out


# --- Flask Blueprint (app.py mounts this) ---------------------------------
bp = Blueprint("overtakes", __name__)


@bp.route("/api/overtakes")
def api_overtakes():
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
    pf = _key(int(y), e, ss).with_suffix(".overtakes.json")
    ovs = out["overtakes"]
    non_pit = sum(1 for o in ovs if not o["pit"])
    print(f"built in {dt:.1f}s | cached={out['cached']} | total_laps={out['total_laps']}")
    print(f"overtakes={len(ovs)} (non-pit={non_pit})")
    for o in ovs[:8]:
        print(f"  L{o['lap']} {o['gained']['code']} > {o['lost']['code']} "
              f"P{o['posAfter']} {'[pit]' if o['pit'] else ''} t={o['sessionTime']}")
    print(f"overtakes JSON size: {pf.stat().st_size/1024:.0f} KB")
