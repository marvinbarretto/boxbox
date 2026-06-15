"""
Core for the per-lap telemetry dashboard feature (Feature 2).

Mirrors playback.py's pattern: load a FastF1 session, transform into compact
JSON-able dicts, cache the heavy per-lap payload to disk under telemetry_cache/.

Produces, for one chosen lap (default the driver's fastest):
  - channels: Speed / Throttle / Brake / nGear / DRS sampled over Distance,
    so the frontend can draw distance-axis line charts with no further maths.
  - track:    the lap's position trace, rotated with the SAME _rotate_flip
    transform playback.py uses, with a per-point Speed value so the frontend
    can colour the racing line by speed.

Supports an OPTIONAL second driver for side-by-side comparison; when present we
also return a speed delta (driver1.speed - driver2.speed) resampled onto a
common distance grid so the two laps line up despite different sample counts.

Exposes a Flask Blueprint `bp`:
  GET /api/telemetry?year=&event=&session=&driver=&lap=&driver2=
"""
import json
import warnings
from pathlib import Path

import numpy as np
import fastf1
from flask import Blueprint, jsonify, request

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(Path.home() / "Library/Caches/fastf1"))

CACHE_DIR = Path(__file__).parent / "telemetry_cache"
CACHE_DIR.mkdir(exist_ok=True)

# FastF1 reports DRS as a status code, not a boolean. Per FastF1's own docs
# (_api.py: "Odd DRS is Disabled, Even DRS is Enabled?"), codes 10/12/14 mean
# the flap is OPEN; 8 = detected/eligible (closed); 0/1/2/3/9 = off. We collapse
# to 0/1. NOTE: some cached laps (e.g. Brazil 2024 Q/R fastest laps) report a
# single constant DRS code for the whole lap, so the DRS strip can legitimately
# read flat — that's the upstream data, not a rendering bug.
DRS_OPEN_CODES = {10, 12, 14}


def _rotate_flip(x, y, angle_deg):
    """Apply FastF1 circuit rotation, then flip Y for SVG (Y-down) coordinates.
    Identical to playback.py so the track map orients the same way as playback."""
    a = angle_deg / 180 * np.pi
    rx = x * np.cos(a) + y * np.sin(a)
    ry = -x * np.sin(a) + y * np.cos(a)
    return rx, -ry  # flip Y so the SVG isn't upside-down


def _key(year, event, session, driver, lap, driver2):
    raw = f"{year}_{event}_{session}_{driver}_{lap}_{driver2 or 'none'}"
    return CACHE_DIR / raw.replace(" ", "-").replace("/", "-")


def _pick_lap(session, driver, lap):
    """Pick a lap for a driver. lap='fastest' -> pick_fastest(), else lap number."""
    drv = session.laps.pick_drivers(driver)
    if drv.empty:
        raise ValueError(f"no laps for driver '{driver}' in this session")
    if str(lap).lower() == "fastest":
        picked = drv.pick_fastest()
        if picked is None or (hasattr(picked, "empty") and picked.empty):
            raise ValueError(f"no valid fastest lap for '{driver}'")
        return picked
    n = int(lap)
    match = drv[drv["LapNumber"] == n]
    if match.empty:
        raise ValueError(f"driver '{driver}' has no lap {n}")
    return match.iloc[0]


def _team_color(session, driver):
    """Bare-hex team colour for a driver code -> '#RRGGBB', fallback '#888'."""
    r = session.results
    row = r[r["Abbreviation"] == driver]
    if not row.empty:
        c = row.iloc[0].get("TeamColor")
        if isinstance(c, str) and c:
            return "#" + c
    return "#888"


def _channels_for_lap(lap):
    """Extract the distance-indexed channels for one lap as plain Python lists.
    Rounded to keep the payload small (speed has no need for 6 decimals)."""
    car = lap.get_car_data().add_distance()
    dist = car["Distance"].to_numpy()
    speed = car["Speed"].to_numpy()
    return {
        # Distance in metres; round to 1dp — sub-metre precision is noise here.
        "dist": [round(float(v), 1) for v in dist],
        "speed": [round(float(v), 1) for v in speed],
        "throttle": [int(round(float(v))) for v in car["Throttle"].to_numpy()],
        # Brake is a bool in FastF1; emit 0/1 so the frontend draws a step trace.
        "brake": [1 if bool(v) else 0 for v in car["Brake"].to_numpy()],
        "gear": [int(v) for v in car["nGear"].to_numpy()],
        "drs": [1 if int(v) in DRS_OPEN_CODES else 0 for v in car["DRS"].to_numpy()],
    }


def _track_for_lap(lap, rotation):
    """Position trace for one lap, rotated/flipped, with a per-point speed value.

    pos_data (X,Y) and car_data (Speed) are separate streams sampled at
    different rates, so we align them on SessionTime: for each position sample
    we interpolate the car's speed at that instant. That gives one speed per
    drawn point, which the frontend maps to a colour."""
    pos = lap.get_pos_data()
    car = lap.get_car_data()
    pt = pos["SessionTime"].dt.total_seconds().to_numpy()
    ct = car["SessionTime"].dt.total_seconds().to_numpy()
    cspeed = car["Speed"].to_numpy()
    # Interpolate car speed onto the position-sample timestamps.
    pspeed = np.interp(pt, ct, cspeed)
    rx, ry = _rotate_flip(pos["X"].to_numpy(), pos["Y"].to_numpy(), rotation)
    return {
        "x": [round(float(v)) for v in rx],
        "y": [round(float(v)) for v in ry],
        "speed": [round(float(v), 1) for v in pspeed],
        "bounds": [round(float(rx.min())), round(float(ry.min())),
                   round(float(rx.max())), round(float(ry.max()))],
    }


def _lap_meta(lap, driver, color):
    lt = lap.get("LapTime")
    return {
        "driver": driver,
        "color": color,
        "lap": int(lap["LapNumber"]) if lap["LapNumber"] == lap["LapNumber"] else None,
        # LapTime is a pandas Timedelta; render as seconds (float) + mm:ss.mmm.
        "lap_time_s": round(lt.total_seconds(), 3) if lt is not None and lt == lt else None,
    }


def _delta(d1, d2):
    """Speed delta d1 - d2 resampled onto a common distance grid (so two laps
    with different sample counts and lengths still line up for plotting)."""
    lo = max(min(d1["dist"]), min(d2["dist"]))
    hi = min(max(d1["dist"]), max(d2["dist"]))
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, 400)
    s1 = np.interp(grid, d1["dist"], d1["speed"])
    s2 = np.interp(grid, d2["dist"], d2["speed"])
    return {
        "dist": [round(float(v), 1) for v in grid],
        "speed_delta": [round(float(a - b), 1) for a, b in zip(s1, s2)],
    }


def build(year, event, session, driver, lap="fastest", driver2=None):
    """Build (or load cached) telemetry payload for one lap (+ optional compare).

    Heavy on first call for a session (downloads car+position telemetry), then
    served from per-(driver,lap) disk cache like playback.build."""
    driver = driver.upper()
    driver2 = driver2.upper() if driver2 else None
    cache_f = _key(year, event, session, driver, lap, driver2).with_suffix(".json")
    if cache_f.exists():
        out = json.loads(cache_f.read_text())
        out["cached"] = True
        return out

    s = fastf1.get_session(year, event, session)
    s.load(telemetry=True, weather=False, messages=False)
    rot = s.get_circuit_info().rotation

    lap1 = _pick_lap(s, driver, lap)
    ch1 = _channels_for_lap(lap1)
    cars = [{
        **_lap_meta(lap1, driver, _team_color(s, driver)),
        "channels": ch1,
    }]
    track = _track_for_lap(lap1, rot)  # speed-coloured line from driver 1's lap

    delta = None
    if driver2 and driver2 != driver:
        # For a fair comparison default driver2 to THEIR fastest lap unless the
        # caller asked for a specific lap number (same `lap` arg applies to both).
        lap2 = _pick_lap(s, driver2, lap)
        ch2 = _channels_for_lap(lap2)
        cars.append({
            **_lap_meta(lap2, driver2, _team_color(s, driver2)),
            "channels": ch2,
        })
        delta = _delta(ch1, ch2)

    out = {
        "meta": {"year": year, "event": event, "session": session},
        "cars": cars,
        "track": track,
        "delta": delta,
        "cached": False,
    }
    cache_f.write_text(json.dumps(out))
    return out


bp = Blueprint("telemetry", __name__)


@bp.route("/api/telemetry")
def api_telemetry():
    """Per-lap telemetry channels + speed-coloured track map for one driver,
    with an optional second driver for comparison."""
    try:
        out = build(
            int(request.args["year"]),
            request.args["event"],
            request.args["session"],
            request.args["driver"],
            request.args.get("lap") or "fastest",
            request.args.get("driver2") or None,
        )
        return jsonify(out)
    except Exception as e:
        # Surface the real error to the UI like app.py does — bad driver codes,
        # missing laps and unknown events are all common, recoverable mistakes.
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import sys, time
    args = (sys.argv[1:] + ["2024", "Brazil", "Q", "VER", "fastest"])[:5]
    y, e, ss, drv, lp = args
    d2 = sys.argv[6] if len(sys.argv) > 6 else None
    t = time.perf_counter()
    out = build(int(y), e, ss, drv, lp, d2)
    dt = time.perf_counter() - t
    cf = _key(int(y), e, ss, drv.upper(), lp, d2.upper() if d2 else None).with_suffix(".json")
    print(f"built in {dt:.1f}s | cached={out['cached']}")
    for c in out["cars"]:
        print(f"  {c['driver']} lap {c['lap']} time={c['lap_time_s']}s "
              f"samples={len(c['channels']['dist'])} color={c['color']}")
    print(f"  track points={len(out['track']['x'])} "
          f"speed range={min(out['track']['speed'])}-{max(out['track']['speed'])}")
    print(f"  delta={'yes' if out['delta'] else 'no'}")
    print(f"  JSON size: {cf.stat().st_size/1024:.0f} KB")
