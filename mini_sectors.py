"""
Core for the mini-sector speed heatmap feature.

Splits a lap into ~N (default 20) equal-distance mini-sectors, then answers
"who's fastest here?" per sector:
  - no driver2: across every driver with a result in the session, whichever
    driver has the highest average speed in a mini-sector "owns" it — colour
    the segment with that driver's team colour (a fastest-driver-by-sector map).
  - driver + driver2: head-to-head — colour each sector by whichever of the
    two is faster there, carrying the speed delta for a tooltip.

Reuses playback.py's _rotate_flip transform (same one telemetry.py and
ghost.py use) so the track outline orients identically to the other maps.

Exposes a Flask Blueprint `bp`:
  GET /api/mini-sectors?year=&event=&session=&driver=&driver2=&n=
"""
import json
import warnings
from pathlib import Path

import numpy as np
import fastf1
from flask import Blueprint, jsonify, request

from playback import _rotate_flip

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(Path.home() / "Library/Caches/fastf1"))

CACHE_DIR = Path(__file__).parent / "mini_sectors_cache"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_N = 20

bp = Blueprint("mini_sectors", __name__)


def _key(year, event, session, driver, driver2, n):
    raw = f"{year}_{event}_{session}_{driver or 'field'}_{driver2 or 'none'}_{n}"
    return CACHE_DIR / raw.replace(" ", "-").replace("/", "-")


def _team_color(results, code):
    """Bare-hex team colour for a driver code -> '#RRGGBB', fallback '#888'."""
    row = results[results["Abbreviation"] == code]
    if len(row):
        c = row.iloc[0].get("TeamColor")
        if isinstance(c, str) and c:
            return "#" + c
    return "#888"


def _fastest_lap(session, code):
    """The driver's fastest lap, or None if they have no valid lap."""
    laps = session.laps.pick_drivers(code)
    if laps.empty:
        return None
    lap = laps.pick_fastest()
    if lap is None or (hasattr(lap, "empty") and lap.empty):
        return None
    return lap


def _dist_speed(lap):
    """Distance + speed arrays for one lap's fastest-lap car data."""
    car = lap.get_car_data().add_distance()
    return car["Distance"].to_numpy(), car["Speed"].to_numpy()


def _sector_bounds(track_len, n):
    edges = np.linspace(0, track_len, n + 1)
    return list(zip(edges[:-1], edges[1:]))


def _avg_speed_in_sector(dist, speed, lo, hi, inclusive_hi):
    mask = (dist >= lo) & (dist <= hi if inclusive_hi else dist < hi)
    if not mask.any():
        return None
    return float(speed[mask].mean())


def build(year, event, session, driver=None, driver2=None, n=DEFAULT_N):
    """Build (or load cached) mini-sector payload.

    Splits the session's reference lap into `n` equal-distance mini-sectors
    and, per sector, determines the fastest presence:
      - driver2 given: head-to-head `driver` vs `driver2` (speed delta/sector)
      - else: fastest driver per sector across everyone with a session result
    """
    driver = driver.upper() if driver else None
    driver2 = driver2.upper() if driver2 else None
    n = int(n)
    cache_f = _key(year, event, session, driver, driver2, n).with_suffix(".json")
    if cache_f.exists():
        out = json.loads(cache_f.read_text())
        out["cached"] = True
        return out

    s = fastf1.get_session(year, event, session)
    s.load(telemetry=True, weather=False, messages=False)
    rot = s.get_circuit_info().rotation

    # Reference lap for the track outline + mini-sector distance grid: the
    # given driver's fastest lap, else the session's outright fastest lap.
    ref_code = driver or s.laps.pick_fastest()["Driver"]
    ref_lap = _fastest_lap(s, ref_code)
    if ref_lap is None:
        raise ValueError(f"no valid fastest lap for '{ref_code}'")
    ref_pos = ref_lap.get_pos_data()
    ref_car = ref_lap.get_car_data().add_distance()
    track_len = float(ref_car["Distance"].to_numpy()[-1])
    bounds = _sector_bounds(track_len, n)

    # Track outline (rotated+flipped), one point per position sample, tagged
    # with the mini-sector it falls in so the frontend can colour it segment
    # by segment. Position and car data are separate streams sampled at
    # different rates, so align on SessionTime like telemetry._track_for_lap.
    pt = ref_pos["SessionTime"].dt.total_seconds().to_numpy()
    ct = ref_car["SessionTime"].dt.total_seconds().to_numpy()
    cdist = ref_car["Distance"].to_numpy()
    pdist = np.interp(pt, ct, cdist)
    rx, ry = _rotate_flip(ref_pos["X"].to_numpy(), ref_pos["Y"].to_numpy(), rot)
    sector_idx = np.clip((pdist / track_len * n).astype(int), 0, n - 1)

    sectors = []
    if driver2:
        # --- head-to-head mode: colour each sector by who's faster + the delta ---
        lap1, lap2 = _fastest_lap(s, driver), _fastest_lap(s, driver2)
        if lap1 is None:
            raise ValueError(f"no valid fastest lap for '{driver}'")
        if lap2 is None:
            raise ValueError(f"no valid fastest lap for '{driver2}'")
        dist1, speed1 = _dist_speed(lap1)
        dist2, speed2 = _dist_speed(lap2)
        c1, c2 = _team_color(s.results, driver), _team_color(s.results, driver2)
        for i, (lo, hi) in enumerate(bounds):
            last = i == n - 1
            a = _avg_speed_in_sector(dist1, speed1, lo, hi, last)
            b = _avg_speed_in_sector(dist2, speed2, lo, hi, last)
            if a is None or b is None:
                sectors.append({"i": i, "lo": round(lo, 1), "hi": round(hi, 1),
                                 "winner": None, "delta": None, "color": "#888"})
                continue
            delta = round(a - b, 1)
            winner, color = (driver, c1) if delta >= 0 else (driver2, c2)
            sectors.append({"i": i, "lo": round(lo, 1), "hi": round(hi, 1),
                             "winner": winner, "delta": delta, "color": color})
        legend = [{"driver": driver, "color": c1}, {"driver": driver2, "color": c2}]
    else:
        # --- field mode: colour each sector by whichever driver was fastest there ---
        codes = [c for c in s.results["Abbreviation"] if isinstance(c, str) and c]
        per_driver = {}
        for code in codes:
            lap = _fastest_lap(s, code)
            if lap is not None:
                per_driver[code] = _dist_speed(lap)
        if not per_driver:
            raise ValueError("no drivers with valid laps in this session")
        colors = {code: _team_color(s.results, code) for code in per_driver}
        for i, (lo, hi) in enumerate(bounds):
            last = i == n - 1
            best_code, best_speed = None, -1.0
            for code, (dist, speed) in per_driver.items():
                avg = _avg_speed_in_sector(dist, speed, lo, hi, last)
                if avg is not None and avg > best_speed:
                    best_code, best_speed = code, avg
            sectors.append({"i": i, "lo": round(lo, 1), "hi": round(hi, 1),
                             "winner": best_code, "delta": None,
                             "color": colors.get(best_code, "#888")})
        legend = [{"driver": code, "color": colors[code]} for code in sorted(per_driver)]

    track = {
        "x": [round(float(v)) for v in rx],
        "y": [round(float(v)) for v in ry],
        "sector": [int(v) for v in sector_idx],
        "bounds": [round(float(rx.min())), round(float(ry.min())),
                   round(float(rx.max())), round(float(ry.max()))],
    }

    out = {
        "meta": {"year": year, "event": event, "session": session,
                  "driver": driver, "driver2": driver2, "n": n, "ref": ref_code},
        "track": track,
        "sectors": sectors,
        "legend": legend,
        "cached": False,
    }
    cache_f.write_text(json.dumps(out))
    return out


@bp.route("/api/mini-sectors")
def api_mini_sectors():
    """Mini-sector speed heatmap: fastest-driver-per-sector (field mode) or
    head-to-head delta-per-sector (driver + driver2 given)."""
    try:
        out = build(
            int(request.args["year"]),
            request.args["event"],
            request.args["session"],
            request.args.get("driver") or None,
            request.args.get("driver2") or None,
            request.args.get("n") or DEFAULT_N,
        )
        return jsonify(out)
    except Exception as e:
        # Surface the real error to the UI like the other features do — bad
        # driver codes and unknown events are common, recoverable mistakes.
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import sys, time
    argv = sys.argv[1:]
    y = argv[0] if len(argv) > 0 else "2024"
    e = argv[1] if len(argv) > 1 else "Brazil"
    ss = argv[2] if len(argv) > 2 else "Q"
    drv = argv[3] if len(argv) > 3 else None
    drv2 = argv[4] if len(argv) > 4 else None
    t = time.perf_counter()
    out = build(int(y), e, ss, drv, drv2)
    dt = time.perf_counter() - t
    print(f"built in {dt:.1f}s | cached={out['cached']}")
    print(f"sectors={len(out['sectors'])} track points={len(out['track']['x'])}")
    for sec in out["sectors"][:5]:
        print(f"  sector {sec['i']}: {sec['lo']}-{sec['hi']}m winner={sec['winner']} delta={sec['delta']}")
