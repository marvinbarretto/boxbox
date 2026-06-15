"""
Core for the ghost-lap duel feature (Feature #2).

Animates TWO drivers' fastest (or chosen) laps on the same track outline, both
starting from the start/finish line at lap-relative time 0, so you SEE the gap
open up — like the F1 qualifying ghost overlay.

Produces, cached to disk as JSON per (year/event/session/driver/driver2/lap/lap2):
  - track: outline (rotated+flipped), corner markers, bounds  (mirrors playback)
  - drivers: [d1, d2] each resampled onto ONE shared LAP-RELATIVE TIME grid
             (every 0.1s from 0 to the slower lap's time). Frame N = both cars
             at the same elapsed lap time -> you watch one pull ahead. Positions
             past a driver's own lap end are null, so the slower ghost just stops.
  - delta: gap-vs-distance array for the live readout.

Unlike playback.py (which uses an absolute SessionTime grid shared across all
cars), here the grid is LAP-RELATIVE: both laps are aligned to t=0 at the line.
We still reuse playback's _rotate_flip transform, track dict shape, and the
nan->null masking idiom.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import fastf1
from flask import Blueprint, jsonify, request

# Reuse playback's coordinate transform verbatim (rotate by circuit rotation,
# then Y-flip for canvas Y-down space) so both features plot identically.
from playback import _rotate_flip

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(Path.home() / "Library/Caches/fastf1"))

CACHE_DIR = Path(__file__).parent / "ghost_cache"
CACHE_DIR.mkdir(exist_ok=True)

bp = Blueprint("ghost", __name__)


def _key(year, event, session, d1, d2, lap, lap2):
    raw = f"{year}_{event}_{session}_{d1}_{d2}_{lap}_{lap2}"
    return CACHE_DIR / raw.replace(" ", "-").replace("/", "-")


def _team_color(results, code):
    """Bare hex from results -> '#RRGGBB'; grey fallback when missing."""
    row = results[results["Abbreviation"] == code]
    if len(row):
        c = row.iloc[0].get("TeamColor")
        if isinstance(c, str) and c:
            return "#" + c
    return "#888"


def _pick_lap(s, code, lap_sel):
    """Return a lap for `code`. lap_sel is 'fastest' or a 1-based lap number."""
    laps = s.laps.pick_drivers(code)
    if len(laps) == 0:
        raise ValueError(f"no laps for driver {code}")
    if lap_sel in (None, "fastest", ""):
        lap = laps.pick_fastest()
    else:
        sub = laps[laps["LapNumber"] == int(lap_sel)]
        if len(sub) == 0:
            raise ValueError(f"driver {code} has no lap {lap_sel}")
        lap = sub.iloc[0]
    if lap is None or (hasattr(lap, "empty") and lap.empty):
        raise ValueError(f"no usable lap for driver {code}")
    return lap


def build(year, event, session, d1, d2, lap1="fastest", lap2="fastest", step=0.1):
    """Build (or load cached) the ghost-duel payload for two drivers.

    step = grid spacing in seconds along the LAP-RELATIVE time axis (0.1 = 10Hz).
    The frontend lerps between frames for smoothness."""
    if not d2:
        raise ValueError("driver2 is required for a duel")
    d1, d2 = d1.upper(), d2.upper()

    out_f = _key(year, event, session, d1, d2, lap1, lap2).with_suffix(".json")
    if out_f.exists():
        payload = json.loads(out_f.read_text())
        payload["cached"] = True
        return payload

    s = fastf1.get_session(year, event, session)
    s.load(telemetry=True, weather=False, messages=False)
    ci = s.get_circuit_info()
    rot = ci.rotation

    # --- track outline from the session's fastest lap pos trace (mirrors playback) ---
    pos = s.laps.pick_fastest().get_pos_data()
    tx, ty = _rotate_flip(pos["X"].to_numpy(), pos["Y"].to_numpy(), rot)
    cx, cy = _rotate_flip(ci.corners["X"].to_numpy(), ci.corners["Y"].to_numpy(), rot)
    track = {
        "line": [[round(x), round(y)] for x, y in zip(tx, ty)],
        "corners": [{"n": int(n), "x": round(x), "y": round(y)}
                    for n, x, y in zip(ci.corners["Number"], cx, cy)],
        "bounds": [round(min(tx.min(), cx.min())), round(min(ty.min(), cy.min())),
                   round(max(tx.max(), cx.max())), round(max(ty.max(), cy.max()))],
    }

    # --- gather each driver's telemetry (Time is already lap-relative, starts at 0) ---
    sel = [(d1, lap1), (d2, lap2)]
    tel = {}
    for code, lap_sel in sel:
        lap = _pick_lap(s, code, lap_sel)
        t = lap.get_telemetry()
        # Time is a timedelta from lap start; convert to seconds. It starts ~0.
        secs = t["Time"].dt.total_seconds().to_numpy()
        dist = t["Distance"].to_numpy()
        xr, yr = _rotate_flip(t["X"].to_numpy(), t["Y"].to_numpy(), rot)
        laptime = float(lap["LapTime"].total_seconds())
        tel[code] = {"secs": secs, "dist": dist, "x": xr, "y": yr,
                     "color": _team_color(s.results, code), "laptime": laptime}

    # --- shared lap-relative TIME grid: 0 .. slower lap's time, both aligned to 0 ---
    max_t = max(v["laptime"] for v in tel.values())
    grid = np.arange(0.0, max_t + step, step)

    drivers = []
    for code, _ in sel:
        d = tel[code]
        # right=nan: past this driver's own lap end -> null -> the ghost stops.
        xi = np.interp(grid, d["secs"], d["x"], left=np.nan, right=np.nan)
        yi = np.interp(grid, d["secs"], d["y"], left=np.nan, right=np.nan)
        di = np.interp(grid, d["secs"], d["dist"], left=np.nan, right=np.nan)
        drivers.append({
            "code": code,
            "color": d["color"],
            "laptime": round(d["laptime"], 3),
            "x": [None if np.isnan(v) else round(v) for v in xi],
            "y": [None if np.isnan(v) else round(v) for v in yi],
            "dist": [None if np.isnan(v) else round(v, 1) for v in di],
        })

    # --- delta-vs-distance: for distance d, delta = t2(d) - t1(d). ---
    # Positive => driver2 reached d later => driver2 is BEHIND. The frontend
    # reads this array at the leader's current distance for the live gap.
    a, b = tel[d1], tel[d2]
    dmax = min(a["dist"].max(), b["dist"].max())  # common distance span
    dgrid = np.arange(0.0, dmax, 10.0)            # every 10 metres
    t1_of_d = np.interp(dgrid, a["dist"], a["secs"])
    t2_of_d = np.interp(dgrid, b["dist"], b["secs"])
    delta = (t2_of_d - t1_of_d)

    payload = {
        "track": track,
        "step": step,
        "frames": len(grid),
        # convention: positive delta => d2 behind d1.
        "d1": d1, "d2": d2,
        "drivers": drivers,
        "delta": {
            "dist": [round(float(v), 1) for v in dgrid],
            "gap": [round(float(v), 3) for v in delta],
        },
        "cached": False,
    }

    out_f.write_text(json.dumps(payload))
    return payload


@bp.route("/api/ghost")
def api_ghost():
    """Two-driver ghost-lap duel payload. Heavy on first call for a session
    (downloads telemetry), then served from disk cache."""
    try:
        out = build(
            int(request.args["year"]),
            request.args["event"],
            request.args["session"],
            request.args["driver"],
            request.args.get("driver2"),
            request.args.get("lap", "fastest"),
            request.args.get("lap2", "fastest"),
        )
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import sys, time
    args = (sys.argv[1:] + ["2024", "Brazil", "Q", "VER", "NOR"])[:5]
    y, e, ss, da, db = args
    t = time.perf_counter()
    out = build(int(y), e, ss, da, db)
    dt = time.perf_counter() - t
    print(f"built in {dt:.1f}s | cached={out['cached']}")
    print(f"frames={out['frames']} step={out['step']}")
    for d in out["drivers"]:
        n = sum(1 for v in d["x"] if v is not None)
        print(f"  {d['code']} laptime={d['laptime']} color={d['color']} non-null={n}/{out['frames']}")
    g = out["delta"]["gap"]
    print(f"delta samples={len(g)} first={g[0]} last={g[-1]} (positive => {out['d2']} behind {out['d1']})")
