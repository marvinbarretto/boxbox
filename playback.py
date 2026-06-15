"""
Core for the track + position playback feature.

Produces two things per session, both cached to disk as JSON:
  build_track(...)    -> track outline (rotated), corner markers, bounds
  build_playback(...) -> every car's X/Y resampled onto ONE shared SessionTime
                         grid (so frame N = all cars at time T), retired/absent
                         cars left as null and skipped by the frontend.

Coordinates are rotated by the circuit's rotation and Y-flipped here (SVG's Y
grows downward), so the frontend can plot raw values with no transform.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import fastf1

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(Path.home() / "Library/Caches/fastf1"))

CACHE_DIR = Path(__file__).parent / "playback_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _rotate_flip(x, y, angle_deg):
    """Apply FastF1 circuit rotation, then flip Y for SVG (Y-down) coordinates."""
    a = angle_deg / 180 * np.pi
    rx = x * np.cos(a) + y * np.sin(a)
    ry = -x * np.sin(a) + y * np.cos(a)
    return rx, -ry  # flip Y so the SVG isn't upside-down


def _key(year, event, session):
    return CACHE_DIR / f"{year}_{event}_{session}".replace(" ", "-").replace("/", "-")


def build(year, event, session, step=0.5):
    """Build (or load cached) {track, playback} for a session.
    step = grid spacing in seconds (0.5 = 2Hz). Frontend lerps between frames."""
    track_f = _key(year, event, session).with_suffix(".track.json")
    play_f = _key(year, event, session).with_suffix(".play.json")
    if track_f.exists() and play_f.exists():
        return {"track": json.loads(track_f.read_text()),
                "playback": json.loads(play_f.read_text()), "cached": True}

    s = fastf1.get_session(year, event, session)
    s.load(telemetry=True, weather=False, messages=False)
    ci = s.get_circuit_info()
    rot = ci.rotation

    # --- track outline from the fastest lap's position trace ---
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

    # --- shared-grid resample of all cars ---
    # Common clock from the union of all cars' position timestamps.
    starts, ends = [], []
    for df in s.pos_data.values():
        st = df["SessionTime"].dt.total_seconds()
        starts.append(st.min()); ends.append(st.max())
    t0, t1 = max(min(starts), 0), max(ends)
    grid = np.arange(t0, t1, step)

    num_to_code, num_to_color = {}, {}
    for _, d in s.results.iterrows():
        num_to_code[str(d["DriverNumber"])] = d["Abbreviation"]
        # results carries the team colour as a bare hex; fall back to grey.
        c = d.get("TeamColor")
        num_to_color[str(d["DriverNumber"])] = "#" + c if isinstance(c, str) and c else "#888"

    drivers = {}
    for num, df in s.pos_data.items():
        code = num_to_code.get(str(num))
        if not code:
            continue
        # The feed reports (0,0) as a "no GPS fix" sentinel (cars on the grid /
        # in the garage). Drop those rows so cars don't park at the origin.
        valid = ~((df["X"] == 0) & (df["Y"] == 0))
        st = df.loc[valid, "SessionTime"].dt.total_seconds().to_numpy()
        if len(st) < 2:
            continue
        # left/right=nan => before-first-fix and after-retirement become null
        xi = np.interp(grid, st, df.loc[valid, "X"].to_numpy(), left=np.nan, right=np.nan)
        yi = np.interp(grid, st, df.loc[valid, "Y"].to_numpy(), left=np.nan, right=np.nan)
        rx, ry = _rotate_flip(xi, yi, rot)
        drivers[code] = {
            "color": num_to_color.get(str(num), "#888"),
            "x": [None if np.isnan(v) else round(v) for v in rx],
            "y": [None if np.isnan(v) else round(v) for v in ry],
        }

    playback = {
        "step": step,
        "frames": len(grid),
        "t0": round(t0, 1),
        "drivers": drivers,
    }

    track_f.write_text(json.dumps(track))
    play_f.write_text(json.dumps(playback))
    return {"track": track, "playback": playback, "cached": False}


if __name__ == "__main__":
    import sys, time
    y, e, ss = (sys.argv[1:] + ["2024", "Brazil", "R"])[:3]
    t = time.perf_counter()
    out = build(int(y), e, ss)
    dt = time.perf_counter() - t
    pf = _key(int(y), e, ss).with_suffix(".play.json")
    print(f"built in {dt:.1f}s | cached={out['cached']}")
    print(f"frames={out['playback']['frames']} drivers={len(out['playback']['drivers'])}")
    print(f"track line points={len(out['track']['line'])} corners={len(out['track']['corners'])}")
    print(f"playback JSON size: {pf.stat().st_size/1024:.0f} KB")
