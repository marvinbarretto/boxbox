"""
Offline decoder for F1 live-timing Position.z messages: the first step of live mode.

The SignalR feed delivers `Position.z` as base64 of a raw-deflate stream holding
JSON: {"Position": [{"Timestamp": ..., "Entries": {"<driver number>": {"X":..,"Y":..,"Z":..}}}]}.
No websocket here; this only turns one message into frames playback.py's renderer can plot.
"""
import base64
import json
import zlib

from playback import _rotate_flip


def decode_position_z(message):
    """Decode one Position.z payload (base64 -> raw deflate -> JSON).

    Returns [{"timestamp": str, "entries": {driver_number: {"X": float, "Y": float}}}],
    one item per sample in the message.
    """
    raw = zlib.decompress(base64.b64decode(message), -15)
    payload = json.loads(raw.decode("utf-8-sig"))
    return [
        {
            "timestamp": sample.get("Timestamp"),
            "entries": {
                str(num): {"X": e["X"], "Y": e["Y"]}
                for num, e in sample.get("Entries", {}).items()
                if "X" in e and "Y" in e
            },
        }
        for sample in payload.get("Position", [])
    ]


def frame_from_entries(entries, rotation, colors):
    """Turn one sample's entries into {code: {color, x, y}}, as playback does.

    colors maps driver number -> {"code": "VER", "color": "#3671C6"}. (0,0) is the
    feed's "no GPS fix" sentinel and is dropped, as are drivers missing from colors.
    """
    frame = {}
    for num, e in entries.items():
        info = colors.get(str(num))
        if not info or (e["X"] == 0 and e["Y"] == 0):
            continue
        rx, ry = _rotate_flip(e["X"], e["Y"], rotation)
        frame[info["code"]] = {
            "color": info.get("color", "#888"),
            "x": [round(float(rx))],
            "y": [round(float(ry))],
        }
    return frame
