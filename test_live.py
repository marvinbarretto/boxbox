import base64
import json
import zlib

from live import decode_position_z, frame_from_entries
from playback import _rotate_flip

COLORS = {
    "1": {"code": "VER", "color": "#3671C6"},
    "44": {"code": "HAM", "color": "#E80020"},
}


def _compress(payload):
    c = zlib.compressobj(wbits=-15)
    raw = c.compress(json.dumps(payload).encode()) + c.flush()
    return base64.b64encode(raw).decode()


def _message():
    return _compress({"Position": [{
        "Timestamp": "2026-06-14T13:00:00.123Z",
        "Entries": {
            "1": {"Status": "OnTrack", "X": 1000, "Y": 2000, "Z": 5},
            "44": {"Status": "OffTrack", "X": 0, "Y": 0, "Z": 0},
        },
    }]})


def test_round_trip_decodes_xy_and_timestamp():
    (sample,) = decode_position_z(_message())
    assert sample["timestamp"] == "2026-06-14T13:00:00.123Z"
    assert sample["entries"]["1"] == {"X": 1000, "Y": 2000}
    assert sample["entries"]["44"] == {"X": 0, "Y": 0}


def test_zero_zero_entry_is_dropped():
    (sample,) = decode_position_z(_message())
    frame = frame_from_entries(sample["entries"], 0, COLORS)
    assert set(frame) == {"VER"}


def test_rotation_matches_rotate_flip():
    (sample,) = decode_position_z(_message())
    frame = frame_from_entries(sample["entries"], 37.5, COLORS)
    rx, ry = _rotate_flip(1000, 2000, 37.5)
    assert frame["VER"] == {"color": "#3671C6", "x": [round(rx)], "y": [round(ry)]}
