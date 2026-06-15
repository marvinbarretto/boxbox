"""
Local web UI for browsing + transcribing F1 team radio.

Run:  .venv/bin/python app.py   then open http://127.0.0.1:5050

It's a thin HTTP layer over f1radio.py:
  GET  /api/clips?year=&event=&session=&driver=  -> list clips
  POST /api/transcribe   {url, model}            -> transcript for one clip
The mp3s are played directly from F1's host in the browser's <audio> tag
(cross-origin media playback doesn't need CORS), so we don't proxy audio.
"""
from flask import Flask, jsonify, request, send_from_directory

# Load .env (OPENROUTER_API_KEY) before feature modules read os.environ.
from dotenv import load_dotenv
load_dotenv()

import f1radio
import playback
import telemetry
import racestory
import radio_ai
import ghost
import overtakes
import radio_timed

app = Flask(__name__, static_folder="static", static_url_path="")
# Don't pretty-print — the playback payload is megabytes; compact ~3x smaller.
app.json.compact = True

# Each feature owns a Blueprint with its own /api routes, kept in its own module.
app.register_blueprint(telemetry.bp)
app.register_blueprint(racestory.bp)
app.register_blueprint(radio_ai.bp)      # LLM-cleaned & classified radio
app.register_blueprint(ghost.bp)         # ghost-lap duel
app.register_blueprint(overtakes.bp)     # overtake detector + reel
app.register_blueprint(radio_timed.bp)   # radio clips placed on the session clock


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/track")
def track_page():
    return send_from_directory("static", "track.html")


@app.route("/telemetry")
def telemetry_page():
    return send_from_directory("static", "telemetry.html")


@app.route("/racestory")
def racestory_page():
    return send_from_directory("static", "racestory.html")


@app.route("/radio-ai")
def radio_ai_page():
    return send_from_directory("static", "radio_ai.html")


@app.route("/ghost")
def ghost_page():
    return send_from_directory("static", "ghost.html")


@app.route("/overtakes")
def overtakes_page():
    return send_from_directory("static", "overtakes.html")


@app.route("/api/playback")
def api_playback():
    """Track outline + all-cars position playback for a session. Heavy on first
    call for a given session (downloads telemetry), then served from disk cache."""
    try:
        out = playback.build(
            int(request.args["year"]),
            request.args["event"],
            request.args["session"],
        )
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/clips")
def clips():
    try:
        rows = f1radio.get_clips(
            int(request.args["year"]),
            request.args["event"],
            request.args["session"],
            request.args.get("driver") or None,
        )
        return jsonify({"clips": rows})
    except Exception as e:
        # Surface the real error to the UI — bad event names etc. are common.
        return jsonify({"error": str(e)}), 400


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    body = request.get_json(force=True)
    try:
        text = f1radio.transcribe_url(body["url"], body.get("model", "base.en"))
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # threaded so a slow transcription doesn't block listing more clips
    app.run(port=5050, debug=True, threaded=True)
