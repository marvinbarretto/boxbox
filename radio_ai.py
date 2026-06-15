"""
Core for Feature #9: LLM-cleaned & classified team radio.

Pipeline per clip:
  f1radio.get_clips      -> the raw mp3 clip URLs (+ driver codes, timestamps)
  f1radio.transcribe_url -> local Whisper raw transcript (slow, CPU)
  OpenRouter chat call   -> a cleaned transcript + category + sentiment + tldr

Marvin's stack routes ALL models through OpenRouter (never a direct provider
API), so the LLM step is a single POST to OpenRouter's OpenAI-compatible
chat/completions endpoint, authed by the OPENROUTER_API_KEY env var.

Two expensive stages, two independent caches (see _load_cache / _save_cache):
  - The Whisper transcript is cached the moment we have it.
  - The LLM result is cached ONLY when the call succeeded.
That split matters: if there's no API key right now, every clip degrades to a
"raw transcript / other / neutral" placeholder. We must NOT freeze that
placeholder to disk as if it were a real LLM result — otherwise adding a key
later would never re-run the LLM (the costly Whisper step stays cached either
way). So on each read we re-attempt the LLM for any clip that has a transcript
but no *successful* LLM result yet.

Mirrors playback.py / racestory.py: enable FastF1 cache, return JSON-able
dicts, cache under a sibling *_cache/ dir, expose a Flask Blueprint `bp`.
"""
import json
import os
from pathlib import Path

import requests
from flask import Blueprint, jsonify, request

import f1radio

CACHE_DIR = Path(__file__).parent / "radio_ai_cache"
CACHE_DIR.mkdir(exist_ok=True)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Cheap + fast default. Configurable via the ?model= query param / form field.
# NOTE: this is the *LLM* model — distinct from the Whisper model below, which
# stays fixed. Don't let one leak into the other.
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"  # verified valid OpenRouter slug (dated API id is NOT)
WHISPER_MODEL = "base.en"

# Allowed enum values — anything off-list from the LLM is coerced to these.
CATEGORIES = {"strategy", "complaint", "incident", "celebration", "info", "other"}
SENTIMENTS = {"positive", "neutral", "negative"}

# Short glossary of common F1 radio terms / STT-correction hints fed to the LLM
# so it knows the domain (e.g. "box" = pit, not a literal box).
GLOSSARY = (
    "F1 team-radio glossary: 'box'/'box box' = come into the pits; "
    "'DRS' = drag reduction system; 'undercut'/'overcut' = pit strategy; "
    "'plank'/'graining'/'deg' = tyre wear; 'delta' = target time gap; "
    "'safety car'/'VSC' = virtual safety car; 'PB' = personal best; "
    "'OK mate', 'get in there' = celebration; 'P1'..'P20' = race position; "
    "'hards/mediums/softs' = tyre compounds. Whisper often mishears 'lap' as "
    "'now/up', driver surnames, and three-letter codes — fix these from context."
)


# --- OpenRouter LLM step ---------------------------------------------------

def _strip_fences(text):
    """Models often wrap JSON in ```json ... ``` fences. Strip them before parse."""
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence line (``` or ```json) and any trailing fence
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _coerce(obj, raw):
    """Force an LLM response dict into our strict shape with safe fallbacks."""
    clean = obj.get("clean") or raw
    cat = str(obj.get("category", "")).lower().strip()
    sent = str(obj.get("sentiment", "")).lower().strip()
    tldr = str(obj.get("tldr", "") or "")[:60]
    return {
        "clean": str(clean).strip() or raw,
        "category": cat if cat in CATEGORIES else "other",
        "sentiment": sent if sent in SENTIMENTS else "neutral",
        "tldr": tldr,
    }


def _degraded(raw):
    """The no-key / failure fallback: surface the raw transcript untouched."""
    return {"clean": raw, "category": "other", "sentiment": "neutral", "tldr": ""}


def _llm_clean(raw, codes, model, api_key):
    """One OpenRouter call -> strict {clean, category, sentiment, tldr}.

    Returns (result_dict, ok). ok=False means caller should degrade and NOT
    cache the result. Never raises — any failure becomes ok=False.
    """
    if not raw:
        # No speech detected — nothing to clean, but it's a legit "success".
        return {"clean": "", "category": "other", "sentiment": "neutral", "tldr": ""}, True

    glossary = GLOSSARY
    if codes:
        glossary += " Drivers in this session (3-letter codes): " + ", ".join(sorted(codes)) + "."

    system = (
        "You clean up Formula 1 team-radio speech-to-text transcripts and classify them. "
        + glossary
        + " Given a raw, error-prone transcript, return STRICT JSON only (no prose, no markdown) "
        'with exactly these keys: {"clean": "<corrected readable transcript>", '
        '"category": "strategy|complaint|incident|celebration|info|other", '
        '"sentiment": "positive|neutral|negative", "tldr": "<=8 word summary"}. '
        "Fix obvious STT errors (misheard words, driver names, 'lap' vs 'now'). "
        "Keep the meaning; do not invent content."
    )
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Raw transcript:\n" + raw},
                ],
                "temperature": 0.2,
                # Ask for a JSON object where the provider supports it.
                "response_format": {"type": "json_object"},
            },
            timeout=40,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        obj = json.loads(_strip_fences(content))
        return _coerce(obj, raw), True
    except Exception:
        # Any HTTP/parse/JSON error -> degrade this one clip, keep endpoint alive.
        return _degraded(raw), False


def _mood_summary(clips, model, api_key):
    """Optional final LLM call: one-line emotional arc of the race radio.
    Returns "" if no key or on any failure (mood counts still render)."""
    if not api_key:
        return ""
    lines = [f"- {c['code']} [{c['category']}/{c['sentiment']}]: {c['tldr'] or c['clean'][:50]}"
             for c in clips if c.get("clean")]
    if not lines:
        return ""
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You summarise the emotional arc of an F1 race "
                     "from its team-radio messages in ONE punchy sentence (max 25 words)."},
                    {"role": "user", "content": "Radio messages in order:\n" + "\n".join(lines)},
                ],
                "temperature": 0.4,
            },
            timeout=40,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


# --- per-session cache (one file per year/event/session/driver) ------------

def _cache_path(year, event, session, driver):
    name = f"{year}_{event}_{session}_{driver or 'ALL'}".replace(" ", "-").replace("/", "-")
    return CACHE_DIR / (name + ".json")


def _load_cache(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(path, cache):
    path.write_text(json.dumps(cache))


# --- build -----------------------------------------------------------------

def build(year, event, session, driver=None, model=None):
    """Return the full radio-AI payload for a session (optionally one driver).

    Cache shape, per clip URL:
      {"utc","code","raw","clean","category","sentiment","tldr","llm_ok": bool}
    raw (Whisper) is cached as soon as known; llm_ok marks whether the cleaned
    fields came from a *successful* LLM call (so we can retry degraded ones).
    """
    model = model or DEFAULT_MODEL
    api_key = os.environ.get("OPENROUTER_API_KEY")
    path = _cache_path(year, event, session, driver)
    cache = _load_cache(path)

    rows = f1radio.get_clips(year, event, session, driver)
    codes = sorted({r["code"] for r in rows})

    clips = []
    dirty = False
    for r in rows:
        url = r["url"]
        entry = cache.get(url)

        # Whisper transcript: compute once, then it's permanent.
        if entry is None or "raw" not in entry:
            raw = f1radio.transcribe_url(url, WHISPER_MODEL)
            entry = {"utc": r["utc"], "code": r["code"], "raw": raw,
                     "clean": raw, "category": "other", "sentiment": "neutral",
                     "tldr": "", "llm_ok": False}
            cache[url] = entry
            dirty = True

        # LLM step: run only if we have a key AND this clip hasn't a good result
        # yet. This lets a later key fill in clips degraded on an earlier run,
        # without ever recomputing the expensive transcription.
        if api_key and not entry.get("llm_ok"):
            res, ok = _llm_clean(entry["raw"], codes, model, api_key)
            entry.update(res)
            entry["llm_ok"] = ok
            cache[url] = entry
            dirty = True

        clips.append({
            "utc": entry["utc"], "code": entry["code"], "url": url,
            "raw": entry["raw"], "clean": entry["clean"],
            "category": entry["category"], "sentiment": entry["sentiment"],
            "tldr": entry["tldr"],
        })

    if dirty:
        _save_cache(path, cache)

    # Mood: counts per category + sentiment (always), plus an optional LLM arc.
    cat_counts, sent_counts = {}, {}
    for c in clips:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
        sent_counts[c["sentiment"]] = sent_counts.get(c["sentiment"], 0) + 1

    return {
        "llm_available": bool(api_key),
        "model": model,
        "count": len(clips),
        "mood": {
            "categories": cat_counts,
            "sentiments": sent_counts,
            "summary": _mood_summary(clips, model, api_key),
        },
        "clips": clips,
    }


# --- Flask Blueprint (app.py mounts this) ---------------------------------
bp = Blueprint("radio_ai", __name__)


@bp.route("/api/radio-ai")
def api_radio_ai():
    try:
        out = build(
            int(request.args["year"]),
            request.args["event"],
            request.args["session"],
            request.args.get("driver") or None,
            request.args.get("model") or None,
        )
        return jsonify(out)
    except Exception as e:
        # Match app.py's pattern — surface the real error to the UI.
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import sys
    import time
    y, e, ss, dr = (sys.argv[1:] + ["2024", "Brazil", "R", "VER"])[:4]
    t = time.perf_counter()
    out = build(int(y), e, ss, dr or None)
    print(f"built in {time.perf_counter()-t:.1f}s | llm_available={out['llm_available']}")
    print(f"clips={out['count']} categories={out['mood']['categories']} "
          f"sentiments={out['mood']['sentiments']}")
    if out["clips"]:
        c = out["clips"][0]
        print(f"first: {c['code']} [{c['category']}/{c['sentiment']}] "
              f"clean={c['clean'][:60]!r}")
