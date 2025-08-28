import os, json, re, base64
from typing import List, Tuple, Optional
from fastapi import FastAPI, Request, UploadFile, Form, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import requests
from requests.exceptions import HTTPError

from spotify_oauth import auth_url, exchange_code_for_token, refresh_token as _refresh_token
from mood_map import analyze_image_to_mood

# ------------------ Setup ------------------

load_dotenv()
app = FastAPI()
templates = Jinja2Templates(directory="templates")

SPOTIFY_API = "https://api.spotify.com/v1"
PLAYLIST_PRIVACY = os.getenv("PLAYLIST_PRIVACY", "private").lower() == "public"

# ------------------ Tiny utils ------------------

def _bearer_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def _sanitize_id(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    tid = str(raw)
    if tid.startswith("spotify:track:"):
        tid = tid.split(":")[-1]
    tid = re.sub(r"\s+", "", tid)
    return tid if re.fullmatch(r"[A-Za-z0-9]{22}", tid) else None

# ------------------ Routes: basic UI + OAuth ------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login")
def login():
    state = "x"  # MVP
    return RedirectResponse(auth_url(state))

@app.get("/callback")
def callback(code: Optional[str] = None, state: Optional[str] = None):
    if not code:
        return HTMLResponse("<p>Missing code.</p>", status_code=400)

    tokens = exchange_code_for_token(code)

    html = """
    <!doctype html>
    <html><body>
      <p>Finishing Spotify sign-in…</p>
      <script>
        (function(){
          try {
            if (window.opener && window.opener !== window) {
              window.opener.postMessage({ type: 'spotify-auth', status: 'success' }, window.location.origin);
            }
          } catch (e) {}
          setTimeout(function(){
            if (window.opener && window.opener !== window) window.close();
            else window.location = "/";
          }, 200);
        })();
      </script>
    </body></html>
    """
    resp = HTMLResponse(html)
    resp.set_cookie("access_token", tokens["access_token"], httponly=True)
    if "refresh_token" in tokens:
        resp.set_cookie("refresh_token", tokens["refresh_token"], httponly=True)
    return resp

# ------------------ Spotify helpers ------------------

def _collect_vibe(access: str, ids: list[str], targets: dict, need: int,
                  pad_steps: list[float] = [0.18, 0.24, 0.30, 0.38, 0.50]) -> list[str]:
    """
    Fetch features for `ids`, then widen the vibe envelope over pad_steps until we have `need`.
    Always returns <= need, ranked by distance; enforces artist diversity at the end.
    """
    feats = _audio_features(access, ids)
    if not feats:
        return ids[:need]  # last-ditch: keep order if features temporarily missing

    for pad in pad_steps:
        filtered = _filter_by_envelope(feats, targets, pad=pad) or []
        if not filtered:
            continue
        ranked = _rank(filtered, targets)
        ranked_ids = [f["id"] for f in ranked if f.get("id")]
        picked = _enforce_diversity(access, ranked_ids, per_artist=2, limit=need)
        if len(picked) >= need or pad == pad_steps[-1]:
            return picked
    return []



def _bounds_from_target(tgt: dict, pad: float = 0.18) -> dict:
    """
    Compute min/max bands around targets. We bias energy/dance tighter for party-ish,
    and acousticness wider for chill-ish to avoid “wrong vibe” pulls.
    """
    v = float(tgt["valence"]); e = float(tgt["energy"])
    d = float(tgt["danceability"]); a = float(tgt["acousticness"])

    # dynamic tightening: if energy+dance high, shrink band; if low, widen acousticness band a bit
    e_pad = pad * (0.85 if (e > 0.7 and d > 0.65) else 1.0)
    d_pad = pad * (0.85 if (e > 0.7 and d > 0.65) else 1.0)
    a_pad = pad * (1.2  if (e < 0.45)                else 1.0)
    v_pad = pad

    clip = lambda x: max(0.0, min(1.0, x))

    return {
        "min_valence":      clip(v - v_pad), "max_valence":      clip(v + v_pad),
        "min_energy":       clip(e - e_pad), "max_energy":       clip(e + e_pad),
        "min_danceability": clip(d - d_pad), "max_danceability": clip(d + d_pad),
        "min_acousticness": clip(a - a_pad), "max_acousticness": clip(a + a_pad),
    }


def _filter_by_envelope(features: list[dict], tgt: dict, pad: float = 0.18) -> list[dict]:
    """
    Keep only tracks whose audio features fall inside the min/max bands.
    """
    b = _bounds_from_target(tgt, pad)
    out = []
    for f in features:
        if not f: 
            continue
        v = float(f.get("valence", 0.5))
        e = float(f.get("energy", 0.5))
        d = float(f.get("danceability", 0.5))
        a = float(f.get("acousticness", 0.5))
        if (b["min_valence"] <= v <= b["max_valence"] and
            b["min_energy"]  <= e <= b["max_energy"]  and
            b["min_danceability"] <= d <= b["max_danceability"] and
            b["min_acousticness"] <= a <= b["max_acousticness"]):
            out.append(f)
    return out


def _ensure_valid_access(request: Request) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (access_token, refresh_token). If /me 401s and we have a refresh,
    refresh once and return the new access.
    """
    access = request.cookies.get("access_token")
    refresh = request.cookies.get("refresh_token")
    if not access:
        return None, refresh
    try:
        r = requests.get(f"{SPOTIFY_API}/me", headers=_bearer_headers(access), timeout=10)
        if r.status_code == 401 and refresh:
            new_tokens = _refresh_token(refresh)
            access = new_tokens["access_token"]
    except requests.RequestException:
        pass
    return access, refresh

def _me(access: str) -> dict:
    r = requests.get(f"{SPOTIFY_API}/me", headers=_bearer_headers(access), timeout=15)
    r.raise_for_status()
    return r.json()

def _my_top_artists(access: str) -> list:
    r = requests.get(f"{SPOTIFY_API}/me/top/artists?limit=10", headers=_bearer_headers(access), timeout=15)
    r.raise_for_status()
    return r.json().get("items", []) or []

def _top_tracks(access: str, cap: int = 100) -> List[str]:
    ids: List[str] = []
    for tr in ("long_term", "medium_term", "short_term"):
        if len(ids) >= cap:
            break
        try:
            # paginate in 50s
            for offset in range(0, cap, 50):
                r = requests.get(
                    f"{SPOTIFY_API}/me/top/tracks",
                    headers=_bearer_headers(access),
                    params={"limit": 50, "offset": offset, "time_range": tr},
                    timeout=15
                )
                if r.status_code != 200:
                    break
                items = r.json().get("items", []) or []
                if not items:
                    break
                for t in items:
                    tid = t.get("id")
                    if tid:
                        ids.append(tid)
                if len(items) < 50:
                    break
        except Exception:
            pass
    # de-dup preserve order and cap
    seen, out = set(), []
    for tid in ids:
        if tid and tid not in seen:
            seen.add(tid); out.append(tid)
        if len(out) >= cap:
            break
    print(f"[PicToBeat] gathered top_tracks={len(out)}")
    return out

def _recently_played(access: str, cap: int = 100) -> List[str]:
    try:
        r = requests.get(
            f"{SPOTIFY_API}/me/player/recently-played",
            headers=_bearer_headers(access),
            params={"limit": min(50, cap)},
            timeout=15
        )
        if r.status_code != 200:
            print("[PicToBeat] recently_played status", r.status_code, r.text[:200])
            return []
        items = r.json().get("items", []) or []
        ids = []
        for it in items:
            tid = ((it.get("track") or {}).get("id")) if isinstance(it, dict) else None
            if tid:
                ids.append(tid)
        print(f"[PicToBeat] gathered recently_played={len(ids)}")
        return ids[:cap]
    except Exception as e:
        print("[PicToBeat] recently_played error", e)
        return []

def _saved_tracks(access: str, cap: int = 300) -> List[str]:
    ids: List[str] = []
    try:
        for offset in range(0, max(0, cap), 50):
            r = requests.get(
                f"{SPOTIFY_API}/me/tracks",
                headers=_bearer_headers(access),
                params={"limit": 50, "offset": offset},
                timeout=15
            )
            if r.status_code != 200:
                print("[PicToBeat] saved_tracks status", r.status_code, r.text[:200])
                break
            items = r.json().get("items", []) or []
            if not items:
                break
            for it in items:
                tr = (it or {}).get("track") or {}
                tid = tr.get("id")
                if tid:
                    ids.append(tid)
            if len(items) < 50:
                break
    except Exception as e:
        print("[PicToBeat] saved_tracks error", e)

    seen, out = set(), []
    for tid in ids:
        if tid and tid not in seen:
            seen.add(tid); out.append(tid)
        if len(out) >= cap:
            break
    print(f"[PicToBeat] gathered saved_tracks={len(out)}")
    return out

SP_API_BASE = "https://api.spotify.com/v1"

def _audio_features(access_token: str, track_ids: List[str] | List[dict]) -> List[dict]:
    """
    Robustly fetch audio features, skipping bad IDs and chunking requests.
    """
    ids = []
    for t in track_ids:
        tid = t if isinstance(t, str) else (t.get("id") if isinstance(t, dict) else None)
        tid = _sanitize_id(tid) if tid else None
        if tid:
            ids.append(tid)

    # dedupe while preserving order
    seen, cleaned = set(), []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            cleaned.append(tid)

    headers = {"Authorization": f"Bearer {access_token}"}
    feats: List[dict] = []

    for i in range(0, len(cleaned), 100):
        chunk = cleaned[i:i+100]
        try:
            r = requests.get(f"{SP_API_BASE}/audio-features", headers=headers,
                             params={"ids": ",".join(chunk)}, timeout=20)
            if r.status_code in (400, 403):
                # fallback one-by-one
                for tid in chunk:
                    r2 = requests.get(f"{SP_API_BASE}/audio-features/{tid}", headers=headers, timeout=15)
                    if r2.status_code == 200:
                        jd = r2.json()
                        if isinstance(jd, dict) and jd:
                            feats.append(jd)
                continue
            r.raise_for_status()
            data = r.json().get("audio_features") or []
            feats.extend([d for d in data if d])
        except requests.RequestException:
            continue
    return feats

def _create_playlist(access: str, user_id: str, name: str, desc: str) -> str:
    r = requests.post(f"{SPOTIFY_API}/users/{user_id}/playlists",
                      headers={**_bearer_headers(access), "Content-Type": "application/json"},
                      json={"name": name, "description": desc, "public": PLAYLIST_PRIVACY},
                      timeout=15)
    r.raise_for_status()
    return r.json()["id"]

def _add_items(access: str, playlist_id: str, track_ids: List[str]) -> dict:
    uris = [f"spotify:track:{i}" for i in track_ids if _sanitize_id(i)]
    if not uris:
        return {"snapshot_id": None}
    r = requests.post(f"{SPOTIFY_API}/playlists/{playlist_id}/tracks",
                      headers={**_bearer_headers(access), "Content-Type": "application/json"},
                      json={"uris": uris}, timeout=15)
    r.raise_for_status()
    return r.json()

def _rank(features: List[dict], target: dict) -> List[dict]:
    # weighted L2 distance to target profile
    W = {"valence": 1.2, "energy": 1.0, "acousticness": 1.0, "danceability": 0.8}
    def score(f: dict) -> float:
        s = 0.0
        for k, w in W.items():
            s += w * (float(f.get(k, 0)) - float(target.get(k, 0))) ** 2
        return s
    return sorted(features, key=score)

def _tracks_meta(access: str, track_ids: List[str]) -> dict:
    """
    Batch-fetch basic track metadata (artists, popularity) for diversity/ranking.
    Returns {track_id: {"artists": [artist_id,...], "popularity": int}}
    """
    meta = {}
    ids = [i for i in track_ids if _sanitize_id(i)]
    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        try:
            r = requests.get(
                f"{SPOTIFY_API}/tracks",
                headers=_bearer_headers(access),
                params={"ids": ",".join(chunk)},
                timeout=15
            )
            if r.status_code != 200:
                continue
            for t in r.json().get("tracks", []) or []:
                if not t:
                    continue
                tid = t.get("id")
                if not tid:
                    continue
                artists = [a.get("id") for a in (t.get("artists") or []) if a.get("id")]
                meta[tid] = {
                    "artists": artists or [],
                    "popularity": int(t.get("popularity", 0))
                }
        except Exception:
            continue
    return meta

def _enforce_diversity(access: str, ids: list[str], per_artist: int = 2, limit: int = 50) -> list[str]:
    """
    Limit to ≤ per_artist per primary artist, preserving order; cap to limit.
    """
    meta = _tracks_meta(access, ids[:600])
    out, seen, per = [], set(), {}
    for tid in ids:
        if tid in seen:
            continue
        artists = (meta.get(tid, {}) or {}).get("artists", []) or ["unknown"]
        primary = artists[0]
        if per.get(primary, 0) >= per_artist:
            continue
        seen.add(tid); out.append(tid)
        per[primary] = per.get(primary, 0) + 1
        if len(out) >= limit: break
    return out

# --- lightweight seeds + robust explorer (discovery slice) ---

def _user_seeds_simple(access: str) -> dict:
    try:
        arts = _my_top_artists(access)
    except Exception:
        arts = []
    artist_ids = [a.get("id") for a in arts if a.get("id")][:2]
    tracks = _top_tracks(access, 10)[:2]
    return {"artists": artist_ids, "tracks": tracks, "genres": ["pop", "dance", "edm"]}

def _explore_recommendations(
    access: str,
    seeds: dict,
    targets: dict,
    limit: int = 40,
    market: Optional[str] = None,
    pad_steps: list[float] = [0.18, 0.24, 0.30, 0.38, 0.50]
) -> list[str]:
    """
    Try recommendations with min/max bands that widen over pad_steps.
    If still thin, fall back to target_* only (no min/max), then to broad pop.
    Everything gets re-ranked client-side and diversity-limited.
    """
    def _call_with_bands(sa, st, sg, use_market, lim, pad):
        bands = _bounds_from_target(targets, pad=pad)
        params = {
            "limit": max(1, min(100, lim)),
            "seed_artists": ",".join(sa[:5]),
            "seed_tracks":  ",".join(st[:5]),
            "seed_genres":  ",".join(sg[:5]) or "pop",
            "min_popularity": 10,
            "target_valence":      float(targets["valence"]),
            "target_energy":       float(targets["energy"]),
            "target_danceability": float(targets["danceability"]),
            "target_acousticness": float(targets["acousticness"]),
            "min_valence":      bands["min_valence"],      "max_valence":      bands["max_valence"],
            "min_energy":       bands["min_energy"],       "max_energy":       bands["max_energy"],
            "min_danceability": bands["min_danceability"], "max_danceability": bands["max_danceability"],
            "min_acousticness": bands["min_acousticness"], "max_acousticness": bands["max_acousticness"],
        }
        if use_market and market:
            params["market"] = market
        r = requests.get(f"{SPOTIFY_API}/recommendations",
                         headers=_bearer_headers(access), params=params, timeout=20)
        if r.status_code != 200:
            return []
        ids = [t.get("id") for t in (r.json().get("tracks") or []) if t.get("id")]
        if not ids:
            return []
        feats = _audio_features(access, ids)
        feats = _filter_by_envelope(feats, targets, pad=pad) or feats
        ranked = _rank(feats, targets)
        return [f["id"] for f in ranked if f.get("id")]

    def _call_soft(sa, st, sg, use_market, lim):
        # no min/max bands, just targets — softer catch-all
        params = {
            "limit": max(1, min(100, lim)),
            "seed_artists": ",".join(sa[:5]),
            "seed_tracks":  ",".join(st[:5]),
            "seed_genres":  ",".join(sg[:5]) or "pop",
            "min_popularity": 10,
            "target_valence":      float(targets["valence"]),
            "target_energy":       float(targets["energy"]),
            "target_danceability": float(targets["danceability"]),
            "target_acousticness": float(targets["acousticness"]),
        }
        if use_market and market:
            params["market"] = market
        r = requests.get(f"{SPOTIFY_API}/recommendations",
                         headers=_bearer_headers(access), params=params, timeout=20)
        if r.status_code != 200:
            return []
        ids = [t.get("id") for t in (r.json().get("tracks") or []) if t.get("id")]
        if not ids:
            return []
        feats = _audio_features(access, ids)
        ranked = _rank(feats, targets) if feats else []
        return [f["id"] for f in (ranked if ranked else ids) if isinstance(f, dict) and f.get("id")] or ids

    sa = seeds.get("artists", []) or []
    st = seeds.get("tracks", [])  or []
    sg = seeds.get("genres", [])  or ["pop"]

    # Try stricter → looser pads
    for pad in pad_steps:
        attempts = [
            lambda: _call_with_bands(sa, st, sg, True,  limit*2, pad),
            lambda: _call_with_bands(sa, st, sg, False, limit*2, pad),
            lambda: _call_with_bands([], st, [], True,  limit*2, pad),
            lambda: _call_with_bands(sa, [], [], True,  limit*2, pad),
            lambda: _call_with_bands([], [], sg, True,  limit*2, pad),
        ]
        for fn in attempts:
            out = fn()
            if out:
                return out[:limit]

    # Soft fallback: just targets (no bands)
    soft_attempts = [
        lambda: _call_soft(sa, st, sg, True,  limit*2),
        lambda: _call_soft(sa, st, sg, False, limit*2),
        lambda: _call_soft([], [], ["pop"], False, limit*2),
    ]
    for fn in soft_attempts:
        out = fn()
        if out:
            return out[:limit]

    return []

# ------------------ Image → targets & AI tag expansion ------------------

def _normalize_targets(t: dict) -> dict:
    keys = ["valence", "energy", "danceability", "acousticness"]
    out = {}
    for k in keys:
        v = float(t.get(k, 0.5))
        out[k] = 0.0 if v < 0 else 1.0 if v > 1 else v
    return out

def _openai_image_targets(img_bytes: bytes) -> tuple[list[str], dict]:
    """
    Optional: uses OpenAI if OPENAI_API_KEY is set; else returns reasonable defaults.
    """
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        return ["upbeat", "party"], {"valence": 0.75, "energy": 0.9, "danceability": 0.9, "acousticness": 0.1}
    b64 = base64.b64encode(img_bytes).decode()
    prompt = (
        "You are selecting music for a Spotify playlist based ONLY on the image. "
        "Return JSON with keys: mood_words (array of 2-5 short words), "
        "valence, energy, danceability, acousticness (floats 0..1). "
        "Respond with only JSON."
    )
    payload = {
        "model": "gpt-4.1-mini",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_data": b64, "mime_type": "image/jpeg"}
            ]
        }]
    }
    try:
        r = requests.post("https://api.openai.com/v1/responses",
                          headers={"Authorization": f"Bearer {api}", "Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        j = r.json()
        text = ""
        for msg in j.get("output", []):
            for c in msg.get("content", []):
                if c.get("type") == "output_text":
                    text += c.get("text", "")
        data = json.loads(text.strip())
        words = [w.strip() for w in data.get("mood_words", []) if w.strip()]
        targets = _normalize_targets(data)
        if not words:
            words = ["bright", "energetic"]
        return words, targets
    except Exception:
        return ["upbeat", "party"], {"valence": 0.75, "energy": 0.9, "danceability": 0.9, "acousticness": 0.1}

def image_to_targets(img_bytes: bytes) -> tuple[list[str], dict]:
    try:
        mm = analyze_image_to_mood(img_bytes)  # {"mood_words":[...], "targets": {...}}
        words = (mm.get("mood_words") or [])
        targets = _normalize_targets(mm.get("targets") or {})
        if not words or sum(targets.values()) == 0:
            return _openai_image_targets(img_bytes)
        return words, targets
    except Exception:
        return _openai_image_targets(img_bytes)

def _ai_expand_tags(image_words: list[str], targets: dict) -> list[str]:
    """
    Uses OpenAI to expand mood words into concise, Gen-Z, playlist-friendly tags.
    Returns up to ~8 tags. If OPENAI_API_KEY missing or error, returns `image_words`.
    """
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        return image_words or []

    sys = (
        "You are helping title a Spotify playlist. Given mood words and audio targets "
        "(valence, energy, danceability, acousticness in 0..1), output ONLY JSON with:"
        '{"tags":[<up to 8 short, contemporary, casual, Gen-Z flavored tags>]} '
        "Keep tags 1–3 words, no hashtags, no punctuation. Example styles: "
        '"lit", "low-key", "dancefloor", "cozy vibes", "late-night", "moody", "feel-good".'
    )
    usr = {
        "image_words": image_words or [],
        "targets": {
            "valence": float(targets.get("valence", 0.5)),
            "energy": float(targets.get("energy", 0.5)),
            "danceability": float(targets.get("danceability", 0.5)),
            "acousticness": float(targets.get("acousticness", 0.5)),
        }
    }
    payload = {
        "model": "gpt-4.1-mini",
        "input": [
            {"role": "system", "content": [{"type": "text", "text": sys}]},
            {"role": "user",   "content": [{"type": "text", "text": json.dumps(usr)}]},
        ],
    }
    try:
        r = requests.post("https://api.openai.com/v1/responses",
                          headers={"Authorization": f"Bearer {api}",
                                   "Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=25)
        r.raise_for_status()
        j = r.json()
        text = ""
        for msg in j.get("output", []):
            for c in msg.get("content", []):
                if c.get("type") == "output_text":
                    text += c.get("text", "")
        data = json.loads(text.strip())
        tags = [t.strip() for t in (data.get("tags") or []) if t and isinstance(t, str)]
        # de-dup & cap
        out, seen = [], set()
        for t in tags:
            if t not in seen:
                seen.add(t); out.append(t)
            if len(out) >= 8: break
        return out or (image_words or [])
    except Exception:
        return image_words or []

# ------------------ Personalization: history selection ------------------

def _user_baseline_vector(access: str) -> dict:
    ids = _recently_played(access, 50) + _top_tracks(access, 30) + _saved_tracks(access, 200)
    feats = _audio_features(access, ids)
    if not feats:
        return {"valence": 0.6, "energy": 0.6, "danceability": 0.6, "acousticness": 0.4}
    acc = {"valence": 0.0, "energy": 0.0, "danceability": 0.0, "acousticness": 0.0}
    n = 0
    for f in feats:
        n += 1
        for k in acc.keys():
            acc[k] += float(f.get(k, 0.0))
    for k in acc.keys():
        acc[k] = acc[k] / max(1, n)
    return acc

def _history_nearest_to_targets(access: str, targets: dict, limit: int = 50) -> List[str]:
    """
    Pull user's recent + top + saved; filter to tracks near the image targets with
    a widening tolerance; rank by weighted distance; enforce artist diversity.
    """
    import random

    # ---- Build pool (recent + top + saved)
    pool: List[str] = []
    try:
        pool.extend(_recently_played(access, 50))
    except Exception: pass
    try:
        pool.extend(_top_tracks(access, 100))
    except Exception: pass
    try:
        pool.extend(_saved_tracks(access, 300))
    except Exception: pass

    # De-dup preserve order
    seen, hist_ids = set(), []
    for tid in pool:
        if tid and tid not in seen:
            seen.add(tid)
            hist_ids.append(tid)

    if not hist_ids:
        return []

    # ---- Fetch features
    feats = _audio_features(access, hist_ids)
    if not feats:
        # No features? Shuffle so it isn't strictly recency-ordered
        random.shuffle(hist_ids)
        return hist_ids[:limit]

    # Build id->feature map and candidate list present in feats
    id2feat = {f.get("id"): f for f in feats if f and f.get("id")}
    cand_ids = [tid for tid in hist_ids if tid in id2feat]

    if not cand_ids:
        random.shuffle(hist_ids)
        return hist_ids[:limit]

    # ---- Widening tolerance filter around targets
    def within(f, t, tol):
        return (
            abs(float(f.get("valence", 0.5))      - t["valence"])      <= tol and
            abs(float(f.get("energy", 0.5))       - t["energy"])       <= tol and
            abs(float(f.get("danceability", 0.5)) - t["danceability"]) <= tol and
            abs(float(f.get("acousticness", 0.5)) - t["acousticness"]) <= tol
        )

    selected = []
    for tol in [0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.50]:
        selected = [tid for tid in cand_ids if within(id2feat[tid], targets, tol)]
        if len(selected) >= max(120, limit*3):
            break
    if not selected:
        selected = cand_ids

    # ---- Rank by weighted distance to targets
    W = {"valence": 1.2, "energy": 1.0, "acousticness": 1.0, "danceability": 0.8}
    def dist(tid: str) -> float:
        f = id2feat[tid]
        s = 0.0
        for k, w in W.items():
            s += w * (float(f.get(k, 0.5)) - float(targets.get(k, 0.5))) ** 2
        return s

    selected.sort(key=dist)

    # ---- Enforce artist diversity (≤2 per artist)
    out: List[str] = []
    per_artist: dict = {}
    meta = _tracks_meta(access, selected[:500])
    for tid in selected:
        arts = (meta.get(tid, {}) or {}).get("artists", []) or ["unknown"]
        primary = arts[0]
        if per_artist.get(primary, 0) >= 2:
            continue
        out.append(tid)
        per_artist[primary] = per_artist.get(primary, 0) + 1
        if len(out) >= limit:
            break

    # If diversity filter trims too hard, top up from remainder (still ranked)
    if len(out) < limit:
        for tid in selected:
            if tid not in out:
                out.append(tid)
                if len(out) >= limit:
                    break

    return out[:limit]

# ------------------ Build endpoint ------------------

@app.post("/build")
async def build(request: Request, image: UploadFile = File(...), ntracks: int = Form(30)):
    access, refresh = _ensure_valid_access(request)
    if not access:
        return RedirectResponse("/login")

    # 1) photo → (words, targets)
    img_bytes = await image.read()
    mood_words, img_targets = image_to_targets(img_bytes)

    # 2) user taste → baseline (for blending)
    try:
        baseline = _user_baseline_vector(access)
    except HTTPError as e:
        if getattr(e.response, "status_code", None) == 401 and refresh:
            new_tokens = _refresh_token(refresh)
            access = new_tokens["access_token"]
            baseline = _user_baseline_vector(access)
        else:
            raise

    # 3) blend targets (image 85% / taste 15%)
    def _blend(a, b, wa, wb):
        return max(0.0, min(1.0, (a*wa + b*wb) / (wa + wb)))
    targets = {
        "valence":      _blend(img_targets["valence"],      baseline["valence"],      0.85, 0.15),
        "energy":       _blend(img_targets["energy"],       baseline["energy"],       0.85, 0.15),
        "danceability": _blend(img_targets["danceability"], baseline["danceability"], 0.85, 0.15),
        "acousticness": _blend(img_targets["acousticness"], baseline["acousticness"], 0.85, 0.15),
    }

    # 3b) AI-expand mood tags for title/description
    mood_words = _ai_expand_tags(mood_words, targets)
    # 4) 60/40 mix, both sides adaptive to avoid empty results
    market = _me(access).get("country", None)

    # Known (history) — pull a big pool, then adaptively narrow to targets
    known_ids_pool = _history_nearest_to_targets(access, targets, limit=400)
    want_known     = max(1, int(round(int(ntracks) * 0.60)))
    known_take     = _collect_vibe(access, known_ids_pool, targets, need=want_known,
                                   pad_steps=[0.18, 0.24, 0.30, 0.38, 0.50])

    # Discovery — adaptive recommendations with widening bands & soft fallback
    seeds          = _user_seeds_simple(access)
    unknown_pool   = _explore_recommendations(access, seeds, targets, limit=240, market=market,
                                              pad_steps=[0.18, 0.24, 0.30, 0.38, 0.50])

    # de-dup and enforce diversity on the discovery slice too
    known_set      = set(known_take)
    unknown_unique = [t for t in unknown_pool if t not in known_set]
    want_unknown   = max(0, int(ntracks) - len(known_take))
    unknown_take   = _enforce_diversity(access, unknown_unique, per_artist=2, limit=want_unknown)

    chosen = known_take + [t for t in unknown_take if t not in known_take]

    # Top-up if still short (use remaining items, still diversity-capped)
    if len(chosen) < int(ntracks):
        remainder = [t for t in (known_ids_pool + unknown_unique) if t not in chosen]
        chosen += _enforce_diversity(access, remainder, per_artist=2, limit=int(ntracks) - len(chosen))

    if not chosen:
        return HTMLResponse("""
          <h2>Couldn’t assemble a vibe-matching set 😕</h2>
          <p>Try a different photo or play/save a few more songs.</p>
          <p><a href="/">Back</a></p>
        """, status_code=200)

    # 5) create playlist + add
    user = _me(access)
    name = f"{', '.join(mood_words) or 'PicToBeat'} • from your photo"
    desc = (
        f"Image mood {mood_words}. "
        f"Targets: valence {targets['valence']:.2f}, energy {targets['energy']:.2f}, "
        f"dance {targets['danceability']:.2f}, acoustic {targets['acousticness']:.2f}. "
        "≈60% from your history, ≈40% discovery."
    )
    pl_id = _create_playlist(access, user["id"], name, desc)
    _add_items(access, pl_id, chosen)

    return HTMLResponse(f"""
      <h2>Done ✅</h2>
      <p>Created: <b>{name}</b></p>
      <p>Mood tags: {mood_words}</p>
      <p>Added <b>{len(chosen)}</b> tracks (~60% known / ~40% fresh).</p>
      <p><a href="https://open.spotify.com/playlist/{pl_id}" target="_blank">Open playlist</a></p>
      <p><a href="/">Make another</a></p>
    """)
    print(f"[PicToBeat] known_pool={len(known_ids_pool)} known_take={len(known_take)}")
    print(f"[PicToBeat] unknown_pool={len(unknown_pool)} unknown_take={len(unknown_take)} chosen={len(chosen)}")



# unknown vs known is incorrect
