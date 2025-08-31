# app.py
import os, json, re, base64
from typing import List, Tuple, Optional
from fastapi import FastAPI, Request, UploadFile, Form, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import requests
from requests.exceptions import HTTPError

from spotify_oauth import auth_url, exchange_code_for_token, refresh_token as _refresh_token

load_dotenv()
app = FastAPI()
templates = Jinja2Templates(directory="templates")

SPOTIFY_API = "https://api.spotify.com/v1"
PLAYLIST_PRIVACY = os.getenv("PLAYLIST_PRIVACY", "private").lower() == "public"

# ------------------ tiny utils ------------------

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

def _ensure_valid_access(request: Request) -> Tuple[Optional[str], Optional[str]]:
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

# ------------------ routes: ui + oauth ------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login")
def login():
    return RedirectResponse(auth_url("x"))

@app.get("/callback")
def callback(code: Optional[str] = None, state: Optional[str] = None):
    if not code:
        return HTMLResponse("<p>Missing code.</p>", status_code=400)
    tokens = exchange_code_for_token(code)
    html = """
    <!doctype html><html><body>
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
    </body></html>"""
    resp = HTMLResponse(html)
    resp.set_cookie("access_token", tokens["access_token"], httponly=True)
    if "refresh_token" in tokens:
        resp.set_cookie("refresh_token", tokens["refresh_token"], httponly=True)
    return resp

# ------------------ spotify helpers ------------------

def _me(access: str) -> dict:
    r = requests.get(f"{SPOTIFY_API}/me", headers=_bearer_headers(access), timeout=15)
    r.raise_for_status()
    return r.json()

def _my_top_artists(access: str, cap: int = 20) -> list[dict]:
    r = requests.get(f"{SPOTIFY_API}/me/top/artists?limit={cap}", headers=_bearer_headers(access), timeout=15)
    r.raise_for_status()
    return r.json().get("items", []) or []

def _top_tracks(access: str, cap: int = 100) -> List[str]:
    ids: List[str] = []
    for tr in ("long_term", "medium_term", "short_term"):
        if len(ids) >= cap:
            break
        try:
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
                if not items: break
                for t in items:
                    tid = t.get("id")
                    if tid: ids.append(tid)
                if len(items) < 50: break
        except Exception:
            pass
    seen, out = set(), []
    for tid in ids:
        if tid and tid not in seen:
            seen.add(tid); out.append(tid)
        if len(out) >= cap: break
    return out

def _recently_played(access: str, cap: int = 100) -> List[str]:
    try:
        r = requests.get(f"{SPOTIFY_API}/me/player/recently-played",
                         headers=_bearer_headers(access),
                         params={"limit": min(50, cap)}, timeout=15)
        if r.status_code != 200:
            return []
        items = r.json().get("items", []) or []
        ids = []
        for it in items:
            tid = ((it.get("track") or {}).get("id")) if isinstance(it, dict) else None
            if tid: ids.append(tid)
        return ids[:cap]
    except Exception:
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
            if r.status_code != 200: break
            items = r.json().get("items", []) or []
            if not items: break
            for it in items:
                tr = (it or {}).get("track") or {}
                tid = tr.get("id")
                if tid: ids.append(tid)
            if len(items) < 50: break
    except Exception:
        pass
    seen, out = set(), []
    for tid in ids:
        if tid and tid not in seen:
            seen.add(tid); out.append(tid)
        if len(out) >= cap: break
    return out

def _tracks_meta(access: str, track_ids: List[str]) -> dict:
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
                if not t: continue
                tid = t.get("id");  arts = [a.get("id") for a in (t.get("artists") or []) if a.get("id")]
                meta[tid] = {"artists": arts or [], "popularity": int(t.get("popularity", 0))}
        except Exception:
            continue
    return meta

SP_API_BASE = "https://api.spotify.com/v1"

def _audio_features(access_token: str, track_ids: List[str] | List[dict]) -> List[dict]:
    ids = []
    for t in track_ids:
        tid = t if isinstance(t, str) else (t.get("id") if isinstance(t, dict) else None)
        tid = _sanitize_id(tid) if tid else None
        if tid: ids.append(tid)
    seen, cleaned = set(), []
    for tid in ids:
        if tid not in seen:
            seen.add(tid); cleaned.append(tid)
    headers = {"Authorization": f"Bearer {access_token}"}
    feats: List[dict] = []
    for i in range(0, len(cleaned), 100):
        chunk = cleaned[i:i+100]
        try:
            r = requests.get(f"{SP_API_BASE}/audio-features", headers=headers,
                             params={"ids": ",".join(chunk)}, timeout=20)
            if r.status_code in (400, 403):
                for tid in chunk:
                    r2 = requests.get(f"{SP_API_BASE}/audio-features/{tid}", headers=headers, timeout=15)
                    if r2.status_code == 200:
                        jd = r2.json()
                        if isinstance(jd, dict) and jd: feats.append(jd)
                continue
            r.raise_for_status()
            data = r.json().get("audio_features") or []
            feats.extend([d for d in data if d])
        except requests.RequestException:
            continue
    return feats

def _rank(features: List[dict], target: dict) -> List[dict]:
    W = {"valence": 1.2, "energy": 1.0, "acousticness": 1.0, "danceability": 0.8}
    def score(f: dict) -> float:
        s = 0.0
        for k, w in W.items():
            s += w * (float(f.get(k, 0)) - float(target.get(k, 0))) ** 2
        return s
    return sorted(features, key=score)

def _enforce_diversity(access: str, ids: list[str], per_artist: int = 2, limit: int = 50) -> list[str]:
    meta = _tracks_meta(access, ids[:600])
    out, seen, per = [], set(), {}
    for tid in ids:
        if tid in seen: continue
        artists = (meta.get(tid, {}) or {}).get("artists", []) or ["unknown"]
        primary = artists[0]
        if per.get(primary, 0) >= per_artist: continue
        seen.add(tid); out.append(tid)
        per[primary] = per.get(primary, 0) + 1
        if len(out) >= limit: break
    return out


# --- add these helpers near spotify helpers ---

def _followed_artist_ids(access: str, cap_pages: int = 20) -> set[str]:
    out=set()
    url = f"{SPOTIFY_API}/me/following"
    params={"type":"artist","limit":50}
    for _ in range(cap_pages):
        r = requests.get(url, headers=_bearer_headers(access), params=params, timeout=15)
        if r.status_code != 200: break
        data = r.json().get("artists", {}) or {}
        items = data.get("items", []) or []
        for a in items:
            aid = (a or {}).get("id")
            if aid: out.add(aid)
        url = data.get("next")
        if not url: break
        params = None  # 'next' already contains params
    return out

def _user_playlist_track_and_artist_ids(access: str, max_playlists: int = 50, max_tracks_per: int = 200) -> tuple[set[str], set[str]]:
    pl_ids=[]
    r = requests.get(f"{SPOTIFY_API}/me/playlists", headers=_bearer_headers(access), params={"limit":50}, timeout=15)
    if r.status_code == 200:
        for it in r.json().get("items", []) or []:
            pid = it.get("id")
            if pid: pl_ids.append(pid)
    pl_ids = pl_ids[:max_playlists]

    track_ids=set(); artist_ids=set()
    for pid in pl_ids:
        url = f"{SPOTIFY_API}/playlists/{pid}/tracks"
        params={"limit":100}
        fetched=0
        while url and fetched < max_tracks_per:
            rr = requests.get(url, headers=_bearer_headers(access), params=params, timeout=20)
            if rr.status_code != 200: break
            js = rr.json()
            for it in js.get("items", []) or []:
                t = (it or {}).get("track") or {}
                tid = t.get("id")
                if tid: track_ids.add(tid)
                for a in (t.get("artists") or []):
                    aid = a.get("id")
                    if aid: artist_ids.add(aid)
                fetched += 1
                if fetched >= max_tracks_per: break
            url = js.get("next"); params=None
    return track_ids, artist_ids

def _pop_cap_for_new_ratio(new_ratio: float) -> Optional[int]:
    # stricter as slider goes to 100% new
    if new_ratio >= 0.95: return 35
    if new_ratio >= 0.80: return 45
    if new_ratio >= 0.60: return 55
    if new_ratio >= 0.40: return 65
    return None


# ------------------ image → mood via prompting (no hard-coded words) ------------------

def _available_seed_genres(access: str) -> list[str]:
    try:
        r = requests.get(f"{SPOTIFY_API}/recommendations/available-genre-seeds",
                         headers=_bearer_headers(access), timeout=10)
        if r.status_code == 200:
            return r.json().get("genres", []) or []
    except Exception:
        pass
    # tiny fallback if Spotify hiccups (not mood words, just seed names)
    return ["pop","dance","edm","house","electropop","hip-hop","rnb","indie","indie-pop","alt-rock","techno","trance","lo-fi","folk","acoustic","ambient"]

def _normalize_targets(t: dict) -> dict:
    out = {}
    for k in ["valence","energy","danceability","acousticness"]:
        v = float(t.get(k, 0.5))
        out[k] = 0.0 if v < 0 else 1.0 if v > 1 else v
    return out

# --- put near your OpenAI helpers ---

def _parse_responses_json(j: dict) -> dict:
    # Responses API: prefer 'output_text'; else stitch from 'output'
    txt = j.get("output_text")
    if not txt:
        parts = []
        for msg in j.get("output", []) or []:
            for c in msg.get("content", []) or []:
                if c.get("type") in ("output_text","text"):
                    parts.append(c.get("text",""))
        txt = "".join(parts)
    txt = (txt or "").strip()
    if txt.startswith("```"):
        import re
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.DOTALL)
    try:
        return json.loads(txt) if txt else {}
    except Exception:
        return {}
    


def _openai_image_mood(img_bytes: bytes, seed_list: list[str]) -> tuple[list[str], dict, list[str]]:
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        return (["vibe", "mood"], {"valence":0.5, "energy":0.45, "danceability":0.45, "acousticness":0.6}, seed_list[:3])

    b64 = base64.b64encode(img_bytes).decode()
    sys = (
        "You create Spotify playlists from a single photo. "
        "Return ONLY JSON with keys:\n"
        '{ "tags":[up to 5 short Gen-Z vibe descriptors], '
        '"targets":{"valence":0..1,"energy":0..1,"danceability":0..1,"acousticness":0..1}, '
        '"seed_genres":[<=3 items from provided list] }\n'
        "Rules: tags are 1–3 words, no hashtags/punctuation; coherent with the scene. "
        "Pick seed_genres ONLY from the list."
    )
    usr = {
        "available_seed_genres": seed_list,
        "hint": "If the photo shows nature/greenery/outdoors, prefer calmer, organic vibes (higher acousticness, lower energy). If it shows neon/club/crowd, prefer high energy/danceability."
    }
    payload = {
        "model": "gpt-4.1-mini",
        "modalities": ["text", "vision"],
        "input": [
            {"role": "system", "content": [{"type": "text", "text": sys}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": json.dumps(usr)},
                {"type": "input_image", "image_base64": b64}
            ]}
        ]
    }
    try:
        r = requests.post("https://api.openai.com/v1/responses",
                          headers={"Authorization": f"Bearer {api}", "Content-Type": "application/json"},
                          json=payload, timeout=40)
        r.raise_for_status()
        data = _parse_responses_json(r.json()) or {}
        tags = [t.strip() for t in (data.get("tags") or []) if isinstance(t, str) and t.strip()]
        tgts = _normalize_targets(data.get("targets") or {})
        seeds = [s for s in (data.get("seed_genres") or []) if s in set(seed_list)]
        if not tags: tags = ["vibe","mood"]
        if not tgts: tgts = {"valence":0.5,"energy":0.45,"danceability":0.45,"acousticness":0.6}
        if not seeds: seeds = seed_list[:3]
        print("[PicToBeat] LLM tags:", tags)
        print("[PicToBeat] LLM targets:", tgts)
        print("[PicToBeat] LLM seeds:", seeds)
        return tags[:5], tgts, seeds[:3]
    except Exception as e:
        print("[PicToBeat] OpenAI error:", e)
        return (["vibe", "mood"], {"valence":0.5, "energy":0.45, "danceability":0.45, "acousticness":0.6}, seed_list[:3])

# ------------------ known / new pools ------------------

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

def _history_nearest_to_targets(access: str, targets: dict, limit: int = 120) -> List[str]:
    import random
    pool: List[str] = []
    try: pool.extend(_recently_played(access, 50))
    except Exception: pass
    try: pool.extend(_top_tracks(access, 100))
    except Exception: pass
    try: pool.extend(_saved_tracks(access, 300))
    except Exception: pass

    seen, hist_ids = set(), []
    for tid in pool:
        if tid and tid not in seen:
            seen.add(tid); hist_ids.append(tid)
    if not hist_ids: return []

    feats = _audio_features(access, hist_ids)
    if not feats:
        random.shuffle(hist_ids); return hist_ids[:limit]

    id2f = {f.get("id"): f for f in feats if f and f.get("id")}
    cand = [tid for tid in hist_ids if tid in id2f]
    if not cand:
        random.shuffle(hist_ids); return hist_ids[:limit]

    def within(f, t, tol):
        return (abs(float(f.get("valence",0.5))-t["valence"])<=tol and
                abs(float(f.get("energy",0.5))-t["energy"])<=tol and
                abs(float(f.get("danceability",0.5))-t["danceability"])<=tol and
                abs(float(f.get("acousticness",0.5))-t["acousticness"])<=tol)

    selected=[]
    for tol in [0.12,0.18,0.24,0.30,0.36,0.42,0.50]:
        selected=[tid for tid in cand if within(id2f[tid],targets,tol)]
        if len(selected)>=max(120, limit*3): break
    if not selected: selected = cand

    W = {"valence":1.2,"energy":1.0,"acousticness":1.0,"danceability":0.8}
    def dist(tid: str)->float:
        f=id2f[tid]; s=0.0
        for k,w in W.items(): s += w * (float(f.get(k,0.5))-float(targets.get(k,0.5)))**2
        return s
    selected.sort(key=lambda tid: dist(tid) if isinstance(tid, str) else float('inf'))
    return _enforce_diversity(access, [tid for tid in selected if tid is not None], per_artist=2, limit=limit)

def _known_universe(access: str) -> set[str]:
    ids = set()
    try:  ids.update(_recently_played(access, 50))
    except: pass
    try:  ids.update(_top_tracks(access, 100))
    except: pass
    try:  ids.update(_saved_tracks(access, 300))
    except: pass
    return ids

def _artist_ids_for_tracks(access: str, track_ids: list[str]) -> set[str]:
    meta = _tracks_meta(access, track_ids)
    out=set()
    for tid, m in meta.items():
        for a in m.get("artists", []): out.add(a)
    return out

def _near_target_filter(access: str, ids: list[str], targets: dict, tol: float = 0.22) -> list[str]:
    feats = _audio_features(access, ids)
    if not feats: return ids
    id2f = {f["id"]: f for f in feats if f and f.get("id")}
    out=[]
    for tid in ids:
        f=id2f.get(tid)
        if not f: continue
        if (abs(float(f.get("valence",0.5))-targets["valence"])<=tol and
            abs(float(f.get("energy",0.5))-targets["energy"])<=tol and
            abs(float(f.get("danceability",0.5))-targets["danceability"])<=tol and
            abs(float(f.get("acousticness",0.5))-targets["acousticness"])<=tol):
            out.append(tid)
    return out

def _recommend_from_seed_genres(access: str, seed_genres: list[str], targets: dict,
                                market: Optional[str], limit: int,
                                pop_max: Optional[int]) -> list[str]:
    """
    Recs with ONLY genre seeds + target features (no user seeds).
    Optionally post-filter by popularity <= pop_max.
    """
    params = {
        "limit": min(100, max(20, limit)),
        "seed_genres": ",".join(seed_genres[:5]) or "pop",
        "target_valence": float(targets["valence"]),
        "target_energy": float(targets["energy"]),
        "target_danceability": float(targets["danceability"]),
        "target_acousticness": float(targets["acousticness"]),
        "min_popularity": 0
    }
    if market: params["market"] = market

    ids: list[str] = []
    # two passes, second relaxes slightly if first is thin
    for tweak in (params, {**params, "min_popularity": 0}):
        r = requests.get(f"{SPOTIFY_API}/recommendations",
                         headers=_bearer_headers(access), params=tweak, timeout=20)
        if r.status_code != 200: continue
        tracks = r.json().get("tracks") or []
        for t in tracks:
            tid = t.get("id")
            if tid: ids.append(tid)
        if len(ids) >= limit: break

    # post-filter by popularity if requested
    if pop_max is not None:
        meta = _tracks_meta(access, ids[:600])
        ids = [tid for tid in ids if (meta.get(tid, {}) or {}).get("popularity", 100) <= pop_max]

    # de-dup preserve order
    seen, out = set(), []
    for tid in ids:
        if tid not in seen:
            seen.add(tid); out.append(tid)
        if len(out) >= limit: break
    return out

# ------------------ playlist ops ------------------

def _create_playlist(access: str, user_id: str, name: str, desc: str) -> str:
    r = requests.post(f"{SPOTIFY_API}/users/{user_id}/playlists",
                      headers={**_bearer_headers(access), "Content-Type": "application/json"},
                      json={"name": name, "description": desc, "public": PLAYLIST_PRIVACY},
                      timeout=15)
    r.raise_for_status()
    return r.json()["id"]

def _add_items(access: str, playlist_id: str, track_ids: List[str]) -> dict:
    uris = [f"spotify:track:{i}" for i in track_ids if _sanitize_id(i)]
    if not uris: return {"snapshot_id": None}
    r = requests.post(f"{SPOTIFY_API}/playlists/{playlist_id}/tracks",
                      headers={**_bearer_headers(access), "Content-Type": "application/json"},
                      json={"uris": uris}, timeout=15)
    r.raise_for_status()
    return r.json()
# ------------------ Build endpoint ------------------
@app.post("/build")
async def build(
    request: Request,
    image: UploadFile = File(...),
    ntracks: int = Form(10),      # default 10
    mix_known: int = Form(60)     # 0..100 slider
):
    access, refresh = _ensure_valid_access(request)
    if not access:
        return RedirectResponse("/login")

    # ----- tiny helper: /recommendations by seed_genres only (optional pop cap)
    def _recommend_from_seed_genres(access: str, seed_genres: list[str], targets: dict,
                                    market: Optional[str], limit: int = 300, pop_max: Optional[int] = None) -> list[str]:
        params = {
            "limit": min(100, max(20, limit)),
            "seed_genres": ",".join((seed_genres or ["pop"])[:5]),
            "target_valence": float(targets["valence"]),
            "target_energy": float(targets["energy"]),
            "target_danceability": float(targets["danceability"]),
            "target_acousticness": float(targets["acousticness"]),
        }
        if market:
            params["market"] = market
        if pop_max is not None:
            params["max_popularity"] = int(pop_max)

        out = []
        # two passes: strict then slight relax
        for p in (params, {**params, "max_popularity": min(100, (pop_max or 100) + 10)}):
            r = requests.get(f"{SPOTIFY_API}/recommendations", headers=_bearer_headers(access), params=p, timeout=25)
            if r.status_code != 200:
                continue
            for t in r.json().get("tracks") or []:
                tid = t.get("id")
                if tid:
                    out.append(tid)
            if len(out) >= limit:
                break
        # de-dup preserve order
        seen, uniq = set(), []
        for tid in out:
            if tid not in seen:
                seen.add(tid); uniq.append(tid)
            if len(uniq) >= limit:
                break
        return uniq

    # --- 1) Image → LLM mood (tags + targets + seed_genres)
    img_bytes = await image.read()
    seed_list = _available_seed_genres(access)
    tags, img_targets, seed_genres = _openai_image_mood(img_bytes, seed_list)
    print("[PicToBeat] final tags:", tags)
    print("[PicToBeat] final targets:", img_targets)
    print("[PicToBeat] final seed_genres:", seed_genres)

    # --- 2) Tiny blend with user taste
    try:
        baseline = _user_baseline_vector(access)
    except HTTPError as e:
        if getattr(e.response, "status_code", None) == 401 and (request.cookies.get("refresh_token")):
            new_tokens = _refresh_token(request.cookies["refresh_token"])
            access = new_tokens["access_token"]
            baseline = _user_baseline_vector(access)
        else:
            raise

    def _blend(a,b,wa,wb): return max(0.0, min(1.0, (a*wa+b*wb)/(wa+wb)))
    targets = {
        "valence":      _blend(img_targets["valence"],      baseline["valence"],      0.85, 0.15),
        "energy":       _blend(img_targets["energy"],       baseline["energy"],       0.85, 0.15),
        "danceability": _blend(img_targets["danceability"], baseline["danceability"], 0.85, 0.15),
        "acousticness": _blend(img_targets["acousticness"], baseline["acousticness"], 0.85, 0.15),
    }

    # --- 3) Targets for counts
    total = max(1, min(100, int(ntracks)))
    want_known = max(0, min(total, int(round(total * (int(mix_known)/100.0)))))
    want_new   = total - want_known
    new_ratio  = 1.0 - (want_known / max(1, total))
    # popularity cap gets stricter as you slide to "new"
    pop_cap = _pop_cap_for_new_ratio(new_ratio)

    # --- 4) Known universe (tracks + artists) to exclude from "new"
    known_ids_all   = _known_universe(access)
    known_meta      = _tracks_meta(access, list(known_ids_all)[:600])
    known_artist_ids = {a for tid in known_meta for a in (known_meta.get(tid, {}).get("artists") or [])}

    # add followed artists and all playlist artists/tracks to the "familiar" set
    followed_ids = _followed_artist_ids(access)
    pl_track_ids, pl_artist_ids = _user_playlist_track_and_artist_ids(access)
    avoid_tracks  = set(known_ids_all) | set(pl_track_ids)
    avoid_artists = set(known_artist_ids) | set(pl_artist_ids) | set(followed_ids)

    # --- 5) Build KNOWN bucket (history-only, vibe-matched)
    known_pool = _history_nearest_to_targets(access, targets, limit=300)
    known_pool = _near_target_filter(access, known_pool, targets, tol=0.18) or known_pool
    known_rank = _audio_features(access, known_pool)
    known_rank = [f["id"] for f in _rank(known_rank, targets)] if known_rank else known_pool
    known_take = _enforce_diversity(access, known_rank, per_artist=2, limit=want_known)

    # --- 6) Build NEW bucket (genre recs → drop familiar → vibe sort → diversity)
    market = _me(access).get("country") or None
    new_raw = _recommend_from_seed_genres(access, seed_genres, targets, market, limit=400, pop_max=pop_cap)

    new_meta = _tracks_meta(access, new_raw[:600])
    fresh = []
    for tid in new_raw:
        if tid in avoid_tracks:
            continue
        arts = (new_meta.get(tid, {}) or {}).get("artists", [])
        if any(a in avoid_artists for a in arts):
            continue
        fresh.append(tid)

    fresh = _near_target_filter(access, fresh, targets, tol=0.18) or fresh
    fresh_feats = _audio_features(access, fresh)
    fresh_rank  = [f["id"] for f in _rank(fresh_feats, targets)] if fresh_feats else fresh
    new_take    = _enforce_diversity(access, fresh_rank, per_artist=2, limit=want_new)

    # --- 7) Merge + top-up if under-filled
    chosen = []
    chosen.extend(known_take)
    for t in new_take:
        if t not in chosen:
            chosen.append(t)

    remaining_new = max(0, want_new - len(new_take))
    remaining_known = max(0, want_known - len(known_take))

    if remaining_new > 0:
        for t in fresh_rank:
            if t not in chosen and t not in avoid_tracks:
                chosen.append(t)
                remaining_new -= 1
            if remaining_new <= 0:
                break

    if remaining_known > 0:
        for t in known_rank:
            if t not in chosen:
                chosen.append(t)
                remaining_known -= 1
            if remaining_known <= 0:
                break

    if len(chosen) < total:
        for t in fresh_rank:
            if len(chosen) >= total:
                break
            if t not in chosen and t not in avoid_tracks:
                chosen.append(t)
        if want_known > 0 and len(chosen) < total:
            for t in known_rank:
                if len(chosen) >= total:
                    break
                if t not in chosen:
                    chosen.append(t)
    if not chosen:
        return HTMLResponse("""
          <h2>Couldn’t assemble a vibe-matching set 😕</h2>
          <p>Try another photo, or slide toward “Known”.</p>
          <p><a href="/">Back</a></p>
        """, status_code=200)

    # --- 8) Create playlist + add
    user = _me(access)
    title = (", ".join(tags[:8]) or "PicToBeat") + " • from your photo"
    desc  = (f"Image vibe → {tags}. "
             f"Targets: valence {targets['valence']:.2f}, energy {targets['energy']:.2f}, "
             f"dance {targets['danceability']:.2f}, acoustic {targets['acousticness']:.2f}. "
             f"Mix: ~{want_known} known / {want_new} new.")
    pl_id = _create_playlist(access, user["id"], title, desc)
    _add_items(access, pl_id, chosen[:total])

    # --- 9) Pretty result page with Spotify embed
    return HTMLResponse(f"""
      <!doctype html>
      <html>
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width, initial-scale=1"/>
          <title>PicToBeat • Done</title>
          <style>
            body{{margin:0;background:#0b0f14;color:#e7f0f7;font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Inter,Helvetica Neue,Arial}}
            .wrap{{min-height:100vh;display:grid;place-items:center;padding:48px 20px;
                   background:radial-gradient(1200px 800px at 90% -10%, rgba(35,166,213,.18),transparent 60%),
                              radial-gradient(1000px 600px at -10% 110%, rgba(0,208,132,.18),transparent 60%),#0b0f14}}
            .card{{max-width:860px;width:100%;background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.02)) ,#121821;
                   border:1px solid rgba(255,255,255,.06);border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.06);
                   padding:22px}}
            .header{{display:flex;align-items:center;gap:12px;margin-bottom:8px}}
            .logo{{width:38px;height:38px;border-radius:10px;background:radial-gradient(80% 80% at 30% 30%, #00d084,#0c9b6f);display:grid;place-items:center;color:#08140f;font-weight:800}}
            .title{{font-size:clamp(20px,2.2vw,28px);font-weight:800}}
            .sub{{color:#8aa0b2;margin-bottom:18px}}
            .row{{display:grid;grid-template-columns:1fr;gap:16px}}
            @media(min-width:860px){{.row{{grid-template-columns:1.1fr .9fr}}}}
            .box{{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:14px}}
            .btn{{display:inline-flex;gap:10px;align-items:center;background:linear-gradient(180deg,#00d084,#08b778);color:#06251b;font-weight:750;border:0;border-radius:12px;padding:12px 16px;text-decoration:none}}
            .btn:hover{{filter:brightness(1.02)}}
            .hint{{color:#8aa0b2}}
            iframe{{width:100%;min-height:520px;border:0;border-radius:12px}}
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="card">
              <div class="header">
                <div class="logo">PB</div>
                <div class="title">PicToBeat</div>
              </div>
              <div class="sub">Playlist created — tuned to your photo and taste.</div>
              <div class="row">
                <div class="box">
                  <iframe style="border-radius:12px"
                    src="https://open.spotify.com/embed/playlist/{pl_id}?utm_source=generator"
                    loading="lazy" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>
                </div>
                <div class="box">
                  <h3 style="margin:0 0 8px 0">{title}</h3>
                  <div class="hint" style="margin-bottom:10px">{desc}</div>
                  <a class="btn" href="https://open.spotify.com/playlist/{pl_id}" target="_blank" rel="noreferrer">
                    Open in Spotify
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                      <path d="M13 5l7 7-7 7M20 12H4" stroke="#06251b" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                  </a>
                  <div style="margin-top:12px"><a class="btn" href="/">Make another</a></div>
                </div>
              </div>
            </div>
          </div>
        </body>
      </html>
    """)
