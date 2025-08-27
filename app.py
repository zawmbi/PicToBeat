import os, time, math, json
from typing import List, Optional
from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import requests

from spotify_oauth import auth_url, exchange_code_for_token, refresh_token
from mood_map import analyze_image_to_mood

load_dotenv()
app = FastAPI()
templates = Jinja2Templates(directory="templates")

SPOTIFY_API = "https://api.spotify.com/v1"
PLAYLIST_PRIVACY = os.getenv("PLAYLIST_PRIVACY","private").lower() == "public"

def _bearer_headers(token): return {"Authorization": f"Bearer {token}"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.get("/login")
def login():
    state = "x"  # (for MVP; you can randomize)
    return RedirectResponse(auth_url(state))

from fastapi.responses import HTMLResponse, RedirectResponse

@app.get("/callback")
def callback(code: str | None = None, state: str | None = None):
    if not code:
        return HTMLResponse("<p>Missing code.</p>", status_code=400)

    tokens = exchange_code_for_token(code)

    # Build a tiny HTML page that notifies opener and closes the popup.
    html = """
    <!doctype html>
    <html><body>
      <p>Finishing Spotify sign-in…</p>
      <script>
        (function(){
          try {
            // Tell the opener we succeeded (same-origin only).
            if (window.opener && window.opener !== window) {
              window.opener.postMessage({ type: 'spotify-auth', status: 'success' }, window.location.origin);
            }
          } catch (e) { /* ignore */ }
          // Close if popup; otherwise go home.
          setTimeout(function(){
            if (window.opener && window.opener !== window) {
              window.close();
            } else {
              window.location = "/";
            }
          }, 200);
        })();
      </script>
    </body></html>
    """

    resp = HTMLResponse(html)
    # set cookies (same as before)
    resp.set_cookie("access_token", tokens["access_token"], httponly=True)
    if "refresh_token" in tokens:
        resp.set_cookie("refresh_token", tokens["refresh_token"], httponly=True)
    return resp


def _me(access): 
    return requests.get(f"{SPOTIFY_API}/me", headers=_bearer_headers(access), timeout=15).json()

def _my_top_artists(access):
    # user-top-read scope :contentReference[oaicite:7]{index=7}
    r = requests.get(f"{SPOTIFY_API}/me/top/artists?limit=10", headers=_bearer_headers(access), timeout=15)
    r.raise_for_status(); return r.json()["items"]

def _search_tracks(access, q:str, limit=25):
    # Search endpoint :contentReference[oaicite:8]{index=8}
    r = requests.get(f"{SPOTIFY_API}/search", headers=_bearer_headers(access),
                     params={"q": q, "type":"track", "limit": limit}, timeout=15)
    r.raise_for_status(); return [t["id"] for t in r.json()["tracks"]["items"]]

import os, requests
from spotify_oauth import refresh_token as _refresh

def _audio_features(access: str, track_ids: list[str], refresh: str | None = None):
    ids = ",".join(track_ids[:100])  # Spotify max 100 per request
    url = f"https://api.spotify.com/v1/audio-features?ids={requests.utils.requote_uri(ids)}"
    headers = {"Authorization": f"Bearer {access}"}

    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code != 200:
        print("AUDIO_FEATURES ERROR 1:", r.status_code, r.text)  # <-- see exact reason
        # If token expired (401) OR sometimes 403 due to token invalid, try refresh once
        if r.status_code in (401, 403) and refresh:
            try:
                new_tokens = _refresh(refresh)
                access = new_tokens["access_token"]
                headers["Authorization"] = f"Bearer {access}"
                r2 = requests.get(url, headers=headers, timeout=15)
                if r2.status_code != 200:
                    print("AUDIO_FEATURES ERROR 2:", r2.status_code, r2.text)
                    r2.raise_for_status()
                return [f for f in r2.json().get("audio_features", []) if f]
            except Exception as e:
                print("REFRESH FAILED:", e)
        r.raise_for_status()

    return [f for f in r.json().get("audio_features", []) if f]

def _create_playlist(access, user_id:str, name:str, desc:str):
    # Create playlist :contentReference[oaicite:10]{index=10}
    r = requests.post(f"{SPOTIFY_API}/users/{user_id}/playlists",
                      headers={**_bearer_headers(access), "Content-Type":"application/json"},
                      json={"name": name, "description": desc, "public": PLAYLIST_PRIVACY},
                      timeout=15)
    r.raise_for_status(); return r.json()["id"]

def _add_items(access, playlist_id:str, track_ids: List[str]):
    # Add items to playlist :contentReference[oaicite:11]{index=11}
    uris = [f"spotify:track:{i}" for i in track_ids]
    r = requests.post(f"{SPOTIFY_API}/playlists/{playlist_id}/tracks",
                      headers={**_bearer_headers(access), "Content-Type":"application/json"},
                      json={"uris": uris}, timeout=15)
    r.raise_for_status(); return r.json()

def _rank(features, target):
    W = {"valence":1.2,"energy":1.0,"acousticness":1.0,"danceability":0.8}
    def score(f):
        s=0.0
        for k,w in W.items():
            s += w*(float(f.get(k,0)) - target[k])**2
        return s
    return sorted(features, key=score)

@app.post("/build")
async def build(request: Request, image: UploadFile, ntracks: int = Form(30)):
    access = request.cookies.get("access_token")
    if not access:
        return RedirectResponse("/login")

    # 1) image -> mood
    img_bytes = await image.read()
    mood = analyze_image_to_mood(img_bytes)  # {"mood_words":[...], "targets": {...}}

    # 2) user taste
    artists = _my_top_artists(access)
    artist_names = [a["name"] for a in artists][:10]

    # 3) build queries (taste + mood words)
    words = " ".join(mood["mood_words"])
    queries = []
    for a in artist_names:
        queries.append(f'artist:"{a}" {words}')
    queries.append(words)

    # 3b) fetch candidates
    cand = set()
    for q in queries:
        cand.update(_search_tracks(access, q, limit=25))
    cand = list(cand)

    # 4) get audio features & rank to target
    feats = _audio_features(access, cand)
    ranked = _rank(feats, mood["targets"])
    chosen = [f["id"] for f in ranked[:max(1, min(ntracks, 100))]]

    # 5) create playlist + add
    user = _me(access)
    name = f"{', '.join(mood['mood_words'])} • from your photo"
    desc = f"Made from your image mood: {mood['mood_words']}. Personalized to your taste."
    pl_id = _create_playlist(access, user["id"], name, desc)
    _add_items(access, pl_id, chosen)

    return HTMLResponse(f"""
      <h2>Done ✅</h2>
      <p>Created: <b>{name}</b></p>
      <p>Mood: {mood['mood_words']} → targets {mood['targets']}</p>
      <p><a href="https://open.spotify.com/playlist/{pl_id}" target="_blank">Open playlist</a></p>
      <p><a href="/">Make another</a></p>
    """)
