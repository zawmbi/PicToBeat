# see_personalized_debug.py
import os, sys, json, urllib.parse, threading, webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8910/callback")

AUTH_URL   = "https://accounts.spotify.com/authorize"
TOKEN_URL  = "https://accounts.spotify.com/api/token"
API_BASE   = "https://api.spotify.com/v1"
SCOPES     = "user-top-read user-read-recently-played"

def _bearer(token: str): return {"Authorization": f"Bearer {token}"}

# ----- tiny one-shot auth server -----
class _OnceHandler(BaseHTTPRequestHandler):
    code = None
    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(p.query)
        if p.path == urllib.parse.urlparse(REDIRECT_URI).path and "code" in qs:
            _OnceHandler.code = qs["code"][0]
            self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
            self.wfile.write(b"<h2>Spotify auth complete. You can close this tab.</h2>")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_response(404); self.end_headers()

def _start_server():
    u = urllib.parse.urlparse(REDIRECT_URI)
    host, port = (u.hostname or "127.0.0.1"), (u.port or 8910)
    httpd = HTTPServer((host, port), _OnceHandler)
    print(f"[auth] Listening on {host}:{port} for {u.path}")
    httpd.serve_forever()

def _auth():
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": "x",
        "show_dialog": "true",
    }
    url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
    print("[auth] Opening browser…")
    webbrowser.open(url)
    _start_server()
    return _OnceHandler.code

def _exchange(code: str):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    r = requests.post(TOKEN_URL, data=data, timeout=20)
    r.raise_for_status()
    return r.json()

def _GET(access: str, path: str, **params):
    url = f"{API_BASE}{path}"
    print(f"\n[GET] {url} {json.dumps(params) if params else ''}")
    r = requests.get(url, headers=_bearer(access), params=params or None, timeout=20)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"[ERROR {r.status_code}] {url}\n{r.text[:600]}")
        raise
    return r.json()

def _recommendations(access: str, **params):
    url = f"{API_BASE}/recommendations"
    print(f"\n[GET] {url} {json.dumps(params)}")
    r = requests.get(url, headers=_bearer(access), params=params, timeout=20)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"[ERROR {r.status_code}] {url}\n{r.text[:600]}")
        raise
    return r.json()

def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET")
        sys.exit(1)

    code = _auth()
    if not code:
        print("No code received.")
        return
    tokens = _exchange(code); access = tokens["access_token"]

    # /me
    me = _GET(access, "/me")
    print(json.dumps({"id": me.get("id"), "display_name": me.get("display_name"), "country": me.get("country")}, indent=2))
    market = me.get("country") or "US"

    # /me/top/tracks
    try:
        top_tracks = _GET(access, "/me/top/tracks", limit=5, time_range="medium_term")
        print(json.dumps([{"id": t["id"], "name": t["name"]} for t in top_tracks.get("items", [])], indent=2))
    except requests.HTTPError:
        top_tracks = {"items": []}

    # /me/top/artists
    try:
        top_artists = _GET(access, "/me/top/artists", limit=5, time_range="medium_term")
        print(json.dumps([{"id": a["id"], "name": a["name"]} for a in top_artists.get("items", [])], indent=2))
    except requests.HTTPError:
        top_artists = {"items": []}

    # /me/player/recently-played
    try:
        recent = _GET(access, "/me/player/recently-played", limit=5)
        compact_recent = [
            {
                "played_at": it.get("played_at"),
                "id":        (it.get("track") or {}).get("id"),
                "name":      (it.get("track") or {}).get("name"),
                "artists":   [a.get("name") for a in (it.get("track") or {}).get("artists", [])],
            }
            for it in (recent.get("items") or [])
        ]
        print(json.dumps(compact_recent, indent=2, ensure_ascii=False))
    except requests.HTTPError:
        pass

    # seeds from your own taste
    seed_artists = [a["id"] for a in top_artists.get("items", [])[:2]]
    seed_tracks  = [t["id"] for t in top_tracks.get("items", [])[:2]]
    seed_genres  = ["pop"] if not (seed_artists or seed_tracks) else []

    # /recommendations (with market)
    try:
        recs = _recommendations(
            access,
            limit=10,
            seed_artists=",".join(seed_artists),
            seed_tracks=",".join(seed_tracks),
            seed_genres=",".join(seed_genres),
            market=market,
            target_valence=0.6,
            target_energy=0.6,
            target_danceability=0.6,
            min_popularity=20,
        )
        compact_recs = [
            {
                "id": t.get("id"),
                "name": t.get("name"),
                "artists": [a.get("name") for a in (t.get("artists") or [])],
                "popularity": t.get("popularity"),
                "preview_url": t.get("preview_url"),
            }
            for t in (recs.get("tracks") or [])
        ]
        print("\nRecommendations:")
        print(json.dumps(compact_recs, indent=2, ensure_ascii=False))
    except requests.HTTPError:
        pass

if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        # final catch just to show status and stop
        status = getattr(e.response, "status_code", "?")
        print(f"\nHTTPError: {status}")
    except Exception as e:
        print("Error:", repr(e))
        
