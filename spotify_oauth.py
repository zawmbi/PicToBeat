import os, uuid, urllib.parse
from dotenv import load_dotenv
from fastapi import Request
load_dotenv()

AUTH_BASE = "https://accounts.spotify.com/authorize"
TOKEN_URL  = "https://accounts.spotify.com/api/token"

SCOPES = [
    "user-top-read",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-read-email",
    "user-read-private",
    "user-read-recently-played",
    "user-library-read"
    
]


CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI")
APP_SECRET    = os.getenv("APP_SECRET", "dev")


def auth_url(state: str):
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "state": state,
        "show_dialog": "true"   # was "false"
    }
    return f"{AUTH_BASE}?{urllib.parse.urlencode(params)}"

def exchange_code_for_token(code: str):
    import requests
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(TOKEN_URL, headers=headers, data=data, timeout=15)
    if r.status_code != 200:
        print("TOKEN ERROR", r.status_code, r.text)  # TEMP: helps debug
    r.raise_for_status()
    return r.json()


def refresh_token(refresh_token: str):
    import requests
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(TOKEN_URL, headers=headers, data=data, timeout=15)
    if r.status_code != 200:
        print("REFRESH ERROR", r.status_code, r.text)  # TEMP
    r.raise_for_status()
    return r.json()



