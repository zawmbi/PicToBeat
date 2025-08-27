import io, base64, os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Simple dictionary you can tune anytime:
WORD_TO_TARGETS = {
  "cozy":       {"valence":0.55, "energy":0.30, "acousticness":0.70, "danceability":0.35},
  "moody":      {"valence":0.30, "energy":0.25, "acousticness":0.60, "danceability":0.30},
  "uplifting":  {"valence":0.75, "energy":0.65, "acousticness":0.20, "danceability":0.60},
  "dreamy":     {"valence":0.55, "energy":0.35, "acousticness":0.50, "danceability":0.40},
  "intimate":   {"valence":0.50, "energy":0.25, "acousticness":0.75, "danceability":0.30},
}

def _avg_targets(words):
    hits = [WORD_TO_TARGETS[w] for w in words if w in WORD_TO_TARGETS]
    if not hits:
        return {"valence":0.5,"energy":0.5,"acousticness":0.5,"danceability":0.5}
    keys = hits[0].keys()
    return {k: sum(h[k] for h in hits)/len(hits) for k in keys}

def analyze_image_to_mood(file_bytes: bytes):
    # You can swap this for the OpenAI SDK; here’s a minimal HTTPS call.
    import requests, json
    img_b64 = base64.b64encode(file_bytes).decode()
    prompt = (
      "You are an image mood tagger. Return JSON with keys: "
      "`mood_words` (2-5 lowercase words like cozy, dreamy, moody) "
      "that fit the image's *vibe*, not objects. No explanations."
    )
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    data = {
      "model": "gpt-4o-mini",
      "messages": [
        {"role":"system","content":"Respond ONLY with a valid JSON object."},
        {"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}
      ],
      "temperature": 0.2
    }
    r = requests.post(url, headers=headers, json=data, timeout=30)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    try:
        js = json.loads(txt)
    except Exception:
        js = {"mood_words":["cozy","intimate"]}
    mood_words = [w.strip().lower() for w in js.get("mood_words", [])][:5]
    targets = _avg_targets(mood_words)
    return {"mood_words": mood_words, "targets": targets}
