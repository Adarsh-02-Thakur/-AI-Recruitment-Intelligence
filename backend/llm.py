import json, os, re
import requests
from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


def get_key(name):
    """Read an API key and remove stray spaces, quotes and line breaks that break authentication."""
    return (os.getenv(name) or "").strip().strip('"').strip("'").strip()


def key_hint(name):
    raw = os.getenv(name) or ""
    k = get_key(name)
    if not k:
        return f"{name} is EMPTY on the server"
    extra = " had hidden spaces/quotes (removed)" if raw != k else ""
    return f"server sees {name}: length {len(k)}, starts with '{k[:4]}'{extra}"


def post(url, headers, payload, key_name=None):
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code >= 300:
        hint = f" | {key_hint(key_name)}" if (key_name and r.status_code in (401, 403)) else ""
        raise RuntimeError(f"{PROVIDER} API error {r.status_code}: {r.text[:200]}{hint}")
    return r.json()


def _call(system, user, max_tokens, json_mode=True):
    if PROVIDER == "groq":
        key = get_key("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set on the server (add it in Render > Environment)")
        payload = {"model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"), "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = post("https://api.groq.com/openai/v1/chat/completions", {"Authorization": f"Bearer {key}"}, payload, "GROQ_API_KEY")
        return data["choices"][0]["message"].get("content") or ""
    if PROVIDER == "gemini":
        key = get_key("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set on the server")
        model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        data = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", {"x-goog-api-key": key},
                    {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"parts": [{"text": user}]}],
                     "generationConfig": {"maxOutputTokens": max_tokens, "responseMimeType": "application/json"}}, "GEMINI_API_KEY")
        return data["candidates"][0]["content"]["parts"][0]["text"]
    from anthropic import Anthropic  # LLM_PROVIDER=anthropic
    msg = Anthropic().messages.create(model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=max_tokens,
                                      system=system, messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if b.type == "text")


def ask_json(system: str, user: str, max_tokens: int = 2500) -> dict:
    system += "\nRespond with ONLY valid JSON."
    try:
        text = _call(system, user, max_tokens)
    except RuntimeError as e:
        if "error 400" not in str(e):
            raise
        text = _call(system, user, max_tokens, json_mode=False)  # some models reject JSON mode
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise RuntimeError("The AI did not return valid JSON. Please try again.")
    return json.loads(m.group(0))