import json, os, re
import requests
from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


def _post(url, headers, payload):
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code >= 300:
        raise RuntimeError(f"{PROVIDER} API error {r.status_code}: {r.text[:300]}")
    return r.json()


def _call(system, user, max_tokens, json_mode=True):
    if PROVIDER == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set on the server (add it in Render > Environment)")
        payload = {"model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"), "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = _post("https://api.groq.com/openai/v1/chat/completions", {"Authorization": f"Bearer {key}"}, payload)
        return data["choices"][0]["message"].get("content") or ""
    if PROVIDER == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set on the server")
        model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        data = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", {"x-goog-api-key": key},
                     {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"parts": [{"text": user}]}],
                      "generationConfig": {"maxOutputTokens": max_tokens, "responseMimeType": "application/json"}})
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