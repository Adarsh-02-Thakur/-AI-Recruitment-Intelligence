import json, os, re, requests
from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("LLM_PROVIDER", "groq")


def _call(system, user, max_tokens):
    if PROVIDER == "groq":
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json={"model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                  "max_tokens": max_tokens, "response_format": {"type": "json_object"}},
            timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if PROVIDER == "gemini":
        model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": os.getenv("GEMINI_API_KEY")},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"parts": [{"text": user}]}],
                  "generationConfig": {"maxOutputTokens": max_tokens, "responseMimeType": "application/json"}},
            timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    from anthropic import Anthropic  # PROVIDER=anthropic
    msg = Anthropic().messages.create(model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=max_tokens,
                                      system=system, messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if b.type == "text")


def ask_json(system: str, user: str, max_tokens: int = 3000) -> dict:
    text = _call(system + "\nRespond with ONLY valid JSON.", user, max_tokens)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)