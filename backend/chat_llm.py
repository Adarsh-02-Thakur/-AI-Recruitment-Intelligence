import os
import requests
from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("LLM_PROVIDER", "groq")


def chat(system: str, history: list, max_tokens: int = 2000) -> str:
    """history = [{"role": "user" or "assistant", "content": "..."}]"""
    if PROVIDER == "groq":
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json={
                "model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                "messages": [{"role": "system", "content": system}] + history,
                "max_tokens": max_tokens,
            },
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    if PROVIDER == "gemini":
        model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in history
        ]
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": os.getenv("GEMINI_API_KEY")},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": contents,
                "generationConfig": {"maxOutputTokens": max_tokens},
            },
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    from anthropic import Anthropic
    msg = Anthropic().messages.create(
        model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=max_tokens, system=system, messages=history,
    )
    return "".join(b.text for b in msg.content if b.type == "text")