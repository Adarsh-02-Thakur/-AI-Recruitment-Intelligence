import os
from llm import PROVIDER, get_key, post


def chat(system: str, history: list, max_tokens: int = 2000) -> str:
    """history = [{"role": "user" or "assistant", "content": "..."}]"""
    if PROVIDER == "groq":
        key = get_key("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set on the server")
        data = post("https://api.groq.com/openai/v1/chat/completions", {"Authorization": f"Bearer {key}"},
                    {"model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                     "messages": [{"role": "system", "content": system}] + history, "max_tokens": max_tokens}, "GROQ_API_KEY")
        return data["choices"][0]["message"].get("content") or ""

    if PROVIDER == "gemini":
        key = get_key("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set on the server")
        model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in history]
        data = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", {"x-goog-api-key": key},
                    {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                     "generationConfig": {"maxOutputTokens": max_tokens}}, "GEMINI_API_KEY")
        return data["candidates"][0]["content"]["parts"][0]["text"]

    from anthropic import Anthropic
    msg = Anthropic().messages.create(model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
                                      max_tokens=max_tokens, system=system, messages=history)
    return "".join(b.text for b in msg.content if b.type == "text")