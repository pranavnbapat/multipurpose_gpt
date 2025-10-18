# app/services/ollama_service.py

from __future__ import annotations

import httpx

from typing import Optional

from app.config.settings import settings, choose_prompt

class OllamaError(RuntimeError):
    pass

def ask_ollama(
    *,
    query: str,
    prompt: Optional[str] = None,
    timeout: float = 60.0,
) -> dict:
    """
    Send a text-only chat request to an Ollama server (DeepSeek 7B).
    Returns: {"answer": "<text>"} to match your ask_gpt shape.
    """
    if not settings.ollama_url:
        raise OllamaError("OLLAMA_URL not configured.")

    system_prompt = choose_prompt(prompt)
    model = settings.ollama_model

    # Ollama /api/chat schema (non-streaming)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query.strip()},
        ],
        "stream": False,
        "options": {
            "temperature": settings.ollama_temperature,
            "num_ctx": settings.ollama_num_ctx,
            "top_k": settings.ollama_top_k,
            "num_predict": settings.ollama_max_tokens,
        },
    }

    url = settings.ollama_url.rstrip("/") + "/api/chat"
    # Use httpx for timeouts and nice errors
    with httpx.Client(timeout=timeout, verify=True) as client:
        resp = client.post(url, json=payload)
    if resp.status_code != 200:
        raise OllamaError(f"Ollama error {resp.status_code}: {resp.text}")

    data = resp.json()
    msg = (data.get("message") or {}).get("content", "")
    return {"answer": msg.strip()}
