# app/services/gpt_service.py

from __future__ import annotations

from typing import Optional
from app.config.settings import client, settings, choose_prompt


def ask_gpt(
    *,
    query: Optional[str] = None,
    prompt: Optional[str] = None,
    summary_model: Optional[str] = None,
    max_retries: int = 3,
) -> dict:
    """
    Text-only GPT call.

    Files (video/audio/docs) are already handled by dedicated services before this is invoked.

    Args:
        query: User input text (required when no file was uploaded). Must be non-empty after trimming.
        prompt: System instruction. If empty/None, the server default from `settings` is used via `choose_prompt`.
        summary_model: The model name to use for the summary; if None, defaults to `settings.summary_model`.
        max_retries: Number of attempts for transient failures (simple retry loop).

    Returns:
        dict: {"answer": <str>}
            The plain-text answer from the model.
    """

    # --- Validate inputs (router already enforces this, but guard anyway) ---
    if not query or not isinstance(query, str) or not query.strip():
        raise ValueError("Provide a non-empty query.")

    effective_prompt = choose_prompt(prompt)
    effective_model = summary_model or settings.summary_model

    messages = [
        {"role": "system", "content": effective_prompt},
        {"role": "user", "content": query.strip()},
    ]

    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.responses.create(
                model=effective_model,
                input=messages,
            )
            answer = (resp.output_text or "").strip()

            usage = getattr(resp, "usage", None)
            # normalise usage to a plain dict if available
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()

            return {"answer": answer}
            # return {"answer": answer, "model": effective_model, "usage": usage}
        except Exception as e:
            last_err = e
            continue  # simple retry; no backoff to keep it minimal

    # If we’re here, all retries failed
    raise last_err if last_err else RuntimeError("Unknown error calling Responses API")

