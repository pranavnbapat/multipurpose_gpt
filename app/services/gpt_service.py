from __future__ import annotations

import os
import mimetypes
from typing import Optional
import json
import time
import random
from openai import OpenAI, OpenAIError, APIConnectionError, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _to_jsonable(obj):
    """
    Best-effort conversion to something JSON serialisable.
    Works with SDK pydantic models (model_dump), dict-like, or falls back to str.
    """
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):       # openai >= 1.40 pydantic model
        return obj.model_dump()
    if hasattr(obj, "to_dict"):          # some SDKs expose this
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    try:
        # last resort: let json handle built-ins; default=str for unknowns
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def _status_from_exc(e: Exception) -> int | None:
    """
    Best-effort HTTP status extractor from OpenAI SDK exceptions.
    Returns None if no HTTP status is available.
    """
    # APIStatusError carries .status_code
    if isinstance(e, APIStatusError):
        try:
            return int(getattr(e, "status_code", None))
        except Exception:
            return None
    # Some generic OpenAIError derivatives may expose .status
    return getattr(e, "status", None)

def _is_retryable(e: Exception) -> bool:
    """
    Decide whether an exception should be retried.
    - Network/connection errors
    - 429 (rate limit)
    - 5xx (server errors)
    """
    if isinstance(e, (APIConnectionError, RateLimitError)):
        return True
    status = _status_from_exc(e)
    if status is None:
        return False
    if status == 429:
        return True
    if 500 <= status <= 599:
        return True
    return False

def ask_gpt(
    *,
    query: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_filename: Optional[str] = None,
    prompt: str = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",
    # Retry knobs
    max_retries: int = 5,
    initial_backoff: float = 1.0,   # seconds
    max_backoff: float = 20.0,      # seconds
    jitter: float = 0.25,           # +/-25% randomisation
    delete_uploaded_file: bool = True,
) -> dict:
    """
    Send optional text + optional file to an OpenAI GPT model via the Responses API and return JSON.
    Built-in retries for 429/5xx/network errors using exponential backoff with jitter.
    Returns:
      {
        "answer": "<model_text>",
        "model": "<model_name>",
        "usage": {...}  # when provided by SDK
      }
    """

    # --- inline helpers (kept local to avoid polluting your module) ---
    def _status_from_exc(e: Exception) -> int | None:
        if isinstance(e, APIStatusError):
            try:
                return int(getattr(e, "status_code", None))
            except Exception:
                return None
        return getattr(e, "status", None)

    def _is_retryable(e: Exception) -> bool:
        if isinstance(e, (APIConnectionError, RateLimitError)):
            return True
        s = _status_from_exc(e)
        if s is None:
            return False
        return s == 429 or (500 <= s <= 599)

    # --- validate inputs and build message parts ---
    content_parts = []
    if query:
        content_parts.append({"type": "input_text", "text": query})

    uploaded_file_id = None
    if file_bytes is not None and file_filename:
        # Best-effort MIME guess (not required by API; helps with diagnostics)
        mime_guess, _ = mimetypes.guess_type(file_filename)

        # Upload the file; the SDK accepts (name, bytes) tuple
        uploaded = _client.files.create(file=(file_filename, file_bytes), purpose="assistants")
        uploaded_file_id = uploaded.id

        # Attach as an input_file content part
        # NEW (only the allowed keys)
        content_parts.append({
            "type": "input_file",
            "file_id": uploaded_file_id,
        })

    if not content_parts:
        raise ValueError("Provide at least one of query or file.")

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content_parts},
    ]

    # --- retry loop ---
    attempt = 0
    while True:
        try:
            resp = _client.responses.create(
                model=model,
                input=messages,
                # temperature=0.3,        # uncomment for more deterministic output
                # max_output_tokens=1200, # set if you want a hard cap
            )

            # Prefer convenience aggregate text
            answer = getattr(resp, "output_text", None)
            if not answer:
                # Fallback: stitch from content parts if needed
                answer = ""
                for item in getattr(resp, "output", []) or []:
                    if item.get("type") == "message":
                        for part in item.get("content", []):
                            if part.get("type") == "output_text":
                                answer += part.get("text", "")
                answer = answer.strip()

            usage_raw = getattr(resp, "usage", None)
            return {
                "answer": answer,
                "model": model,
                "usage": _to_jsonable(usage_raw),
            }


        except OpenAIError as e:
            # Decide if we should retry; if not (or out of retries), raise
            should_retry = _is_retryable(e) and attempt < max_retries
            if not should_retry:
                raise RuntimeError(f"OpenAI call failed after {attempt} retries: {e}") from e

            # Exponential backoff with jitter
            base = min(max_backoff, initial_backoff * (2 ** attempt))
            factor = 1.0 + random.uniform(-jitter, jitter)  # 0.75x .. 1.25x by default
            delay = max(0.1, base * factor)
            time.sleep(delay)
            attempt += 1

        finally:
            # Optional clean-up: remove uploaded file from OpenAI once we're done
            if uploaded_file_id and delete_uploaded_file:
                try:
                    _client.files.delete(uploaded_file_id)
                except Exception:
                    # Non-fatal; don't mask the main result/error
                    pass
                else:
                    uploaded_file_id = None  # so we don't try to delete twice on a retry

