# app/services/image_service.py

from __future__ import annotations

import base64
import tempfile

from pathlib import Path

from app.config.settings import client, settings

_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".gif": "image/gif",
    ".bmp": "image/bmp",  ".webp": "image/webp",
    ".tiff": "image/tiff",".tif": "image/tiff",
    ".heic": "image/heic",
}

def _guess_mime(suffix: str) -> str:
    return _MIME_BY_EXT.get(suffix.lower(), "application/octet-stream")

def summarise_image_file(
    file_bytes: bytes,
    filename: str,
    prompt: str | None = None,
    summary_model: str | None = None,
) -> str:
    """
    Send the image via Responses 'input_image' using a data: URL.
    This avoids 'input_file' (PDF-only) and the invalid 'image' field.
    """
    suffix = Path(filename).suffix or ".png"
    mime = _guess_mime(suffix)

    # Base64-encode the bytes and wrap as a data URL for image_url
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    eff_model = summary_model or settings.summary_model
    resp = client.responses.create(
        model=eff_model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": (prompt or "")},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    return resp.output_text
