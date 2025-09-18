# app/services/doc_service.py

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from app.config.settings import settings, client


def _ensure_pdf(src_path: Path) -> Path:
    """
    If src is PDF, return it. Otherwise, convert to PDF via LibreOffice headless.
    Returns the path to a PDF inside the same (temporary) directory.
    """
    if src_path.suffix.lower() == ".pdf":
        return src_path

    if not shutil.which("soffice"):
        raise RuntimeError("LibreOffice not found. Install: sudo apt-get install -y libreoffice")

    out_dir = src_path.parent  # keep outputs in the same temp dir
    cmd = [
        "soffice", "--headless", "--convert-to", "pdf",
        "--outdir", str(out_dir), str(src_path),
    ]
    # Timeout guards against corrupt files hanging soffice
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        err = (res.stderr or res.stdout or "").strip()
        raise RuntimeError(f"LibreOffice conversion failed: {err}")

    pdf_path = out_dir / (src_path.stem + ".pdf")
    if not pdf_path.exists():
        raise RuntimeError("Conversion reported success but PDF not found.")
    return pdf_path


def summarise_document_file(
    file_bytes: bytes,
    filename: str,
    prompt: str | None = None,
    summary_model: str | None = None,
) -> str:
    """
    Save uploaded doc to a temp dir, convert to PDF if needed (via LibreOffice),
    upload PDF to OpenAI Files, then summarise using the chosen text model.

    Returns plain-text summary.
    """
    # Work inside an isolated temp directory so we can point soffice --outdir here.
    with tempfile.TemporaryDirectory(prefix="docsum_") as td:
        tmpdir = Path(td)

        # 1) Persist the uploaded file so soffice / uploader can read it
        suffix = Path(filename).suffix or ".pdf"
        src_path = tmpdir / f"upload{suffix}"
        src_path.write_bytes(file_bytes)

        # 2) Ensure we have a PDF for the input_file path
        pdf_path = _ensure_pdf(src_path)

        # 3) Upload the PDF (purpose=user_data) so Responses can reference it
        with pdf_path.open("rb") as f:
            up = client.files.create(file=f, purpose="user_data")
        file_id = up.id

        # 4) Summarise with user-chosen model; fall back to settings.summary_model
        eff_model = summary_model or settings.summary_model

        resp = client.responses.create(
            model=eff_model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": prompt},
                ],
            }],
        )
        return resp.output_text
