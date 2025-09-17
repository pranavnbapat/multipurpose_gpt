# app/api/routes.py

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from app.models.enums import ModelName
from app.services.gpt_service import ask_gpt
from app.utils.file_utils import extract_ext_category

router = APIRouter()

@router.get("/healthz")
def healthz():
    return {"ok": True}

def _nonblank(s: str) -> str:
    # Enforce at least one non-space character
    if not s or not s.strip():
        raise ValueError("Prompt cannot be empty or whitespace.")
    return s.strip()

def _normalise_query(v: str | None) -> str | None:
    """
    Trim, convert empty/placeholder to None.
    Swagger UI often sends 'string' as a placeholder.
    """
    if v is None:
        return None
    v = v.strip()
    if not v or v.lower() == "string":
        return None
    return v

@router.post("/ask")
async def ask(
    # — Form fields (no Pydantic models) —
    prompt: str = Form(
        "You are a precise assistant. Answer concisely.",
        description="System prompt (required; default pre-filled in UI)."
    ),
    query: str | None = Form(
        None,
        description="Optional user query (leave blank if you will upload a file).",
        example="What is AI?"
    ),
    model: ModelName = Form(
        ...,
        description="Select a GPT model (required)."
    ),
    file: UploadFile | None | str = File(
        None,
        description="Optional file upload (PDF, DOCX, TXT, etc.).",
    ),
):
    # --- Normalise Swagger oddities ---
    # Swagger can send 'file' as "" (string) or an UploadFile with empty filename.
    if isinstance(file, str) or (file and getattr(file, "filename", "") == ""):
        file = None

    # --- Field-level validation (manual; no Pydantic models) ---
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")

    query = _normalise_query(query)

    # Cross-field rule: at least one of (query, file)
    if not query and not file:
        raise HTTPException(status_code=422, detail="Provide at least query or file.")

    # If a file is provided, validate its type and read it
    file_bytes, filename = None, None
    if file:
        filename = file.filename
        ext, category = extract_ext_category(filename)
        if not ext:
            raise HTTPException(
                status_code=422,
                detail="Unsupported file type. Allowed: video, audio, text, image, archive."
            )
        file_bytes = await file.read()

    try:
        result = ask_gpt(
            query=query,
            file_bytes=file_bytes,
            file_filename=filename,
            prompt=prompt,
            model=model.value,  # ModelName is an Enum; pass its value
        )
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        # Avoid leaking internals; return the message for now
        raise HTTPException(status_code=500, detail=str(e))
