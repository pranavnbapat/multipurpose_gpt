# app/api/routes.py

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config.settings import choose_prompt
from app.models.enums import ModelName
from app.services.audio_service import summarise_audio
from app.services.doc_service import summarise_document_file
from app.services.image_service import summarise_image_file
from app.services.gpt_service import ask_gpt
from app.services.video_service import summarise_video
from app.utils.file_utils import extract_ext_category

router = APIRouter()

VISION_MODELS = {"gpt-4o", "gpt-5", "gpt-4o-mini"}

def _is_vision_model(name: str) -> bool:
    # Allow "-mini" variants if they support vision in your stack
    return any(name.startswith(m) for m in VISION_MODELS)

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
    prompt: str = Form(
        None,
        description="System prompt. Leave blank to use the server default.",
        example="You are a precise assistant. Answer concisely."
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

    query = _normalise_query(query)

    # Cross-field rule: at least one of (query, file)
    if not query and not file:
        raise HTTPException(status_code=422, detail="Provide at least query or file.")

    if model.value.endswith("-transcribe"):
        raise HTTPException(
            status_code=422,
            detail="Selected model is a speech-to-text model. Choose from the models provided."
        )

    category: str | None = None

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

    prompt = _normalise_query(prompt)
    if not prompt:
        prompt = choose_prompt(prompt)

    # print("File category is: ", category)
    # print("Filename is: ", filename)
    # print("Prompt is: ", prompt)

    try:
        # --- If it's a video, call video service and return immediately ---
        if file_bytes and category == "video":
            # Run the blocking ffmpeg/transcription pipeline off the event loop
            summary = await run_in_threadpool(
                summarise_video,
                file_bytes,
                filename,
                prompt,
                model.value,
            )
            return PlainTextResponse(content=summary)

        # --- If it's an audio file, call audio service and return immediately ---
        if file_bytes and category == "audio":
            summary = await run_in_threadpool(
                summarise_audio,
                file_bytes,
                filename,
                prompt,
                model.value,
            )
            # return JSONResponse(content=jsonable_encoder({"summary": summary}))
            return PlainTextResponse(content=summary)

        # # --- If an AV file is uploaded, route to the appropriate service once ---
        # if file_bytes and category in {"video", "audio"}:
        #     # Service registry: add more handlers here (e.g. "image": summarise_image)
        #     handlers = {
        #         "video": summarise_video,
        #         "audio": summarise_audio,
        #     }
        #     handler = handlers.get(category)
        #
        #     summary = await run_in_threadpool(
        #         handler,
        #         file_bytes,
        #         filename,
        #         prompt,
        #         model.value,
        #     )
        #     return JSONResponse(content=jsonable_encoder({"summary": summary}))

        # --- If it's a document (PDF/Office/Text), call doc service and return immediately ---
        if file_bytes and category == "text":
            summary = await run_in_threadpool(
                summarise_document_file,
                file_bytes,
                filename,
                prompt,
                model.value,
            )
            return PlainTextResponse(content=summary)

        if file_bytes and category == "image":
            if not _is_vision_model(model.value):
                raise HTTPException(
                    status_code=422,
                    detail=f"Model '{model.value}' is not vision-capable. Select a vision model (e.g., gpt-4o or gpt-5)."
                )
            summary = await run_in_threadpool(
                summarise_image_file, file_bytes, filename, prompt, model.value
            )
            return PlainTextResponse(content=summary)

        result = ask_gpt(
            query=query,
            prompt=prompt,
            summary_model=model.value
        )
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        # Avoid leaking internals; return the message for now
        raise HTTPException(status_code=500, detail=str(e))
