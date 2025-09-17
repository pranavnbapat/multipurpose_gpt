# app/main.py

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.api.routes import router as api_router

API_TITLE = "Multipurpose GPT"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Single endpoint to ask GPT with an optional file upload."

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)

# Mount API routes
app.include_router(api_router, prefix="/api")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
    )

    # Patch the /api/ask file param to a clean binary description
    try:
        props = schema["paths"]["/api/ask"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]
        if "file" in props:
            props["file"] = {
                "type": "string",
                "format": "binary",
                "description": "Optional file upload (PDF, DOCX, TXT, etc.)"
            }
    except Exception:
        pass
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi
