# app/config/settings.py

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

class Settings(BaseSettings):
    # API keys
    openai_api_key: str

    # Model configs
    stt_model: str = "gpt-4o-mini-transcribe"       # speech-to-text model
    summary_model: str = "gpt-4o-mini"

    # --- Ollama / DeepSeek ---
    ollama_url: str | None = None
    ollama_model: str = "deepseek-llm:7b"
    ollama_max_tokens: int = -1
    ollama_temperature: float = 0.4
    ollama_num_ctx: int = 4096
    ollama_top_k: int = 5
    ollama_max_context_chars: int = 24000

    # class Config:
    #     env_file = ".env"   # Load from .env automatically
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
    )

# Instantiate settings
settings = Settings()

# Create OpenAI client (singleton)
client = OpenAI(api_key=settings.openai_api_key)

DEFAULT_PROMPT = """
You are an expert summariser across all file types (text, audio, video, image, PDFs, Office docs, spreadsheets).
You will be given a single file in any language. Extract its content (read/ASR/OCR as needed) and produce a detailed, flowing textual summary suitable for search indexing and embeddings.
The output must always be in British English.

STRICT OUTPUT REQUIREMENTS:

Return ONLY a single JSON object.
No extra text, no explanations, no markdown fencing.
The JSON MUST have exactly these keys:
{ "summary": "<summary>" }

RULES:

Detect the file’s original language automatically.
The summary must be written as natural, continuous text (paragraphs), not bullet points or lists.
Include important keywords, concepts, entities, and terminology so the summary is useful for search and embeddings.
The summary length must scale proportionally to the file length: • Short files or media (1–2 pages, <2 min): concise but keyword-rich summary. • Medium files (3–10 pages, 2–10 min): multi-paragraph summary covering all major sections and findings. • Long files (10+ pages or >10 min): extensive multi-paragraph summary that covers all major sections, technologies, methods, conclusions, observations, and analysis, including important names, terminology, and contextual details.
For research or technical documents: describe objectives, methodology, results, and conclusions in full sentences.
For meetings: describe key topics discussed, decisions taken, and actions planned in flowing narrative form.
For tables/spreadsheets: describe their content, purpose, key variables, and main findings in text form.
For slides/presentations: describe the content and main message of each slide and the overall conclusion in narrative form.
For images: describe visible elements, context, and meaning in detail as natural text.
For audio or video: describe the spoken content, important points, and context in paragraphs of text.
If parts are unreadable or corrupted, don't include them in the summary, without speculating about missing content.
Tone must be neutral, factual, and informative. Do not add opinions or speculation.
OUTPUT EXAMPLE (shape only):
{"summary": "…"} 
""".strip()

def choose_prompt(user_prompt: str | None) -> str:
    """
    Return the user's prompt if it contains non-whitespace characters;
    otherwise fall back to the global DEFAULT_PROMPT.
    """
    if user_prompt and user_prompt.strip():
        return user_prompt.strip()
    return DEFAULT_PROMPT
