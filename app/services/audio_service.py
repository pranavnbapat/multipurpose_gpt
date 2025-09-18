# app/services/audio_service.py

from pathlib import Path
from tempfile import NamedTemporaryFile

from app.config.settings import settings, client


def summarise_audio(
    file_bytes: bytes,
    filename: str,
    prompt: str | None = None,
    summary_model: str | None = None,
) -> str:
    """
    Transcribe an uploaded audio file and summarise the transcript.
    No ffmpeg step: we pass the original audio to STT.
    Returns plain-text summary.
    """
    # Persist to a temp file so the SDK gets a normal file handle
    suffix = Path(filename).suffix or ".mp3"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio:
        tmp_audio.write(file_bytes)
        audio_path = Path(tmp_audio.name)

    try:
        # 1) Transcribe with server-side STT model
        with open(audio_path, "rb") as f:
            stt = client.audio.transcriptions.create(
                model=settings.stt_model,  # keep STT model in settings
                file=f,
            )
        transcript = stt.text

        # 2) Summarise with user-chosen model
        eff_model = summary_model or settings.summary_model

        resp = client.responses.create(
            model=eff_model,
            input=f"{prompt}\n\nTRANSCRIPT:\n{transcript}",
        )
        return resp.output_text

    finally:
        # Cleanup temp file
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass
