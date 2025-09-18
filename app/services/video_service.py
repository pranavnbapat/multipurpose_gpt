# app/services/video_service.py

import shlex
import subprocess

from pathlib import Path
from tempfile import NamedTemporaryFile

from app.config.settings import settings, client

def summarise_video(
        file_bytes: bytes,
        filename: str,
        prompt: str | None = None,
        summary_model: str | None = None
    ) -> str:
    """
    Extract audio from uploaded video, transcribe, and summarise.
    """
    with NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_vid:
        tmp_vid.write(file_bytes)
        video_path = Path(tmp_vid.name)

    audio_path = video_path.with_suffix(".wav")

    try:
        # --- Extract audio ---
        cmd = (
            f'ffmpeg -i {shlex.quote(str(video_path))} '
            f'-vn -ac 1 -ar 16000 -y {shlex.quote(str(audio_path))}'
        )
        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # --- Transcribe ---
        with open(audio_path, "rb") as f:
            stt = client.audio.transcriptions.create(
                model=settings.stt_model,
                file=f,
            )
        transcript = stt.text

        # --- Summarise ---
        effective_model = summary_model or settings.summary_model

        resp = client.responses.create(
            model=effective_model,
            input=f"{prompt}\n\nTRANSCRIPT:\n{transcript}",
        )

        return resp.output_text

    finally:
        # Cleanup
        for p in (video_path, audio_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
