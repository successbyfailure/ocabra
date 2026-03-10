"""
POST /v1/audio/transcriptions — Whisper speech-to-text
POST /v1/audio/speech — TTS text-to-speech
"""
from __future__ import annotations

from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import StreamingResponse

from ._deps import check_capability, ensure_loaded, get_model_manager

router = APIRouter()
logger = structlog.get_logger(__name__)

_AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


@router.post("/audio/transcriptions", summary="Transcribe audio")
async def transcriptions(
    request: Request,
    file: UploadFile,
) -> Any:
    """
    Transcribe an audio file using a Whisper model.
    Accepts multipart/form-data with 'file' and optional parameters.
    Requires a model with capability audio_transcription=True.

    Supported response_format: json (default), text, srt, vtt, verbose_json.
    """
    form = await request.form()
    model_id: str = form.get("model", "")
    language: str | None = form.get("language")
    response_format: str = form.get("response_format", "json")
    prompt: str | None = form.get("prompt")
    temperature: float = float(form.get("temperature", 0.0))

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "audio_transcription", "audio transcription")

    worker_pool = request.app.state.worker_pool
    worker = worker_pool.get_worker(model_id)
    if not worker:
        from ._deps import _openai_error
        raise _openai_error("Model worker not found.", "server_error", status_code=503)

    audio_bytes = await file.read()
    url = f"http://127.0.0.1:{worker.port}/transcribe"

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            url,
            files={"file": (file.filename or "audio", audio_bytes, file.content_type or "audio/mpeg")},
            data={
                "language": language or "",
                "response_format": response_format,
                "prompt": prompt or "",
                "temperature": str(temperature),
            },
        )
        resp.raise_for_status()
        result = resp.json()

    # OpenAI format: {"text": "..."}
    if response_format == "text":
        return result.get("text", "")
    return result


@router.post("/audio/speech", summary="Generate speech")
async def speech(request: Request) -> StreamingResponse:
    """
    Generate speech audio from text using a TTS model.
    Requires a model with capability tts=True.

    Supported response_format: mp3 (default), opus, aac, flac, wav, pcm.
    Supported voices: alloy, echo, fable, onyx, nova, shimmer (mapped per model).
    """
    body = await request.json()
    model_id: str = body.get("model", "")
    input_text: str = body.get("input", "")
    voice: str = body.get("voice", "alloy")
    response_format: str = body.get("response_format", "mp3")
    speed: float = float(body.get("speed", 1.0))

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "tts", "text-to-speech")

    worker_pool = request.app.state.worker_pool
    worker = worker_pool.get_worker(model_id)
    if not worker:
        from ._deps import _openai_error
        raise _openai_error("Model worker not found.", "server_error", status_code=503)

    url = f"http://127.0.0.1:{worker.port}/synthesize"
    content_type = _AUDIO_CONTENT_TYPES.get(response_format, "audio/mpeg")

    async def _stream_audio():
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                url,
                json={
                    "input": input_text,
                    "voice": voice,
                    "response_format": response_format,
                    "speed": speed,
                },
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        _stream_audio(),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="speech.{response_format}"'},
    )
