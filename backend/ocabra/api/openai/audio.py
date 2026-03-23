"""
POST /v1/audio/transcriptions — Whisper speech-to-text
POST /v1/audio/speech — TTS text-to-speech
"""
from __future__ import annotations

from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from ocabra.config import settings

from ._deps import (
    _openai_error,
    check_capability,
    ensure_loaded,
    get_model_manager,
    raise_upstream_http_error,
)

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
    form = await request.form(
        max_part_size=max(1, int(settings.openai_audio_max_part_size_mb)) * 1024 * 1024
    )
    model_id: str = form.get("model", "")
    language: str | None = form.get("language")
    response_format: str = form.get("response_format", "json")
    prompt: str | None = form.get("prompt")
    temperature: float = float(form.get("temperature", 0.0))

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "audio_transcription", "audio transcription")
    model_id = state.model_id
    request.state.stats_model_id = model_id

    worker_pool = request.app.state.worker_pool
    audio_bytes = await file.read()

    resp: httpx.Response | None = None
    last_error: Exception | None = None

    for attempt in range(2):
        worker = worker_pool.get_worker(model_id)
        backend = await worker_pool.get_backend(state.backend_type)
        worker_healthy = bool(worker) and await backend.health_check(state.backend_model_id)

        if not worker or not worker_healthy:
            if attempt == 0:
                reason = "worker_missing" if not worker else "worker_unhealthy"
                # Recover stale state where ModelManager says LOADED but worker is missing/dead.
                await model_manager.unload(model_id, reason=reason)
                await model_manager.load(model_id)
                continue
            raise _openai_error(
                f"Whisper worker unavailable for model '{model_id}'.",
                "server_error",
                code="worker_unavailable",
                status_code=503,
            )

        url = f"http://127.0.0.1:{worker.port}/transcribe"

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                form_data: dict[str, str] = {
                    "response_format": response_format,
                    "temperature": str(temperature),
                }
                if language:
                    form_data["language"] = language
                if prompt:
                    form_data["prompt"] = prompt

                resp = await client.post(
                    url,
                    files={"file": (file.filename or "audio", audio_bytes, file.content_type or "audio/mpeg")},
                    data=form_data,
                )
                resp.raise_for_status()
            break
        except httpx.TransportError as exc:
            last_error = exc
            logger.warning(
                "whisper_worker_transport_error",
                model_id=model_id,
                worker_port=worker.port,
                attempt=attempt + 1,
                error=str(exc),
            )
            if attempt == 0:
                await model_manager.unload(model_id, reason="worker_transport_error")
                await model_manager.load(model_id)
                continue
            raise _openai_error(
                f"Whisper worker unavailable for model '{model_id}'.",
                "server_error",
                code="worker_unavailable",
                status_code=503,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise_upstream_http_error(exc)

    if resp is None:
        raise _openai_error(
            f"Failed to reach Whisper worker for model '{model_id}': {last_error}",
            "server_error",
            code="worker_unavailable",
            status_code=503,
        )

    # OpenAI format: {"text": "..."}
    normalized_format = str(response_format).lower()
    if normalized_format in {"text", "srt", "vtt"}:
        return PlainTextResponse(resp.text)
    return resp.json()


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
    model_id = state.model_id

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
