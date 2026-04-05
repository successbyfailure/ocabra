"""
POST /v1/audio/transcriptions — Whisper speech-to-text
POST /v1/audio/speech        — TTS text-to-speech (streams sentence-by-sentence)
GET  /v1/audio/voices        — List available voices for a TTS model
POST /v1/audio/generate      — ACE-Step music generation
"""
from __future__ import annotations

from typing import Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from ocabra.api._deps_auth import UserContext
from ocabra.config import settings

from ._deps import (
    _openai_error,
    check_capability,
    ensure_loaded,
    get_model_manager,
    get_openai_user,
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
    user: Annotated[UserContext, Depends(get_openai_user)],
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
    diarize: str | None = form.get("diarize")

    # Log a warning if the client sends a format that may not be handled by
    # all Whisper backends (e.g. M4A/AAC from Android MediaRecorder).
    # faster-whisper decodes via ffmpeg so these work; NeMo models may not.
    audio_ct = (file.content_type or "").lower()
    if audio_ct and audio_ct not in {
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/ogg", "audio/flac",
        "audio/webm",
        "audio/mp4", "audio/m4a", "audio/x-m4a", "video/mp4",  # M4A/AAC
        "application/octet-stream",
    }:
        logger.warning(
            "transcription_unusual_content_type",
            model_id=model_id,
            content_type=audio_ct,
            filename=file.filename,
        )

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id, user=user)
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
                if diarize is not None:
                    form_data["diarize"] = diarize

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


@router.get("/audio/voices", summary="List TTS voices for a model")
async def list_voices(
    request: Request,
    model: str,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Return available voices, speakers, and model_type for a loaded TTS model.

    Query params:
        model — canonical model ID (must be loaded and have capability tts=True)

    Response:
        voices       — list of voice/speaker names
        model_type   — "base" | "custom_voice" | "placeholder"
        languages    — list of supported languages
        supports_voice_clone — bool
    """
    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model, user=user)
    check_capability(state, "tts", "text-to-speech")
    model_id = state.model_id

    worker_pool = request.app.state.worker_pool
    worker = worker_pool.get_worker(model_id)
    if not worker:
        return {
            "voices": ["alloy", "echo", "fable", "nova", "onyx", "shimmer"],
            "model_type": "placeholder",
            "languages": ["Auto"],
            "supports_voice_clone": False,
        }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://127.0.0.1:{worker.port}/voices")
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {
            "voices": ["alloy", "echo", "fable", "nova", "onyx", "shimmer"],
            "model_type": "placeholder",
            "languages": ["Auto"],
            "supports_voice_clone": False,
        }


@router.post("/audio/speech", summary="Generate speech")
async def speech(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> StreamingResponse:
    """
    Generate speech audio from text using a TTS model.
    Requires a model with capability tts=True.

    The worker synthesises text sentence-by-sentence and streams audio chunks
    progressively, reducing time-to-first-audio for long texts.

    Supported response_format: mp3 (default), opus, aac, flac, wav, pcm.
      - mp3/opus/aac/flac : per-sentence encoded chunks (concatenatable stream).
      - wav               : streaming RIFF header + raw PCM frames.
      - pcm               : raw PCM16LE frames, no header.
    Supported voices: alloy, echo, fable, onyx, nova, shimmer (mapped per model).
    """
    body = await request.json()
    model_id: str = body.get("model", "")
    input_text: str = body.get("input", "")
    voice: str = body.get("voice", "alloy")
    response_format: str = body.get("response_format", "mp3")
    speed: float = float(body.get("speed", 1.0))
    language: str = body.get("language", "Auto")
    reference_audio: str | None = body.get("reference_audio")
    reference_text: str | None = body.get("reference_text")
    speaker: str | None = body.get("speaker")
    instruct: str | None = body.get("instruct")

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id, user=user)
    check_capability(state, "tts", "text-to-speech")
    model_id = state.model_id

    worker_pool = request.app.state.worker_pool
    worker = worker_pool.get_worker(model_id)
    if not worker:
        raise _openai_error(
            f"TTS worker unavailable for model '{model_id}'.",
            "server_error",
            code="worker_unavailable",
            status_code=503,
        )

    # Use /synthesize/stream for formats that support concatenation (mp3, pcm, opus,
    # flac).  Fall back to /synthesize for WAV which requires a correct RIFF header.
    stream_safe = response_format in {"mp3", "pcm", "opus", "flac", "aac"}
    endpoint = "/synthesize/stream" if stream_safe else "/synthesize"
    url = f"http://127.0.0.1:{worker.port}{endpoint}"
    content_type = _AUDIO_CONTENT_TYPES.get(response_format, "audio/mpeg")

    async def _stream_audio():
        async with httpx.AsyncClient(timeout=300.0) as client:
            payload: dict = {
                "input": input_text,
                "voice": voice,
                "response_format": response_format,
                "speed": speed,
                "language": language,
            }
            if reference_audio:
                payload["reference_audio"] = reference_audio
            if reference_text:
                payload["reference_text"] = reference_text
            if speaker:
                payload["speaker"] = speaker
            if instruct:
                payload["instruct"] = instruct
            async with client.stream(
                "POST",
                url,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        _stream_audio(),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="speech.{response_format}"'},
    )


@router.post("/audio/generate", summary="Generate music")
async def generate_music(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Response:
    """
    Generate music audio from a text prompt using an ACE-Step model.
    Requires a model with capability music_generation=True.

    Request body (JSON):
        model           — canonical model ID, e.g. "acestep/turbo" (required)
        prompt          — music description, e.g. "upbeat jazz with piano" (required)
        lyrics          — optional song lyrics
        audio_duration  — duration in seconds (10–600, default chosen by model)
        bpm             — tempo in BPM (30–300)
        key_scale       — e.g. "C Major"
        time_signature  — e.g. "4"
        inference_steps — diffusion steps (default 8 for turbo)
        thinking        — bool, enable LM-based song planning (default false)
        vocal_language  — lyrics language code, e.g. "en", "es" (default "en")
        seed            — int, -1 for random
        response_format — "mp3" (default), "wav", or "flac"

    Returns the generated audio file as binary.
    """
    body = await request.json()
    model_id: str = body.get("model", "")
    response_format: str = str(body.get("response_format") or "mp3").lower()
    if response_format not in {"mp3", "wav", "flac"}:
        response_format = "mp3"

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id, user=user)
    check_capability(state, "music_generation", "music generation")
    model_id = state.model_id
    request.state.stats_model_id = model_id

    worker_pool = request.app.state.worker_pool
    worker = worker_pool.get_worker(model_id)
    backend = await worker_pool.get_backend(state.backend_type)

    if not worker or not await backend.health_check(state.backend_model_id):
        raise _openai_error(
            f"ACE-Step worker unavailable for model '{model_id}'.",
            "server_error",
            code="worker_unavailable",
            status_code=503,
        )

    try:
        result = await backend.forward_request(model_id, "/generate", body)
    except TimeoutError as exc:
        raise _openai_error(str(exc), "server_error", code="generation_timeout", status_code=504) from exc
    except RuntimeError as exc:
        raise _openai_error(str(exc), "server_error", code="generation_failed", status_code=500) from exc

    if not isinstance(result, tuple) or len(result) < 2:
        raise _openai_error(
            "Unexpected response from ACE-Step backend.",
            "server_error",
            code="generation_failed",
            status_code=500,
        )

    audio_bytes, content_type = result[0], result[1]
    filename = f"music.{response_format}"
    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
