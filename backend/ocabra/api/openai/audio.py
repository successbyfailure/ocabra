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
    compute_worker_key,
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    merge_profile_defaults,
    raise_upstream_http_error,
    resolve_profile,
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
    audio_ct = (file.content_type or "").lower()
    if audio_ct and audio_ct not in {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/flac",
        "audio/webm",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "video/mp4",  # M4A/AAC
        "application/octet-stream",
    }:
        logger.warning(
            "transcription_unusual_content_type",
            model_id=model_id,
            content_type=audio_ct,
            filename=file.filename,
        )

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
    )
    check_capability(state, "audio_transcription", "audio transcription")
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
    request.state.stats_model_id = worker_key

    worker_pool = request.app.state.worker_pool
    audio_bytes = await file.read()

    resp: httpx.Response | None = None
    last_error: Exception | None = None

    for attempt in range(2):
        worker = worker_pool.get_worker(worker_key)
        backend = await worker_pool.get_backend(state.backend_type)
        worker_healthy = bool(worker) and await backend.health_check(state.backend_model_id)

        if not worker or not worker_healthy:
            if attempt == 0:
                reason = "worker_missing" if not worker else "worker_unhealthy"
                await model_manager.unload(worker_key, reason=reason)
                await model_manager.load(worker_key)
                continue
            raise _openai_error(
                f"Whisper worker unavailable for model '{worker_key}'.",
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
                    files={
                        "file": (
                            file.filename or "audio",
                            audio_bytes,
                            file.content_type or "audio/mpeg",
                        )
                    },
                    data=form_data,
                )
                resp.raise_for_status()
            break
        except httpx.TransportError as exc:
            last_error = exc
            logger.warning(
                "whisper_worker_transport_error",
                model_id=worker_key,
                worker_port=worker.port,
                attempt=attempt + 1,
                error=str(exc),
            )
            if attempt == 0:
                await model_manager.unload(worker_key, reason="worker_transport_error")
                await model_manager.load(worker_key)
                continue
            raise _openai_error(
                f"Whisper worker unavailable for model '{worker_key}'.",
                "server_error",
                code="worker_unavailable",
                status_code=503,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise_upstream_http_error(exc)

    if resp is None:
        raise _openai_error(
            f"Failed to reach Whisper worker for model '{worker_key}': {last_error}",
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
        model — profile_id (or legacy canonical model ID)

    Response:
        voices       — list of voice/speaker names
        model_type   — "base" | "custom_voice" | "placeholder"
        languages    — list of supported languages
        supports_voice_clone — bool
    """
    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    profile, state = await resolve_profile(
        model,
        model_manager,
        profile_registry,
        user=user,
    )
    check_capability(state, "tts", "text-to-speech")
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)

    worker_pool = request.app.state.worker_pool
    worker = worker_pool.get_worker(worker_key)
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

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    # --- Federation hook ---
    federation_manager = get_federation_manager(request)
    if federation_manager is not None:
        from ocabra.config import settings as _settings

        if _settings.federation_enabled:
            from ocabra.core.federation import resolve_federated

            target, peer = await resolve_federated(model_id, model_manager, federation_manager)
            if target == "remote":
                request.state.federation_remote_node_id = peer.peer_id
                # TTS is streaming audio — proxy as stream
                return StreamingResponse(
                    federation_manager.proxy_stream(peer, "POST", request.url.path, body),
                    media_type="audio/mpeg",
                )
    # --- End federation hook ---

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
    )
    check_capability(state, "tts", "text-to-speech")

    merged_body = merge_profile_defaults(profile, body)
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)

    input_text: str = merged_body.get("input", "")
    voice: str = merged_body.get("voice", "alloy")
    response_format: str = merged_body.get("response_format", "mp3")
    speed: float = float(merged_body.get("speed", 1.0))
    language: str = merged_body.get("language", "Auto")
    reference_audio: str | None = merged_body.get("reference_audio") or merged_body.get("voice_ref")
    reference_text: str | None = merged_body.get("reference_text")
    speaker: str | None = merged_body.get("speaker")
    instruct: str | None = merged_body.get("instruct")

    worker_pool = request.app.state.worker_pool
    worker = worker_pool.get_worker(worker_key)
    if not worker:
        raise _openai_error(
            f"TTS worker unavailable for model '{worker_key}'.",
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

    if not stream_safe:
        # Non-streaming (WAV): fetch full response so errors propagate correctly
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                detail = resp.text[:300] if resp.text else f"Worker error {resp.status_code}"
                raise _openai_error(detail, "server_error", status_code=resp.status_code)
            return StreamingResponse(
                iter([resp.content]),
                media_type=content_type,
                headers={"Content-Disposition": f'attachment; filename="speech.{response_format}"'},
            )

    async def _stream_audio():
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise _openai_error(
                        body.decode(errors="replace")[:300],
                        "server_error",
                        status_code=resp.status_code,
                    )
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
        model           — profile_id or canonical model ID (required)
        prompt          — music description, e.g. "upbeat jazz with piano" (required)
        lyrics          — optional song lyrics
        audio_duration  — duration in seconds (10-600, default chosen by model)
        bpm             — tempo in BPM (30-300)
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
    profile_registry = get_profile_registry(request)

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
    )
    check_capability(state, "music_generation", "music generation")

    merged_body = merge_profile_defaults(profile, body)
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
    request.state.stats_model_id = worker_key

    worker_pool = request.app.state.worker_pool
    worker = worker_pool.get_worker(worker_key)
    backend = await worker_pool.get_backend(state.backend_type)

    if not worker or not await backend.health_check(state.backend_model_id):
        raise _openai_error(
            f"ACE-Step worker unavailable for model '{worker_key}'.",
            "server_error",
            code="worker_unavailable",
            status_code=503,
        )

    try:
        result = await backend.forward_request(worker_key, "/generate", merged_body)
    except TimeoutError as exc:
        raise _openai_error(
            str(exc), "server_error", code="generation_timeout", status_code=504
        ) from exc
    except RuntimeError as exc:
        raise _openai_error(
            str(exc), "server_error", code="generation_failed", status_code=500
        ) from exc

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
