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
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from ocabra.api._deps_auth import UserContext
from ocabra.config import settings

from ._deps import (
    NON_STREAMING_LOAD_RESPONSE_DOC,
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
from ._federation import try_proxy_json, try_proxy_multipart

router = APIRouter()
logger = structlog.get_logger(__name__)


def _worker_timeout() -> float:
    """Read timeout for a transcription forwarded to a Whisper worker.

    Follows the server-wide inference timeout instead of a hard-coded 300 s:
    long recordings with diarization (pyannote on top of large-v3) routinely
    take longer than five minutes, and the old cap failed them every time.
    """
    try:
        return float(max(60, int(settings.inference_request_timeout_seconds)))
    except (TypeError, ValueError):
        return 1500.0

# faster-whisper wires diarize=true to pyannote in-process
# (workers/whisper_worker.py); whisper_cpp forwards it to a dedicated pyannote
# sidecar instead (workers/diarization_worker.py, launched lazily by
# WhisperCppBackend — see its module docstring). VibeASR has no
# speaker-diarization code at all — it just drops the field, so a client gets
# a normal transcription back with zero speakers identified and no error,
# which reads as "diarization silently did nothing" (reported live 2026-11-05:
# an agent got diarize=true accepted by two whisper.cpp profiles with no
# speaker labels in the response — whisper_cpp support added the same day).
_DIARIZATION_CAPABLE_BACKENDS = {"whisper", "whisper_cpp"}


def _is_diarize_requested(raw: str | None) -> bool:
    return raw is not None and raw.strip().lower() not in {"", "0", "false", "no"}


_AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


@router.post(
    "/audio/transcriptions", summary="Transcribe audio", responses=NON_STREAMING_LOAD_RESPONSE_DOC
)
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
    # One audio per request, like OpenAI's own endpoint. Several parts named
    # "file" used to be accepted with a 200 that transcribed only the last
    # one: the other audios were dropped with nothing in the response saying
    # so (reported 2026-11-08 by a harness that sent four WAVs and got one
    # transcription back). Transcribing batches goes through /v1/batches, one
    # line per audio — see the Batches section of GET /docs.
    if len(form.getlist("file")) > 1:
        raise _openai_error(
            "This endpoint transcribes one audio per request, and "
            f"{len(form.getlist('file'))} 'file' parts were sent. Post them one "
            "at a time, or use /v1/batches with one line per audio (upload each "
            'via POST /v1/files with purpose="batch_input_audio").',
            "invalid_request_error",
            param="file",
            code="multiple_files",
        )

    model_id: str = form.get("model", "")
    language: str | None = form.get("language")
    response_format: str = form.get("response_format", "json")
    prompt: str | None = form.get("prompt")
    temperature: float = float(form.get("temperature", 0.0))
    diarize: str | None = form.get("diarize")
    # ASR hallucination-loop mitigations, forwarded to faster-whisper. Left unset
    # here so the worker's defaults (vad_filter=true, condition_on_previous_text=
    # false) apply unless the client overrides them.
    vad_filter: str | None = form.get("vad_filter")
    condition_on_previous_text: str | None = form.get("condition_on_previous_text")
    # OpenAI sends repeated `timestamp_granularities[]` fields; accept the
    # bracketed and plain spellings so word/segment timestamps reach the worker.
    granularities: list[str] = [
        str(value)
        for value in (
            form.getlist("timestamp_granularities[]") or form.getlist("timestamp_granularities")
        )
    ]

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

    audio_bytes = await file.read()

    # Proxy whisper requests to a peer when it already has the model warm —
    # avoids loading a local worker just for one transcription.
    fed_form_data: dict[str, Any] = {
        "model": model_id,
        "response_format": response_format,
        "temperature": str(temperature),
    }
    if language:
        fed_form_data["language"] = language
    if prompt:
        fed_form_data["prompt"] = prompt
    if diarize is not None:
        fed_form_data["diarize"] = diarize
    if granularities:
        fed_form_data["timestamp_granularities"] = granularities

    fed_resp = await try_proxy_multipart(
        request,
        model_id=model_id,
        files={
            "file": (
                file.filename or "audio",
                audio_bytes,
                file.content_type or "audio/mpeg",
            )
        },
        data=fed_form_data,
        federation_manager=get_federation_manager(request),
        model_manager=model_manager,
        profile_registry=profile_registry,
        logger=logger,
    )
    if fed_resp is not None:
        normalized_format = str(response_format).lower()
        if fed_resp.status_code >= 400:
            # Non-recoverable peer error → propagate (recoverable ones
            # would have made the helper return None for local fallback).
            if normalized_format in {"text", "srt", "vtt"}:
                return PlainTextResponse(fed_resp.text, status_code=fed_resp.status_code)
            return Response(
                content=fed_resp.content,
                status_code=fed_resp.status_code,
                media_type=fed_resp.headers.get("content-type"),
            )
        # Peer succeeded — return its body, or fall through to local if the
        # body fails to JSON-decode (e.g. an upstream proxy returned HTML).
        if normalized_format in {"text", "srt", "vtt"}:
            return PlainTextResponse(fed_resp.text)
        try:
            return fed_resp.json()
        except Exception as exc:
            logger.warning(
                "federation_peer_non_json_response_fallback_local",
                model_id=model_id,
                error=str(exc),
                body_preview=fed_resp.text[:200],
            )
            # Fall through to local processing.

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
        router_resolver=getattr(request.app.state, "router_resolver", None),
        request_state=request.state,
    )
    check_capability(state, "audio_transcription", "audio transcription")
    diarize_requested = _is_diarize_requested(diarize)
    if diarize_requested and state.backend_type not in _DIARIZATION_CAPABLE_BACKENDS:
        raise _openai_error(
            f"The model '{state.model_id}' (backend '{state.backend_type}') does not "
            f"support diarize=true — speaker diarization is only available on "
            f"{sorted(_DIARIZATION_CAPABLE_BACKENDS)} backends.",
            "invalid_request_error",
            param="diarize",
            code="diarization_not_supported",
        )
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
    request.state.stats_model_id = worker_key

    worker_pool = request.app.state.worker_pool
    # whisper.cpp has no in-process pyannote to lazy-load diarization into
    # (whisper-server is a native binary) — WhisperCppBackend.forward_request()
    # owns that orchestration (launching/reusing the diarization sidecar and
    # merging its output), so route through it instead of this endpoint's own
    # direct worker call, which only knows how to talk to whisper-server.
    via_backend_forward = diarize_requested and state.backend_type == "whisper_cpp"

    resp: httpx.Response | None = None
    sidecar_result: Any = None
    last_error: Exception | None = None

    inflight_request_id = model_manager.begin_request(worker_key)
    if state.backend_type == "whisper":
        # Diarizing or many transcriptions at once grow whisper's footprint past
        # what it is sized for: ask a llama.cpp model on its card to shrink its
        # context first, instead of whisper running out of memory.
        await model_manager.make_room_for_whisper(worker_key, diarize=diarize_requested)
    try:
        for attempt in range(2):
            worker = worker_pool.get_worker(worker_key)
            backend = await worker_pool.get_backend(state.backend_type)
            worker_healthy = bool(worker) and await backend.health_check(state.backend_model_id)

            if not worker or not worker_healthy:
                if attempt == 0:
                    reason = "worker_missing" if not worker else "worker_unhealthy"
                    await model_manager.unload(worker_key, reason=reason)
                    await model_manager.load(worker_key)
                    await model_manager.touch_last_request_at(worker_key)
                    continue
                raise _openai_error(
                    f"Whisper worker unavailable for model '{worker_key}'.",
                    "server_error",
                    code="worker_unavailable",
                    status_code=503,
                    headers={"Retry-After": "10"},
                )

            try:
                if via_backend_forward:
                    body: dict[str, Any] = {
                        "file": (
                            file.filename or "audio",
                            audio_bytes,
                            file.content_type or "audio/mpeg",
                        ),
                        "response_format": response_format,
                        "temperature": temperature,
                        "diarize": diarize,
                    }
                    if language:
                        body["language"] = language
                    if prompt:
                        body["prompt"] = prompt
                    if granularities:
                        body["timestamp_granularities"] = granularities
                    # WhisperCppBackend (like AceStep below) keys its own
                    # _processes by backend_model_id (the short suffix), not
                    # the canonical worker_key — see the ACE-Step call further
                    # down for the same gotcha.
                    sidecar_result = await backend.forward_request(
                        state.backend_model_id, "/transcribe", body
                    )
                else:
                    url = f"http://127.0.0.1:{worker.port}/transcribe"
                    async with httpx.AsyncClient(timeout=_worker_timeout()) as client:
                        form_data: dict[str, Any] = {
                            "response_format": response_format,
                            "temperature": str(temperature),
                        }
                        if language:
                            form_data["language"] = language
                        if prompt:
                            form_data["prompt"] = prompt
                        if diarize is not None:
                            form_data["diarize"] = diarize
                        if granularities:
                            form_data["timestamp_granularities"] = granularities
                        if vad_filter is not None:
                            form_data["vad_filter"] = vad_filter
                        if condition_on_previous_text is not None:
                            form_data["condition_on_previous_text"] = condition_on_previous_text

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
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                # The worker is alive and busy: it accepted the request and is
                # still transcribing. Killing and reloading it (the transport
                # error path below) threw away that work, evicted whatever it
                # had to reload next to, and re-ran the same audio only to hit
                # the same wall again (2026-11-10: a diarized large-v3 batch
                # item failed twice at exactly 300 s each).
                logger.warning(
                    "whisper_worker_timeout",
                    model_id=worker_key,
                    worker_port=worker.port,
                    timeout_s=_worker_timeout(),
                )
                raise _openai_error(
                    f"Transcription with '{worker_key}' did not finish within "
                    f"{_worker_timeout():.0f} s. Try a shorter recording, no diarization "
                    "or a faster model.",
                    "server_error",
                    code="transcription_timeout",
                    status_code=504,
                ) from exc
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
                    await model_manager.touch_last_request_at(worker_key)
                    continue
                raise _openai_error(
                    f"Whisper worker unavailable for model '{worker_key}'.",
                    "server_error",
                    code="worker_unavailable",
                    status_code=503,
                    headers={"Retry-After": "10"},
                ) from exc
            except httpx.HTTPStatusError as exc:
                if (
                    exc.response.status_code == 503
                    and "diarization_out_of_memory" in exc.response.text
                ):
                    # The worker already retried at smaller batch sizes and the
                    # context shrink above was not enough: evict what is on
                    # whisper's card, as a load under pressure would, and retry.
                    if attempt == 0 and await model_manager.evict_for_whisper(worker_key):
                        continue
                    raise _openai_error(
                        f"No GPU memory to diarize with '{worker_key}' right now. Retry "
                        "later or send the request without diarization.",
                        "server_error",
                        code="diarization_out_of_memory",
                        status_code=503,
                        headers={"Retry-After": "30"},
                    ) from exc
                raise_upstream_http_error(exc)
    finally:
        model_manager.end_request(worker_key, inflight_request_id)

    if resp is None and sidecar_result is None:
        raise _openai_error(
            f"Failed to reach Whisper worker for model '{worker_key}': {last_error}",
            "server_error",
            code="worker_unavailable",
            status_code=503,
            headers={"Retry-After": "10"},
        )

    from ocabra.api.openai._deps import ocabra_response_headers

    stt_headers = await ocabra_response_headers(
        request, model_manager, worker_key, profile.base_model_id
    )
    # OpenAI format: {"text": "..."}
    normalized_format = str(response_format).lower()
    if via_backend_forward:
        if isinstance(sidecar_result, str):
            return PlainTextResponse(sidecar_result, headers=stt_headers)
        return JSONResponse(content=sidecar_result, headers=stt_headers)
    if normalized_format in {"text", "srt", "vtt"}:
        return PlainTextResponse(resp.text, headers=stt_headers)
    return JSONResponse(content=resp.json(), headers=stt_headers)


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
        router_resolver=getattr(request.app.state, "router_resolver", None),
        request_state=request.state,
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


@router.post("/audio/speech", summary="Generate speech", responses=NON_STREAMING_LOAD_RESPONSE_DOC)
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

            target, peer = await resolve_federated(
                model_id, model_manager, federation_manager, profile_registry
            )
            if target == "remote":
                request.state.federation_remote_node_id = peer.peer_id
                # TTS is streaming audio — proxy as stream.
                response_format = str(body.get("response_format") or "mp3").lower()
                media_type = _AUDIO_CONTENT_TYPES.get(response_format, "audio/mpeg")
                return StreamingResponse(
                    federation_manager.proxy_stream(
                        peer=peer,
                        path=request.url.path,
                        body=body,
                        headers=dict(request.headers),
                    ),
                    media_type=media_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="speech.{response_format}"',
                    },
                )
    # --- End federation hook ---

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
        router_resolver=getattr(request.app.state, "router_resolver", None),
        request_state=request.state,
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
            headers={"Retry-After": "10"},
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

    from ocabra.api.openai._deps import ocabra_response_headers

    tts_headers = await ocabra_response_headers(
        request, model_manager, worker_key, profile.base_model_id
    )
    tts_headers["Content-Disposition"] = f'attachment; filename="speech.{response_format}"'

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
                headers=tts_headers,
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
        headers=tts_headers,
    )


@router.post("/audio/generate", summary="Generate music", responses=NON_STREAMING_LOAD_RESPONSE_DOC)
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

    fed_resp = await try_proxy_json(
        request,
        model_id=model_id,
        body=body,
        federation_manager=get_federation_manager(request),
        model_manager=model_manager,
        profile_registry=profile_registry,
        logger=logger,
    )
    if fed_resp is not None:
        fed_resp.headers["Content-Disposition"] = f'attachment; filename="music.{response_format}"'
        return fed_resp

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
        router_resolver=getattr(request.app.state, "router_resolver", None),
        request_state=request.state,
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
            headers={"Retry-After": "10"},
        )

    try:
        # ACE-Step backend stores _workers keyed by backend_model_id (the
        # short suffix like "turbo"), not the canonical worker_key
        # ("acestep/turbo"). Use backend_model_id to find the loaded worker.
        result = await backend.forward_request(state.backend_model_id, "/generate", merged_body)
    except TimeoutError as exc:
        raise _openai_error(
            str(exc),
            "server_error",
            code="generation_timeout",
            status_code=504,
            headers={"Retry-After": "5"},
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
    from ocabra.api.openai._deps import ocabra_response_headers

    music_headers = await ocabra_response_headers(
        request, model_manager, worker_key, profile.base_model_id
    )
    music_headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers=music_headers,
    )
