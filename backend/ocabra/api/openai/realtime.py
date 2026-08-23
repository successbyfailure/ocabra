"""
WebSocket endpoint for the OpenAI Realtime API.

    GET /v1/realtime?model=<model_id>

Upgrades to WebSocket and runs a :class:`RealtimeSession` that coordinates
the bidirectional audio pipeline (STT -> LLM -> TTS).

Authentication follows the same pattern as the internal WebSocket endpoint:
checks ``Authorization: Bearer <key>`` header, ``ocabra_session`` cookie,
or allows anonymous access when ``require_api_key_openai`` is disabled.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ocabra.api._deps_auth import UserContext
from ocabra.core.realtime_session import RealtimeSession
from ocabra.stats.collector import record_realtime_request

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["OpenAI Realtime"])


async def _authenticate_ws(websocket: WebSocket) -> UserContext | None:
    """Authenticate a WebSocket connection.

    Checks, in order:
    1. ``Authorization: Bearer <key>`` header
    2. ``ocabra_session`` cookie (JWT)
    3. Anonymous access if ``require_api_key_openai`` is disabled

    Returns the resolved identity when authorized, otherwise ``None``. Raw
    credentials are never retained on the WebSocket or written to logs.
    """
    from ocabra.config import settings

    # 1. Bearer token from header (or query param for WebSocket clients)
    auth_header = websocket.headers.get("authorization", "")
    bearer_token = ""
    if auth_header.startswith("Bearer "):
        bearer_token = auth_header[len("Bearer ") :]

    # Some WebSocket clients pass the key as a query parameter
    if not bearer_token:
        bearer_token = websocket.query_params.get("api_key", "")

    if bearer_token:
        from ocabra.api._deps_auth import _resolve_api_key
        from ocabra.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            return await _resolve_api_key(bearer_token, session)

    # 2. JWT cookie
    cookie_token = websocket.cookies.get("ocabra_session")
    if cookie_token:
        from ocabra.api._deps_auth import _resolve_jwt_cookie
        from ocabra.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            return await _resolve_jwt_cookie(cookie_token, session)

    # 3. Anonymous access
    if not settings.require_api_key_openai:
        from ocabra.api._deps_auth import _build_anonymous_context
        from ocabra.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            return await _build_anonymous_context(session)

    return None


@router.websocket("/realtime")
async def realtime_ws(
    websocket: WebSocket,
    model: str = Query(..., description="Model ID: the LLM for the session, or the Whisper STT model when intent=transcription"),
    intent: str = Query(default="", description="Set to 'transcription' for a transcription-only session (audio -> transcript events, no LLM/TTS)."),
    conversation_id: str = Query(default="", description="Resume the speaker namespace of a prior transcription session (reconnections) — stable speaker ids across sessions of the same audio."),
    num_speakers: int = Query(default=0, description="Expected distinct speakers; caps the diarization namespace to avoid id drift in long sessions (0 = unbounded)."),
) -> None:
    """OpenAI Realtime API WebSocket endpoint.

    Establishes a bidirectional audio session. The client sends PCM16 audio
    and receives transcriptions, LLM text, and synthesized audio back.

    Query params:
        model: Canonical model ID for the LLM (required).

    Protocol:
        Client events: session.update, input_audio_buffer.append,
            input_audio_buffer.commit, input_audio_buffer.clear,
            response.create, response.cancel.
        Server events: session.created, session.updated,
            input_audio_buffer.speech_started, input_audio_buffer.speech_stopped,
            input_audio_buffer.committed, conversation.item.created,
            response.created, response.audio.delta, response.audio.done,
            response.audio_transcript.delta, response.audio_transcript.done,
            response.done, error.

    Note:
        Tool calls are not yet implemented. The ``tools`` and ``tool_choice``
        fields in session.update are accepted but ignored.
    """
    # Authenticate before accepting the WebSocket
    user = await _authenticate_ws(websocket)
    if user is None:
        await websocket.close(code=1008, reason="Authentication required")
        return

    websocket.state.auth_user = user
    await websocket.accept()

    worker_pool = websocket.app.state.worker_pool
    model_manager = websocket.app.state.model_manager
    profile_registry = websocket.app.state.profile_registry

    # Resolve profile_id → canonical model_id (same as REST endpoints)
    resolved_model_id = model
    profile = await profile_registry.get(model)
    if profile and profile.enabled:
        resolved_model_id = profile.base_model_id
    elif "/" not in model:
        # Not a profile and not a canonical id — try legacy fallback
        states = await model_manager.list_states()
        match = next(
            (s for s in states if s.backend_model_id == model or s.model_id == model),
            None,
        )
        if match:
            resolved_model_id = match.model_id

    async def _record_realtime(**kwargs: object) -> None:
        try:
            await record_realtime_request(websocket, **kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - stats must not break Realtime
            logger.warning(
                "realtime_stats_record_failed",
                model_id=resolved_model_id,
                request_kind=kwargs.get("request_kind"),
                error=str(exc),
            )

    session = RealtimeSession(
        ws=websocket,
        model_id=resolved_model_id,
        worker_pool=worker_pool,
        model_manager=model_manager,
        user=user,
        transcription_only=(intent.strip().lower() == "transcription"),
        request_recorder=_record_realtime,
    )

    # Auto-enable diarization when a transcription session is opened with a diarized
    # whisper profile (matches the batch endpoint behaviour); the client can still
    # override via session.update input_audio_transcription.diarize.
    if session.transcription_only:
        try:
            from ocabra.backends.whisper_backend import _should_enable_diarization

            overrides = (profile.load_overrides if profile else None) or {}
            if _should_enable_diarization(model, overrides):
                session._stt_diarize = True  # noqa: SLF001
        except Exception:  # noqa: BLE001 — never block the session on this
            pass
        if num_speakers and num_speakers > 0:
            session._max_speakers = num_speakers  # noqa: SLF001
        if conversation_id.strip():
            session._conversation_id = conversation_id.strip()  # noqa: SLF001
            session._load_registry()  # noqa: SLF001

    logger.info(
        "realtime_session_started",
        model=model,
        resolved_model_id=resolved_model_id,
        session_id=session._session_id,
        username=user.username,
        api_key_name=user.api_key_name,
        client_addr=websocket.client.host if websocket.client else None,
        intent="transcription" if session.transcription_only else "conversation",
    )

    session_started_at = datetime.now(UTC)
    session_started_monotonic = time.monotonic()
    session_status = 101
    session_error: str | None = None
    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info(
            "realtime_session_disconnected",
            session_id=session._session_id,
            username=user.username,
            api_key_name=user.api_key_name,
        )
    except asyncio.CancelledError:
        session_status = 499
        session_error = "Realtime session cancelled"
        raise
    except Exception as exc:
        session_status = 500
        session_error = str(exc)
        logger.warning(
            "realtime_session_error",
            session_id=session._session_id,
            username=user.username,
            api_key_name=user.api_key_name,
            error=str(exc),
        )
    finally:
        # Cancel any in-progress response and partial-transcription loop
        if session._response_task and not session._response_task.done():
            session._response_task.cancel()
        if session._partial_task and not session._partial_task.done():
            session._partial_task.cancel()
        if session._background_load_task and not session._background_load_task.done():
            session._background_load_task.cancel()
        await _record_realtime(
            session_id=session._session_id,
            model_id=resolved_model_id,
            started_at=session_started_at,
            duration_ms=(time.monotonic() - session_started_monotonic) * 1000,
            request_kind="realtime_session",
            status_code=session_status,
            error_message=session_error,
            input_tokens=None,
            output_tokens=None,
        )
        logger.info(
            "realtime_session_ended",
            session_id=session._session_id,
            username=user.username,
            api_key_name=user.api_key_name,
        )
