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

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ocabra.core.realtime_session import RealtimeSession

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["OpenAI Realtime"])


async def _authenticate_ws(websocket: WebSocket) -> bool:
    """Authenticate a WebSocket connection.

    Checks, in order:
    1. ``Authorization: Bearer <key>`` header
    2. ``ocabra_session`` cookie (JWT)
    3. Anonymous access if ``require_api_key_openai`` is disabled

    Returns True if the connection is authorized, False otherwise.
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
            ctx = await _resolve_api_key(bearer_token, session)
            if ctx is not None:
                return True
        return False

    # 2. JWT cookie
    cookie_token = websocket.cookies.get("ocabra_session")
    if cookie_token:
        from ocabra.core.auth_manager import AuthError, decode_access_token

        try:
            decode_access_token(cookie_token)
            return True
        except AuthError:
            return False

    # 3. Anonymous access
    if not settings.require_api_key_openai:
        return True

    return False


@router.websocket("/realtime")
async def realtime_ws(
    websocket: WebSocket,
    model: str = Query(..., description="LLM model ID for the session"),
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
    if not await _authenticate_ws(websocket):
        await websocket.close(code=1008, reason="Authentication required")
        return

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

    session = RealtimeSession(
        ws=websocket,
        model_id=resolved_model_id,
        worker_pool=worker_pool,
        model_manager=model_manager,
    )

    logger.info(
        "realtime_session_started",
        model=model,
        session_id=session._session_id,
    )

    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info(
            "realtime_session_disconnected",
            session_id=session._session_id,
        )
    except Exception as exc:
        logger.warning(
            "realtime_session_error",
            session_id=session._session_id,
            error=str(exc),
        )
    finally:
        # Cancel any in-progress response
        if session._response_task and not session._response_task.done():
            session._response_task.cancel()
        logger.info(
            "realtime_session_ended",
            session_id=session._session_id,
        )
