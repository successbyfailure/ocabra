"""``POST /ocabra/sessions/{id}/kill`` must really end the Realtime session.

Antes solo borraba la entrada del registro: el WebSocket seguía abierto, la
respuesta en curso seguía generando y el bucle de recepción seguía esperando
al cliente. Ahora el registro llama al ``close`` de la propia sesión, que hace
el mismo desmontaje que una desconexión del cliente.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.websockets import WebSocketDisconnect, WebSocketState

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.api.internal import sessions as sessions_api
from ocabra.core.realtime_session import RealtimeSession, RealtimeSessionClosed
from ocabra.core.session_registry import SessionRegistry


def _make_session(registry: SessionRegistry) -> RealtimeSession:
    ws = MagicMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    ws.client_state = WebSocketState.CONNECTED

    async def _never_answers(*args, **kwargs):
        await asyncio.sleep(999)

    ws.receive_text = _never_answers
    model_manager = MagicMock()
    model_manager.get_state = AsyncMock(
        return_value=SimpleNamespace(
            capabilities=SimpleNamespace(audio_input=False, chat=True),
            backend_model_id="m",
        )
    )
    worker_pool = MagicMock()
    worker_pool.get_worker = MagicMock(return_value=None)
    session = RealtimeSession(
        ws=ws,
        model_id="whisper/openai/whisper-large-v3",
        worker_pool=worker_pool,
        model_manager=model_manager,
        transcription_only=True,
    )
    session.stt_model_id = "whisper/openai/whisper-large-v3"
    session._send_session_created = AsyncMock()
    # Que la carga de STT no haga nada: el test va del cierre, no de cargar.
    session._load_stt_then_ready = AsyncMock()
    session.set_session_registry(registry)
    return session


async def _wait_registered(registry: SessionRegistry, session_id: str) -> None:
    for _ in range(100):
        if registry.get(session_id) is not None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("session never registered")


async def test_registry_close_ends_the_live_session() -> None:
    registry = SessionRegistry()
    session = _make_session(registry)
    run_task = asyncio.create_task(session.run())
    await _wait_registered(registry, session._session_id)

    # A response in flight must be cancelled, not left generating.
    response = asyncio.create_task(asyncio.sleep(999))
    session._response_task = response

    closed = await registry.close(session._session_id)

    assert closed is True
    with pytest.raises(RealtimeSessionClosed):
        await asyncio.wait_for(run_task, timeout=2.0)
    assert response.cancelled()
    session.ws.close.assert_awaited_once()
    assert session.ws.close.await_args.kwargs["code"] == 4001
    sent = [call.args[0] for call in session.ws.send_text.await_args_list]
    assert any('"session_closed"' in payload for payload in sent)
    # Workers released: no entry, no holder, no closer left behind.
    assert registry.get(session._session_id) is None
    assert not registry.has_active_session_holding(session.stt_model_id)
    assert session._session_id not in registry._closers


def test_closed_session_is_an_ordinary_disconnect() -> None:
    """The endpoint's teardown keys on ``WebSocketDisconnect``."""
    assert issubclass(RealtimeSessionClosed, WebSocketDisconnect)


async def test_close_is_idempotent() -> None:
    registry = SessionRegistry()
    session = _make_session(registry)
    await session.close()
    await session.close()
    session.ws.close.assert_awaited_once()


async def test_close_without_closer_only_unregisters() -> None:
    """Entries rehydrated from Redis have no socket in this process."""
    registry = SessionRegistry()
    await registry.register("sess_x", ["w1"])
    assert await registry.close("sess_x") is False
    assert registry.get("sess_x") is None


async def test_close_drops_entry_even_if_closer_fails() -> None:
    registry = SessionRegistry()
    closer = AsyncMock(side_effect=RuntimeError("boom"))
    await registry.register("sess_y", ["w1"], closer=closer)
    assert await registry.close("sess_y") is False
    closer.assert_awaited_once_with("admin_kill")
    assert registry.get("sess_y") is None
    assert not registry.has_active_session_holding("w1")


async def test_kill_endpoint_closes_connection() -> None:
    registry = SessionRegistry()
    closer = AsyncMock()
    await registry.register("sess_z", ["w1"], closer=closer)

    app = FastAPI()
    app.include_router(sessions_api.router, prefix="/ocabra")
    app.state.session_registry = registry
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id=None, username="admin", role="system_admin"
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.post("/ocabra/sessions/sess_z/kill")
        missing = await client.post("/ocabra/sessions/sess_z/kill")

    assert resp.status_code == 200
    assert resp.json() == {"session_id": "sess_z", "killed": True, "connection_closed": True}
    closer.assert_awaited_once()
    assert missing.status_code == 404
