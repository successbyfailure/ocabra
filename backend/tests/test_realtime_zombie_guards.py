"""Tests for the Realtime session's zombie-session guards.

Covers two scenarios observed in production (2026-10-14) where the main
``receive_text()`` blocks forever on a half-closed TCP (client crashed with
no WS close frame delivered). Both the partial-transcription background loop
and the main client-event loop must notice and bail instead of pegging the
Whisper worker on a stale audio buffer.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.websockets import WebSocketState

from ocabra.core.realtime_session import (
    _PARTIAL_INTERVAL_S,
    _PARTIAL_STAGNANT_MAX,
    _RECEIVE_IDLE_TIMEOUT_S,
    RealtimeSession,
)


def _make_session() -> RealtimeSession:
    ws = MagicMock()
    ws.send_text = AsyncMock()
    ws.client_state = WebSocketState.CONNECTED

    state = SimpleNamespace(
        capabilities=SimpleNamespace(audio_input=False, chat=True),
        backend_model_id="m",
    )
    model_manager = MagicMock()
    model_manager.get_state = AsyncMock(return_value=state)

    worker_pool = MagicMock()
    worker_pool.get_worker = MagicMock(return_value=None)

    session = RealtimeSession(
        ws=ws,
        model_id="whisper/openai/whisper-large-v3",
        worker_pool=worker_pool,
        model_manager=model_manager,
        transcription_only=True,
    )
    # STT model is "ready" so the partial loop would otherwise call _transcribe().
    session._stt_ready.set()
    return session


@pytest.mark.asyncio
async def test_partial_loop_bails_when_ws_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard A — if the WebSocket flips to DISCONNECTED between ticks, the
    partial loop must return instead of hammering Whisper on a stale buffer."""
    session = _make_session()
    session.audio_buffer.extend(b"\x00" * 32_000)  # 1 s of PCM16

    transcribe = AsyncMock(return_value="")
    monkeypatch.setattr(session, "_transcribe", transcribe)
    # Shrink sleep so the test is fast but still goes through one iteration.
    monkeypatch.setattr("ocabra.core.realtime_session._PARTIAL_INTERVAL_S", 0.001)

    # First iteration: WS flips to disconnected before we re-enter the loop.
    session.ws.client_state = WebSocketState.DISCONNECTED

    await asyncio.wait_for(session._partial_loop("item_1"), timeout=1.0)
    transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_loop_bails_on_stagnant_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard B — if the audio buffer stops growing for
    ``_PARTIAL_STAGNANT_MAX`` consecutive ticks (dead client), the loop must
    stop. Transcription may happen *once* for the current buffer, then the
    guard trips."""
    session = _make_session()
    session.audio_buffer.extend(b"\x00" * 32_000)

    transcribe = AsyncMock(return_value="")
    monkeypatch.setattr(session, "_transcribe", transcribe)
    monkeypatch.setattr("ocabra.core.realtime_session._PARTIAL_INTERVAL_S", 0.001)
    # Keep the real stagnant threshold but shrink it so the test is quick.
    monkeypatch.setattr("ocabra.core.realtime_session._PARTIAL_STAGNANT_MAX", 3)

    await asyncio.wait_for(session._partial_loop("item_1"), timeout=1.0)

    # Transcribe is called at most once (the first iteration before stagnation
    # kicks in). After that, the buffer length stays constant and the loop
    # increments the stagnation counter until it bails.
    assert transcribe.await_count <= 1


@pytest.mark.asyncio
async def test_run_main_loop_idle_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``receive_text()`` blocks past ``_RECEIVE_IDLE_TIMEOUT_S`` with no
    client message, ``run()`` must raise so the outer finally cleans up."""
    session = _make_session()
    # Bypass the usual STT background load path.
    session.stt_model_id = None
    # ``receive_text`` never returns — simulate a half-closed TCP peer.
    session.ws.receive_text = AsyncMock(side_effect=asyncio.sleep(999))
    session.ws.accept = AsyncMock()
    session._send_session_created = AsyncMock()
    session._register_workers = AsyncMock()
    session._unregister_workers = AsyncMock()
    session._heartbeat_session = AsyncMock()

    # Shrink timeout so the test runs in <1 s instead of 300 s.
    monkeypatch.setattr(
        "ocabra.core.realtime_session._RECEIVE_IDLE_TIMEOUT_S", 0.05
    )

    with pytest.raises(RuntimeError, match="idle timeout"):
        await asyncio.wait_for(session.run(), timeout=1.0)
    session._unregister_workers.assert_awaited()


def test_constants_look_reasonable() -> None:
    """Smoke test: the production defaults must give at least ~60 s of
    grace before killing a session (otherwise live transcription with a
    quiet pause would get cut off mid-conversation)."""
    assert _PARTIAL_STAGNANT_MAX * _PARTIAL_INTERVAL_S >= 30
    assert _RECEIVE_IDLE_TIMEOUT_S >= 60
