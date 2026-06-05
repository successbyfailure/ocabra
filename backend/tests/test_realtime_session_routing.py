"""Tests for RealtimeSession audio-input routing.

Covers the T4 contract: when the LLM advertises ``audio_input``, the session
must skip Whisper and embed the audio as an ``input_audio`` content part for
the LLM. Otherwise the legacy Whisper -> text -> LLM flow is preserved.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ocabra.backends.base import BackendCapabilities
from ocabra.core.realtime_session import RealtimeSession


def _make_session(audio_input_capable: bool) -> RealtimeSession:
    ws = MagicMock()
    ws.send_text = AsyncMock()

    state = SimpleNamespace(
        capabilities=BackendCapabilities(chat=True, audio_input=audio_input_capable),
        backend_model_id="m",
    )
    model_manager = MagicMock()
    model_manager.get_state = AsyncMock(return_value=state)

    worker_pool = MagicMock()
    worker_pool.get_worker = MagicMock(return_value=None)

    return RealtimeSession(
        ws=ws,
        model_id="vllm/some-llm",
        worker_pool=worker_pool,
        model_manager=model_manager,
    )


@pytest.mark.asyncio
async def test_should_use_native_audio_true_when_capability_set() -> None:
    session = _make_session(audio_input_capable=True)
    assert await session._should_use_native_audio() is True


@pytest.mark.asyncio
async def test_should_use_native_audio_false_when_capability_missing() -> None:
    session = _make_session(audio_input_capable=False)
    assert await session._should_use_native_audio() is False


@pytest.mark.asyncio
async def test_should_use_native_audio_respects_explicit_stt_override() -> None:
    session = _make_session(audio_input_capable=True)
    session._input_audio_routing = "stt"
    assert await session._should_use_native_audio() is False


@pytest.mark.asyncio
async def test_commit_native_audio_embeds_input_audio_part() -> None:
    session = _make_session(audio_input_capable=True)
    session.audio_buffer.extend(b"\x00\x01" * 1600)  # 0.1s of PCM16 @ 16kHz

    # Should NOT touch Whisper at all.
    session._transcribe = AsyncMock(side_effect=AssertionError("STT must not run"))

    await session._commit_audio_only()

    assert len(session.conversation) == 1
    msg = session.conversation[0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    part = msg["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    assert isinstance(part["input_audio"]["data"], str)
    assert part["input_audio"]["data"]  # non-empty base64

    session._transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_commit_stt_path_calls_whisper_and_stores_text() -> None:
    session = _make_session(audio_input_capable=False)
    session.audio_buffer.extend(b"\x00\x01" * 1600)
    session._transcribe = AsyncMock(return_value="hello there")

    await session._commit_audio_only()

    session._transcribe.assert_awaited_once()
    assert session.conversation == [{"role": "user", "content": "hello there"}]
