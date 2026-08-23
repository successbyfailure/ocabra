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
from ocabra.core.model_manager import ModelStatus
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
    # Disable the optional parallel STT for this specific assertion — we
    # want to verify the strict "no Whisper" path.
    session._transcribe_user_audio = False

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


@pytest.mark.asyncio
async def test_native_audio_with_parallel_transcript_emits_event() -> None:
    """Native mode + transcribe_user_audio=true also runs Whisper async."""
    import asyncio

    session = _make_session(audio_input_capable=True)
    session.audio_buffer.extend(b"\x00\x01" * 1600)
    session.stt_model_id = "ollama/whisper-tiny"
    session._transcribe_user_audio = True
    session._transcribe = AsyncMock(return_value="parallel transcript")
    session._send_event = AsyncMock()

    await session._commit_audio_only()

    # Drain any scheduled fire-and-forget tasks.
    pending = [
        t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()
    ]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    session._transcribe.assert_awaited_once()
    transcript_event = [
        c.kwargs
        for c in session._send_event.await_args_list
        if c.args
        and c.args[0] == "conversation.item.input_audio_transcription.completed"
    ]
    assert transcript_event, "parallel STT did not emit transcription event"
    assert transcript_event[0]["transcript"] == "parallel transcript"


@pytest.mark.asyncio
async def test_native_audio_with_transcribe_disabled_skips_parallel_stt() -> None:
    """transcribe_user_audio=false keeps the strict no-Whisper guarantee."""
    session = _make_session(audio_input_capable=True)
    session.audio_buffer.extend(b"\x00\x01" * 1600)
    session.stt_model_id = "ollama/whisper-tiny"
    session._transcribe_user_audio = False
    session._transcribe = AsyncMock(side_effect=AssertionError("STT must not run"))

    await session._commit_audio_only()

    session._transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_should_use_native_audio_output_reads_capability() -> None:
    """Symmetric helper for output bypass — defaults to False today."""
    ws = MagicMock()
    ws.send_text = AsyncMock()
    state = SimpleNamespace(
        capabilities=BackendCapabilities(chat=True, audio_output=True),
        backend_model_id="m",
    )
    mm = MagicMock()
    mm.get_state = AsyncMock(return_value=state)
    wp = MagicMock()
    wp.get_worker = MagicMock(return_value=None)
    session = RealtimeSession(
        ws=ws, model_id="vllm/omni", worker_pool=wp, model_manager=mm
    )

    assert await session._should_use_native_audio_output() is True


@pytest.mark.asyncio
async def test_transcribe_holds_realtime_activity_lease(monkeypatch) -> None:
    session = _make_session(audio_input_capable=False)
    session._request_recorder = AsyncMock()
    session.stt_model_id = "whisper/faster-whisper-small"
    worker = SimpleNamespace(port=12345)
    session._ensure_stt_worker = AsyncMock(return_value=worker)
    session._model_manager.begin_request.return_value = "request-1"

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"text": "hola"}
    client = AsyncMock()
    client.post.return_value = response
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "ocabra.core.realtime_session.httpx.AsyncClient",
        MagicMock(return_value=context),
    )

    assert await session._transcribe(b"\x00\x01" * 100) == "hola"
    session._model_manager.begin_request.assert_called_once_with(
        session.stt_model_id,
        source="realtime_stt",
    )
    session._model_manager.end_request.assert_called_once_with(
        session.stt_model_id,
        "request-1",
    )
    assert session._request_recorder.await_args.kwargs["request_kind"] == (
        "realtime_transcription"
    )
    assert session._request_recorder.await_args.kwargs["status_code"] == 200
    assert session._request_recorder.await_args.kwargs["output_tokens"] == 1


@pytest.mark.asyncio
async def test_stream_llm_records_realtime_chat_operation() -> None:
    session = _make_session(audio_input_capable=False)
    session._request_recorder = AsyncMock()
    session._worker_pool.get_worker.return_value = SimpleNamespace(port=12345)

    async def _stream(*args, **kwargs):
        yield b'data: {"choices":[{"delta":{"content":"hola mundo"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    session._worker_pool.forward_stream = _stream
    chunks = [chunk async for chunk in session._stream_llm(
        [{"role": "user", "content": "di hola"}]
    )]

    assert chunks == ["hola mundo"]
    recorded = session._request_recorder.await_args.kwargs
    assert recorded["request_kind"] == "realtime_chat"
    assert recorded["status_code"] == 200
    assert recorded["input_tokens"] == 2
    assert recorded["output_tokens"] == 2


@pytest.mark.asyncio
async def test_synthesize_records_realtime_tts_operation(monkeypatch) -> None:
    session = _make_session(audio_input_capable=False)
    session.tts_model_id = "tts/kokoro"
    session._tts_ready.set()
    session._request_recorder = AsyncMock()
    session._worker_pool.get_worker.return_value = SimpleNamespace(port=12345)
    session._send_event = AsyncMock()

    response = MagicMock(content=b"audio")
    response.raise_for_status = MagicMock()
    client = AsyncMock()
    client.post.return_value = response
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "ocabra.core.realtime_session.httpx.AsyncClient",
        MagicMock(return_value=context),
    )

    await session._synthesize_and_send("hola mundo", "resp_1", "item_1")

    recorded = session._request_recorder.await_args.kwargs
    assert recorded["request_kind"] == "realtime_tts"
    assert recorded["status_code"] == 200
    assert recorded["input_tokens"] == 2
    assert recorded["output_tokens"] == 0


@pytest.mark.asyncio
async def test_ensure_stt_worker_waits_for_unload_transition(monkeypatch) -> None:
    session = _make_session(audio_input_capable=False)
    session.stt_model_id = "whisper/faster-whisper-small"
    worker = SimpleNamespace(port=12345)
    session._model_manager.get_state = AsyncMock(
        side_effect=[
            SimpleNamespace(status=ModelStatus.UNLOADING),
            SimpleNamespace(status=ModelStatus.UNLOADED),
        ]
    )
    session._model_manager.load = AsyncMock()
    session._worker_pool.get_worker = MagicMock(side_effect=[None, worker])
    monkeypatch.setattr(
        "ocabra.config.settings.model_load_wait_timeout_s",
        2,
    )

    assert await session._ensure_stt_worker() is worker
    session._model_manager.load.assert_awaited_once_with(session.stt_model_id)
