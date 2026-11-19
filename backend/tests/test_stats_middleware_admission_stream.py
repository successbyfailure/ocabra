"""StatsMiddleware: incremental stream parsing (BUG-36), no buffering of
uploads (BUG-37) and admission for multipart/WebSocket (BUG-38, now in
``AdmissionMiddleware``, REF-08)."""

from __future__ import annotations

import asyncio
import json
import random
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest
from starlette.testclient import WebSocketDenialResponse

from ocabra.core import admission
from ocabra.core.model_manager import ModelStatus
from ocabra.core.request_telemetry import RequestRuntimeTelemetry
from ocabra.stats import collector
from ocabra.stats.collector import (
    _audio_duration_s,
    _extract_last_payload_from_stream,
    _extract_stream_error,
    _resolve_inflight_model_id,
    _stream_output_tokens,
    _StreamAccumulator,
)
from tests._request_stack import install_request_stack

# ── BUG-36: the running parse gives what the whole-body parse gave ──────────


def _sse(*events: object) -> bytes:
    return b"".join(
        b"data: " + (e if isinstance(e, bytes) else json.dumps(e).encode()) + b"\n\n"
        for e in events
    )


_SSE_STREAM = _sse(
    {"choices": [{"delta": {"role": "assistant"}}]},
    {"choices": [{"delta": {"reasoning_content": "pienso en ello"}}]},
    {"choices": [{"delta": {"content": "hola mundo ñandú 😀"}}]},
    {"choices": [{"delta": {"tool_calls": [{"function": {"name": "f", "arguments": '{"a":'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": " 1}"}}]}}]},
    {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 11}},
    {"error": {"message": "algo falló", "code": "stream_error"}},
    b"[DONE]",
)
_SSE_CRLF = _SSE_STREAM.replace(b"\n", b"\r\n")
_NDJSON_STREAM = (
    b'{"message":{"role":"assistant","thinking":"a ver"},"done":false}\n'
    b'{"message":{"content":"una respuesta larga"},"done":false}\n'
    b'{"message":{"tool_calls":[{"function":{"name":"x","arguments":{"k":"v"}}}]},"done":false}\n'
    b'{"response":"generate style","done":false}\n'
    b'{"done":true,"prompt_eval_count":3,"eval_count":9,"eval_duration":1000000}'
)  # no trailing newline: the last line must still count


def _feed_split(body: bytes, content_type: str, cuts: list[int]) -> _StreamAccumulator:
    acc = _StreamAccumulator(content_type)
    prev = 0
    for cut in [*cuts, len(body)]:
        acc.feed(body[prev:cut])
        prev = cut
    acc.finish()
    return acc


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (_SSE_STREAM, "text/event-stream"),
        (_SSE_CRLF, "text/event-stream; charset=utf-8"),
        (_NDJSON_STREAM, "application/x-ndjson"),
        (b'{"error": "oops"}\n' + b"\x00\xffbinary\n", "audio/mpeg"),
    ],
)
def test_accumulator_matches_whole_body_parsers_at_any_split(body: bytes, content_type: str):
    expected = (
        _stream_output_tokens(body, content_type),
        _stream_output_tokens(body, content_type, include_reasoning=True),
        _extract_last_payload_from_stream(body, content_type),
        _extract_stream_error(body, content_type),
    )
    rng = random.Random(1234)
    splits = [[i] for i in range(1, len(body))]  # every byte boundary (mid-UTF-8 too)
    splits += [sorted(rng.sample(range(1, len(body)), 12)) for _ in range(20)]
    for cuts in splits:
        acc = _feed_split(body, content_type, cuts)
        got = (
            acc.output_tokens(),
            acc.output_tokens(include_reasoning=True),
            acc.last_payload(),
            acc.stream_error(),
        )
        assert got == expected, cuts


def test_accumulator_memory_is_bounded_and_keeps_no_body_by_default():
    acc = _StreamAccumulator("text/event-stream")
    event = _sse({"choices": [{"delta": {"content": "x " * 50}}]})
    for _ in range(2000):
        acc.feed(event)
    assert acc._body == []
    assert len(acc._pending) == 0
    assert acc.output_tokens() == 100_000
    # A line that never ends is dropped once over the cap, not grown forever.
    other = _StreamAccumulator("audio/pcm")
    for _ in range(100):
        other.feed(b"\0" * 4096)
    assert len(other._pending) <= _StreamAccumulator._MAX_LINE_OTHER


def _streaming_app(chunks: list[bytes], media_type: str, path: str = "/v1/chat/completions"):
    app = FastAPI()
    app.state.profile_registry = SimpleNamespace(
        get=AsyncMock(return_value=None), list_by_model=AsyncMock(return_value=[])
    )
    mm = MagicMock()
    mm.is_in_maintenance.return_value = False
    mm.try_begin_request.return_value = "req-1"
    mm.get_state = AsyncMock(return_value=None)
    app.state.model_manager = mm

    @app.post(path)
    async def _endpoint():
        async def gen():
            for chunk in chunks:
                yield chunk

        return StreamingResponse(gen(), media_type=media_type)

    install_request_stack(app)
    return app, mm


def test_stream_parse_work_is_linear_in_stream_length():
    """BUG-36: before, every chunk re-parsed the whole body until the first
    token (and every 0.5 s after it), O(n²) on the event loop. A stream that
    takes a while to emit visible tokens (role chunks, prefill progress)
    made that explicit. Count the JSON parses: one per event now."""
    n = 300
    role = _sse({"choices": [{"delta": {"role": "assistant"}}]})
    chunks = [role] * n + [_sse({"choices": [{"delta": {"content": "fin"}}]})]
    app, _mm = _streaming_app(chunks, "text/event-stream")
    calls = 0

    def counting_loads(data, *args, **kwargs):
        nonlocal calls
        calls += 1
        return json.loads(data, *args, **kwargs)

    record_stat = AsyncMock()
    fake_json = SimpleNamespace(loads=counting_loads, dumps=json.dumps)
    with (
        patch("ocabra.stats.collector._record_stat", new=record_stat),
        patch.object(collector, "json", fake_json),
    ):
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "vllm/x", "messages": [], "stream": True},
        )
    assert response.status_code == 200
    assert response.content == b"".join(chunks)  # untouched, chunk by chunk
    assert record_stat.await_args.kwargs["output_tokens"] == 1
    # Old code: ~n²/2 = 45.000 parses. Linear: one per event (+ slack).
    assert calls <= 2 * (n + 1) + 10, calls


def test_tts_stream_still_hands_the_whole_audio_to_record_stat():
    """``audio_seconds`` of generated speech comes from the streamed body."""
    chunks = [b"RIFF....WAVE", b"\x01" * 1000, b"\n\x02" * 500, b"tail"]
    app, _mm = _streaming_app(chunks, "audio/wav", path="/v1/audio/speech")
    record_stat = AsyncMock()
    with patch("ocabra.stats.collector._record_stat", new=record_stat):
        response = TestClient(app).post("/v1/audio/speech", json={"model": "tts/x", "input": "hi"})
    assert response.content == b"".join(chunks)
    assert record_stat.await_args.kwargs["output_audio"] == (b"".join(chunks), "audio/wav")


def test_text_stream_does_not_keep_the_body():
    chunks = [_sse({"choices": [{"delta": {"content": "a b"}}]})] * 3
    app, _mm = _streaming_app(chunks, "text/event-stream")
    record_stat = AsyncMock()
    with patch("ocabra.stats.collector._record_stat", new=record_stat):
        TestClient(app).post("/v1/chat/completions", json={"model": "m/x", "stream": True})
    kwargs = record_stat.await_args.kwargs
    assert kwargs["output_audio"] is None
    assert kwargs["output_tokens"] == 6


def _wav_header(byte_rate: int) -> bytes:
    return (
        b"RIFF"
        + struct.pack("<I", 0)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, byte_rate // 2, byte_rate, 2, 16)
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )


@pytest.mark.asyncio
async def test_truncated_audio_keeps_wav_duration_from_total_size():
    head = _wav_header(48000) + b"\0" * 100
    total = len(_wav_header(48000)) + 48000 * 3
    assert await _audio_duration_s(head, "audio/wav", total) == 3.0
    assert await _audio_duration_s(b"\0" * 10, "audio/pcm", 48000 * 2) == 2.0
    assert await _audio_duration_s(b"ID3...", "audio/mpeg", 10_000_000) is None


def test_stream_ttft_still_marked_on_the_chunk_with_the_first_token():
    acc = _StreamAccumulator("text/event-stream")
    request = SimpleNamespace(state=SimpleNamespace(runtime_telemetry=RequestRuntimeTelemetry(0.0)))
    acc.feed(_sse({"choices": [{"delta": {"role": "assistant"}}]}))
    collector._mark_stream_first_token_from(request, acc)
    assert request.state.runtime_telemetry.ttft_ms is None
    # The JSON is complete but its newline has not arrived yet.
    acc.feed(b'data: {"choices":[{"delta":{"content":"hola"}}]}')
    collector._mark_stream_first_token_from(request, acc)
    assert request.state.runtime_telemetry.ttft_ms is not None


# ── BUG-37: uploads are not read by the middleware ─────────────────────────


def _multipart(fields: list[tuple[str, bytes, str | None]], boundary: str = "XyZ") -> bytes:
    out = b""
    for name, value, filename in fields:
        out += f"--{boundary}\r\n".encode()
        disp = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disp += f'; filename="{filename}"\r\nContent-Type: audio/wav'
        out += disp.encode() + b"\r\n\r\n" + value + b"\r\n"
    return out + f"--{boundary}--\r\n".encode()


def _upload_app(mm) -> tuple[FastAPI, dict]:
    app = FastAPI()
    profile = SimpleNamespace(
        profile_id="whisper-public",
        base_model_id="whisper/large",
        load_overrides=None,
        enabled=True,
        is_default=True,
        profile_kind="execution",
    )
    app.state.profile_registry = SimpleNamespace(
        get=AsyncMock(return_value=profile), list_by_model=AsyncMock(return_value=[profile])
    )
    app.state.model_manager = mm
    seen: dict = {}

    @app.post("/v1/audio/transcriptions")
    async def _transcribe(request: Request):
        form = await request.form()
        seen["model"] = form["model"]
        seen["file"] = await form["file"].read()
        request.state.stats_model_id = "whisper-public"
        return JSONResponse({"text": "hola que tal"})

    install_request_stack(app)
    return app, seen


def _plain_mm(*, maintenance: bool = False, inflight: int = 0, cap: int = 4) -> MagicMock:
    mm = MagicMock()
    mm.is_in_maintenance.return_value = maintenance
    mm.inflight_count.return_value = inflight
    mm.effective_inflight_cap.return_value = cap
    mm.get_state = AsyncMock(return_value=None)
    return mm


def test_multipart_upload_is_not_read_by_the_middleware():
    """BUG-37: ``request.json()`` read the whole audio into memory again."""
    audio = bytes(range(256)) * 8192  # 2 MiB
    body = _multipart([("model", b"whisper-public", None), ("file", audio, "a.wav")])
    app, seen = _upload_app(_plain_mm())
    json_calls = 0
    original_json = StarletteRequest.json

    async def spy_json(self):
        nonlocal json_calls
        json_calls += 1
        return await original_json(self)

    record_stat = AsyncMock()
    with (
        patch.object(StarletteRequest, "json", spy_json),
        patch("ocabra.stats.collector._record_stat", new=record_stat),
    ):
        response = TestClient(app).post(
            "/v1/audio/transcriptions",
            content=body,
            headers={"content-type": "multipart/form-data; boundary=XyZ"},
        )
    assert response.status_code == 200
    assert json_calls == 0
    assert seen == {"model": "whisper-public", "file": audio}  # replayed intact
    kwargs = record_stat.await_args.kwargs
    assert kwargs["model_id"] == "whisper-public"
    assert kwargs["output_tokens"] == 3


@pytest.mark.asyncio
async def test_peek_stops_at_the_file_part_and_replays_everything():
    body = _multipart([("file", b"A" * 300_000, "a.wav"), ("model", b"late", None)])
    messages = [
        {"type": "http.request", "body": body[i : i + 65536], "more_body": i + 65536 < len(body)}
        for i in range(0, len(body), 65536)
    ]
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        return messages[reads - 1]

    model, replay = await admission.peek_multipart_field(
        receive, "multipart/form-data; boundary=XyZ"
    )
    assert model is None  # after the file: not looked for
    assert reads == 1  # the file was not read
    got = b""
    while True:
        msg = await replay()
        got += msg["body"]
        if not msg["more_body"]:
            break
    assert got == body


@pytest.mark.asyncio
async def test_peek_finds_model_split_across_messages():
    body = _multipart([("model", b"whisper-public", None), ("file", b"x" * 10, "a.wav")])
    parts = [body[:30], body[30:45], body[45:]]
    queue = [{"type": "http.request", "body": p, "more_body": i < 2} for i, p in enumerate(parts)]

    async def receive():
        return queue.pop(0)

    model, _replay = await admission.peek_multipart_field(
        receive, 'multipart/form-data; boundary="XyZ"'
    )
    assert model == "whisper-public"


# ── BUG-38: multipart and WebSocket under maintenance and the cap ──────────


def _post_upload(app):
    body = _multipart([("model", b"whisper-public", None), ("file", b"RIFF" * 10, "a.wav")])
    return TestClient(app).post(
        "/v1/audio/transcriptions",
        content=body,
        headers={"content-type": "multipart/form-data; boundary=XyZ"},
    )


def test_multipart_gets_503_under_maintenance():
    mm = _plain_mm(maintenance=True)
    app, seen = _upload_app(mm)
    with patch("ocabra.stats.collector._record_stat", new=AsyncMock()):
        response = _post_upload(app)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    assert response.json()["error"]["code"] == "model_maintenance"
    assert seen == {}
    mm.is_in_maintenance.assert_called_with("whisper/large")


def test_multipart_gets_429_at_capacity_and_is_counted_once():
    mm = _plain_mm(inflight=4, cap=4)
    mm.try_begin_request.return_value = None  # at the cap
    app, seen = _upload_app(mm)
    with patch("ocabra.stats.collector._record_stat", new=AsyncMock()):
        response = _post_upload(app)
    assert response.status_code == 429
    assert response.headers["retry-after"] == "10"
    assert response.json()["error"]["code"] == "model_at_capacity"
    assert seen == {}
    assert mm.effective_inflight_cap.call_args.kwargs["request_kind"] == "audio_transcription"
    mm.end_request.assert_not_called()

    mm = _plain_mm(inflight=3, cap=4)
    mm.try_begin_request.return_value = "req-1"
    app, seen = _upload_app(mm)
    with patch("ocabra.stats.collector._record_stat", new=AsyncMock()):
        response = _post_upload(app)
    assert response.status_code == 200
    assert seen["model"] == "whisper-public"
    # REF-08: admission counts the upload (check and count in one step), once,
    # and releases it after the response; the endpoint takes that slot.
    mm.try_begin_request.assert_called_once()
    assert mm.try_begin_request.call_args.args[0] == "whisper/large"
    assert mm.try_begin_request.call_args.kwargs["request_kind"] == "audio_transcription"
    assert mm.try_begin_request.call_args.kwargs["profile_id"] == "whisper-public"
    mm.begin_request.assert_not_called()
    mm.end_request.assert_called_once_with("whisper/large", "req-1")


def _ws_app(mm) -> FastAPI:
    app = FastAPI()
    app.state.profile_registry = SimpleNamespace(
        get=AsyncMock(return_value=None), list_by_model=AsyncMock(return_value=[])
    )
    app.state.model_manager = mm

    @app.websocket("/v1/realtime")
    async def _realtime(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"type": "session.created"})
        await websocket.close()

    install_request_stack(app)
    return app


def test_realtime_handshake_denied_with_503_under_maintenance():
    mm = _plain_mm(maintenance=True)
    with patch("ocabra.stats.collector._record_stat", new=AsyncMock()) as record_stat:
        with pytest.raises(WebSocketDenialResponse) as denied:
            with TestClient(_ws_app(mm)).websocket_connect("/v1/realtime?model=llm/x"):
                pass
    assert denied.value.status_code == 503
    assert denied.value.headers["retry-after"] == "30"
    assert denied.value.json()["error"]["code"] == "model_maintenance"
    assert record_stat.call_args.kwargs["status_code"] == 503


def test_realtime_handshake_denied_with_429_at_capacity():
    mm = _plain_mm(inflight=8, cap=8)
    with patch("ocabra.stats.collector._record_stat", new=AsyncMock()):
        with pytest.raises(WebSocketDenialResponse) as denied:
            with TestClient(_ws_app(mm)).websocket_connect(
                "/v1/realtime?model=whisper/x&intent=transcription"
            ):
                pass
    assert denied.value.status_code == 429
    assert denied.value.headers["retry-after"] == "2"
    assert mm.effective_inflight_cap.call_args.kwargs["request_kind"] == "realtime_transcription"


def test_realtime_session_still_opens_with_room():
    mm = _plain_mm(inflight=1, cap=4)
    with TestClient(_ws_app(mm)).websocket_connect("/v1/realtime?model=llm/x") as ws:
        assert ws.receive_json() == {"type": "session.created"}
    mm.try_begin_request.assert_not_called()


def test_websocket_denial_without_extension_closes_before_accept():
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "websocket.connect"}

    asyncio.run(
        admission.send_websocket_denial(
            admission.capacity_response("m", 1), {"type": "websocket"}, receive, send
        )
    )
    assert sent == [{"type": "websocket.close", "code": 1013, "reason": "try again later"}]


@pytest.mark.asyncio
async def test_router_admission_targets_the_warm_target_the_router_will_pick():
    """BUG-38 (second half): the cap was checked against the router's first
    target even when ``RouterResolver.pick`` sends the request to a later,
    already loaded one."""
    profiles = {
        "router": SimpleNamespace(
            profile_id="router",
            enabled=True,
            profile_kind="router",
            routing_targets=["cold", "warm"],
        ),
        "cold": SimpleNamespace(
            profile_id="cold", enabled=True, base_model_id="llama/cold", load_overrides=None
        ),
        "warm": SimpleNamespace(
            profile_id="warm", enabled=True, base_model_id="llama/warm", load_overrides=None
        ),
    }
    registry = SimpleNamespace(get=AsyncMock(side_effect=lambda pid: profiles.get(pid)))
    loaded = {"llama/warm"}
    mm = MagicMock()
    mm.get_state = AsyncMock(
        side_effect=lambda key: SimpleNamespace(
            status=ModelStatus.LOADED if key in loaded else ModelStatus.CONFIGURED
        )
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(profile_registry=registry, model_manager=mm))
    )
    assert await _resolve_inflight_model_id(request, "router") == "llama/warm"
    loaded.clear()  # nothing warm: list order, as before
    assert await _resolve_inflight_model_id(request, "router") == "llama/cold"
