import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from ocabra.core.request_telemetry import RequestRuntimeTelemetry, mark_prefill_since
from ocabra.core.worker_pool import InferenceTimeoutError
from ocabra.stats.collector import (
    StatsMiddleware,
    _classify_request_kind,
    _client_addr,
    _count_input_tokens,
    _extract_error_message,
    _extract_generation_ms,
    _extract_last_payload_from_stream,
    _extract_response_payload_and_rebuild,
    _extract_speculative_stats,
    _extract_stream_error,
    _extract_usage_tokens,
    _mark_stream_first_token,
    _prefill_only_prompt_tps,
    _resolve_inflight_model_id,
    _stream_error_status,
    _stream_output_tokens,
    _update_engine_runtime_telemetry,
    record_realtime_request,
)


def _fake_request(
    headers: dict[str, str], *, peer_host: str | None = "172.22.0.13"
) -> SimpleNamespace:
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host=peer_host) if peer_host else None,
    )


class TestClientAddr:
    """2026-10-31: the proxy in front of Caddy for the public domain sends
    BOTH headers, but X-Forwarded-For carries its own address, not the
    client's — only X-Real-IP has the genuine origin. _client_addr must
    prefer X-Real-IP, not just fall back to it when X-Forwarded-For is
    absent."""

    def test_prefers_x_real_ip_over_x_forwarded_for(self) -> None:
        req = _fake_request({"x-real-ip": "79.117.117.14", "x-forwarded-for": "10.0.8.5"})
        assert _client_addr(req) == "79.117.117.14"

    def test_falls_back_to_x_forwarded_for_without_x_real_ip(self) -> None:
        req = _fake_request({"x-forwarded-for": "203.0.113.9, 10.0.8.5"})
        assert _client_addr(req) == "203.0.113.9"

    def test_falls_back_to_direct_peer_without_either_header(self) -> None:
        req = _fake_request({}, peer_host="198.51.100.2")
        assert _client_addr(req) == "198.51.100.2"

    def test_returns_none_with_no_signal_at_all(self) -> None:
        req = _fake_request({}, peer_host=None)
        assert _client_addr(req) is None


def test_classify_request_kind_known_paths() -> None:
    assert _classify_request_kind("/v1/chat/completions") == "chat"
    assert _classify_request_kind("/v1/audio/transcriptions") == "audio_transcription"
    assert _classify_request_kind("/api/generate") == "ollama_generate"


class TestStreamOutputTokensReasoningVisibility:
    """2026-10-29 — verified live that a reasoning model's entire thinking
    phase was invisible to the live-progress counter (model/GPU card showed
    outputTokensEstimate: null for 25+ seconds of active work) because the
    default behaviour only counts delta.content, never delta.reasoning_content
    / message.thinking. include_reasoning=True (used only by the live-
    progress tee, not the persisted request_stats row) fixes that without
    changing what "output_tokens" means in the stats the user already
    relies on."""

    def test_sse_reasoning_only_delta_is_invisible_by_default(self) -> None:
        body = b'data: {"choices":[{"delta":{"reasoning_content":"pensando mucho"}}]}\n\n'
        assert _stream_output_tokens(body, "text/event-stream") is None

    def test_sse_reasoning_delta_counted_when_requested(self) -> None:
        body = b'data: {"choices":[{"delta":{"reasoning_content":"pensando mucho rato"}}]}\n\n'
        result = _stream_output_tokens(body, "text/event-stream", include_reasoning=True)
        assert result == 3

    def test_sse_content_and_reasoning_both_counted(self) -> None:
        body = (
            b'data: {"choices":[{"delta":{"reasoning_content":"pensando"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"hola mundo"}}]}\n\n'
        )
        result = _stream_output_tokens(body, "text/event-stream", include_reasoning=True)
        assert result == 3

    def test_ndjson_thinking_only_is_invisible_by_default(self) -> None:
        body = b'{"message":{"thinking":"pensando mucho"}}\n'
        assert _stream_output_tokens(body, "application/x-ndjson") is None

    def test_ndjson_thinking_counted_when_requested(self) -> None:
        body = b'{"message":{"thinking":"pensando mucho rato"}}\n'
        result = _stream_output_tokens(body, "application/x-ndjson", include_reasoning=True)
        assert result == 3


class TestStreamOutputCountsToolCallArguments:
    """2026-11-12: un agente escribia un fichero largo en una tool call de
    ravenx; el contador en vivo se quedo en 156 tokens (≈1 t/s en el
    dashboard) mientras llama-server llevaba 8.300 generados, porque solo se
    contaban content y reasoning_content."""

    def _sse(self, *pieces: str) -> bytes:
        import json as _json

        out = b""
        for i, piece in enumerate(pieces):
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        **({"id": "c1", "type": "function"} if i == 0 else {}),
                        "function": {"name": "write_file", "arguments": piece}
                        if i == 0
                        else {"arguments": piece},
                    }
                ]
            }
            out += b"data: " + _json.dumps({"choices": [{"delta": delta}]}).encode() + b"\n\n"
        return out

    def test_los_argumentos_de_una_tool_call_cuentan_como_salida(self) -> None:
        body = self._sse('{"path": "hola.py", ', '"content": "print(\'hola mundo\')\\n"}')
        assert (_stream_output_tokens(body, "text/event-stream") or 0) >= 8

    def test_el_contador_crece_mientras_la_tool_call_crece(self) -> None:
        corto = self._sse('{"content": "', "linea uno ")
        largo = self._sse('{"content": "', *(["linea ", "otra "] * 50))
        a = _stream_output_tokens(corto, "text/event-stream", include_reasoning=True)
        b = _stream_output_tokens(largo, "text/event-stream", include_reasoning=True)
        assert a is not None and b is not None and b > a + 50

    def test_ollama_tool_call_con_argumentos_como_objeto(self) -> None:
        body = (
            b'{"message":{"tool_calls":[{"function":{"name":"write_file",'
            b'"arguments":{"path":"hola.py","content":"print(1)"}}}]}}\n'
        )
        assert (_stream_output_tokens(body, "application/x-ndjson") or 0) >= 5


class TestCountInputTokensIncludesTools:
    """2026-10-29 — reproduced live: a Home Assistant request's live-progress
    card showed '~128 tok entrada' while the real prompt (confirmed via the
    preset selector's own required_tokens) was 18,240 — _count_input_tokens
    only looked at messages[].content, never tools. Tool-heavy clients are
    exactly the ones a reviewer most needs an honest size for."""

    def test_tools_words_are_counted(self) -> None:
        payload = {
            "messages": [{"role": "user", "content": "hola"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "turn_on_light",
                        "description": "Turns on a light in a given room of the house",
                        "parameters": {"room": {"type": "string"}},
                    },
                }
            ],
        }
        with_tools = _count_input_tokens(payload, "chat")
        without_tools = _count_input_tokens({"messages": payload["messages"]}, "chat")
        assert with_tools > without_tools

    def test_no_tools_field_is_unaffected(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hola mundo"}]}
        assert _count_input_tokens(payload, "chat") == 2

    def test_empty_tools_list_is_unaffected(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hola mundo"}], "tools": []}
        assert _count_input_tokens(payload, "chat") == 2

    def test_completion_kind_still_ignores_tools(self) -> None:
        """Only chat/ollama_chat carry a 'tools' field in practice; completion
        kind keeps its own prompt-only estimate unchanged."""
        payload = {"prompt": "hola mundo", "tools": [{"function": {"name": "x"}}]}
        assert _count_input_tokens(payload, "completion") == 2


class TestExtractSpeculativeStats:
    """2026-10-29 — persisted so a future 'this model feels slow' question
    can be answered from history instead of needing a perfectly uncontended
    live moment to test (which real GPU1 contention made impossible to get
    the day this was added — see gpu-bench/README.md)."""

    def test_reads_draft_and_speed_fields_from_timings(self) -> None:
        payload = {
            "timings": {
                "draft_n": 16,
                "draft_n_accepted": 13,
                "predicted_per_second": 82.4,
                "prompt_per_second": 410.2,
            }
        }
        draft_n, draft_n_accepted, predicted_tps, prompt_tps = _extract_speculative_stats(payload)
        assert draft_n == 16
        assert draft_n_accepted == 13
        assert predicted_tps == 82.4
        assert prompt_tps == 410.2

    def test_non_speculative_model_has_no_draft_fields(self) -> None:
        """A normal (non-self-speculative) llama.cpp load still reports
        predicted_per_second/prompt_per_second, just no draft_n."""
        payload = {"timings": {"predicted_per_second": 34.1, "prompt_per_second": 900.0}}
        draft_n, draft_n_accepted, predicted_tps, prompt_tps = _extract_speculative_stats(payload)
        assert draft_n is None
        assert draft_n_accepted is None
        assert predicted_tps == 34.1
        assert prompt_tps == 900.0

    def test_missing_timings_block_returns_all_none(self) -> None:
        assert _extract_speculative_stats({"usage": {"completion_tokens": 5}}) == (
            None,
            None,
            None,
            None,
        )

    def test_none_payload_returns_all_none(self) -> None:
        assert _extract_speculative_stats(None) == (None, None, None, None)

    def test_non_numeric_values_are_dropped_not_raised(self) -> None:
        payload = {"timings": {"draft_n": "not-a-number", "predicted_per_second": None}}
        draft_n, draft_n_accepted, predicted_tps, prompt_tps = _extract_speculative_stats(payload)
        assert draft_n is None
        assert draft_n_accepted is None
        assert predicted_tps is None
        assert prompt_tps is None


def test_extract_usage_tokens_openai_payload() -> None:
    input_tokens, output_tokens = _extract_usage_tokens(
        {"usage": {"prompt_tokens": 12, "completion_tokens": 34}}
    )
    assert input_tokens == 12
    assert output_tokens == 34


def test_extract_usage_tokens_ollama_payload() -> None:
    input_tokens, output_tokens = _extract_usage_tokens({"prompt_eval_count": 7, "eval_count": 11})
    assert input_tokens == 7
    assert output_tokens == 11


def test_extract_usage_tokens_audio_transcription_payload() -> None:
    input_tokens, output_tokens = _extract_usage_tokens(
        {"text": "hola mundo desde whisper"},
        request_kind="audio_transcription",
    )
    assert input_tokens == 0
    assert output_tokens == 4


def test_extract_generation_ms_from_llama_cpp_timings() -> None:
    """2026-10-27 tok/s bug: duration_ms includes queue wait, so dividing
    output_tokens by it understates real throughput under contention. The
    llama_cpp backend's own response reports pure generation time directly —
    use it instead when present."""
    assert _extract_generation_ms({"timings": {"predicted_ms": 839.827}}) == 839


def test_extract_generation_ms_from_ollama_eval_duration_nanoseconds() -> None:
    # 2_500_000_000 ns = 2500 ms
    assert _extract_generation_ms({"eval_duration": 2_500_000_000}) == 2500


def test_extract_generation_ms_none_when_backend_does_not_report_it() -> None:
    assert _extract_generation_ms({"usage": {"completion_tokens": 5}}) is None
    assert _extract_generation_ms(None) is None
    assert _extract_generation_ms({"timings": {}}) is None


def test_engine_runtime_telemetry_extracts_prefill_and_slot() -> None:
    telemetry = RequestRuntimeTelemetry(started_monotonic=0.0)
    request = SimpleNamespace(state=SimpleNamespace(runtime_telemetry=telemetry))

    _update_engine_runtime_telemetry(
        request,
        {"slot_id": 3, "timings": {"prompt_ms": 1234.9}},
    )

    assert telemetry.prefill_ms == 1234
    assert telemetry.slot_id == 3


def test_embedding_prompt_tps_uses_worker_time_not_wall_clock() -> None:
    # 2026-11-12: los embeddings no guardaban velocidad (el worker no
    # devuelve ``timings``). Se mide la llamada al worker, sin cola ni carga.
    telemetry = RequestRuntimeTelemetry(started_monotonic=0.0)
    request = SimpleNamespace(state=SimpleNamespace(runtime_telemetry=telemetry))
    telemetry.mark_prefill_since(5.0, now=5.5)

    assert telemetry.prefill_ms == 500
    assert _prefill_only_prompt_tps(request, "embedding", 1000) == 2000.0
    assert _prefill_only_prompt_tps(request, "ollama_embedding", 250) == 500.0


def test_embedding_prompt_tps_none_without_tokens_time_or_wrong_kind() -> None:
    telemetry = RequestRuntimeTelemetry(started_monotonic=0.0)
    request = SimpleNamespace(state=SimpleNamespace(runtime_telemetry=telemetry))
    assert _prefill_only_prompt_tps(request, "embedding", 100) is None  # sin prefill

    telemetry.prefill_ms = 200
    assert _prefill_only_prompt_tps(request, "embedding", None) is None
    assert _prefill_only_prompt_tps(request, "embedding", 0) is None
    assert _prefill_only_prompt_tps(request, "chat", 100) is None
    no_telemetry = SimpleNamespace(state=SimpleNamespace())
    assert _prefill_only_prompt_tps(no_telemetry, "embedding", 100) is None


def test_mark_prefill_since_keeps_engine_reported_prefill() -> None:
    telemetry = RequestRuntimeTelemetry(started_monotonic=0.0, prefill_ms=42)
    mark_prefill_since(SimpleNamespace(runtime_telemetry=telemetry), 0.0)
    assert telemetry.prefill_ms == 42
    # Sin telemetría (petición fuera del middleware) no rompe.
    mark_prefill_since(SimpleNamespace(), 0.0)


def test_stream_ttft_is_captured_before_progress_throttle() -> None:
    telemetry = RequestRuntimeTelemetry(started_monotonic=10.0)
    request = SimpleNamespace(state=SimpleNamespace(runtime_telemetry=telemetry))
    chunks = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n']

    _mark_stream_first_token(request, chunks, "text/event-stream", now_monotonic=10.1)
    assert telemetry.ttft_ms is None

    chunks.append(b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n')
    _mark_stream_first_token(request, chunks, "text/event-stream", now_monotonic=10.25)
    assert telemetry.ttft_ms == 250


def test_extract_error_message_preserves_openai_detail() -> None:
    payload = {
        "detail": {
            "error": {
                "message": "Failed to load Raven: cudaMalloc failed",
                "code": "model_load_failed",
            }
        }
    }

    assert _extract_error_message(payload, 503) == (
        "Failed to load Raven: cudaMalloc failed [model_load_failed]"
    )
    assert _extract_error_message(None, 502) == "HTTP 502"


def test_extract_last_payload_from_sse_stream_uses_usage_chunk() -> None:
    body = b"""data: {\"choices\":[{\"delta\":{\"content\":\"hola\"}}]}\n\ndata: {\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n"""

    payload = _extract_last_payload_from_stream(body, "text/event-stream")

    assert payload == {"usage": {"prompt_tokens": 3, "completion_tokens": 5}}


def test_extract_last_payload_from_ndjson_stream_uses_done_chunk() -> None:
    body = b'{"message":{"content":"hola"},"done":false}\n{"prompt_eval_count":7,"eval_count":11,"done":true}\n'

    payload = _extract_last_payload_from_stream(body, "application/x-ndjson")

    assert payload == {"prompt_eval_count": 7, "eval_count": 11, "done": True}


def test_extract_stream_error_from_sse() -> None:
    body = (
        b'data: {"error":{"message":"not enough memory","type":"server_error",'
        b'"code":"insufficient_vram"}}\n\ndata: [DONE]\n\n'
    )

    assert _extract_stream_error(body, "text/event-stream") == (
        "not enough memory",
        "insufficient_vram",
    )
    assert _stream_error_status("insufficient_vram") == 409
    assert _stream_error_status("model_load_timeout") == 503
    assert _stream_error_status("upstream_invalid_request") == 400
    assert _stream_error_status("upstream_rate_limited") == 429
    assert _stream_error_status("upstream_server_error") == 502
    assert _stream_error_status("unknown_error") == 500


@pytest.mark.asyncio
async def test_extract_response_payload_json_only() -> None:
    json_resp = JSONResponse({"usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    payload, rebuilt = await _extract_response_payload_and_rebuild(json_resp)
    assert payload == {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    assert rebuilt.status_code == 200

    streaming_resp = StreamingResponse(iter([b"chunk"]))
    payload, rebuilt = await _extract_response_payload_and_rebuild(streaming_resp)
    assert payload is None
    assert isinstance(rebuilt, StreamingResponse)


@pytest.mark.asyncio
async def test_resolve_inflight_model_id_maps_profile_to_worker_key() -> None:
    profile = SimpleNamespace(
        profile_id="gemma-public",
        base_model_id="ollama/gemma:12b",
        load_overrides={"max_model_len": 65536},
        enabled=True,
        is_default=True,
    )
    registry = SimpleNamespace(
        get=AsyncMock(return_value=profile),
        list_by_model=AsyncMock(return_value=[profile]),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                profile_registry=registry,
                model_manager=None,
            )
        )
    )

    resolved = await _resolve_inflight_model_id(request, "gemma-public")

    assert resolved == "ollama/gemma:12b::13a8ca1bf587"


@pytest.mark.asyncio
async def test_resolve_inflight_model_id_follows_hidden_alias() -> None:
    """Regression test: a hidden compatibility alias (e.g. the client-facing
    name "gemma4:26b" after it was migrated to the "gemma4:26b-llamacpp"
    profile) isn't itself a profile_id, so registry.get() alone misses it.
    Without resolve_alias() as a fallback, stats attribution fell through to
    a backend_model_id string match further down and mis-blamed every
    request on the OLD backend even though it was actually served by the
    new one — found 2026-10-27 debugging a report that looked like a broken
    router but was actually just wrong stats."""
    new_profile = SimpleNamespace(
        profile_id="gemma4:26b-llamacpp",
        base_model_id="llama_cpp/gemma4-26b",
        load_overrides={},
        enabled=True,
        is_default=True,
    )
    registry = SimpleNamespace(
        get=AsyncMock(return_value=None),
        resolve_alias=AsyncMock(return_value=(new_profile, None)),
        list_by_model=AsyncMock(return_value=[]),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                profile_registry=registry,
                model_manager=None,
            )
        )
    )

    resolved = await _resolve_inflight_model_id(request, "gemma4:26b")

    assert resolved is not None
    assert resolved.startswith("llama_cpp/gemma4-26b")


@pytest.mark.asyncio
async def test_record_realtime_request_reuses_attributed_stats_path() -> None:
    auth_user = SimpleNamespace(username="makespace", api_key_name="voice-client")
    websocket = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace()),
        state=SimpleNamespace(auth_user=auth_user),
        headers={"user-agent": "test-client"},
        client=SimpleNamespace(host="127.0.0.1"),
    )
    record_stat = AsyncMock()

    with patch("ocabra.stats.collector._record_stat", new=record_stat):
        await record_realtime_request(
            websocket,
            session_id="sess_test",
            model_id="whisper/faster-whisper-small",
            started_at=SimpleNamespace(),
            duration_ms=42.8,
            request_kind="realtime_transcription",
            status_code=200,
            input_tokens=0,
            output_tokens=3,
        )

    assert record_stat.await_args.kwargs["request"] is websocket
    assert record_stat.await_args.kwargs["endpoint_path"] == "/v1/realtime"
    assert record_stat.await_args.kwargs["request_kind"] == "realtime_transcription"
    assert record_stat.await_args.kwargs["output_tokens"] == 3


def test_stats_middleware_returns_504_and_releases_canonical_worker() -> None:
    app = FastAPI()
    profile = SimpleNamespace(
        profile_id="gemma-public",
        base_model_id="ollama/gemma:12b",
        load_overrides=None,
        enabled=True,
        is_default=True,
    )
    app.state.profile_registry = SimpleNamespace(
        get=AsyncMock(return_value=profile),
        list_by_model=AsyncMock(return_value=[profile]),
    )
    model_manager = MagicMock()
    model_manager.try_begin_request.return_value = "request-id"
    model_manager.is_in_maintenance.return_value = False
    app.state.model_manager = model_manager

    @app.post("/v1/chat/completions")
    async def _timeout() -> None:
        raise InferenceTimeoutError(
            "ollama/gemma:12b",
            "/v1/chat/completions",
            900,
        )

    app.add_middleware(StatsMiddleware)

    with (
        patch("ocabra.stats.collector._record_stat", new=AsyncMock()),
        patch("ocabra.stats.collector.settings.max_inflight_per_model", 0),
    ):
        response = TestClient(app, raise_server_exceptions=False).post(
            "/v1/chat/completions",
            json={"model": "gemma-public", "messages": []},
        )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "generation_timeout"
    model_manager.try_begin_request.assert_called_once_with(
        "ollama/gemma:12b",
        0,
        user_id=None,
        username=None,
        api_key_name=None,
        profile_id="gemma-public",
        input_tokens_estimate=None,
        request_kind="chat",
        remote_bound=False,
        # /v1 sin "stream" en el cuerpo: OpenAI responde de una vez.
        streaming=False,
        max_output_tokens=None,
    )
    model_manager.end_request.assert_called_once_with(
        "ollama/gemma:12b",
        "request-id",
    )


def test_stats_middleware_returns_503_when_model_is_in_maintenance() -> None:
    """A model under maintenance must reject new requests with 503 before
    ever calling try_begin_request — no admission, no load, no contention
    for whatever's using the GPU for manual testing."""
    app = FastAPI()
    profile = SimpleNamespace(
        profile_id="gemma-public",
        base_model_id="ollama/gemma:12b",
        load_overrides=None,
        enabled=True,
        is_default=True,
    )
    app.state.profile_registry = SimpleNamespace(
        get=AsyncMock(return_value=profile),
        list_by_model=AsyncMock(return_value=[profile]),
    )
    model_manager = MagicMock()
    model_manager.is_in_maintenance.return_value = True
    app.state.model_manager = model_manager

    @app.post("/v1/chat/completions")
    async def _unreachable() -> None:
        raise AssertionError("maintenance check must short-circuit before the endpoint")

    app.add_middleware(StatsMiddleware)

    with patch("ocabra.stats.collector._record_stat", new=AsyncMock()):
        response = TestClient(app, raise_server_exceptions=False).post(
            "/v1/chat/completions",
            json={"model": "gemma-public", "messages": []},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_maintenance"
    model_manager.try_begin_request.assert_not_called()


def test_stats_middleware_records_in_band_sse_error_status() -> None:
    app = FastAPI()
    app.state.profile_registry = SimpleNamespace(
        get=AsyncMock(return_value=None),
        list_by_model=AsyncMock(return_value=[]),
    )
    app.state.model_manager = None

    @app.post("/v1/chat/completions")
    async def _stream_error() -> StreamingResponse:
        async def _body():
            yield (
                b'data: {"error":{"message":"not enough memory",'
                b'"code":"insufficient_vram"}}\n\ndata: [DONE]\n\n'
            )

        return StreamingResponse(_body(), media_type="text/event-stream")

    app.add_middleware(StatsMiddleware)
    record_stat = AsyncMock()

    with patch("ocabra.stats.collector._record_stat", new=record_stat):
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "vllm/test-model", "messages": []},
        )

    assert response.status_code == 200
    assert record_stat.await_args.kwargs["status_code"] == 409
    assert record_stat.await_args.kwargs["error_message"] == "not enough memory"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("streamed_chunk", "expected_output_tokens"),
    [
        (b"", None),
        (b'data: {"choices":[{"delta":{"content":"hola mundo"}}]}\n\n', 2),
    ],
)
async def test_stats_middleware_records_cancelled_stream_as_499(
    streamed_chunk: bytes,
    expected_output_tokens: int | None,
) -> None:
    """A disconnect happens while Starlette consumes the response body, after
    ``dispatch`` has already returned the initial HTTP 200 response. The body
    tee must catch that cancellation itself and retain any streamed usage."""
    app = FastAPI()
    app.state.profile_registry = SimpleNamespace(
        get=AsyncMock(return_value=None),
        list_by_model=AsyncMock(return_value=[]),
    )
    app.state.model_manager = None

    request_body = b'{"model":"ollama/qwen3-coder:30b","messages":[]}'
    request_sent = False

    async def _receive() -> dict:
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": request_body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "app": app,
        },
        _receive,
    )
    keep_stream_open = asyncio.Event()
    body_closed = asyncio.Event()

    async def _body():
        try:
            if streamed_chunk:
                yield streamed_chunk
            await keep_stream_open.wait()
        finally:
            body_closed.set()

    async def _call_next(_request: Request) -> StreamingResponse:
        return StreamingResponse(_body(), media_type="text/event-stream")

    middleware = StatsMiddleware(app)
    record_stat = AsyncMock()

    with patch("ocabra.stats.collector._record_stat", new=record_stat):
        response = await middleware.dispatch(request, _call_next)

        async def _consume() -> None:
            async for _chunk in response.body_iterator:
                pass

        consumer = asyncio.create_task(_consume())
        await asyncio.sleep(0)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await asyncio.sleep(0)

    assert body_closed.is_set()
    assert record_stat.await_args.kwargs["status_code"] == 499
    assert record_stat.await_args.kwargs["error_message"] == "stream_cancelled"
    assert record_stat.await_args.kwargs["output_tokens"] == expected_output_tokens


# ── _request_bound_for_peer ─────────────────────────────────────────────


def _peer_request(federation_manager):
    """Minimal stand-in for a Request carrying app.state."""
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(federation_manager=federation_manager))
    )


def _mm_with_status(status):
    mm = MagicMock()
    mm.get_state = AsyncMock(return_value=SimpleNamespace(status=status) if status else None)
    return mm


@pytest.mark.asyncio
async def test_request_bound_for_peer_true_when_local_cold_and_peer_warm():
    """Step 2 of resolve_federated: not loaded here, loaded on a peer."""
    from ocabra.core.model_manager import ModelStatus
    from ocabra.stats.collector import _request_bound_for_peer

    fed = MagicMock()
    fed.get_remote_models.return_value = {"vllm/some-model": [MagicMock()]}
    mm = _mm_with_status(ModelStatus.CONFIGURED)

    with patch("ocabra.stats.collector.settings.federation_enabled", True):
        assert await _request_bound_for_peer(_peer_request(fed), mm, "vllm/some-model") is True


@pytest.mark.asyncio
async def test_request_bound_for_peer_false_when_loaded_locally():
    """Deliberately conservative: a locally-loaded model keeps its slot cap,
    even though select_target could in principle still choose a peer."""
    from ocabra.core.model_manager import ModelStatus
    from ocabra.stats.collector import _request_bound_for_peer

    fed = MagicMock()
    fed.get_remote_models.return_value = {"vllm/some-model": [MagicMock()]}
    mm = _mm_with_status(ModelStatus.LOADED)

    with patch("ocabra.stats.collector.settings.federation_enabled", True):
        assert await _request_bound_for_peer(_peer_request(fed), mm, "vllm/some-model") is False


@pytest.mark.asyncio
async def test_request_bound_for_peer_false_when_no_peer_has_it():
    from ocabra.core.model_manager import ModelStatus
    from ocabra.stats.collector import _request_bound_for_peer

    fed = MagicMock()
    fed.get_remote_models.return_value = {}
    mm = _mm_with_status(ModelStatus.CONFIGURED)

    with patch("ocabra.stats.collector.settings.federation_enabled", True):
        assert await _request_bound_for_peer(_peer_request(fed), mm, "vllm/some-model") is False


@pytest.mark.asyncio
async def test_request_bound_for_peer_respects_kill_switch_and_failures():
    """Off when federation is disabled or absent, and never raises into the
    admission path — federation stays best effort here."""
    from ocabra.core.model_manager import ModelStatus
    from ocabra.stats.collector import _request_bound_for_peer

    fed = MagicMock()
    fed.get_remote_models.return_value = {"vllm/some-model": [MagicMock()]}
    mm = _mm_with_status(ModelStatus.CONFIGURED)

    with patch("ocabra.stats.collector.settings.federation_enabled", False):
        assert await _request_bound_for_peer(_peer_request(fed), mm, "vllm/some-model") is False

    with patch("ocabra.stats.collector.settings.federation_enabled", True):
        assert await _request_bound_for_peer(_peer_request(None), mm, "vllm/some-model") is False

        exploding = MagicMock()
        exploding.get_state = AsyncMock(side_effect=RuntimeError("boom"))
        assert (
            await _request_bound_for_peer(_peer_request(fed), exploding, "vllm/some-model") is False
        )
