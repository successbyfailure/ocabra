from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from ocabra.core.worker_pool import InferenceTimeoutError
from ocabra.stats.collector import (
    StatsMiddleware,
    _classify_request_kind,
    _extract_last_payload_from_stream,
    _extract_response_payload_and_rebuild,
    _extract_usage_tokens,
    _resolve_inflight_model_id,
)


def test_classify_request_kind_known_paths() -> None:
    assert _classify_request_kind("/v1/chat/completions") == "chat"
    assert _classify_request_kind("/v1/audio/transcriptions") == "audio_transcription"
    assert _classify_request_kind("/api/generate") == "ollama_generate"


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
    assert input_tokens is None
    assert output_tokens == 4


def test_extract_last_payload_from_sse_stream_uses_usage_chunk() -> None:
    body = b"""data: {\"choices\":[{\"delta\":{\"content\":\"hola\"}}]}\n\ndata: {\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n"""

    payload = _extract_last_payload_from_stream(body, "text/event-stream")

    assert payload == {"usage": {"prompt_tokens": 3, "completion_tokens": 5}}


def test_extract_last_payload_from_ndjson_stream_uses_done_chunk() -> None:
    body = b'{"message":{"content":"hola"},"done":false}\n{"prompt_eval_count":7,"eval_count":11,"done":true}\n'

    payload = _extract_last_payload_from_stream(body, "application/x-ndjson")

    assert payload == {"prompt_eval_count": 7, "eval_count": 11, "done": True}


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
    )
    model_manager.end_request.assert_called_once_with(
        "ollama/gemma:12b",
        "request-id",
    )
