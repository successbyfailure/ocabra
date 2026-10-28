"""Tests for Ollama API compatibility layer."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Profile helpers (shared pattern with test_openai_api)
# ---------------------------------------------------------------------------

def _make_test_profile(profile_id, base_model_id, category="llm", **kwargs):
    """Create a SimpleNamespace that mimics a ModelProfile for tests."""
    return SimpleNamespace(
        profile_id=profile_id,
        base_model_id=base_model_id,
        display_name=kwargs.get("display_name", profile_id),
        description=None,
        category=category,
        load_overrides=kwargs.get("load_overrides", {}),
        request_defaults=kwargs.get("request_defaults", {}),
        assets=kwargs.get("assets", {}),
        enabled=True,
        is_default=True,
    )


class FakeProfileRegistry:
    def __init__(self, profiles=None):
        self._profiles = {p.profile_id: p for p in (profiles or [])}

    async def get(self, profile_id):
        return self._profiles.get(profile_id)

    async def list_enabled(self):
        return [p for p in self._profiles.values() if p.enabled]

    async def list_by_model(self, base_model_id):
        return [p for p in self._profiles.values() if p.base_model_id == base_model_id]

    async def list_all(self):
        return list(self._profiles.values())


def _make_app(model_state=None, worker_result=None) -> FastAPI:
    from ocabra.api._deps_auth import UserContext, get_current_user
    from ocabra.api.ollama import router as ollama_router
    from ocabra.api.ollama._shared import get_ollama_user
    from ocabra.api.openai._deps import get_openai_user
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    _admin_ctx = UserContext(
        user_id=None, username=None, role="system_admin",
        group_ids=[], accessible_model_ids=set(), is_anonymous=False,
    )

    app = FastAPI()
    app.include_router(ollama_router)

    # Override all auth callables so no real DB/Redis is needed
    async def _fake_user():
        return _admin_ctx
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_ollama_user] = _fake_user
    app.dependency_overrides[get_openai_user] = _fake_user

    if model_state is None:
        model_state = ModelState(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            display_name="Llama 3.2 3B",
            backend_type="vllm",
            status=ModelStatus.LOADED,
            load_policy=LoadPolicy.ON_DEMAND,
            capabilities=BackendCapabilities(chat=True, completion=True, embeddings=True, streaming=True),
        )

    mm = MagicMock()
    mm.get_state = AsyncMock(return_value=model_state)
    mm.list_states = AsyncMock(return_value=[model_state])
    mm.load = AsyncMock(return_value=model_state)
    mm.delete_model = AsyncMock(return_value=None)
    mm.touch_last_request_at = AsyncMock()

    wp = MagicMock()
    wp.forward_request = AsyncMock(return_value=worker_result or {})
    wp.forward_stream = MagicMock(return_value=_async_gen([b"data: [DONE]\\n\\n"]))

    app.state.model_manager = mm
    app.state.worker_pool = wp

    # Profile registry — empty by default; Ollama endpoints fall back to
    # legacy name resolution when no profile matches.
    app.state.profile_registry = FakeProfileRegistry()
    return app


async def _async_gen(items):
    for item in items:
        yield item


def test_name_mapper_round_trip() -> None:
    from ocabra.api.ollama._mapper import OllamaNameMapper

    mapper = OllamaNameMapper()
    internal = mapper.to_internal("llama3.2:3b")
    assert internal == "vllm/meta-llama/Llama-3.2-3B-Instruct"
    assert mapper.to_ollama(internal) == "llama3.2:3b"


def test_name_resolution_prefers_native_ollama_model() -> None:
    import asyncio

    from ocabra.api.ollama._mapper import resolve_model

    native_state = object()
    mapped_state = object()

    class FakeModelManager:
        async def get_state(self, model_id: str):
            if model_id == "ollama/llama3.2:3b":
                return native_state
            if model_id == "vllm/meta-llama/Llama-3.2-3B-Instruct":
                return mapped_state
            return None

    with patch(
        "ocabra.api.ollama._mapper.resolve_openai_model",
        new=AsyncMock(side_effect=AssertionError("openai fallback should not run for native ollama matches")),
    ):
        model_id, state = asyncio.run(resolve_model(FakeModelManager(), "llama3.2:3b"))

    assert model_id == "ollama/llama3.2:3b"
    assert state is native_state


def test_name_resolution_delegates_to_openai_resolution_when_no_native_ollama_match() -> None:
    import asyncio

    from ocabra.api.ollama import _mapper

    class FakeModelManager:
        async def get_state(self, model_id: str):
            return None

    delegated_state = object()

    async def fake_openai_resolve(model_manager, requested_name, user=None):
        assert requested_name == "some-alias"
        return "vllm/delegated-model", delegated_state

    with patch.object(_mapper, "resolve_openai_model", new=fake_openai_resolve):
        model_id, state = asyncio.run(_mapper.resolve_model(FakeModelManager(), "some-alias"))

    assert model_id == "vllm/delegated-model"
    assert state is delegated_state


def test_generate_passthroughs_native_ollama_body() -> None:
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    model_state = ModelState(
        model_id="ollama/llama3.2:3b",
        display_name="Llama 3.2 3B",
        backend_type="ollama",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(completion=True, streaming=True),
        backend_model_id="llama3.2:3b",
    )

    app = _make_app(
        model_state=model_state,
        worker_result={"response": "ok", "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
    )
    client = TestClient(app)

    resp = client.post(
        "/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": "hello",
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 12},
        },
    )

    assert resp.status_code == 200
    assert resp.json()["response"] == "ok"

    called = app.state.worker_pool.forward_request.await_args
    assert called.args[1] == "/api/generate"
    assert called.args[2] == {
        "model": "llama3.2:3b",
        "prompt": "hello",
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 12},
    }


def test_native_backend_model_name_prefers_backend_model_id() -> None:
    from ocabra.api.ollama.chat import _native_backend_model_name

    assert _native_backend_model_name("ollama/qwen3:8b", "qwen3:8b") == "qwen3:8b"
    assert _native_backend_model_name("qwen3:8b", None) == "qwen3:8b"


def test_build_native_passthrough_body_promotes_top_level_limits_to_options() -> None:
    from ocabra.api.ollama._shared import build_native_passthrough_body

    body = build_native_passthrough_body(
        {
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "think": False,
            "max_tokens": 32,
            "temperature": 0.1,
        },
        model="qwen3:8b",
        stream=False,
        content_keys=("messages",),
        passthrough_keys=("keep_alive", "format", "think", "tools"),
    )

    assert body == {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "think": False,
        "options": {"num_predict": 32, "temperature": 0.1},
    }


def test_pull_delegates_to_download_manager() -> None:
    app = _make_app()
    client = TestClient(app)

    fake_job = MagicMock()
    fake_job.job_id = "job-1"

    with patch("ocabra.api.ollama.pull.download_manager.enqueue", new=AsyncMock(return_value=fake_job)) as enqueue:
        with patch(
            "ocabra.api.ollama.pull.download_manager.get_job",
            new=AsyncMock(return_value=MagicMock(status="completed", model_dump=lambda mode="json": {"status": "completed"})),
        ):
            resp = client.post("/api/pull", json={"name": "llama3.2:3b", "stream": False})

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    enqueue.assert_awaited_once_with(source="ollama", model_ref="llama3.2:3b")


def test_chat_stream_translates_sse_to_ndjson() -> None:
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    model_state = ModelState(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        display_name="Llama 3.2 3B",
        backend_type="vllm",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(chat=True, completion=True, embeddings=True, streaming=True),
    )

    app = _make_app(model_state=model_state)
    app.state.worker_pool.forward_stream = MagicMock(
        return_value=_async_gen(
            [
                b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
    )

    client = TestClient(app)
    with client.stream(
        "POST",
        "/api/chat",
        json={
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]

    payloads = [json.loads(line) for line in lines]
    assert payloads[0]["message"]["content"] == "Hi"
    assert payloads[1]["message"]["content"] == " there"
    assert payloads[-1]["done"] is True


def test_chat_non_streaming_converts_tool_calls_to_ollama_shape() -> None:
    """Regression test: non-ollama backends (vllm/llama_cpp/...) used to drop
    tool_calls entirely when translating the OpenAI-shaped worker response
    back into Ollama's /api/chat shape — the HTTP request succeeded but the
    tool call silently vanished. See gpu-bench README, 2026-10-27."""
    worker_result = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Madrid"}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    app = _make_app(worker_result=worker_result)
    client = TestClient(app)

    resp = client.post(
        "/api/chat",
        json={
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "weather in Madrid?"}],
            "stream": False,
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        },
    )

    assert resp.status_code == 200
    message = resp.json()["message"]
    assert message["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Madrid"}}}
    ]


def test_chat_stream_accumulates_and_converts_tool_call_deltas() -> None:
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    model_state = ModelState(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        display_name="Llama 3.2 3B",
        backend_type="vllm",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(chat=True, completion=True, embeddings=True, streaming=True),
    )
    app = _make_app(model_state=model_state)
    app.state.worker_pool.forward_stream = MagicMock(
        return_value=_async_gen(
            [
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x","type":"function","function":{"name":"get_weather","arguments":"{"}}]}}]}\n\n',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"city\\":"}}]}}]}\n\n',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"Madrid\\"}"}}]}}]}\n\n',
                b'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
    )

    client = TestClient(app)
    with client.stream(
        "POST",
        "/api/chat",
        json={
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "weather in Madrid?"}],
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        },
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]

    payloads = [json.loads(line) for line in lines]
    final = payloads[-1]
    assert final["done"] is True
    assert final["message"]["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Madrid"}}}
    ]


def test_generate_translates_options_to_vllm_params() -> None:
    app = _make_app(worker_result={"choices": [{"text": "ok"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
    client = TestClient(app)

    resp = client.post(
        "/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": "hello",
            "stream": False,
            "options": {"num_predict": 12, "repeat_penalty": 1.2},
        },
    )

    assert resp.status_code == 200
    assert resp.json()["response"] == "ok"

    called = app.state.worker_pool.forward_request.await_args
    assert called.args[1] == "/v1/completions"
    assert called.args[2]["max_tokens"] == 12
    assert called.args[2]["repetition_penalty"] == 1.2


def test_build_vllm_chat_body_preserves_top_level_max_tokens() -> None:
    from ocabra.api.ollama.chat import _build_vllm_chat_body

    body = _build_vllm_chat_body(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "max_tokens": 12,
        },
        "meta-llama/Llama-3.2-3B-Instruct",
        False,
    )

    assert body["max_tokens"] == 12


def test_openai_tool_calls_to_ollama_parses_stringified_arguments() -> None:
    from ocabra.api.ollama.chat import _openai_tool_calls_to_ollama

    out = _openai_tool_calls_to_ollama(
        [{"id": "x", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}]
    )

    assert out == [{"function": {"name": "f", "arguments": {"a": 1}}}]


def test_openai_tool_calls_to_ollama_handles_malformed_json_and_dict_args() -> None:
    from ocabra.api.ollama.chat import _openai_tool_calls_to_ollama

    out = _openai_tool_calls_to_ollama(
        [
            {"function": {"name": "broken", "arguments": "{not json"}},
            {"function": {"name": "already_dict", "arguments": {"a": 1}}},
            {"function": {"name": "", "arguments": "{}"}},
        ]
    )

    assert out == [
        {"function": {"name": "broken", "arguments": {}}},
        {"function": {"name": "already_dict", "arguments": {"a": 1}}},
    ]


def test_normalize_messages_coerces_null_content_to_empty_string() -> None:
    """Regression test: gemma4's llama.cpp worker hard-crashes (process dies,
    not just a 500) when a historical assistant message has content: null
    alongside tool_calls — the standard OpenAI shape for a tool-calls-only
    message. Reproduced live 2026-10-27; content: "" with identical
    tool_calls does not crash it. See gpu-bench README for the isolation."""
    from ocabra.api.ollama.chat import _normalize_messages_for_backend

    out = _normalize_messages_for_backend(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "result"},
        ]
    )

    assert out[0]["content"] == "hi"
    assert out[1]["content"] == ""
    assert out[1]["tool_calls"] == [
        {"function": {"name": "f", "arguments": "{}"}, "type": "function"}
    ]
    assert out[2]["content"] == "result"


def test_build_vllm_chat_body_normalizes_null_content_in_messages() -> None:
    from ocabra.api.ollama.chat import _build_vllm_chat_body

    body = _build_vllm_chat_body(
        {"messages": [{"role": "assistant", "content": None, "tool_calls": []}], "stream": False},
        "meta-llama/Llama-3.2-3B-Instruct",
        False,
    )

    assert body["messages"][0]["content"] == ""


def test_ollama_messages_to_openai_coerces_null_content() -> None:
    from ocabra.api.ollama.chat import _ollama_messages_to_openai

    out = _ollama_messages_to_openai(
        [{"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f"}}]}]
    )

    assert out[0]["content"] == ""


def test_normalize_tool_call_backfills_type_and_stringifies_arguments() -> None:
    """Regression test: oCabra hands clients Ollama-native tool_calls
    (no "type", arguments as a plain object). A client that echoes that
    exact shape back as history on the next turn gets forwarded straight
    to llama-server's OpenAI-compatible endpoint, which rejects it with
    "Failed to parse messages: Missing tool call type". Reproduced live
    2026-10-27 via a real Home Assistant conversation (intent__HassTurnOn),
    caught by the diagnostic logging added for the content:null crash."""
    from ocabra.api.ollama.chat import _normalize_tool_call_for_backend

    out = _normalize_tool_call_for_backend(
        {"function": {"name": "intent__HassTurnOn", "arguments": {"name": "Lámpara"}}}
    )

    assert out["type"] == "function"
    assert out["function"]["arguments"] == '{"name": "L\\u00e1mpara"}'


def test_normalize_tool_call_preserves_existing_type_and_string_arguments() -> None:
    from ocabra.api.ollama.chat import _normalize_tool_call_for_backend

    out = _normalize_tool_call_for_backend(
        {"id": "x", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
    )

    assert out["type"] == "function"
    assert out["function"]["arguments"] == '{"a":1}'
    assert out["id"] == "x"


def test_build_vllm_chat_body_normalizes_historical_tool_calls() -> None:
    from ocabra.api.ollama.chat import _build_vllm_chat_body

    body = _build_vllm_chat_body(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
                }
            ],
            "stream": False,
        },
        "meta-llama/Llama-3.2-3B-Instruct",
        False,
    )

    tc = body["messages"][0]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["arguments"] == '{"a": 1}'
