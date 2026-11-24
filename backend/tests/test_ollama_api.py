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
        user_id=None,
        username=None,
        role="system_admin",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
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
            capabilities=BackendCapabilities(
                chat=True, completion=True, embeddings=True, streaming=True
            ),
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
        new=AsyncMock(
            side_effect=AssertionError("openai fallback should not run for native ollama matches")
        ),
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

    with patch(
        "ocabra.api.ollama.pull.download_manager.enqueue", new=AsyncMock(return_value=fake_job)
    ) as enqueue:
        with patch(
            "ocabra.api.ollama.pull.download_manager.get_job",
            new=AsyncMock(
                return_value=MagicMock(
                    status="completed", model_dump=lambda mode="json": {"status": "completed"}
                )
            ),
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
        capabilities=BackendCapabilities(
            chat=True, completion=True, embeddings=True, streaming=True
        ),
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
        capabilities=BackendCapabilities(
            chat=True, completion=True, embeddings=True, streaming=True
        ),
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
    app = _make_app(
        worker_result={
            "choices": [{"text": "ok"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
    )
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


def test_chat_non_streaming_forwards_fairness_user_key_and_slots() -> None:
    """Wiring check for the fair-share admission gate (2026-10-28 design):
    chat() must pass user_key/slots through to forward_request so the
    worker pool can actually gate on them — without this the gate class
    exists but nothing ever engages it."""
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    model_state = ModelState(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        display_name="Llama 3.2 3B",
        backend_type="vllm",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(chat=True, completion=True, streaming=True),
        extra_config={"_runtime": {"parallelism": 3}},
    )
    app = _make_app(
        model_state=model_state, worker_result={"choices": [{"message": {"content": "ok"}}]}
    )
    client = TestClient(app)

    resp = client.post(
        "/api/chat",
        json={
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )

    assert resp.status_code == 200
    called = app.state.worker_pool.forward_request.await_args
    assert called.kwargs["user_key"] == "anonymous"  # _make_app's admin ctx has no user_id/username
    assert called.kwargs["slots"] == 3


def test_chat_non_streaming_surfaces_done_reason() -> None:
    """Regression test: /api/chat never set done_reason for any backend
    other than native Ollama (which passes it through untouched), so a
    client had no way to tell a normal completion from one truncated by
    max_tokens or by running out of context — both looked identical
    (done: true, no hint). OpenAI's finish_reason maps 1:1 onto Ollama's
    done_reason vocabulary (stop/length/tool_calls)."""
    worker_result = {
        "choices": [
            {"message": {"role": "assistant", "content": "partial..."}, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    app = _make_app(worker_result=worker_result)
    client = TestClient(app)

    resp = client.post(
        "/api/chat",
        json={
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["done_reason"] == "length"


def test_chat_stream_surfaces_done_reason_in_final_chunk() -> None:
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    model_state = ModelState(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        display_name="Llama 3.2 3B",
        backend_type="vllm",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(
            chat=True, completion=True, embeddings=True, streaming=True
        ),
    )
    app = _make_app(model_state=model_state)
    app.state.worker_pool.forward_stream = MagicMock(
        return_value=_async_gen(
            [
                b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
                b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
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
    assert payloads[-1]["done_reason"] == "length"


def _llama_state():
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    return ModelState(
        model_id="llamacpp_mks/gemma4:26b",
        display_name="Gemma 4 26B",
        backend_type="llamacpp_mks",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(chat=True, completion=True, streaming=True),
    )


def _stream_chat(app, chunks: list[bytes]) -> tuple[list[dict], dict]:
    app.state.worker_pool.forward_stream = MagicMock(return_value=_async_gen(chunks))
    client = TestClient(app)
    with client.stream(
        "POST",
        "/api/chat",
        json={
            "model": "gemma4:26b",
            "messages": [{"role": "user", "content": "hola"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.iter_lines() if line]
    sent_body = app.state.worker_pool.forward_stream.call_args.args[2]
    return lines, sent_body


def test_chat_stream_asks_the_engine_for_usage_and_reports_it() -> None:
    """Regression 2026-11-09: the Ollama translation built its own upstream
    body without stream_options.include_usage, llama-server sent no usage
    chunk, and every streamed /api/chat answered prompt_eval_count=0 /
    eval_count=0 to the client (and to request_stats)."""
    app = _make_app(model_state=_llama_state())
    lines, sent = _stream_chat(
        app,
        [
            b'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":21,"completion_tokens":7}}\n\n',
            b"data: [DONE]\n\n",
        ],
    )

    assert sent["stream_options"] == {"include_usage": True}
    assert lines[-1]["done"] is True
    assert lines[-1]["prompt_eval_count"] == 21
    assert lines[-1]["eval_count"] == 7


def test_chat_stream_falls_back_to_llama_timings_when_no_usage_chunk() -> None:
    app = _make_app(model_state=_llama_state())
    lines, _sent = _stream_chat(
        app,
        [
            b'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            b'"timings":{"prompt_n":5,"cache_n":16,"predicted_n":7,"prompt_ms":10,"predicted_ms":40}}\n\n',
            b"data: [DONE]\n\n",
        ],
    )

    assert lines[-1]["prompt_eval_count"] == 21  # prompt_n + cache_n
    assert lines[-1]["eval_count"] == 7


def _app_as(role: str) -> FastAPI:
    from ocabra.api._deps_auth import UserContext
    from ocabra.api.ollama._shared import get_ollama_user

    app = _make_app()
    app.dependency_overrides[get_ollama_user] = lambda: UserContext(
        user_id="u", username="u", role=role
    )
    return app


def test_pull_and_delete_require_model_manager() -> None:
    """SEG-05: any API key could delete any model or fill the disk with pulls."""
    client = TestClient(_app_as("user"))

    with (
        patch("ocabra.api.ollama.pull.download_manager.enqueue", new=AsyncMock()) as enqueue,
        patch("ocabra.api.ollama.delete.resolve_model", new=AsyncMock()) as resolve,
    ):
        pull = client.post("/api/pull", json={"name": "llama3.2:3b", "stream": False})
        delete = client.request("DELETE", "/api/delete", json={"name": "llama3.2:3b"})

    assert pull.status_code == 403
    assert delete.status_code == 403
    assert "error" in pull.json()["detail"]
    enqueue.assert_not_awaited()
    resolve.assert_not_awaited()


def test_pull_is_allowed_for_model_manager() -> None:
    client = TestClient(_app_as("model_manager"))
    fake_job = MagicMock()
    fake_job.job_id = "job-1"

    with (
        patch(
            "ocabra.api.ollama.pull.download_manager.enqueue",
            new=AsyncMock(return_value=fake_job),
        ),
        patch(
            "ocabra.api.ollama.pull.download_manager.get_job",
            new=AsyncMock(
                return_value=MagicMock(
                    status="completed", model_dump=lambda mode="json": {"status": "completed"}
                )
            ),
        ),
    ):
        resp = client.post("/api/pull", json={"name": "llama3.2:3b", "stream": False})

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# BUG-10: dispatch errors keep their status; stream failures end in-band
# ---------------------------------------------------------------------------


def _chat_body(stream: bool) -> dict:
    return {
        "model": "gemma4:26b",
        "messages": [{"role": "user", "content": "hola"}],
        "stream": stream,
    }


def _upstream_error(status: int, body: str):
    import httpx

    req = httpx.Request("POST", "http://worker/v1/chat/completions")
    return httpx.HTTPStatusError(
        "upstream", request=req, response=httpx.Response(status, text=body, request=req)
    )


def test_chat_timeout_is_a_504_with_retry_after_not_a_500() -> None:
    from ocabra.core.worker_pool import InferenceTimeoutError

    app = _make_app(model_state=_llama_state())
    app.state.worker_pool.forward_request = AsyncMock(
        side_effect=InferenceTimeoutError("llamacpp_mks/gemma4:26b", "/v1/chat/completions", 900)
    )
    resp = TestClient(app, raise_server_exceptions=False).post("/api/chat", json=_chat_body(False))

    assert resp.status_code == 504
    assert resp.headers.get("retry-after")


def test_chat_upstream_400_keeps_its_status() -> None:
    app = _make_app(model_state=_llama_state())
    app.state.worker_pool.forward_request = AsyncMock(
        side_effect=_upstream_error(400, '{"error": {"message": "Missing tool call type"}}')
    )
    resp = TestClient(app, raise_server_exceptions=False).post("/api/chat", json=_chat_body(False))

    assert resp.status_code == 400
    assert "Missing tool call type" in resp.text


def test_chat_stream_rejected_before_the_first_byte_is_a_real_status() -> None:
    from ocabra.core.worker_pool import WorkerDegradedError

    async def _refuse():
        raise WorkerDegradedError("llamacpp_mks/gemma4:26b", 3)
        yield b""  # pragma: no cover

    app = _make_app(model_state=_llama_state())
    app.state.worker_pool.forward_stream = MagicMock(return_value=_refuse())
    resp = TestClient(app, raise_server_exceptions=False).post("/api/chat", json=_chat_body(True))

    assert resp.status_code == 503
    assert resp.headers.get("retry-after")


def test_chat_stream_failing_midway_ends_with_an_ollama_error_line() -> None:
    from ocabra.core.worker_pool import InferenceTimeoutError

    async def _half():
        yield b'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n'
        raise InferenceTimeoutError("llamacpp_mks/gemma4:26b", "/v1/chat/completions", 900)

    app = _make_app(model_state=_llama_state())
    app.state.worker_pool.forward_stream = MagicMock(return_value=_half())
    with TestClient(app).stream("POST", "/api/chat", json=_chat_body(True)) as resp:
        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.iter_lines() if line]

    assert lines[0]["message"]["content"] == "Hola"
    assert lines[-1]["code"] == "generation_timeout" and lines[-1]["error"]


def test_generate_timeout_is_a_504_not_a_500() -> None:
    from ocabra.core.worker_pool import InferenceTimeoutError

    app = _make_app(model_state=_llama_state())
    app.state.worker_pool.forward_request = AsyncMock(
        side_effect=InferenceTimeoutError("llamacpp_mks/gemma4:26b", "/v1/completions", 900)
    )
    resp = TestClient(app, raise_server_exceptions=False).post(
        "/api/generate", json={"model": "gemma4:26b", "prompt": "hola", "stream": False}
    )

    assert resp.status_code == 504


# ---------------------------------------------------------------------------
# BUG-11: reasoning controls reach the backend; thinking comes back
# ---------------------------------------------------------------------------


def test_chat_think_false_turns_reasoning_off_on_llama_cpp() -> None:
    app = _make_app(model_state=_llama_state())
    _lines, sent = _stream_chat_with(app, {**_chat_body(True), "think": False})

    assert sent.get("chat_template_kwargs", {}).get("enable_thinking") is False
    assert "think" not in sent


def test_chat_stream_returns_reasoning_as_thinking() -> None:
    app = _make_app(model_state=_llama_state())
    lines, _sent = _stream_chat(
        app,
        [
            b'data: {"choices":[{"delta":{"reasoning_content":"pienso"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )

    assert lines[0]["message"] == {"role": "assistant", "content": "", "thinking": "pienso"}
    assert lines[1]["message"]["content"] == "Hola"


def test_chat_non_stream_returns_reasoning_as_thinking() -> None:
    app = _make_app(
        model_state=_llama_state(),
        worker_result={
            "choices": [
                {
                    "message": {"content": "Hola", "reasoning_content": "pienso"},
                    "finish_reason": "stop",
                }
            ]
        },
    )
    resp = TestClient(app).post("/api/chat", json=_chat_body(False))

    assert resp.json()["message"]["thinking"] == "pienso"


def _stream_chat_with(app, body: dict) -> tuple[list[dict], dict]:
    app.state.worker_pool.forward_stream = MagicMock(return_value=_async_gen([b"data: [DONE]\n\n"]))
    with TestClient(app).stream("POST", "/api/chat", json=body) as resp:
        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.iter_lines() if line]
    return lines, app.state.worker_pool.forward_stream.call_args.args[2]


# ---------------------------------------------------------------------------
# BUG-12: /api/embed
# ---------------------------------------------------------------------------


def _ollama_embed_state():
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    return ModelState(
        model_id="ollama/nomic-embed-text",
        display_name="nomic",
        backend_type="ollama",
        backend_model_id="nomic-embed-text:latest",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(embeddings=True),
    )


def test_embed_does_not_silently_embed_only_the_first_input() -> None:
    """Any /api/embed failure used to retry on /api/embeddings with input[0]:
    three inputs, one vector, 200."""
    app = _make_app(model_state=_ollama_embed_state())
    app.state.worker_pool.forward_request = AsyncMock(
        side_effect=_upstream_error(500, '{"error": "input too long"}')
    )
    resp = TestClient(app, raise_server_exceptions=False).post(
        "/api/embed", json={"model": "nomic-embed-text", "input": ["a", "b", "c"]}
    )

    assert resp.status_code == 500
    assert app.state.worker_pool.forward_request.await_count == 1


def test_embed_on_an_old_ollama_embeds_every_input() -> None:
    calls = []

    async def _forward(key, path, payload, **kw):
        calls.append(path)
        if path == "/api/embed":
            raise _upstream_error(404, "404 page not found")
        return {"embedding": [float(len(payload["prompt"]))]}

    app = _make_app(model_state=_ollama_embed_state())
    app.state.worker_pool.forward_request = _forward
    resp = TestClient(app).post(
        "/api/embed", json={"model": "nomic-embed-text", "input": ["a", "bb", "ccc"]}
    )

    assert resp.status_code == 200
    assert resp.json()["embeddings"] == [[1.0], [2.0], [3.0]]
    assert calls == ["/api/embed", "/api/embeddings", "/api/embeddings", "/api/embeddings"]


def test_embed_uses_the_backend_model_name_and_forwards_dimensions_to_ollama() -> None:
    app = _make_app(model_state=_ollama_embed_state(), worker_result={"embeddings": [[0.1, 0.2]]})
    TestClient(app).post(
        "/api/embed", json={"model": "nomic-embed-text", "input": "a", "dimensions": 2}
    )
    payload = app.state.worker_pool.forward_request.call_args.args[2]

    assert payload["model"] == "nomic-embed-text:latest"
    assert payload["dimensions"] == 2


def test_embed_truncates_and_renormalises_when_the_backend_ignores_dimensions() -> None:
    import math

    app = _make_app(worker_result={"data": [{"embedding": [3.0, 4.0, 12.0]}]})
    resp = TestClient(app).post(
        "/api/embed",
        json={"model": "meta-llama/Llama-3.2-3B-Instruct", "input": "a", "dimensions": 2},
    )
    vector = resp.json()["embeddings"][0]

    assert len(vector) == 2
    assert math.isclose(math.hypot(*vector), 1.0)
    assert app.state.worker_pool.forward_request.call_args.args[2]["dimensions"] == 2


def test_embed_rejects_invalid_dimensions() -> None:
    app = _make_app()
    resp = TestClient(app).post(
        "/api/embed",
        json={"model": "meta-llama/Llama-3.2-3B-Instruct", "input": "a", "dimensions": 0},
    )
    assert resp.status_code == 400


def test_embed_does_not_turn_a_403_into_a_legacy_404() -> None:
    from fastapi import HTTPException

    app = _make_app()
    with patch(
        "ocabra.api.ollama.embeddings.resolve_profile",
        AsyncMock(side_effect=HTTPException(status_code=403, detail="no access")),
    ):
        resp = TestClient(app).post(
            "/api/embed", json={"model": "recomendado-embedding", "input": "a"}
        )
    assert resp.status_code == 403


def test_embed_passes_the_router_resolver_and_federation_gets_the_registry() -> None:
    import inspect

    from ocabra.api.ollama import embeddings

    source = inspect.getsource(embeddings._run_embeddings)
    assert "router_resolver=" in source
    assert "federation_manager, profile_registry" in source


def test_legacy_embeddings_answers_ollamas_single_vector_field() -> None:
    app = _make_app(worker_result={"data": [{"embedding": [0.5, 0.5]}]})
    body = (
        TestClient(app)
        .post("/api/embeddings", json={"model": "meta-llama/Llama-3.2-3B-Instruct", "prompt": "a"})
        .json()
    )

    assert body["embedding"] == [0.5, 0.5]
    assert body["embeddings"] == [[0.5, 0.5]]


# ---------------------------------------------------------------------------
# REF-01 fase C: con un PERFIL resuelto (no la vía legacy), el núcleo
# no-streaming pasa por ``run_request`` (``preresolved``) y la traducción de
# vuelta a la forma Ollama se conserva byte a byte.
# ---------------------------------------------------------------------------


def _profile_state():
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    return ModelState(
        model_id="llamacpp_mks/gemma4:26b",
        display_name="Gemma 4 26B",
        backend_type="llamacpp_mks",
        backend_model_id="gemma4:26b",
        status=ModelStatus.LOADED,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(chat=True, completion=True, streaming=True, tools=True),
    )


def test_chat_profile_path_dispatches_via_engine_and_keeps_tool_calls() -> None:
    state = _profile_state()
    profile = _make_test_profile("gemma4:26b", "llamacpp_mks/gemma4:26b")
    worker_result = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Madrid"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    app = _make_app(model_state=state, worker_result=worker_result)
    with patch(
        "ocabra.api.ollama.chat.resolve_profile", new=AsyncMock(return_value=(profile, state))
    ):
        resp = TestClient(app).post(
            "/api/chat",
            json={
                "model": "gemma4:26b",
                "messages": [{"role": "user", "content": "weather?"}],
                "stream": False,
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["done_reason"] == "tool_calls"
    assert body["message"]["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Madrid"}}}
    ]
    called = app.state.worker_pool.forward_request.await_args
    assert called.args[1] == "/v1/chat/completions"
    assert called.args[2]["model"] == "gemma4:26b"


def test_chat_profile_path_maps_worker_errors_to_ollama_error_shape() -> None:
    state = _profile_state()
    profile = _make_test_profile("gemma4:26b", "llamacpp_mks/gemma4:26b")
    app = _make_app(model_state=state)
    app.state.worker_pool.forward_request = AsyncMock(
        side_effect=_upstream_error(400, '{"error": {"message": "Missing tool call type"}}')
    )
    with patch(
        "ocabra.api.ollama.chat.resolve_profile", new=AsyncMock(return_value=(profile, state))
    ):
        resp = TestClient(app, raise_server_exceptions=False).post(
            "/api/chat", json=_chat_body(False)
        )

    # El motor mapea el error de worker al mismo estado que /v1; la app mínima
    # del test no monta el handler de main.py que lo envuelve en ``{"error":
    # ...}`` (familia /api), así que aquí se comprueba el estado y el mensaje
    # conservados (el envoltorio Ollama lo cubren los tests de ``_errors``).
    assert resp.status_code == 400
    assert "Missing tool call type" in resp.text


def test_generate_profile_path_dispatches_via_engine() -> None:
    state = _profile_state()
    profile = _make_test_profile("gemma4:26b", "llamacpp_mks/gemma4:26b")
    app = _make_app(
        model_state=state,
        worker_result={
            "choices": [{"text": "hola", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
    )
    with patch(
        "ocabra.api.ollama.generate.resolve_profile", new=AsyncMock(return_value=(profile, state))
    ):
        resp = TestClient(app).post(
            "/api/generate", json={"model": "gemma4:26b", "prompt": "hola", "stream": False}
        )

    assert resp.status_code == 200
    assert resp.json()["response"] == "hola"
    assert resp.json()["done_reason"] == "stop"
    called = app.state.worker_pool.forward_request.await_args
    assert called.args[1] == "/v1/completions"
