"""Tests for Ollama API compatibility layer."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(model_state=None, worker_result=None) -> FastAPI:
    from ocabra.api.ollama import router as ollama_router
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus

    app = FastAPI()
    app.include_router(ollama_router)

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

    wp = MagicMock()
    wp.forward_request = AsyncMock(return_value=worker_result or {})
    wp.forward_stream = MagicMock(return_value=_async_gen([b"data: [DONE]\\n\\n"]))

    app.state.model_manager = mm
    app.state.worker_pool = wp
    return app


async def _async_gen(items):
    for item in items:
        yield item


def test_name_mapper_round_trip() -> None:
    from ocabra.api.ollama._mapper import OllamaNameMapper

    mapper = OllamaNameMapper()
    internal = mapper.to_internal("llama3.2:3b")
    assert internal == "meta-llama/Llama-3.2-3B-Instruct"
    assert mapper.to_ollama(internal) == "llama3.2:3b"


def test_name_resolution_prefers_native_ollama_model() -> None:
    import asyncio

    from ocabra.api.ollama._mapper import resolve_model

    native_state = object()
    mapped_state = object()

    class FakeModelManager:
        async def get_state(self, model_id: str):
            if model_id == "llama3.2:3b":
                return native_state
            if model_id == "meta-llama/Llama-3.2-3B-Instruct":
                return mapped_state
            return None

    model_id, state = asyncio.run(resolve_model(FakeModelManager(), "llama3.2:3b"))
    assert model_id == "llama3.2:3b"
    assert state is native_state


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
