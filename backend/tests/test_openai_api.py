"""
Tests for OpenAI API compatibility layer.

Uses FastAPI TestClient with mocked ModelManager and WorkerPool.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------

def _make_app(model_state=None, worker_result=None):
    """Create a minimal FastAPI app with mocked state."""
    from ocabra.api.openai import router as openai_router
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import ModelState, ModelStatus, LoadPolicy

    app = FastAPI()
    app.include_router(openai_router, prefix="/v1")

    # Mock model manager
    mm = MagicMock()
    if model_state is None:
        caps = BackendCapabilities(
            chat=True, completion=True, tools=True, streaming=True,
            embeddings=False, image_generation=False, audio_transcription=False, tts=False,
            context_length=8192,
        )
        model_state = ModelState(
            model_id="test-model",
            display_name="Test Model",
            backend_type="vllm",
            status=ModelStatus.LOADED,
            load_policy=LoadPolicy.ON_DEMAND,
            capabilities=caps,
        )

    mm.get_state = AsyncMock(return_value=model_state)
    mm.list_states = AsyncMock(return_value=[model_state])
    mm.load = AsyncMock(return_value=model_state)

    # Mock worker pool
    wp = MagicMock()
    wp.forward_request = AsyncMock(return_value=worker_result or {})
    wp.forward_stream = AsyncMock(return_value=_async_gen([b"data: chunk\n\n", b"data: [DONE]\n\n"]))
    worker = MagicMock()
    worker.port = 18001
    wp.get_worker = MagicMock(return_value=worker)

    app.state.model_manager = mm
    app.state.worker_pool = wp

    return app


async def _async_gen(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_returns_loaded_models(self):
        from ocabra.backends.base import BackendCapabilities
        from ocabra.core.model_manager import ModelState, ModelStatus, LoadPolicy

        caps = BackendCapabilities(chat=True, streaming=True, context_length=4096)
        state = ModelState(
            model_id="llama-3-8b",
            display_name="Llama 3 8B",
            backend_type="vllm",
            status=ModelStatus.LOADED,
            load_policy=LoadPolicy.WARM,
            capabilities=caps,
            current_gpu=[1],
            vram_used_mb=8192,
        )
        app = _make_app(model_state=state)
        client = TestClient(app)

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        model = data["data"][0]
        assert model["id"] == "llama-3-8b"
        assert model["owned_by"] == "ocabra"
        assert model["ocabra"]["status"] == "loaded"

    def test_excludes_unloaded_models(self):
        from ocabra.backends.base import BackendCapabilities
        from ocabra.core.model_manager import ModelState, ModelStatus, LoadPolicy

        state = ModelState(
            model_id="ghost-model",
            display_name="Ghost",
            backend_type="vllm",
            status=ModelStatus.UNLOADED,
            load_policy=LoadPolicy.ON_DEMAND,
            capabilities=BackendCapabilities(),
        )
        app = _make_app(model_state=state)
        client = TestClient(app)

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json()["data"] == []


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

class TestChatCompletions:
    def test_non_stream_returns_json(self):
        fake_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        app = _make_app(worker_result=fake_response)
        client = TestClient(app)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        assert resp.json()["object"] == "chat.completion"

    def test_model_not_found_returns_404(self):
        app = _make_app()
        app.state.model_manager.get_state = AsyncMock(return_value=None)
        client = TestClient(app)

        resp = client.post("/v1/chat/completions", json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"]["code"] == "model_not_found"

    def test_capability_check_fails_for_embedding_model(self):
        from ocabra.backends.base import BackendCapabilities
        from ocabra.core.model_manager import ModelState, ModelStatus, LoadPolicy

        # Embedding model — no chat capability
        state = ModelState(
            model_id="bert-base",
            display_name="BERT",
            backend_type="vllm",
            status=ModelStatus.LOADED,
            load_policy=LoadPolicy.ON_DEMAND,
            capabilities=BackendCapabilities(embeddings=True),
        )
        app = _make_app(model_state=state)
        client = TestClient(app)

        resp = client.post("/v1/chat/completions", json={
            "model": "bert-base",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"]["code"] == "model_not_capable"

    def test_on_demand_load_triggered(self):
        """A CONFIGURED model should be loaded on first request."""
        from ocabra.backends.base import BackendCapabilities
        from ocabra.core.model_manager import ModelState, ModelStatus, LoadPolicy

        configured_state = ModelState(
            model_id="lazy-model",
            display_name="Lazy",
            backend_type="vllm",
            status=ModelStatus.CONFIGURED,
            load_policy=LoadPolicy.ON_DEMAND,
            capabilities=BackendCapabilities(chat=True, streaming=True),
        )
        loaded_state = ModelState(
            model_id="lazy-model",
            display_name="Lazy",
            backend_type="vllm",
            status=ModelStatus.LOADED,
            load_policy=LoadPolicy.ON_DEMAND,
            capabilities=BackendCapabilities(chat=True, streaming=True),
        )
        app = _make_app(worker_result={"object": "chat.completion", "choices": []})

        # First call returns CONFIGURED, after load returns LOADED
        app.state.model_manager.get_state = AsyncMock(side_effect=[
            configured_state,  # initial check
            loaded_state,      # after load call
        ])
        app.state.model_manager.load = AsyncMock(return_value=loaded_state)

        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={
            "model": "lazy-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        app.state.model_manager.load.assert_called_once_with("lazy-model")


# ---------------------------------------------------------------------------
# POST /v1/embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_embeddings_success(self):
        from ocabra.backends.base import BackendCapabilities
        from ocabra.core.model_manager import ModelState, ModelStatus, LoadPolicy

        state = ModelState(
            model_id="bert-base",
            display_name="BERT",
            backend_type="vllm",
            status=ModelStatus.LOADED,
            load_policy=LoadPolicy.ON_DEMAND,
            capabilities=BackendCapabilities(embeddings=True),
        )
        fake_response = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        app = _make_app(model_state=state, worker_result=fake_response)
        client = TestClient(app)

        resp = client.post("/v1/embeddings", json={
            "model": "bert-base",
            "input": "Hello world",
        })
        assert resp.status_code == 200
        assert resp.json()["object"] == "list"

    def test_chat_model_cannot_embed(self):
        app = _make_app()  # default state has chat=True, embeddings=False
        client = TestClient(app)

        resp = client.post("/v1/embeddings", json={
            "model": "test-model",
            "input": "Hello",
        })
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"]["code"] == "model_not_capable"


# ---------------------------------------------------------------------------
# POST /v1/images/generations
# ---------------------------------------------------------------------------

class TestImageGenerations:
    def test_size_parsing(self):
        from ocabra.api.openai.images import _parse_size
        assert _parse_size("1024x1024") == (1024, 1024)
        assert _parse_size("1792x1024") == (1792, 1024)
        assert _parse_size("1024x1792") == (1024, 1792)
        assert _parse_size("256x256") == (256, 256)
        assert _parse_size("unknown") == (1024, 1024)

    def test_image_generation_requires_capability(self):
        app = _make_app()  # default has image_generation=False
        client = TestClient(app)

        resp = client.post("/v1/images/generations", json={
            "model": "test-model",
            "prompt": "a cat",
        })
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"]["code"] == "model_not_capable"
