"""Permissive request-body validation (REF-11, closes BUG-18).

The handlers under ``api/openai`` and ``api/ollama`` validate the JSON body
against the Pydantic schemas in ``api/openai/_schemas.py`` as an **early gate**:
a genuinely malformed body (missing a required field, a known field with the
wrong JSON type) is answered with the family's error format (REF-10) instead of
reaching the backend and failing there. The gate must stay permissive — unknown
fields and loosely-typed values a real OpenAI/Ollama client sends have to pass
through untouched — so these tests pin both halves: malformed input is rejected,
and a valid request with extra fields reaches the backend unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.api.openai._schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    OllamaChatRequest,
    validate_body,
)

# ── Unit: the gate itself ────────────────────────────────────────


def test_valid_body_passes_through_unchanged_same_object():
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        # Unknown fields a real client/SDK adds must survive verbatim.
        "x_vendor_ext": {"a": 1},
        "num_ctx": 32768,
    }
    out = validate_body(ChatCompletionRequest, body)
    assert out is body
    assert out["x_vendor_ext"] == {"a": 1}
    assert out["num_ctx"] == 32768


def test_loose_values_are_accepted():
    # Strings for numeric fields, token-id arrays for prompt: all coerced/kept
    # by the permissive schema, never rejected.
    validate_body(ChatCompletionRequest, {"model": "m", "messages": [], "temperature": "0.7"})
    validate_body(CompletionRequest, {"model": "m", "prompt": [1, 2, 3]})
    validate_body(EmbeddingRequest, {"model": "m", "input": [[1, 2], [3, 4]]})


def test_missing_required_field_raises_422_openai_shape():
    with pytest.raises(HTTPException) as exc_info:
        validate_body(ChatCompletionRequest, {"messages": [{"role": "user", "content": "hi"}]})
    exc = exc_info.value
    assert exc.status_code == 422
    assert exc.detail["error"]["param"] == "model"
    assert "model" in exc.detail["error"]["message"]


def test_wrong_type_for_known_field_raises_422():
    with pytest.raises(HTTPException) as exc_info:
        validate_body(ChatCompletionRequest, {"model": "m", "messages": "not-a-list"})
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["error"]["param"] == "messages"


def test_ollama_only_model_required():
    # A model-load call (model, no messages) is valid.
    out = validate_body(OllamaChatRequest, {"model": "m"})
    assert out == {"model": "m"}
    with pytest.raises(HTTPException) as exc_info:
        validate_body(OllamaChatRequest, {"messages": []})
    assert exc_info.value.status_code == 422


# ── Integration: malformed → family error, through the real app ──


@pytest.mark.asyncio
async def test_v1_malformed_renders_openai_error(async_client):
    # No ``model``: a top-level ``error`` object (OpenAI shape), status 422.
    resp = await async_client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 422, resp.text
    err = resp.json()["error"]
    assert isinstance(err, dict)
    assert err["type"] == "invalid_request_error"
    assert err["param"] == "model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/completions", {"model": "m"}),  # no prompt
        ("/v1/embeddings", {"model": "m"}),  # no input
        ("/v1/images/generations", {"model": "m"}),  # no prompt
        ("/v1/audio/generate", {"model": "m"}),  # no prompt
        ("/v1/pooling", {"input": "x"}),  # no model
        ("/v1/chat/completions", {"model": "m", "messages": "bad"}),  # wrong type
    ],
)
async def test_v1_malformed_endpoints(async_client, path, body):
    resp = await async_client.post(path, json=body)
    assert resp.status_code == 422, resp.text
    assert isinstance(resp.json()["error"], dict)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/api/chat", "/api/generate", "/api/embed", "/api/embeddings"],
)
async def test_api_malformed_renders_ollama_error(async_client, path):
    # No ``model``: Ollama's top-level ``error`` *string*, status 422.
    resp = await async_client.post(path, json={"messages": []})
    assert resp.status_code == 422, resp.text
    assert isinstance(resp.json()["error"], str)
    assert "model" in resp.json()["error"]


# ── Integration: a valid request with extras reaches the backend ─


def _admin_ctx() -> UserContext:
    return UserContext(user_id="u1", username="tester", role="system_admin")


def test_embeddings_forwards_extra_fields_unchanged():
    """A valid /v1/embeddings body with an unknown field reaches the worker
    with that field intact — the validation gate never strips it."""
    from ocabra.api.openai import router as openai_router

    app = FastAPI()
    app.include_router(openai_router, prefix="/v1")
    app.dependency_overrides[get_current_user] = _admin_ctx
    pool = MagicMock()
    pool.forward_request = AsyncMock(return_value={"data": [], "model": "e"})
    app.state.worker_pool = pool
    app.state.model_manager = MagicMock()
    app.state.profile_registry = MagicMock()
    app.state.router_resolver = MagicMock(is_router=MagicMock(return_value=False))

    profile = SimpleNamespace(
        profile_id="e", base_model_id="be", load_overrides=None, request_defaults=None, assets=None
    )
    state = SimpleNamespace(
        model_id="be",
        backend_type="llamacpp_mks",
        backend_model_id="e",
        capabilities=SimpleNamespace(embeddings=True),
    )
    mod = "ocabra.api.openai.embeddings"
    pipeline = "ocabra.core.inference.pipeline"
    with (
        patch(f"{mod}.try_proxy_json", AsyncMock(return_value=None)),
        patch(f"{pipeline}.resolve_profile", AsyncMock(return_value=(profile, state))),
        patch(f"{pipeline}.check_capability"),
        patch(f"{pipeline}.merge_profile_defaults", lambda _p, body: body),
        patch(f"{pipeline}.to_backend_body", lambda _s, body: body),
        patch(f"{pipeline}.ocabra_response_headers", AsyncMock(return_value={})),
    ):
        resp = TestClient(app, raise_server_exceptions=True).post(
            "/v1/embeddings",
            json={"model": "e", "input": ["hola"], "x_vendor": "keep-me"},
        )

    assert resp.status_code == 200, resp.text
    forwarded = pool.forward_request.await_args.args[2]
    assert forwarded["x_vendor"] == "keep-me"
    assert forwarded["input"] == ["hola"]


def test_chat_forwards_extra_fields_unchanged():
    """A valid non-streaming /v1/chat/completions body with an unknown field
    reaches the worker intact."""
    from ocabra.api.openai import router as openai_router

    app = FastAPI()
    app.include_router(openai_router, prefix="/v1")
    app.dependency_overrides[get_current_user] = _admin_ctx
    pool = MagicMock()
    pool.forward_request = AsyncMock(return_value={"choices": [], "usage": {}})
    pool.peek_queue_state = MagicMock(return_value=None)
    app.state.worker_pool = pool
    app.state.model_manager = MagicMock()
    app.state.profile_registry = MagicMock()
    app.state.router_resolver = MagicMock(is_router=MagicMock(return_value=False))

    profile = SimpleNamespace(profile_id="c", base_model_id="bc")
    state = SimpleNamespace(model_id="bc", backend_type="llamacpp_mks", backend_model_id="c")
    mod = "ocabra.api.openai.chat"
    with (
        patch("ocabra.api.openai._media_urls.check_media_urls", AsyncMock()),
        patch(f"{mod}.resolve_profile", AsyncMock(return_value=(profile, state))),
        patch(f"{mod}.check_capability"),
        patch(f"{mod}.merge_profile_defaults", lambda _p, body: body),
        patch(f"{mod}.apply_declared_output_limit"),
        patch(f"{mod}.to_backend_body", lambda _s, body: body),
        patch(f"{mod}.fairness_slots", lambda _s: 1),
        patch(f"{mod}.fairness_user_key", lambda _u: None),
        patch(f"{mod}.fairness_kv_budget", lambda *a, **k: (None, 0)),
        patch(f"{mod}.is_batch_origin_request", lambda _r: False),
        # chat.py imports this one lazily from ``_deps`` inside the handler.
        patch("ocabra.api.openai._deps.ocabra_response_headers", AsyncMock(return_value={})),
    ):
        resp = TestClient(app, raise_server_exceptions=True).post(
            "/v1/chat/completions",
            json={
                "model": "c",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "x_vendor": "keep-me",
            },
        )

    assert resp.status_code == 200, resp.text
    forwarded = pool.forward_request.await_args.args[2]
    assert forwarded["x_vendor"] == "keep-me"
    assert forwarded["messages"] == [{"role": "user", "content": "hi"}]
