"""BUG-16: the legacy ``ensure_loaded`` (Ollama layer) waited the whole load
timeout for a load that had already failed."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from ocabra.backends.base import BackendCapabilities
from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus


def _state(status: ModelStatus) -> ModelState:
    return ModelState(
        model_id="ollama/qwen3:8b",
        display_name="qwen3",
        backend_type="ollama",
        status=status,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(chat=True),
    )


@pytest.mark.asyncio
async def test_a_load_that_fails_while_waiting_answers_at_once():
    # ensure_loaded lives in core.inference.resolution and delegates the wait
    # loop to core.inference.loading (REF-06); patch each in its own module.
    from ocabra.core.inference import loading as _loading
    from ocabra.core.inference import resolution as _resolution

    loading, failed = _state(ModelStatus.LOADING), _state(ModelStatus.ERROR)
    failed.error_message = "CUDA out of memory"
    mm = MagicMock()
    mm.get_state = AsyncMock(side_effect=[loading, failed, failed, failed])
    sleeps = AsyncMock()

    with (
        patch.object(
            _resolution, "resolve_model", AsyncMock(return_value=("ollama/qwen3:8b", loading))
        ),
        patch.object(_loading, "_wait_for_eviction_claim", AsyncMock(side_effect=lambda m, i, s: s)),
        patch.object(
            _loading, "_wait_for_unloading_transition", AsyncMock(side_effect=lambda m, i, s: s)
        ),
        patch.object(_loading, "_ensure_load_timeout_s", return_value=600),
        patch.object(_loading.asyncio, "sleep", sleeps),
    ):
        with pytest.raises(HTTPException) as exc:
            await _resolution.ensure_loaded(mm, "qwen3:8b")

    assert sleeps.await_count <= 2  # not 600 one-second polls
    assert "CUDA out of memory" in str(exc.value.detail)
    assert "did not finish loading" not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_an_unknown_legacy_name_is_a_404():
    from ocabra.core.inference import resolution as _resolution

    with patch.object(_resolution, "resolve_model", AsyncMock(return_value=("x", None))):
        with pytest.raises(HTTPException) as exc:
            await _resolution.ensure_loaded(MagicMock(), "x")
    assert exc.value.status_code == 404


def test_upstream_retry_after_is_kept():
    """BUG-19: a backend's own 429/503 lost its Retry-After."""
    import httpx

    from ocabra.api.openai._deps import raise_upstream_http_error

    req = httpx.Request("POST", "http://worker/v1/chat/completions")
    for body in ('{"error": {"message": "busy"}}', "plain busy"):
        resp = httpx.Response(503, text=body, request=req, headers={"Retry-After": "7"})
        with pytest.raises(HTTPException) as exc:
            raise_upstream_http_error(httpx.HTTPStatusError("x", request=req, response=resp))
        assert exc.value.status_code == 503
        assert exc.value.headers == {"retry-after": "7"}


@pytest.mark.asyncio
async def test_insufficient_vram_says_when_to_retry():
    from ocabra.core.inference import loading as _loading
    from ocabra.core.scheduler import InsufficientVRAMError

    mm = MagicMock()
    mm.get_state = AsyncMock(return_value=_state(ModelStatus.UNLOADED))
    mm.load = AsyncMock(side_effect=InsufficientVRAMError("no room"))
    with (
        patch.object(_loading, "_wait_for_eviction_claim", AsyncMock(side_effect=lambda m, i, s: s)),
        patch.object(
            _loading, "_wait_for_unloading_transition", AsyncMock(side_effect=lambda m, i, s: s)
        ),
        patch.object(
            _loading, "_wait_for_service_gpu_and_retry_load", AsyncMock(return_value=None)
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await _loading._do_ensure_loaded(mm, "ollama/qwen3:8b")
    assert exc.value.status_code == 409
    assert exc.value.headers.get("Retry-After")


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"])
@pytest.mark.parametrize("raw", [b"{not json", b"[1, 2]", b'"text"'])
def test_a_body_that_is_not_a_json_object_is_a_400_not_a_500(path, raw):
    """BUG-18."""
    from fastapi.testclient import TestClient

    from tests.test_openai_api import _make_app

    resp = TestClient(_make_app(), raise_server_exceptions=False).post(
        path, content=raw, headers={"content-type": "application/json"}
    )
    assert resp.status_code == 400, resp.text
    assert "invalid_json" in resp.text


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate", "/api/embed"])
def test_ollama_paths_reject_a_non_object_body_with_400(path):
    from fastapi.testclient import TestClient

    from tests.test_ollama_api import _make_app

    resp = TestClient(_make_app(), raise_server_exceptions=False).post(
        path, content=b"[]", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 400, resp.text
