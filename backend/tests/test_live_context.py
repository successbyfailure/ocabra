"""Changing a loaded llama.cpp worker's context in place (fork, POST /props {"context"})."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from ocabra.api.openai._deps import fairness_kv_budget, resolve_profile_runtime_plan
from ocabra.backends.base import BackendCapabilities, WorkerInfo
from ocabra.backends.llama_cpp_backend import (
    LiveContextBusy,
    LiveContextFailed,
    LiveContextUnsupported,
    LlamaCppBackend,
)
from ocabra.backends.llamacpp_mks_backend import LlamaCppMksBackend
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus
from ocabra.core.model_manager_helpers import resolve_worker_slots
from ocabra.core.runtime_coordinator import RuntimePlan
from ocabra.core.worker_pool import WorkerPool
from ocabra.schemas.live_context import LiveContextRequest

RESULT = {
    "n_ctx": 16384,
    "n_ctx_slot": 16384,
    "n_parallel": 2,
    "n_ubatch": 512,
    "n_batch": 2048,
    "t_ms": 131.9,
}


def _http(status: int, payload: dict | None = None, text: str = ""):
    response = MagicMock(status_code=status, text=text)
    response.json.return_value = payload or {}
    client = AsyncMock()
    client.post.return_value = response
    context = AsyncMock()
    context.__aenter__.return_value = client
    return client, patch(
        "ocabra.backends.llama_cpp_backend.httpx.AsyncClient", return_value=context
    )


def _mks_with_worker() -> LlamaCppMksBackend:
    backend = LlamaCppMksBackend()
    backend._processes["m"] = SimpleNamespace(port=18050)
    backend._model_configs["m"] = {
        "ctx_size": 65536,
        "parallel_slots": 4,
        "ubatch_size": 128,
        "batch_size": 512,
    }
    return backend


# -- backend -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconfigure_posts_only_given_sizes_and_updates_options() -> None:
    backend = _mks_with_worker()
    client, patcher = _http(200, {"success": True, "context": RESULT})
    with patcher:
        result = await backend.reconfigure_context("m", n_ctx=16384, n_parallel=2)

    client.post.assert_awaited_once_with(
        "http://127.0.0.1:18050/props", json={"context": {"n_ctx": 16384, "n_parallel": 2}}
    )
    assert result["t_ms"] == 131.9
    assert backend._model_configs["m"]["ctx_size"] == 16384
    assert backend._model_configs["m"]["parallel_slots"] == 2
    assert backend._model_configs["m"]["ubatch_size"] == 512
    assert backend.context_options("m") == {
        "n_ctx": 16384,
        "n_parallel": 2,
        "n_ubatch": 512,
        "n_batch": 2048,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error"),
    [
        (501, LiveContextUnsupported),
        (404, LiveContextUnsupported),
        (503, LiveContextBusy),
        (500, LiveContextFailed),
    ],
)
async def test_reconfigure_maps_errors_and_keeps_options(status, error) -> None:
    backend = _mks_with_worker()
    _, patcher = _http(status, text="nope")
    with patcher, pytest.raises(error):
        await backend.reconfigure_context("m", n_ctx=1_048_576)
    assert backend._model_configs["m"]["ctx_size"] == 65536


@pytest.mark.asyncio
async def test_upstream_backend_does_not_support_live_context() -> None:
    backend = LlamaCppBackend()
    backend._processes["m"] = SimpleNamespace(port=18050)
    with pytest.raises(LiveContextUnsupported):
        await backend.reconfigure_context("m", n_ctx=16384)
    assert LlamaCppMksBackend.SUPPORTS_LIVE_CONTEXT is True


# -- readers -------------------------------------------------------------------


def _state(**kw) -> ModelState:
    state = ModelState(
        model_id="llamacpp_mks/qwen3.8:27b::abc",
        display_name="q",
        backend_type="llamacpp_mks",
        load_policy=LoadPolicy.ON_DEMAND,
        extra_config={
            "llama_cpp": {"ctx_size": 262144, "parallel_slots": 4, "kv_unified": True},
            "_runtime": {
                "preset_key": "extended",
                "parallelism": 4,
                "effective_context": 262144,
                "context_is_pool": True,
            },
        },
    )
    for key, value in kw.items():
        setattr(state, key, value)
    return state


def test_worker_slots_follow_live_context() -> None:
    assert resolve_worker_slots(_state()) == 4
    assert resolve_worker_slots(_state(live_context={"n_parallel": 2, "n_ctx": 65536})) == 2


def test_kv_budget_is_capped_by_live_pool() -> None:
    assert fairness_kv_budget(_state(), required_tokens=1000)[0] == 262144
    capped = fairness_kv_budget(
        _state(live_context={"n_ctx": 65536, "n_parallel": 4}),
        required_tokens=1000,
        effective_context=262144,
    )
    assert capped[0] == 65536


def test_request_schema() -> None:
    assert LiveContextRequest(context=65536).context == 65536
    assert LiveContextRequest(restore=True).restore is True
    with pytest.raises(ValidationError):
        LiveContextRequest()
    with pytest.raises(ValidationError):
        LiveContextRequest(restore=True, context=1)


# -- ModelManager.resize_context ------------------------------------------------


def _manager(monkeypatch, *, enabled=True, busy=False):
    monkeypatch.setattr(
        "ocabra.core.model_manager.settings.llama_cpp_live_context_enabled", enabled
    )
    monkeypatch.setattr(
        "ocabra.core.model_manager.settings.llama_cpp_live_context_drain_timeout_s", 0.05
    )
    backend = MagicMock()
    backend.SUPPORTS_LIVE_CONTEXT = True
    backend.context_options.return_value = {
        "n_ctx": 262144,
        "n_parallel": 4,
        "n_ubatch": 128,
        "n_batch": 512,
    }
    backend.reconfigure_context = AsyncMock(return_value=dict(RESULT))
    backend.get_capabilities = AsyncMock(
        return_value=BackendCapabilities(chat=True, context_length=16384)
    )
    pool = WorkerPool()
    pool.get_backend = AsyncMock(return_value=backend)
    gpu = MagicMock()
    gpu.get_process_vram_mb = MagicMock(side_effect=lambda idx: {4242: {0: 8000, 1: 20000}[idx]})
    gpu.lock_vram = AsyncMock()
    mm = ModelManager(pool, gpu_manager=gpu)
    state = _state(status=ModelStatus.LOADED, current_gpu=[0, 1], vram_used_mb=31000)
    state.worker_info = WorkerInfo(
        backend_type="llamacpp_mks",
        model_id=state.model_id,
        gpu_indices=[0, 1],
        port=18050,
        pid=4242,
        vram_used_mb=31000,
    )
    mm._states[state.model_id] = state
    mm._load_locks[state.model_id] = asyncio.Lock()
    if busy:
        mm._in_flight[state.model_id] = 1
    mm._publish_event = AsyncMock()
    return mm, state, backend, gpu


@pytest.mark.asyncio
async def test_resize_records_live_context_and_real_vram(monkeypatch) -> None:
    mm, state, backend, gpu = _manager(monkeypatch)

    await mm.resize_context(state.model_id, context=16384, parallel=2, ubatch=512, batch=2048)

    backend.reconfigure_context.assert_awaited_once_with(
        state.backend_model_id, n_ctx=16384, n_parallel=2, n_ubatch=512, n_batch=2048
    )
    assert state.live_context["n_ctx"] == 16384
    assert state.live_context["nominal"]["n_ctx"] == 262144
    assert state.extra_config["llama_cpp"]["ctx_size"] == 262144  # nominal kept
    assert state.capabilities.context_length == 16384
    gpu.lock_vram.assert_any_await(0, 8000, state.model_id)
    gpu.lock_vram.assert_any_await(1, 20000, state.model_id)
    assert state.vram_used_mb == 28000


@pytest.mark.asyncio
async def test_restore_goes_back_to_nominal_and_clears_live_context(monkeypatch) -> None:
    mm, state, backend, _ = _manager(monkeypatch)
    state.live_context = {
        **RESULT,
        "nominal": {"n_ctx": 262144, "n_parallel": 4, "n_ubatch": 128, "n_batch": 512},
    }
    backend.reconfigure_context.return_value = {
        "n_ctx": 262144,
        "n_ctx_slot": 262144,
        "n_parallel": 4,
        "n_ubatch": 128,
        "n_batch": 512,
        "t_ms": 140.0,
    }

    await mm.resize_context(state.model_id, restore=True)

    backend.reconfigure_context.assert_awaited_once_with(
        state.backend_model_id, n_ctx=262144, n_parallel=4, n_ubatch=128, n_batch=512
    )
    assert state.live_context is None


@pytest.mark.asyncio
async def test_resize_refuses_when_disabled_or_busy(monkeypatch) -> None:
    mm, state, backend, _ = _manager(monkeypatch, enabled=False)
    with pytest.raises(RuntimeError, match="disabled"):
        await mm.resize_context(state.model_id, context=16384)

    mm, state, backend, _ = _manager(monkeypatch, busy=True)
    with pytest.raises(RuntimeError, match="busy"):
        await mm.resize_context(state.model_id, context=16384)
    backend.reconfigure_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_resize_leaves_state_untouched(monkeypatch) -> None:
    mm, state, backend, gpu = _manager(monkeypatch)
    backend.reconfigure_context.side_effect = LiveContextFailed("did not fit; previous restored")

    with pytest.raises(LiveContextFailed):
        await mm.resize_context(state.model_id, context=1_048_576)

    assert state.live_context is None
    gpu.lock_vram.assert_not_awaited()


# -- planner / request path ------------------------------------------------------


def _plan(**kw) -> RuntimePlan:
    base = dict(
        worker_key="llamacpp_mks/qwen3.8:27b::abc",
        load_overrides=None,
        preferred_gpu=None,
        load_policy=None,
        preset_key="extended",
        runtime_key="rk",
        effective_context=262144,
        required_tokens=100_000,
        selection_reason="loaded",
        kv_reservation_tokens=100_000,
        live_effective_context=65536,
    )
    base.update(kw)
    return RuntimePlan(**base)


@pytest.mark.asyncio
async def test_request_larger_than_live_context_grows_worker_back() -> None:
    mm = MagicMock()
    mm.resize_context = AsyncMock()
    request_state = SimpleNamespace()
    with patch("ocabra.api.openai._deps.plan_profile_runtime", AsyncMock(return_value=_plan())):
        await resolve_profile_runtime_plan(MagicMock(), mm, {}, request_state=request_state)

    mm.resize_context.assert_awaited_once_with("llamacpp_mks/qwen3.8:27b::abc", restore=True)
    assert request_state.runtime_effective_context == 262144


@pytest.mark.asyncio
async def test_request_that_fits_live_context_uses_it_without_regrowing() -> None:
    mm = MagicMock()
    mm.resize_context = AsyncMock()
    request_state = SimpleNamespace()
    with patch(
        "ocabra.api.openai._deps.plan_profile_runtime",
        AsyncMock(return_value=_plan(required_tokens=4000)),
    ):
        await resolve_profile_runtime_plan(MagicMock(), mm, {}, request_state=request_state)

    mm.resize_context.assert_not_awaited()
    assert request_state.runtime_effective_context == 65536


@pytest.mark.asyncio
async def test_failed_regrow_is_a_retryable_503() -> None:
    mm = MagicMock()
    mm.resize_context = AsyncMock(side_effect=RuntimeError("no VRAM"))
    with (
        patch("ocabra.api.openai._deps.plan_profile_runtime", AsyncMock(return_value=_plan())),
        pytest.raises(HTTPException) as caught,
    ):
        await resolve_profile_runtime_plan(MagicMock(), mm, {}, request_state=SimpleNamespace())

    assert caught.value.status_code == 503
    assert caught.value.headers["Retry-After"] == "5"


# -- internal endpoint -----------------------------------------------------------


def _request(states, resize=None):
    mm = MagicMock()
    mm.get_state = AsyncMock(return_value=states[0] if states else None)
    mm.list_states = AsyncMock(return_value=states)
    mm.resize_context = resize or AsyncMock()
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model_manager=mm))), mm


def _loaded(model_id: str, status: str = "loaded"):
    return SimpleNamespace(
        model_id=model_id, status=SimpleNamespace(value=status), live_context=None, vram_used_mb=1
    )


@pytest.mark.asyncio
async def test_endpoint_resolves_canonical_id_to_its_loaded_worker() -> None:
    from ocabra.api.internal.models import change_model_context

    states = [_loaded("llamacpp_mks/q", "configured"), _loaded("llamacpp_mks/q::abc")]
    request, mm = _request(states, resize=AsyncMock(return_value=states[1]))

    out = await change_model_context(
        "llamacpp_mks/q", LiveContextRequest(context=65536), request, None
    )

    mm.resize_context.assert_awaited_once_with(
        "llamacpp_mks/q::abc", context=65536, parallel=None, ubatch=None, batch=None, restore=False
    )
    assert out["worker_id"] == "llamacpp_mks/q::abc"


@pytest.mark.asyncio
async def test_endpoint_errors() -> None:
    from ocabra.api.internal.models import change_model_context

    request, _ = _request([_loaded("llamacpp_mks/q", "configured")])
    with pytest.raises(HTTPException) as caught:
        await change_model_context("llamacpp_mks/q", LiveContextRequest(context=1), request, None)
    assert caught.value.status_code == 404

    request, _ = _request([_loaded("llamacpp_mks/q::a"), _loaded("llamacpp_mks/q::b")])
    with pytest.raises(HTTPException) as caught:
        await change_model_context("llamacpp_mks/q", LiveContextRequest(context=1), request, None)
    assert caught.value.status_code == 409

    request, _ = _request(
        [_loaded("llamacpp_mks/q")], resize=AsyncMock(side_effect=LiveContextFailed("did not fit"))
    )
    with pytest.raises(HTTPException) as caught:
        await change_model_context("llamacpp_mks/q", LiveContextRequest(context=1), request, None)
    assert caught.value.status_code == 409
