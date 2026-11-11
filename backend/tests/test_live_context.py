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


# -- shrink before evict / grow back (phase 2) ---------------------------------


def _elastic_manager(monkeypatch, *, free=None, kv_mb_per_64k=(500, 1000)):
    """Manager whose worker's VRAM follows its context, like the real fork:
    every 64k tokens of KV cost ``kv_mb_per_64k[gpu]`` on each GPU."""
    mm, state, backend, gpu = _manager(monkeypatch)
    monkeypatch.setattr("ocabra.core.model_manager.settings.llama_cpp_shrink_before_evict", True)
    monkeypatch.setattr("ocabra.core.model_manager.settings.llama_cpp_shrink_min_context", 32768)
    mm._worker_pool._backends["llamacpp_mks"] = backend
    ctx = {"n": 262144}
    weights = {0: 6000, 1: 16000}

    def vram(idx):
        return {4242: weights[idx] + kv_mb_per_64k[idx] * ctx["n"] // 65536}

    async def reconfigure(_model, *, n_ctx, n_parallel, n_ubatch, n_batch):
        ctx["n"] = n_ctx
        return {
            "n_ctx": n_ctx,
            "n_ctx_slot": n_ctx,
            "n_parallel": n_parallel,
            "n_ubatch": n_ubatch,
            "n_batch": n_batch,
            "t_ms": 150.0,
        }

    gpu.get_process_vram_mb = MagicMock(side_effect=vram)
    free_mb = dict(free or {0: 0, 1: 0})
    full = {idx: vram(idx)[4242] for idx in weights}

    # What the GPU reports free: the base figure plus whatever the worker
    # released relative to its loaded size (as NVML would show it).
    gpu.get_free_vram = AsyncMock(
        side_effect=lambda idx: free_mb[idx] + full[idx] - vram(idx)[4242]
    )
    gpu.get_physical_free_vram = gpu.get_free_vram
    # What a direct reading gives (ModelManager._fresh_free_vram): the card's
    # usable ceiling, consistent with the free figure above, minus processes.
    gpu.admission_ceiling_mb = MagicMock(
        side_effect=lambda idx, physical=True: free_mb[idx] + full[idx] + 512
    )
    backend.reconfigure_context = AsyncMock(side_effect=reconfigure)
    return mm, state, backend, ctx, free_mb


@pytest.mark.asyncio
async def test_shrink_halves_until_the_gpu_frees_enough(monkeypatch) -> None:
    mm, state, _, ctx, _ = _elastic_manager(monkeypatch)

    # GPU 1 frees 1000 MB per 64k: 262k -> 131k frees 2000, -> 65k frees 3000.
    freed = await mm._try_shrink_for_space(state.model_id, 1, 2500)

    assert freed == 3000
    assert ctx["n"] == 65536
    assert state.live_context["n_ctx"] == 65536
    assert state.live_context["nominal_vram_mb"] == {0: 8000, 1: 20000}


@pytest.mark.asyncio
async def test_shrink_that_cannot_free_enough_puts_the_context_back(monkeypatch) -> None:
    mm, state, backend, ctx, _ = _elastic_manager(monkeypatch)

    # The 3060 holds little KV: even at the 32k floor it frees 1750 MB.
    freed = await mm._try_shrink_for_space(state.model_id, 0, 6000)

    assert freed == 0
    assert ctx["n"] == 262144
    assert state.live_context is None


@pytest.mark.asyncio
async def test_shrink_skips_busy_or_foreign_gpu_or_disabled(monkeypatch) -> None:
    mm, state, backend, ctx, _ = _elastic_manager(monkeypatch)
    assert await mm._try_shrink_for_space(state.model_id, 3, 100) == 0
    mm._in_flight[state.model_id] = 1
    assert await mm._try_shrink_for_space(state.model_id, 1, 100) == 0
    mm._in_flight.pop(state.model_id)
    monkeypatch.setattr("ocabra.core.model_manager.settings.llama_cpp_shrink_before_evict", False)
    assert await mm._try_shrink_for_space(state.model_id, 1, 100) == 0
    backend.reconfigure_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_evict_for_space_shrinks_instead_of_unloading(monkeypatch) -> None:
    mm, state, _, ctx, _ = _elastic_manager(monkeypatch)
    mm.unload = AsyncMock()
    mm._get_eviction_candidates = MagicMock(return_value=[state.model_id])

    freed = await mm._evict_for_space(1, 1500)

    assert freed == 2000
    mm.unload.assert_not_awaited()
    assert ctx["n"] == 131072


@pytest.mark.asyncio
async def test_evict_for_space_still_unloads_when_shrinking_is_not_enough(monkeypatch) -> None:
    mm, state, _, ctx, _ = _elastic_manager(monkeypatch)
    mm.unload = AsyncMock()
    mm._get_eviction_candidates = MagicMock(return_value=[state.model_id])

    await mm._evict_for_space(0, 20000)

    mm.unload.assert_awaited_once_with(state.model_id, reason="vram_pressure")
    assert ctx["n"] == 262144


@pytest.mark.asyncio
async def test_grow_back_evicts_what_took_the_room(monkeypatch) -> None:
    mm, state, _, ctx, free_mb = _elastic_manager(monkeypatch)
    await mm._try_shrink_for_space(state.model_id, 1, 2500)  # now 65k
    evicted = []

    async def evict(requesting, gpu_idx, needed):
        evicted.append((requesting, gpu_idx, needed))
        free_mb[gpu_idx] += needed
        return needed

    mm._evict_last_resort_for_space = AsyncMock(side_effect=evict)
    # Others took the room: GPU 1 only shows 1000 free beyond what it held.
    free_mb.update({0: 5000, 1: 1000 - 3000})

    await mm.grow_context_back(state.model_id)

    # GPU 1 needs 3000 back and has 1000 free; GPU 0 has room.
    assert evicted == [(state.model_id, 1, 2000)]
    assert ctx["n"] == 262144
    assert state.live_context is None


@pytest.mark.asyncio
async def test_idle_regrow_only_when_there_is_room(monkeypatch) -> None:
    mm, state, _, ctx, free_mb = _elastic_manager(monkeypatch)
    await mm._try_shrink_for_space(state.model_id, 1, 2500)
    mm._evict_last_resort_for_space = AsyncMock()
    free_mb.update({0: -1500, 1: -3000})  # another model moved into what it freed

    await mm._regrow_if_room(state.model_id)
    assert ctx["n"] == 65536  # no room: stays small, nothing evicted
    mm._evict_last_resort_for_space.assert_not_awaited()

    free_mb.update({0: 5000, 1: 5000})
    await mm._regrow_if_room(state.model_id)
    assert ctx["n"] == 262144
    assert state.live_context is None
    assert state.model_id not in mm._regrow_pending


@pytest.mark.asyncio
async def test_shrink_stops_early_when_the_projection_cannot_reach_it(monkeypatch) -> None:
    mm, state, backend, ctx, _ = _elastic_manager(monkeypatch)

    # One halving frees 2000 on GPU 1, so the 32k floor frees 3500 at most.
    assert await mm._try_shrink_for_space(state.model_id, 1, 21000) == 0

    sizes = [call.kwargs["n_ctx"] for call in backend.reconfigure_context.await_args_list]
    assert sizes == [131072, 262144]  # one step and back, not three
    assert ctx["n"] == 262144


@pytest.mark.asyncio
async def test_worker_without_live_context_is_not_retried(monkeypatch) -> None:
    mm, state, backend, _, _ = _elastic_manager(monkeypatch)
    backend.reconfigure_context.side_effect = LiveContextUnsupported("start it with --props")

    assert await mm._try_shrink_for_space(state.model_id, 1, 100) == 0
    assert await mm._try_shrink_for_space(state.model_id, 1, 100) == 0

    assert backend.reconfigure_context.await_count == 1


@pytest.mark.asyncio
async def test_pressure_load_shrinks_on_another_gpu_before_evicting(monkeypatch) -> None:
    """2026-11-10 live: qwen3-embedding:4b preferring the 3060 evicted qwen3.8
    there, though shrinking qwen3.8 on the 3090 had room to spare."""
    from ocabra.core.scheduler import InsufficientVRAMError

    mm, state, _, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 1000, 1: 3000})
    mm._gpu_manager.get_all_states = AsyncMock(
        return_value=[SimpleNamespace(index=0), SimpleNamespace(index=1)]
    )
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model.side_effect = [InsufficientVRAMError("full"), [1]]
    mm._gpu_scheduler = scheduler
    mm._get_pressure_eviction_candidates = MagicMock(return_value=[state.model_id])
    mm.unload = AsyncMock()

    gpus = await mm._assign_gpus_for_load(
        "llamacpp_mks/qwen3-embedding:4b",
        vram_needed=6000,
        preferred_gpu=0,
        enforce_vllm_headroom=False,
    )

    assert gpus == [1]
    mm.unload.assert_not_awaited()
    assert ctx["n"] == 65536  # 3060 could not give 5000; the 3090 gave 3000


@pytest.mark.asyncio
async def test_shrink_waits_until_the_gpu_shows_the_release(monkeypatch) -> None:
    """hw-monitor publishes free VRAM every ~2 s: the scheduler must not judge
    the GPU on the figure from before the shrink (seen live 2026-11-10)."""
    mm, state, _, _, _ = _elastic_manager(monkeypatch)
    stale = {"reads": 0}
    real = mm._gpu_manager.get_free_vram.side_effect

    async def lagging(idx):
        stale["reads"] += 1
        return 0 if stale["reads"] <= 3 else real(idx)

    mm._gpu_manager.get_free_vram = AsyncMock(side_effect=lagging)
    monkeypatch.setattr("ocabra.core.model_manager.asyncio.sleep", AsyncMock())

    assert await mm._try_shrink_for_space(state.model_id, 1, 1500) == 2000
    assert stale["reads"] >= 4  # kept polling until the release showed up


def _with_idle_whisper(monkeypatch, mm, free_mb, *, whisper_cost=10, qwen_cost=60):
    """Add an idle whisper (~2,250 MB, no live context) on GPU 0 next to qwen3.8."""
    whisper = ModelState(
        model_id="whisper/large-v3-turbo",
        display_name="w",
        backend_type="whisper",
        load_policy=LoadPolicy.PIN,
        status=ModelStatus.LOADED,
        current_gpu=[0],
        vram_used_mb=2250,
    )
    mm._states[whisper.model_id] = whisper

    async def unload(model_id, reason=None):
        if model_id == whisper.model_id:
            free_mb[0] += 2250
            whisper.status = ModelStatus.UNLOADED

    mm.unload = AsyncMock(side_effect=unload)
    mm._get_vram_free_snapshot = AsyncMock(return_value=None)
    mm._wait_for_vram_released = AsyncMock()
    costs = {whisper.model_id: whisper_cost}
    mm.get_expected_load_seconds = AsyncMock(side_effect=lambda m, **_: costs.get(m, qwen_cost))
    return whisper


@pytest.mark.asyncio
async def test_load_on_the_3060_evicts_idle_whisper_and_shrinks_qwen(monkeypatch) -> None:
    """Neither evicting whisper (2,250) nor shrinking qwen3.8 there (1,750 at
    the floor) frees 3,500 alone. Today's order would evict qwen3.8 (60 s to
    reload); evicting whisper (10 s) and shrinking qwen3.8 is cheaper."""
    from ocabra.core.scheduler import InsufficientVRAMError

    mm, state, _, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 0, 1: -100_000})
    whisper = _with_idle_whisper(monkeypatch, mm, free_mb)
    mm._gpu_manager.get_all_states = AsyncMock(
        return_value=[SimpleNamespace(index=0), SimpleNamespace(index=1)]
    )
    scheduler = AsyncMock()
    scheduler.find_gpu_for_model.side_effect = [InsufficientVRAMError("full"), [0]]
    mm._gpu_scheduler = scheduler
    mm._get_pressure_eviction_candidates = MagicMock(
        side_effect=lambda _req, gpu_indices=None: (
            [state.model_id, whisper.model_id] if gpu_indices == [0] else [state.model_id]
        )
    )

    gpus = await mm._assign_gpus_for_load(
        "llamacpp_mks/qwen3-embedding:4b",
        vram_needed=3500,
        preferred_gpu=0,
        enforce_vllm_headroom=False,
    )

    assert gpus == [0]
    mm.unload.assert_awaited_once_with(whisper.model_id, reason="vram_pressure")
    assert ctx["n"] == 65536  # qwen3.8 stays loaded, shrunk
    assert state.status == ModelStatus.LOADED


@pytest.mark.asyncio
async def test_cheap_eviction_is_skipped_when_it_costs_more_than_the_big_model(
    monkeypatch,
) -> None:
    mm, state, _, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 0, 1: 0})
    whisper = _with_idle_whisper(monkeypatch, mm, free_mb, whisper_cost=90, qwen_cost=60)
    mm._get_pressure_eviction_candidates = MagicMock(
        return_value=[state.model_id, whisper.model_id]
    )
    mm._shrink_capacity[(state.model_id, 0)] = 1750

    assert await mm._evict_cheap_then_shrink("x", 0, 3500) is False
    mm.unload.assert_not_awaited()
    assert ctx["n"] == 262144


@pytest.mark.asyncio
async def test_busy_models_are_never_evicted_by_the_cheap_plan(monkeypatch) -> None:
    mm, state, _, _, free_mb = _elastic_manager(monkeypatch, free={0: 0, 1: 0})
    whisper = _with_idle_whisper(monkeypatch, mm, free_mb)
    mm._in_flight[whisper.model_id] = 1
    mm._get_pressure_eviction_candidates = MagicMock(
        return_value=[state.model_id, whisper.model_id]
    )
    mm._shrink_capacity[(state.model_id, 0)] = 1750

    assert await mm._evict_cheap_then_shrink("x", 0, 3500) is False
    mm.unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_forced_gpu_load_also_evicts_cheap_before_big(monkeypatch) -> None:
    """Same case with the 3060 forced by a preset (required_gpu_indices), which
    takes the admission cascade instead of _assign_gpus_for_load: it must not
    evict qwen3.8 while whisper plus a shrink is enough."""
    mm, state, _, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 0, 1: 0})
    whisper = _with_idle_whisper(monkeypatch, mm, free_mb)
    mm._reconcile_ollama_state = AsyncMock()
    mm._evict_for_space = AsyncMock(return_value=0)
    mm._evict_last_resort_for_space = AsyncMock(return_value=0)
    mm._get_pressure_eviction_candidates = MagicMock(
        return_value=[state.model_id, whisper.model_id]
    )

    gpus = await mm._admit_gpu_load(
        model_id="llamacpp_mks/qwen3-embedding:4b",
        vram_needed=3500,
        preferred_gpu=None,
        gpu_managed=True,
        enforce_vllm_headroom=False,
        vllm_gpu_memory_utilization=None,
        required_gpu_indices=[0],
    )

    assert gpus == [0]
    mm.unload.assert_awaited_once_with(whisper.model_id, reason="vram_pressure")
    mm._evict_for_space.assert_not_awaited()
    assert ctx["n"] == 65536
    assert state.status == ModelStatus.LOADED


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
    mm.grow_context_back = AsyncMock()
    request_state = SimpleNamespace()
    with patch("ocabra.api.openai._deps.plan_profile_runtime", AsyncMock(return_value=_plan())):
        await resolve_profile_runtime_plan(MagicMock(), mm, {}, request_state=request_state)

    mm.grow_context_back.assert_awaited_once_with("llamacpp_mks/qwen3.8:27b::abc")
    assert request_state.runtime_effective_context == 262144


@pytest.mark.asyncio
async def test_request_that_fits_live_context_uses_it_without_regrowing() -> None:
    mm = MagicMock()
    mm.grow_context_back = AsyncMock()
    request_state = SimpleNamespace()
    with patch(
        "ocabra.api.openai._deps.plan_profile_runtime",
        AsyncMock(return_value=_plan(required_tokens=4000)),
    ):
        await resolve_profile_runtime_plan(MagicMock(), mm, {}, request_state=request_state)

    mm.grow_context_back.assert_not_awaited()
    assert request_state.runtime_effective_context == 65536


@pytest.mark.asyncio
async def test_failed_regrow_is_a_retryable_503() -> None:
    mm = MagicMock()
    mm.grow_context_back = AsyncMock(side_effect=RuntimeError("no VRAM"))
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


# -- vision on a full card (phase B, 2026-11-11) --------------------------------


def _with_vision(mm, state, backend, free_mb, tmp_path, *, projector_mb=1, measured=None):
    """Give the fake worker an on-demand projector that takes ``load_mb`` on GPU 0
    when activated, visible both in its process VRAM and in the free figure."""
    extra = {0: 0, 1: 0}
    gm = mm._gpu_manager
    base_vram = gm.get_process_vram_mb.side_effect
    base_free = gm.get_physical_free_vram.side_effect
    gm.get_process_vram_mb = MagicMock(
        side_effect=lambda idx: {4242: base_vram(idx)[4242] + extra[idx]}
    )

    async def free(idx):
        return base_free(idx) - extra[idx]

    gm.get_physical_free_vram = AsyncMock(side_effect=free)
    gm.get_free_vram = gm.get_physical_free_vram
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"\0" * (projector_mb * 1024 * 1024))
    active = {"on": False}

    async def activate(_model):
        active["on"] = True
        extra[0] += 900

    backend._contains_image = lambda body: "image" in str(body)
    backend.vision_is_active = lambda _m: active["on"]
    backend.vision_projector_path = lambda _m: str(projector)
    backend.activate_vision = AsyncMock(side_effect=activate)
    if measured:
        state.extra_config["_runtime"]["context_probe_vision_mb"] = measured
    return active, extra


IMAGE = {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "data:"}}]}]}


@pytest.mark.asyncio
async def test_vision_with_room_loads_without_touching_the_context(monkeypatch, tmp_path):
    mm, state, backend, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 4000, 1: 0})
    active, _ = _with_vision(mm, state, backend, free_mb, tmp_path)

    await mm.make_room_for_vision(state.model_id, IMAGE)

    assert active["on"] is True
    assert ctx["n"] == 262144


@pytest.mark.asyncio
async def test_vision_on_a_full_card_shrinks_first_with_its_own_request_in_flight(
    monkeypatch, tmp_path
):
    """gemma4:26b 2026-11-10: the first image on a full 3090 killed the worker
    (cudaMalloc). Now the context gives way first; the image request itself is
    already counted in flight and must not block the change."""
    mm, state, backend, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 0, 1: 0})
    active, extra = _with_vision(mm, state, backend, free_mb, tmp_path)
    mm._in_flight[state.model_id] = 1  # the image request

    await mm.make_room_for_vision(state.model_id, IMAGE)

    assert ctx["n"] < 262144
    assert active["on"] is True
    assert state.live_context["vision_mb"] == {"0": 900}


@pytest.mark.asyncio
async def test_vision_that_cannot_fit_is_a_clear_error_not_a_crash(monkeypatch, tmp_path):
    mm, state, backend, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 0, 1: 0})
    # The probe measured 5 GB for this projector; GPU 0 frees 1.75 GB at most.
    active, _ = _with_vision(mm, state, backend, free_mb, tmp_path, measured={"0": 5000})

    with pytest.raises(RuntimeError, match="vision projector needs"):
        await mm.make_room_for_vision(state.model_id, IMAGE)

    backend.activate_vision.assert_not_awaited()
    assert ctx["n"] == 262144  # put back


@pytest.mark.asyncio
async def test_requests_without_images_never_touch_vision(monkeypatch, tmp_path):
    mm, state, backend, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 0, 1: 0})
    _with_vision(mm, state, backend, free_mb, tmp_path)

    await mm.make_room_for_vision(state.model_id, {"messages": [{"content": "hola"}]})

    backend.activate_vision.assert_not_awaited()
    assert ctx["n"] == 262144


@pytest.mark.asyncio
async def test_growing_back_counts_the_projector_that_stays_loaded(monkeypatch, tmp_path):
    mm, state, backend, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 0, 1: 0})
    _with_vision(mm, state, backend, free_mb, tmp_path)
    mm._in_flight[state.model_id] = 1
    await mm.make_room_for_vision(state.model_id, IMAGE)
    mm._in_flight.pop(state.model_id)

    missing = await mm._missing_vram_to_grow(state)

    # GPU 0 gave up KV for the projector; growing back needs those MB again
    # although the worker's total footprint now includes the projector.
    assert missing.get(0, 0) >= 900 - 256


@pytest.mark.asyncio
async def test_vision_room_ignores_a_stale_free_figure_right_after_a_load(monkeypatch, tmp_path):
    """2026-11-11 live: the image arrived seconds after qwen3.8 filled the 3090;
    hw-monitor still showed the card empty, so no room was made and the worker
    died with cudaMalloc when the projector loaded."""
    mm, state, backend, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 0, 1: 0})
    active, _ = _with_vision(mm, state, backend, free_mb, tmp_path)
    # hw-monitor: stale, from before the load.
    mm._gpu_manager.get_physical_free_vram = AsyncMock(return_value=20000)
    mm._in_flight[state.model_id] = 1

    await mm.make_room_for_vision(state.model_id, IMAGE)

    assert ctx["n"] < 262144  # it shrank anyway: the direct reading is full
    assert active["on"] is True


@pytest.mark.asyncio
async def test_with_a_probe_vision_takes_just_the_context_it_needs(monkeypatch, tmp_path):
    """2026-11-11 live: halving to fit a 1.1 GB projector left qwen3.8 at 57k
    although 90k fitted. With the probe's KV cost per GPU it cuts just enough."""
    mm, state, backend, ctx, free_mb = _elastic_manager(
        monkeypatch, free={0: 0, 1: 0}, kv_mb_per_64k=(2000, 1000)
    )
    active, _ = _with_vision(mm, state, backend, free_mb, tmp_path, measured={"0": 1500})
    state.extra_config["_runtime"]["context_probe_kv_mb_per_1k"] = {"0": 2000 / 64, "1": 1000 / 64}
    mm._in_flight[state.model_id] = 1

    await mm.make_room_for_vision(state.model_id, IMAGE)

    assert active["on"] is True
    assert ctx["n"] > 131072  # halving would have stopped at 131072
    assert ctx["n"] >= 262144 - (1500 / (2000 / 64) * 1024) - 2 * 4096
