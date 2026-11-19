"""Evicting a model must not kill a request admitted while the eviction runs,
and an admission must not trust a stale free-VRAM reading (2026-11-16)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.core.model_manager import ModelManager, ModelState, ModelStatus
from ocabra.core.worker_pool import WorkerPool
from tests.test_vram_eviction import _add_loaded_model, _make_mm


def test_claim_is_atomic_with_admission():
    mm = _make_mm()
    _add_loaded_model(mm, "m")
    rid = mm.try_begin_request("m", 4)
    assert rid is not None
    assert mm.claim_for_eviction("m") is False  # a request is in flight
    mm._in_flight.pop("m", None)  # the request finished
    assert mm.claim_for_eviction("m") is True
    assert mm.is_eviction_claimed("m")
    mm.release_eviction_claim("m")
    assert not mm.is_eviction_claimed("m")


@pytest.mark.asyncio
async def test_request_arriving_mid_eviction_sees_the_claim():
    """The real race: idle when listed, a request lands while the shrink
    attempt is awaited, then the worker is unloaded."""
    mm = _make_mm()
    _add_loaded_model(mm, "victim", vram_mb=8000)
    seen_during: list[bool] = []

    async def slow_shrink(model_id, gpu_index, needed, **_kwargs):
        mm.try_begin_request("victim", 4)  # a request arrives now
        seen_during.append(mm.is_eviction_claimed("victim"))
        return 0

    mm._try_shrink_for_space = slow_shrink
    mm.unload = AsyncMock()
    await mm._evict_for_space(0, 4000)
    # The request saw the claim (so ensure_loaded makes it wait for the
    # unload) and the claim is released afterwards.
    assert seen_during == [True]
    mm.unload.assert_awaited_once()
    assert not mm.is_eviction_claimed("victim")


@pytest.mark.asyncio
async def test_eviction_skips_a_model_that_became_busy():
    mm = _make_mm()
    _add_loaded_model(mm, "victim")
    mm._get_eviction_candidates = lambda gpu: ["victim"]  # listed while idle…
    mm._in_flight["victim"] = 1  # …busy by the time it is evicted
    mm.unload = AsyncMock()
    assert await mm._evict_for_space(0, 4000) == 0
    mm.unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_released_even_if_unload_fails():
    mm = _make_mm()
    _add_loaded_model(mm, "victim")
    mm._try_shrink_for_space = AsyncMock(return_value=0)
    mm.unload = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await mm._evict_for_space(0, 4000)
    assert not mm.is_eviction_claimed("victim")


@pytest.mark.asyncio
async def test_ensure_loaded_waits_for_the_claimed_model_then_reloads():
    from ocabra.api.openai._deps import _do_ensure_loaded

    loaded = ModelState(
        model_id="m", display_name="m", backend_type="mock", status=ModelStatus.LOADED
    )
    gone = ModelState(
        model_id="m", display_name="m", backend_type="mock", status=ModelStatus.UNLOADED
    )
    state = {"current": loaded}
    claimed = {"on": True}

    class MM:
        def is_eviction_claimed(self, mid):
            return claimed["on"]

        async def get_state(self, mid):
            return state["current"]

        async def load(self, mid):
            state["current"] = loaded

        def touch_last_request_at(self, *a):
            return None

    async def evict_later():
        await asyncio.sleep(0.25)
        state["current"] = gone  # the eviction unloaded it…
        claimed["on"] = False  # …and released the claim

    task = asyncio.create_task(evict_later())
    result = await _do_ensure_loaded(MM(), "m")
    await task
    # It did not take the dying worker: it waited and loaded it again.
    assert result.status == ModelStatus.LOADED and state["current"] is loaded


@pytest.mark.asyncio
@pytest.mark.parametrize("live_free, admitted", [(1_767, False), (9_000, True)])
async def test_validated_admission_trusts_the_live_reading_when_the_poll_is_stale(
    live_free, admitted
):
    """Polled free said 9,076 MB; the embedding model had just taken 8 GB.
    The live reading (1,767) must win, so nemotron's 6 GB shard is not
    admitted onto a full card. With room really there, it still is."""
    from ocabra.core.scheduler import InsufficientVRAMError

    gpu_manager = AsyncMock()
    gpu_manager.get_free_vram = AsyncMock(return_value=9_000)
    gpu_manager.get_state = AsyncMock(return_value=SimpleNamespace(free_vram_mb=9_076))
    gpu_manager.live_free_vram_mb = lambda gpu: live_free
    mm = ModelManager(WorkerPool(), gpu_manager=gpu_manager)
    model_id = "llama_cpp/nemotron::x"
    mm._states[model_id] = ModelState(
        model_id=model_id,
        display_name="nemotron",
        backend_type="llama_cpp",
        extra_config={"_runtime": {"capacity_validated": True}},
    )
    kwargs = dict(
        model_id=model_id,
        vram_needed=5_980,
        preferred_gpu=0,
        gpu_managed=True,
        enforce_vllm_headroom=False,
        vllm_gpu_memory_utilization=None,
        required_gpu_indices=[0],
        allow_pressure_eviction=False,
    )
    if admitted:
        assert await mm._admit_gpu_load(**kwargs) == [0]
    else:
        with pytest.raises(InsufficientVRAMError):
            await mm._admit_gpu_load(**kwargs)
