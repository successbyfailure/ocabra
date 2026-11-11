"""Two-GPU presets that keep a reserve free (whisper on the 3060), phase D."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ocabra.backends.llamacpp_mks_backend import LlamaCppMksBackend
from ocabra.core.runtime_coordinator import _apply_multi_gpu_reserve, multi_gpu_launch_context

# qwen3.8 extended-like: at 262k the 3060 holds 11,000 of 11,500 usable.
PROBE = {
    "used_mb": {"0": 11000, "1": 22600},
    "limit_mb": {"0": 11500, "1": 23600},
    "kv_mb_per_1k": {"0": 14.0, "1": 33.0},
}


def test_launch_context_leaves_the_reserve_free_on_the_card_that_needs_it():
    launch, at_launch = multi_gpu_launch_context(262144, PROBE, {"0": 2700})

    # 2,700 wanted, 500 left at the maximum: give back 2,200 MB of KV on GPU 0.
    assert launch % 4096 == 0
    assert 11500 - at_launch["0"] >= 2700
    assert 262144 - launch <= 2200 / 14.0 * 1024 + 4096
    assert at_launch["1"] < 22600  # the other card holds less too


def test_no_reserve_needed_or_none_possible_means_no_change():
    assert multi_gpu_launch_context(262144, PROBE, {"0": 400}) is None  # already free
    assert multi_gpu_launch_context(262144, PROBE, {}) is None
    # Asking for more than the whole KV on that card can give: not worth starting at <32k.
    assert multi_gpu_launch_context(262144, PROBE, {"0": 9000}) is None


def _overrides():
    return {
        "llama_cpp": {"ctx_size": 262144, "parallel_slots": 4},
        "_runtime": {"gpu_indices": [0, 1], "capacity_validated": True, "capacity_vram_mb": 33600},
    }


def test_coordinator_starts_probed_two_gpu_presets_below_their_maximum():
    overrides = _overrides()
    _apply_multi_gpu_reserve(overrides, {"context_probe": {"extended": PROBE}}, "extended", [0, 1])

    launch = overrides["llama_cpp"]["launch_ctx_size"]
    assert overrides["llama_cpp"]["ctx_size"] == 262144  # the key stays the maximum's
    assert overrides["_runtime"]["launch_context"] == launch < 262144
    # Admission reserves what the start takes, not the maximum.
    per_gpu = overrides["_runtime"]["capacity_vram_by_gpu_mb"]
    assert overrides["_runtime"]["capacity_vram_mb"] == sum(per_gpu.values()) < 33600


def test_a_per_preset_reserve_overrides_the_global_one_and_one_gpu_is_untouched():
    overrides = _overrides()
    policy = {"context_probe": {"extended": PROBE}, "reserve_mb_by_preset": {"extended": {}}}
    _apply_multi_gpu_reserve(overrides, policy, "extended", [0, 1])
    assert "launch_ctx_size" not in overrides["llama_cpp"]

    single = _overrides()
    _apply_multi_gpu_reserve(single, {"context_probe": {"normal": PROBE}}, "normal", [1])
    assert "launch_ctx_size" not in single["llama_cpp"]


def test_backend_starts_at_the_launch_context_and_remembers_the_maximum():
    backend = LlamaCppMksBackend()
    backend._model_configs["m"] = {
        "ctx_size": 98304,
        "nominal_ctx_size": 262144,
        "parallel_slots": 4,
        "ubatch_size": 128,
        "batch_size": 512,
    }
    assert backend.context_options("m")["n_ctx"] == 98304
    assert backend.nominal_context_options("m")["n_ctx"] == 262144
    backend._model_configs["n"] = {"ctx_size": 98304, "parallel_slots": 4}
    assert backend.nominal_context_options("n") is None


# -- model manager behaviour ------------------------------------------------------


def _reserved_worker(monkeypatch):
    from tests.test_live_context import _elastic_manager

    mm, state, backend, ctx, free_mb = _elastic_manager(monkeypatch, free={0: 3000, 1: 3000})
    monkeypatch.setattr(
        "ocabra.core.model_manager.settings.llama_cpp_return_to_initial_seconds", 120
    )
    ctx["n"] = 131072
    backend._model_configs = {}
    backend.context_options.return_value = {
        "n_ctx": 131072,
        "n_parallel": 4,
        "n_ubatch": 128,
        "n_batch": 512,
    }
    backend.nominal_context_options = MagicMock(
        return_value={"n_ctx": 262144, "n_parallel": 4, "n_ubatch": 128, "n_batch": 512}
    )
    state.extra_config["_runtime"]["context_probe_used_mb"] = {"0": 8000, "1": 20000}
    state.live_context = mm._initial_live_context(state, backend)
    return mm, state, backend, ctx


@pytest.mark.asyncio
async def test_a_reserved_worker_starts_live_below_its_nominal(monkeypatch):
    mm, state, backend, ctx = _reserved_worker(monkeypatch)
    assert state.live_context["n_ctx"] == 131072
    assert state.live_context["initial_ctx"] == 131072
    assert state.live_context["nominal"]["n_ctx"] == 262144


@pytest.mark.asyncio
async def test_idle_regrow_never_takes_it_past_its_starting_context(monkeypatch):
    mm, state, backend, ctx = _reserved_worker(monkeypatch)
    await mm._regrow_if_room(state.model_id)
    assert ctx["n"] == 131072  # room on the cards, but the reserve is the point


@pytest.mark.asyncio
async def test_a_large_request_grows_it_and_idle_brings_it_back(monkeypatch):
    mm, state, backend, ctx = _reserved_worker(monkeypatch)
    mm._evict_last_resort_for_space = AsyncMock(return_value=0)
    await mm.grow_context_back(state.model_id)
    assert ctx["n"] == 262144
    assert state.live_context["initial_ctx"] == 131072  # kept at the maximum too

    state.last_request_at = datetime.now(UTC) - timedelta(seconds=60)
    await mm.check_idle_evictions()
    await asyncio.sleep(0)
    assert ctx["n"] == 262144  # 60 s idle: not yet

    state.last_request_at = datetime.now(UTC) - timedelta(seconds=121)
    await mm.check_idle_evictions()
    for _ in range(5):
        await asyncio.sleep(0)
    assert ctx["n"] == 131072


# -- whisper asks for room before it grows ----------------------------------------


def _whisper_next_to_qwen(monkeypatch, free_gpu0):
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus
    from tests.test_live_context import _elastic_manager

    mm, state, backend, ctx, free_mb = _elastic_manager(
        monkeypatch, free={0: free_gpu0, 1: 0}, kv_mb_per_64k=(2000, 1000)
    )
    whisper = ModelState(
        model_id="whisper/whisper-large-v3-turbo",
        display_name="w",
        backend_type="whisper",
        load_policy=LoadPolicy.PIN,
        status=ModelStatus.LOADED,
        current_gpu=[0],
    )
    mm._states[whisper.model_id] = whisper
    mm._get_pressure_eviction_candidates = MagicMock(return_value=[state.model_id])
    return mm, state, whisper, ctx


@pytest.mark.asyncio
async def test_a_diarization_on_a_full_3060_shrinks_the_big_model_first(monkeypatch):
    mm, qwen, whisper, ctx = _whisper_next_to_qwen(monkeypatch, free_gpu0=0)
    mm._in_flight[whisper.model_id] = 1

    await mm.make_room_for_whisper(whisper.model_id, diarize=True)

    assert ctx["n"] < 262144  # qwen3.8 gave ~3.4 GB of KV on GPU 0


@pytest.mark.asyncio
async def test_two_plain_transcriptions_fit_in_whisper_s_own_estimate(monkeypatch):
    mm, qwen, whisper, ctx = _whisper_next_to_qwen(monkeypatch, free_gpu0=0)
    mm._in_flight[whisper.model_id] = 2

    await mm.make_room_for_whisper(whisper.model_id, diarize=False)

    assert ctx["n"] == 262144


@pytest.mark.asyncio
async def test_a_third_concurrent_one_asks_only_for_its_share(monkeypatch):
    mm, qwen, whisper, ctx = _whisper_next_to_qwen(monkeypatch, free_gpu0=500)
    mm._in_flight[whisper.model_id] = 3

    await mm.make_room_for_whisper(whisper.model_id, diarize=False)

    assert ctx["n"] == 262144  # 350 MB needed, 500 free: nothing to do
