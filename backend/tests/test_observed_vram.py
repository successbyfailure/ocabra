"""Learning llama.cpp VRAM from real loads (ocabra.core.observed_vram)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.backends._mock import MockBackend
from ocabra.core import observed_vram
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState
from ocabra.core.worker_pool import WorkerPool

MARGIN = observed_vram.OBSERVED_VRAM_MARGIN_MB


def test_fingerprint_changes_with_ubatch_and_ignores_runtime_bookkeeping() -> None:
    base = {
        "llama_cpp": {"ctx_size": 262144, "ubatch_size": 128},
        "_runtime": {"gpu_indices": [0, 1]},
    }
    same = {
        "llama_cpp": {"ubatch_size": 128, "ctx_size": 262144},
        "_runtime": {"gpu_indices": [0, 1], "tracking_worker_key": "x"},
    }
    other_ubatch = {
        "llama_cpp": {"ctx_size": 262144, "ubatch_size": 512},
        "_runtime": {"gpu_indices": [0, 1]},
    }
    other_gpus = {
        "llama_cpp": {"ctx_size": 262144, "ubatch_size": 128},
        "_runtime": {"gpu_indices": [1]},
    }

    fp = observed_vram.footprint_fingerprint(base)
    assert fp == observed_vram.footprint_fingerprint(same)
    assert fp != observed_vram.footprint_fingerprint(other_ubatch)
    assert fp != observed_vram.footprint_fingerprint(other_gpus)


def test_raise_to_observed_lifts_an_estimate_that_falls_short() -> None:
    """flint:40b 262k on two GPUs: estimated 25,787, measured 9,972 + 18,926."""
    total, per_gpu = observed_vram.raise_to_observed(
        25_787, {0: 8_596, 1: 17_191}, {0: 9_972, 1: 18_926}
    )

    assert total == 9_972 + 18_926 + MARGIN
    assert per_gpu == {0: 9_972 + MARGIN // 2, 1: 18_926 + MARGIN // 2}


def test_raise_to_observed_never_lowers() -> None:
    total, per_gpu = observed_vram.raise_to_observed(23_600, {1: 23_600}, {1: 21_000})

    assert total == 23_600
    assert per_gpu == {1: 23_600}


def test_raise_to_observed_without_split_only_touches_total() -> None:
    total, per_gpu = observed_vram.raise_to_observed(20_000, None, {1: 22_856})

    assert total == 22_856 + MARGIN
    assert per_gpu is None


@pytest.mark.asyncio
async def test_lookup_ignores_a_reading_for_another_configuration() -> None:
    stored = {"fingerprint": "old", "per_gpu_mb": {"1": 22_856}, "total_mb": 22_856}
    with patch("ocabra.core.observed_vram.get_key", new=AsyncMock(return_value=stored)):
        assert await observed_vram.lookup("m", fingerprint="new") is None
        assert await observed_vram.lookup("m", fingerprint="old") == {1: 22_856}


@pytest.mark.asyncio
async def test_record_skips_incomplete_readings() -> None:
    with patch("ocabra.core.observed_vram.set_key", new=AsyncMock()) as set_key:
        await observed_vram.record("m", fingerprint="f", per_gpu_mb={0: 0, 1: 18_000})
        await observed_vram.record("m", fingerprint="f", per_gpu_mb={})
    set_key.assert_not_awaited()


def _llama_manager(*, pid_usage: dict[int, dict[int, int]]):
    wp = WorkerPool()
    wp.register_backend("llamacpp_mks", MockBackend(vram_mb=4096))
    gpu_manager = AsyncMock()
    gpu_manager.get_process_vram_mb = MagicMock(side_effect=lambda idx: pid_usage.get(idx, {}))
    mm = ModelManager(wp, gpu_manager=gpu_manager)
    model_id = "llamacpp_mks/flint:40b::abc"
    mm._states[model_id] = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="llamacpp_mks",
        load_policy=LoadPolicy.ON_DEMAND,
        extra_config={"llama_cpp": {"gpu_layers": 99, "ubatch_size": 512}},
    )
    mm._load_locks[model_id] = asyncio.Lock()
    return mm, model_id


async def _load(mm, model_id, *, observed, admitted_gpus):
    admit = AsyncMock(return_value=admitted_gpus)
    with (
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch("ocabra.core.observed_vram.lookup", new=AsyncMock(return_value=observed)),
        patch("ocabra.core.observed_vram.record", new=AsyncMock()) as record,
        patch.object(mm, "_admit_gpu_load", new=admit),
    ):
        await mm.load(model_id)
    return admit.await_args.kwargs, record


@pytest.mark.asyncio
async def test_load_raises_estimate_to_what_was_observed() -> None:
    mm, model_id = _llama_manager(pid_usage={1: {12345: 22_880}})

    kwargs, record = await _load(mm, model_id, observed={1: 22_880}, admitted_gpus=[1])

    assert kwargs["vram_needed"] == 22_880 + MARGIN


@pytest.mark.asyncio
async def test_load_records_the_measured_footprint() -> None:
    """Also the path with no calibrated preset: it used to need the flag."""
    mm, model_id = _llama_manager(pid_usage={0: {12345: 9_972}, 1: {12345: 18_926}})

    _, record = await _load(mm, model_id, observed=None, admitted_gpus=[0, 1])

    record.assert_awaited_once()
    assert record.await_args.kwargs["per_gpu_mb"] == {0: 9_972, 1: 18_926}
    assert record.await_args.args == (model_id,)


@pytest.mark.asyncio
async def test_load_does_not_record_a_partial_reading() -> None:
    mm, model_id = _llama_manager(pid_usage={1: {12345: 18_926}})

    _, record = await _load(mm, model_id, observed=None, admitted_gpus=[0, 1])

    record.assert_not_awaited()


@pytest.mark.asyncio
async def test_manually_validated_preset_keeps_its_figure() -> None:
    """qwen3.8 "normal" is validated at 24,000 of a 24,159 card: margin on top
    of a reading would make it unloadable."""
    mm, model_id = _llama_manager(pid_usage={1: {12345: 23_900}})
    mm._states[model_id].extra_config["_runtime"] = {
        "capacity_validated": True,
        "capacity_vram_mb": 24_000,
    }

    kwargs, _ = await _load(mm, model_id, observed={1: 23_900}, admitted_gpus=[1])

    assert kwargs["vram_needed"] == 24_000
