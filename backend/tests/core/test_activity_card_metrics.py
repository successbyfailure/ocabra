"""Metrics the dashboard's active-models card consumes (2026-11-06).

Everything here was already computed somewhere and simply not surfaced on
``/ocabra/models/activity``: the fair-share gate knew the KV pool occupancy
and the queue depth, and the per-request phases existed but could only be read
one line at a time. Before this, the card could not tell a busy worker from
one about to queue, nor a model generating from one stuck in prompt eval.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocabra.api.internal import models as models_api
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState
from ocabra.core.worker_pool import WorkerPool

GATE = {
    "slots": 4,
    "in_flight": 2,
    "queued": 3,
    "slot_saturated": False,
    "admission_limit": 3,
    "compute_saturated": False,
    "kv_capacity_tokens": 245760,
    "kv_reserved_tokens": 27525,
    "kv_free_tokens": 218235,
}


def _mm(model_id: str, *, backend: str = "ollama", extra: dict | None = None) -> ModelManager:
    mm = ModelManager(WorkerPool())
    mm._persisted_model_ids.add(model_id)
    mm._states[model_id] = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type=backend,
        load_policy=LoadPolicy.ON_DEMAND,
        extra_config=extra or {},
    )
    return mm


def _request(mm: ModelManager, *, gate: dict | None = None) -> SimpleNamespace:
    state = SimpleNamespace(model_manager=mm)
    if gate is not None:
        state.worker_pool = SimpleNamespace(peek_queue_state=lambda _key: gate)
    return SimpleNamespace(app=SimpleNamespace(state=state))


@pytest.mark.asyncio
async def test_aggregates_phases_and_tokens_per_second() -> None:
    mm = _mm("ollama/m")
    generating = mm.try_begin_request("ollama/m", max_inflight=32, input_tokens_estimate=10)
    mm.update_request_progress(generating, output_tokens_estimate=120)
    mm.try_begin_request("ollama/m", max_inflight=32, input_tokens_estimate=5000)

    entry = (await models_api.get_models_activity(_request(mm), _user=None))["activity"]["ollama/m"]

    assert entry["phases"] == {"generating": 1, "prompt_eval": 1}
    # Only the generating one contributes; the count is what stops the card
    # from reading "0 t/s" on a model saturated with prompt-eval work.
    assert entry["generatingCount"] == 1
    assert entry["tokensPerSecondTotal"] is not None


@pytest.mark.asyncio
async def test_tokens_per_second_is_null_when_nothing_is_generating() -> None:
    """Null rather than 0: a model busy in prompt eval is at full load, and
    reporting zero would read as idle."""
    mm = _mm("ollama/m")
    mm.try_begin_request("ollama/m", max_inflight=32, input_tokens_estimate=5000)

    entry = (await models_api.get_models_activity(_request(mm), _user=None))["activity"]["ollama/m"]

    assert entry["phases"] == {"prompt_eval": 1}
    assert entry["generatingCount"] == 0
    assert entry["tokensPerSecondTotal"] is None


@pytest.mark.asyncio
async def test_surfaces_the_gate_queue_and_kv_pool() -> None:
    mm = _mm("llamacpp_mks/m", backend="llamacpp_mks")
    mm.try_begin_request("llamacpp_mks/m", max_inflight=32)

    result = await models_api.get_models_activity(_request(mm, gate=GATE), _user=None)
    adm = result["activity"]["llamacpp_mks/m"]["admission"]

    assert adm["queuedRequests"] == 3
    assert adm["kvCapacityTokens"] == 245760
    assert adm["kvReservedTokens"] == 27525
    assert adm["admissionLimit"] == 3
    assert adm["slotSaturated"] is False
    assert adm["computeSaturated"] is False


@pytest.mark.asyncio
async def test_gate_slots_fill_in_when_runtime_parallelism_is_missing() -> None:
    """The gate's slot count is a real concurrency measure, so it is a valid
    fallback — unlike ``max_inflight_per_model``, which is an admission
    ceiling and once made whisper models read as "1/32"."""
    mm = _mm("llamacpp_mks/m", backend="llamacpp_mks")
    mm.try_begin_request("llamacpp_mks/m", max_inflight=32)

    result = await models_api.get_models_activity(_request(mm, gate=GATE), _user=None)

    assert result["activity"]["llamacpp_mks/m"]["slots"] == 4


@pytest.mark.asyncio
async def test_no_gate_means_no_fabricated_slots_or_admission() -> None:
    """Backends with no slot pool never create a gate. They must keep
    reporting ``slots: None`` so clients render an honest "N en curso"."""
    mm = _mm("whisper/large", backend="whisper")
    mm.try_begin_request("whisper/large", max_inflight=32)

    entry = (await models_api.get_models_activity(_request(mm), _user=None))["activity"][
        "whisper/large"
    ]

    assert entry["slots"] is None
    assert "admission" not in entry


@pytest.mark.asyncio
async def test_exposes_the_idle_timeout_for_the_unload_countdown() -> None:
    """Shipped in the payload like stuckThresholdSeconds, so the card does not
    need server config threaded through GpuCard/Dashboard for one number."""
    mm = _mm("ollama/m")
    result = await models_api.get_models_activity(_request(mm), _user=None)
    assert isinstance(result["idleTimeoutSeconds"], int)
    assert result["idleTimeoutSeconds"] > 0
