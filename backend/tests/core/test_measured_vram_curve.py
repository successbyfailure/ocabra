"""La capacidad de un preset usa la VRAM medida cuando la hay (2026-11-13).

El planificador apilaba constantes calibradas con otro modelo u otra
configuracion (el borrador MTP medido en qwen3.8, el gemma4 de octubre con 1
slot y el proyector residente) y daba a gemma4:26b 129.564 tokens en la 3090
mientras corria a 262.144 con 2 GB libres: la guarda rechazaba un peldano de
156.672. Medidas reales de gemma4:26b (4 slots, q8, kv_unified, ubatch 128,
MTP, vision bajo demanda): 32.768 -> 18.108 MiB, 131.072 -> ~19.960,
139.584 -> 20.112, 262.144 -> ~22.410.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.core import observed_vram
from ocabra.core.observed_vram import VramCurve, curve_key
from ocabra.core.runtime_preset_capacity import (
    _preset_llama_section,
    resolve_profile_runtime_capacities,
)
from ocabra.core.vram_capacity import ArchResolution
from ocabra.core.vram_planner import ModelArch

GEMMA_POINTS = {32768: 18108, 131072: 19960, 139584: 20112, 262144: 22410}
GEMMA_LLAMA = {
    "ctx_size": 262144,
    "batch_size": 512,
    "flash_attn": True,
    "gpu_layers": 99,
    "kv_unified": True,
    "split_mode": "none",
    "speculative": {"spec_type": "draft-mtp"},
    "ubatch_size": 128,
    "vision_mode": "on_demand",
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "tensor_split": None,
    "disabled_gpus": [0],
    "parallel_slots": 4,
    "vram_estimate_mb": 23600,
}


def test_the_curve_follows_the_measurements():
    curve = VramCurve(GEMMA_POINTS)
    margin = observed_vram.OBSERVED_VRAM_MARGIN_MB
    for ctx, mb in GEMMA_POINTS.items():
        assert abs(curve.predict_mb(ctx) - margin - mb) <= 60
    # La 3090 que planifica la admision: 24576 - no atribuida - buffer.
    assert curve.max_context_within(23391) == 262144


def test_the_curve_never_predicts_below_a_smaller_context_reading():
    curve = VramCurve({32768: 18000, 65536: 25000, 262144: 22000})
    assert curve.predict_mb(100000) >= 25000


def test_a_small_capacity_gets_a_context_inside_the_measured_range():
    curve = VramCurve(GEMMA_POINTS)
    ctx = curve.max_context_within(21000)
    assert 131072 < ctx < 262144
    assert curve.predict_mb(ctx) <= 21000


@pytest.mark.asyncio
async def test_lookup_needs_two_points_spread_out(monkeypatch):
    stored = {"points": {"131072": 19960}}
    monkeypatch.setattr(observed_vram, "get_key", AsyncMock(side_effect=lambda k: stored))
    assert await observed_vram.lookup_curve("m", key="k") is None
    stored["points"]["135000"] = 20050  # menos de 16k de separacion
    assert await observed_vram.lookup_curve("m", key="k") is None
    stored["points"]["262144"] = 22410
    assert await observed_vram.lookup_curve("m", key="k") is not None


def _gemma_profile(tokens: int):
    preset = SimpleNamespace(
        preset_key="normal",
        context_mode="fixed",
        context_tokens=tokens,
        parallelism=4,
        kv_cache_dtype="q8",
        placement_policy="forced",
        preferred_gpus=[1],
        priority=10,
        placement=None,
    )
    profile = SimpleNamespace(
        base_model_id="llamacpp_mks/gemma4:26b",
        load_overrides=None,
        runtime_policy={"max_context_tokens": 262144},
        runtime_presets=[preset],
    )
    state = SimpleNamespace(
        model_id="llamacpp_mks/gemma4:26b",
        backend_type="llamacpp_mks",
        preferred_gpu=1,
        extra_config={"llama_cpp": copy.deepcopy(GEMMA_LLAMA)},
    )
    return profile, preset, state


def test_the_base_worker_and_its_preset_share_the_curve_key():
    """Lo que mide el worker base tiene que servir al preset equivalente."""
    profile, preset, state = _gemma_profile(156672)
    recorded = curve_key(GEMMA_LLAMA, [1])
    planned = curve_key(
        _preset_llama_section(
            state, profile, preset, effective_parallelism=4, backend="llamacpp_mks", placement={}
        ),
        [1],
    )
    assert recorded == planned
    # Otro ubatch es otra configuracion: sus medidas no valen.
    assert curve_key({**GEMMA_LLAMA, "ubatch_size": 512}, [1]) != recorded


async def _rows(monkeypatch, tokens, curve):
    # Arquitectura con la sobreestimacion que tenia el planificador.
    arch = ModelArch(30, 8, 512, 512, 2816, 262144, 2.0, 23636, 0.0, 1266.0)
    monkeypatch.setattr(
        "ocabra.core.runtime_preset_capacity.resolve_arch_and_weights",
        AsyncMock(return_value=ArchResolution(arch, 16283, "")),
    )
    monkeypatch.setattr(observed_vram, "lookup_curve", AsyncMock(return_value=curve))
    profile, _preset, state = _gemma_profile(tokens)
    gpus = [
        SimpleNamespace(index=0, total_vram_mb=12288, free_vram_mb=12288, processes=[]),
        SimpleNamespace(index=1, total_vram_mb=24576, free_vram_mb=24159, processes=[]),
    ]
    return await resolve_profile_runtime_capacities(profile, state, gpus)


@pytest.mark.asyncio
async def test_without_measurements_the_formula_still_decides(monkeypatch):
    rows = await _rows(monkeypatch, 156672, None)
    assert rows[0]["fits"] is False


@pytest.mark.asyncio
async def test_with_measurements_the_156k_step_fits(monkeypatch):
    rows = await _rows(monkeypatch, 156672, VramCurve(GEMMA_POINTS))
    row = rows[0]
    assert row["fits"] is True
    assert row["effective_context"] == 156672
    assert row["vram_source"] == "measured"
    assert 20000 < row["estimated_vram_mb"] < 21500


@pytest.mark.asyncio
async def test_measurements_never_lower_what_the_formula_allows(monkeypatch):
    tiny = VramCurve({16384: 30000, 65536: 31000})  # medida absurda: no cabe nada
    rows = await _rows(monkeypatch, 65536, tiny)
    assert rows[0]["fits"] is True  # la formula ya lo admitia


@pytest.mark.asyncio
async def test_a_live_context_change_is_recorded_under_the_planner_key(monkeypatch):
    from ocabra.core.model_manager import ModelManager, ModelState

    recorded = []

    async def fake_record(model_id, *, key, context_tokens, total_mb):
        recorded.append((model_id, key, context_tokens, total_mb))

    monkeypatch.setattr(observed_vram, "record_curve_point", fake_record)
    mm = ModelManager(worker_pool=None)
    state = ModelState(
        model_id="llamacpp_mks/gemma4:26b",
        display_name="g",
        backend_type="llamacpp_mks",
        extra_config={"llama_cpp": copy.deepcopy(GEMMA_LLAMA)},
    )
    mm._states[state.model_id] = state
    live = {"n_ctx": 131072, "n_parallel": 4, "n_ubatch": 128, "n_batch": 512}

    await mm._record_vram_curve_point(state.model_id, state, {1: 19960}, [1], live=live)

    assert recorded == [("llamacpp_mks/gemma4:26b", curve_key(GEMMA_LLAMA, [1]), 131072, 19960)]


@pytest.mark.asyncio
async def test_no_point_while_an_on_demand_projector_is_loaded(monkeypatch):
    """Con el proyector cargado en caliente la lectura lleva ~1,1 GB mas: en la
    misma curva aplanaria la pendiente y sobrestimaria lo que cabe."""
    from ocabra.core.model_manager import ModelManager, ModelState

    recorded = []

    async def fake_record(model_id, **kw):
        recorded.append(model_id)

    monkeypatch.setattr(observed_vram, "record_curve_point", fake_record)
    mm = ModelManager(worker_pool=None)
    backend = SimpleNamespace(vision_is_active=lambda model_id: True)
    monkeypatch.setattr(mm, "_registered_backend", lambda backend_type: backend)
    state = ModelState(
        model_id="llamacpp_mks/gemma4:26b",
        display_name="g",
        backend_type="llamacpp_mks",
        backend_model_id="gemma4:26b",
        extra_config={"llama_cpp": copy.deepcopy(GEMMA_LLAMA)},
    )
    mm._states[state.model_id] = state

    await mm._record_vram_curve_point(
        state.model_id, state, {1: 21100}, [1], live={"n_ctx": 250000, "n_parallel": 4}
    )

    assert recorded == []
