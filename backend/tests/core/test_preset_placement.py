"""Preset placement (GPU/CPU per runtime preset), split policies and smoke test."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from ocabra.core.vram_capacity import ArchResolution
from ocabra.core.vram_planner import ModelArch
from ocabra.schemas.profiles import (
    PresetPlacement,
    RuntimePresetCreate,
    RuntimePresetOut,
    RuntimePresetResolutionOut,
    RuntimePresetUpdate,
    normalize_placement,
)

# ── Schema ──────────────────────────────────────────────────────────────────


def test_placement_accepts_every_field_and_forbids_extras():
    placement = PresetPlacement(
        split_policy="when_needed",
        split_mode="row",
        tensor_split=[1, 3],
        gpu_layers=40,
        n_cpu_moe=8,
    )
    assert placement.tensor_split == [1.0, 3.0]
    with pytest.raises(ValidationError):
        PresetPlacement(main_gpu=1)
    with pytest.raises(ValidationError):
        PresetPlacement(split_policy="sometimes")
    with pytest.raises(ValidationError):
        PresetPlacement(split_mode="none")


@pytest.mark.parametrize(
    "bad",
    [{"gpu_layers": -1}, {"n_cpu_moe": -2}, {"tensor_split": [1, -1]}, {"tensor_split": []},
     {"tensor_split": [0, 0]}],
)
def test_placement_rejects_negative_or_empty_values(bad):
    with pytest.raises(ValidationError):
        PresetPlacement(**bad)


def test_normalize_placement_drops_nulls_and_empty_objects():
    assert normalize_placement(None) is None
    assert normalize_placement({}) is None
    assert normalize_placement({"split_policy": None, "gpu_layers": None}) is None
    assert normalize_placement({"gpu_layers": 0, "split_mode": None}) == {"gpu_layers": 0}
    with pytest.raises(ValidationError):
        normalize_placement({"bogus": 1})


def test_create_update_and_out_carry_placement():
    created = RuntimePresetCreate(
        preset_key="normal", label="Normal", context_tokens=32768,
        placement={"split_policy": "always", "tensor_split": [1, 3]},
    )
    assert created.placement.split_policy == "always"
    patch_body = RuntimePresetUpdate(placement=None)
    assert patch_body.model_dump(exclude_unset=True) == {"placement": None}
    patch_body = RuntimePresetUpdate(placement={"gpu_layers": 20})
    assert patch_body.model_dump(exclude_unset=True)["placement"]["gpu_layers"] == 20

    row = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001", profile_id="p", preset_key="normal",
        label="Normal", context_mode="fixed", context_tokens=32768, parallelism=2,
        kv_cache_dtype="q8", placement_policy="auto", preferred_gpus=None,
        execution_policy="auto", placement={"gpu_layers": 20}, enabled=True,
        is_default=True, priority=100, created_at=None, updated_at=None,
    )
    out = RuntimePresetOut.model_validate(row).model_dump(mode="json")
    assert out["placement"] == {
        "split_policy": None, "split_mode": None, "tensor_split": None,
        "gpu_layers": 20, "n_cpu_moe": None,
    }
    row.placement = None
    assert RuntimePresetOut.model_validate(row).model_dump(mode="json")["placement"] is None


def test_resolution_out_exposes_cpu_mb():
    base = {"preset_key": "k", "resolved": True, "requested_parallelism": 1,
            "effective_parallelism": 1, "fits": True}
    assert RuntimePresetResolutionOut.model_validate(base).cpu_mb is None
    assert RuntimePresetResolutionOut.model_validate({**base, "cpu_mb": 1234}).cpu_mb == 1234


# ── Registry round-trip (no database) ───────────────────────────────────────


def _registry_profile():
    return SimpleNamespace(
        profile_id="m", profile_kind="execution", base_model_id="llamacpp_mks/m",
        runtime_presets=[],
    )


@pytest.mark.asyncio
async def test_registry_create_and_update_store_normalized_placement():
    from ocabra.core.profile_registry import ProfileRegistry

    registry = ProfileRegistry()
    profile = _registry_profile()
    registry._profiles[profile.profile_id] = profile
    registry._reload_profile = AsyncMock(return_value=profile)
    session = AsyncMock()
    session.add = MagicMock()

    preset = await registry.create_runtime_preset(
        session, "m",
        {"preset_key": "big", "label": "Big", "context_tokens": 65536,
         "placement": {"split_policy": "when_needed", "gpu_layers": None, "n_cpu_moe": 4}},
    )
    assert preset.placement == {"split_policy": "when_needed", "n_cpu_moe": 4}

    registry._get_runtime_preset_row = AsyncMock(return_value=preset)
    updated = await registry.update_runtime_preset(
        session, "m", "big", {"placement": {"split_mode": "row", "tensor_split": [1, 2]}}
    )
    assert updated.placement == {"split_mode": "row", "tensor_split": [1.0, 2.0]}
    # A patch that does not mention placement keeps it.
    updated = await registry.update_runtime_preset(session, "m", "big", {"label": "Grande"})
    assert updated.placement == {"split_mode": "row", "tensor_split": [1.0, 2.0]}
    # placement: null clears it.
    updated = await registry.update_runtime_preset(session, "m", "big", {"placement": None})
    assert updated.placement is None
    with pytest.raises(ValidationError):
        await registry.update_runtime_preset(session, "m", "big", {"placement": {"x": 1}})


# ── Runtime key ─────────────────────────────────────────────────────────────


def test_runtime_key_unchanged_without_placement_and_distinct_with_it():
    from ocabra.core.runtime_presets import build_runtime_key

    args = dict(model_id="m", backend_type="llamacpp_mks", effective_context=32768,
                parallelism=4, kv_cache_dtype="q8", placement_policy="auto", gpu_indices=(1,))
    plain = build_runtime_key(**args)
    assert build_runtime_key(**args, placement=None) == plain
    assert build_runtime_key(**args, placement={}) == plain
    with_layers = build_runtime_key(**args, placement={"gpu_layers": 20})
    assert with_layers != plain
    assert build_runtime_key(**args, placement={"gpu_layers": 30}) != with_layers


# ── Capacity planner ────────────────────────────────────────────────────────

_ARCH = ModelArch(layers=40, n_kv_heads=8, key_length=128, value_length=128,
                  context_length=262_144)
_GPUS = [SimpleNamespace(index=0, total_vram_mb=12_288), SimpleNamespace(index=1, total_vram_mb=24_576)]


def _cap_preset(key="p", *, tokens=32_768, mode="fixed", placement=None, policy="auto",
                gpus=None, parallelism=1):
    return SimpleNamespace(
        preset_key=key, context_mode=mode, context_tokens=tokens if mode == "fixed" else None,
        parallelism=parallelism, kv_cache_dtype="q8", placement_policy=policy,
        preferred_gpus=gpus, placement=placement, priority=10,
    )


async def _resolve(monkeypatch, presets, *, weights_mb=10_000, backend="llamacpp_mks"):
    from ocabra.core.runtime_preset_capacity import resolve_profile_runtime_capacities

    monkeypatch.setattr(
        "ocabra.core.runtime_preset_capacity.resolve_arch_and_weights",
        AsyncMock(return_value=ArchResolution(_ARCH, weights_mb, "")),
    )
    profile = SimpleNamespace(base_model_id=f"{backend}/m", load_overrides=None,
                              runtime_policy=None, runtime_presets=presets)
    state = SimpleNamespace(backend_type=backend, preferred_gpu=1, extra_config={})
    return await resolve_profile_runtime_capacities(profile, state, _GPUS)


@pytest.mark.asyncio
async def test_planner_gpu_layers_offload_cuts_gpu_mb_and_reports_cpu_mb(monkeypatch):
    full, half = await _resolve(monkeypatch, [
        _cap_preset("full", gpus=[1], policy="forced"),
        _cap_preset("half", gpus=[1], policy="forced", placement={"gpu_layers": 20}),
    ])
    assert "cpu_mb" not in full
    assert full["fits"] and half["fits"]
    gpu_full = full["vram_by_gpu"][1]
    gpu_half = half["vram_by_gpu"][1]
    assert gpu_half < gpu_full
    # Half the weights and half the KV leave the GPU.
    assert half["vram_breakdown_by_gpu"][1]["weights"] == 5_000
    assert half["vram_breakdown_by_gpu"][1]["kv"] == full["vram_breakdown_by_gpu"][1]["kv"] // 2
    assert half["cpu_mb"] >= 5_000
    assert half["cpu_mb"] == pytest.approx(
        5_000 + full["vram_breakdown_by_gpu"][1]["kv"] / 2, abs=2
    )
    assert half["runtime_key"] != full["runtime_key"]
    out = RuntimePresetResolutionOut.model_validate(half).model_dump(mode="json")
    assert out["cpu_mb"] == half["cpu_mb"]


@pytest.mark.asyncio
async def test_planner_offload_raises_the_max_context(monkeypatch):
    full, half = await _resolve(monkeypatch, [
        _cap_preset("full", mode="max", gpus=[0], policy="forced"),
        _cap_preset("half", mode="max", gpus=[0], policy="forced", placement={"gpu_layers": 20}),
    ])
    assert half["effective_context"] > full["effective_context"]


@pytest.mark.asyncio
async def test_planner_cpu_moe_approximates_expert_weights_and_warns(monkeypatch):
    (row,) = await _resolve(monkeypatch, [
        _cap_preset("moe", gpus=[1], policy="forced", placement={"n_cpu_moe": 20}),
    ])
    # 20/40 layers x 85 % of the weights go to RAM.
    assert row["vram_breakdown_by_gpu"][1]["weights"] == 10_000 - 4_250
    assert row["cpu_mb"] == 4_250
    assert "estimación aproximada: expertos MoE en CPU" in row["warnings"]
    assert row["fits"] is True  # the estimate warning does not make it unfit


@pytest.mark.asyncio
async def test_planner_cpu_moe_only_counts_experts_of_layers_on_gpu(monkeypatch):
    (row,) = await _resolve(monkeypatch, [
        _cap_preset("moe", gpus=[1], policy="forced",
                    placement={"gpu_layers": 30, "n_cpu_moe": 20}),
    ])
    # 10 layers on CPU already; of the 20 MoE layers only 10 were on GPU.
    assert row["vram_breakdown_by_gpu"][1]["weights"] == int(10_000 * 30 / 40 - 10_000 * 10 / 40 * 0.85)


@pytest.mark.asyncio
async def test_planner_when_needed_resolves_multi_gpu_when_no_single_gpu_fits(monkeypatch):
    single, when_needed = await _resolve(monkeypatch, [
        _cap_preset("a"),
        _cap_preset("b", placement={"split_policy": "when_needed"}),
    ], weights_mb=26_000)
    assert single["fits"] is False and len(single["target_gpus"]) == 1
    assert when_needed["fits"] is True
    assert when_needed["target_gpus"] == [0, 1]
    assert set(when_needed["vram_by_gpu"]) == {0, 1}


@pytest.mark.asyncio
async def test_planner_when_needed_stays_single_when_it_fits(monkeypatch):
    (row,) = await _resolve(monkeypatch, [_cap_preset("b", placement={"split_policy": "when_needed"})])
    assert row["fits"] and row["target_gpus"] == [1]


@pytest.mark.asyncio
async def test_planner_always_splits_even_when_one_gpu_fits(monkeypatch):
    auto, preferred = await _resolve(monkeypatch, [
        _cap_preset("a", placement={"split_policy": "always"}),
        _cap_preset("b", policy="preferred", gpus=[1], placement={"split_policy": "always"}),
    ])
    assert auto["target_gpus"] == [0, 1] and auto["fits"]
    # The preferred GPU goes first (it becomes llama.cpp's main GPU order).
    assert preferred["target_gpus"] == [1, 0]


@pytest.mark.asyncio
async def test_planner_tensor_split_drives_the_per_gpu_breakdown(monkeypatch):
    (row,) = await _resolve(monkeypatch, [
        _cap_preset("a", policy="forced", gpus=[0, 1], placement={"tensor_split": [1, 3]}),
    ])
    weights = {g: parts["weights"] for g, parts in row["vram_breakdown_by_gpu"].items()}
    assert weights == {0: 2_500, 1: 7_500}


@pytest.mark.asyncio
async def test_planner_ignores_placement_for_other_backends(monkeypatch):
    plain, placed = await _resolve(monkeypatch, [
        _cap_preset("a", gpus=[1], policy="forced"),
        _cap_preset("b", gpus=[1], policy="forced", placement={"gpu_layers": 10}),
    ], backend="ollama")
    assert "cpu_mb" not in placed
    assert placed["vram_by_gpu"] == plain["vram_by_gpu"]


@pytest.mark.asyncio
async def test_planner_without_placement_is_unchanged(monkeypatch):
    """A preset with placement=None must resolve exactly as one without the attribute."""
    with_attr, = await _resolve(monkeypatch, [_cap_preset("a", gpus=[1], policy="forced")])
    legacy = _cap_preset("a", gpus=[1], policy="forced")
    del legacy.placement
    without_attr, = await _resolve(monkeypatch, [legacy])
    assert with_attr == without_attr


# ── Coordinator ─────────────────────────────────────────────────────────────


def _coord_preset(gpus, placement):
    return SimpleNamespace(
        preset_key="p", context_mode="fixed", context_tokens=65_536, parallelism=2,
        kv_cache_dtype="q8", placement_policy="forced", preferred_gpus=gpus,
        execution_policy="auto", enabled=True, is_default=True, priority=10,
        placement=placement,
    )


async def _plan(monkeypatch, gpus, placement, *, runtime_key="runtime:p"):
    from ocabra.core.runtime_coordinator import plan_profile_runtime

    profile = SimpleNamespace(
        profile_id="m", base_model_id="llamacpp_mks/m", load_overrides=None,
        runtime_policy={"selection": "prefer_loaded"},
        runtime_presets=[_coord_preset(gpus, placement)],
    )
    state = SimpleNamespace(
        model_id=profile.base_model_id, backend_type="llamacpp_mks",
        status=SimpleNamespace(value="configured"), load_policy=SimpleNamespace(value="warm"),
        extra_config={"llama_cpp": {"split_mode": "none", "disabled_gpus": [0],
                                    "gpu_layers": 99, "n_cpu_moe": 2}},
    )
    manager = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=_GPUS)),
        list_states=AsyncMock(return_value=[state]),
    )
    monkeypatch.setattr(
        "ocabra.core.runtime_coordinator.runtime_capacity_cache.resolve",
        AsyncMock(return_value=[{
            "preset_key": "p", "resolved": True, "fits": True, "effective_context": 65_536,
            "effective_parallelism": 2, "target_gpus": list(gpus), "runtime_key": runtime_key,
        }]),
    )
    return await plan_profile_runtime(
        profile, manager,
        {"messages": [{"role": "user", "content": "hola"}], "max_tokens": 1},
    )


@pytest.mark.asyncio
async def test_coordinator_applies_multi_gpu_placement_into_llama_section(monkeypatch):
    plan = await _plan(monkeypatch, [0, 1], {
        "split_mode": "row", "tensor_split": [1, 2], "gpu_layers": 30, "n_cpu_moe": 6,
    })
    llama = plan.load_overrides["llama_cpp"]
    assert llama["disabled_gpus"] == []
    assert llama["split_mode"] == "row"
    assert llama["tensor_split"] == [1.0, 2.0]
    assert llama["gpu_layers"] == 30
    assert llama["n_cpu_moe"] == 6
    assert plan.load_overrides["_runtime"]["placement"]["gpu_layers"] == 30


@pytest.mark.asyncio
async def test_coordinator_ignores_split_options_on_one_gpu(monkeypatch):
    plan = await _plan(monkeypatch, [1], {
        "split_mode": "row", "tensor_split": [1, 2], "gpu_layers": 20,
    })
    llama = plan.load_overrides["llama_cpp"]
    assert llama["split_mode"] == "none"
    assert llama["tensor_split"] is None
    assert llama["gpu_layers"] == 20
    # The preset owns its experts: none listed = none on CPU.
    assert llama["n_cpu_moe"] is None


@pytest.mark.asyncio
async def test_coordinator_tensor_split_needs_one_ratio_per_gpu(monkeypatch):
    plan = await _plan(monkeypatch, [0, 1], {"tensor_split": [1, 2, 3]})
    assert plan.load_overrides["llama_cpp"]["tensor_split"] == [1.0, 2.0]  # by GPU size
    assert plan.load_overrides["llama_cpp"]["gpu_layers"] == 999  # null = all


@pytest.mark.asyncio
async def test_coordinator_without_placement_keeps_the_worker_key(monkeypatch):
    plain = await _plan(monkeypatch, [1], None)
    assert "placement" not in plain.load_overrides["_runtime"]
    assert plain.load_overrides["llama_cpp"]["gpu_layers"] == 99
    assert plain.load_overrides["llama_cpp"]["n_cpu_moe"] == 2
    placed = await _plan(monkeypatch, [1], {"gpu_layers": 20}, runtime_key="runtime:other")
    assert placed.worker_key != plain.worker_key
    assert placed.runtime_key == "runtime:other"


# ── Scheduler ───────────────────────────────────────────────────────────────


def _scheduler(free_by_gpu, total_by_gpu):
    from ocabra.core.gpu_manager import GPUState
    from ocabra.core.scheduler import GPUScheduler

    states = [
        GPUState(index=i, name=f"GPU-{i}", total_vram_mb=total_by_gpu[i], free_vram_mb=free,
                 used_vram_mb=0, utilization_pct=0, temperature_c=40, power_draw_w=50,
                 power_limit_w=370, locked_vram_mb=0)
        for i, free in free_by_gpu.items()
    ]
    gm = AsyncMock()
    gm.get_all_states = AsyncMock(return_value=states)
    gm.get_free_vram = AsyncMock(side_effect=lambda i: free_by_gpu[i])
    gm.has_service_lock = MagicMock(return_value=False)
    return GPUScheduler(gm)


@pytest.mark.asyncio
async def test_scheduler_never_picks_an_excluded_gpu():
    from ocabra.core.scheduler import InsufficientVRAMError

    scheduler = _scheduler({0: 12_000, 1: 2_000}, {0: 12_288, 1: 24_576})
    assert await scheduler.find_gpu_for_model(8_000, preferred_gpu=1) == [0]
    with pytest.raises(InsufficientVRAMError):
        await scheduler.find_gpu_for_model(8_000, preferred_gpu=1, allowed_gpus=[1])
    scheduler = _scheduler({0: 12_000, 1: 20_000}, {0: 12_288, 1: 24_576})
    assert await scheduler.find_gpu_for_model(8_000, preferred_gpu=0, allowed_gpus=[1]) == [1]


@pytest.mark.asyncio
async def test_scheduler_allow_split_false_never_spreads():
    from ocabra.core.scheduler import InsufficientVRAMError

    scheduler = _scheduler({0: 12_000, 1: 24_000}, {0: 12_288, 1: 24_576})
    assert await scheduler.find_gpu_for_model(30_000) == [0, 1]
    with pytest.raises(InsufficientVRAMError):
        await scheduler.find_gpu_for_model(30_000, allow_split=False)


# ── Base config split policy (models without presets) ───────────────────────


def _llama_state(llama, model_id="llamacpp_mks/base"):
    from ocabra.core.model_manager import LoadPolicy, ModelState

    return ModelState(model_id=model_id, display_name=model_id, backend_type="llamacpp_mks",
                      load_policy=LoadPolicy.ON_DEMAND, extra_config={"llama_cpp": llama})


def _mm_with_gpus(scheduler=None):
    from ocabra.core.model_manager import ModelManager

    gm = MagicMock()
    gm.get_all_states = AsyncMock(return_value=_GPUS)
    return ModelManager(MagicMock(), gpu_manager=gm, gpu_scheduler=scheduler)


def test_split_policy_legacy_defaults():
    from ocabra.core.model_manager_helpers import resolve_llama_cpp_split_policy

    assert resolve_llama_cpp_split_policy(_llama_state({"split_mode": "none"})) == "single"
    assert resolve_llama_cpp_split_policy(_llama_state({"split_mode": "layer"})) == "when_needed"
    assert resolve_llama_cpp_split_policy(_llama_state({})) == "when_needed"
    assert resolve_llama_cpp_split_policy(
        _llama_state({"split_mode": "none", "split_policy": "always"})
    ) == "always"
    assert resolve_llama_cpp_split_policy(_llama_state({"split_policy": "bogus"})) == "when_needed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("llama", "expected"),
    [
        # production shape: one GPU, the 3060 excluded
        ({"split_mode": "none", "disabled_gpus": [0]}, (None, [1], False)),
        ({"split_mode": "none", "disabled_gpus": []}, (None, None, False)),
        ({"split_mode": "layer", "tensor_split": [1, 3]}, (None, None, True)),
        ({"split_policy": "always"}, ([0, 1], None, True)),
        ({"split_policy": "always", "disabled_gpus": [0]}, (None, [1], True)),
        ({"split_policy": "when_needed", "disabled_gpus": [1]}, (None, [0], True)),
        ({"split_policy": "single"}, (None, None, False)),
    ],
)
async def test_base_placement_by_policy(llama, expected):
    mm = _mm_with_gpus()
    assert await mm._llama_cpp_base_placement(_llama_state(llama)) == expected


@pytest.mark.asyncio
async def test_assign_gpus_passes_restrictions_to_the_scheduler_only_when_set():
    scheduler = MagicMock()
    scheduler.find_gpu_for_model = AsyncMock(return_value=[1])
    mm = _mm_with_gpus(scheduler)
    await mm._assign_gpus_for_load("m", 1000, None, False)
    assert scheduler.find_gpu_for_model.await_args.kwargs == {
        "enforce_vllm_headroom": False, "vllm_gpu_memory_utilization": None,
    }
    await mm._assign_gpus_for_load("m", 1000, None, False, allowed_gpus=[1], allow_split=False)
    assert scheduler.find_gpu_for_model.await_args.kwargs["allowed_gpus"] == [1]
    assert scheduler.find_gpu_for_model.await_args.kwargs["allow_split"] is False


class _Stop(Exception):
    pass


async def _admission_kwargs(llama, runtime_gpus=None):
    """Run the real load path up to admission and return what it asked for."""
    from ocabra.core.worker_pool import WorkerPool

    pool = WorkerPool()
    backend = MagicMock()
    backend.get_vram_estimate_mb = AsyncMock(return_value=8_000)
    pool._backends["llamacpp_mks"] = backend
    mm = _mm_with_gpus(MagicMock())
    mm._worker_pool = pool
    state = _llama_state({"gpu_layers": 99, **llama})
    if runtime_gpus is not None:
        state.extra_config["_runtime"] = {"gpu_indices": runtime_gpus}
    mm._states[state.model_id] = state
    mm._load_locks[state.model_id] = asyncio.Lock()
    captured = {}

    async def fake_admit(**kwargs):
        captured.update(kwargs)
        raise _Stop

    with (
        patch.object(mm, "_admit_gpu_load", new=fake_admit),
        patch("ocabra.core.model_manager.publish", new=AsyncMock()),
        patch("ocabra.core.model_manager.set_key", new=AsyncMock()),
        patch("ocabra.core.model_manager.observed_vram.lookup", new=AsyncMock(return_value=None)),
        patch.object(mm, "_record_model_load_stat", new=AsyncMock()),
    ):
        with pytest.raises(Exception):  # noqa: B017 — the load stops at admission on purpose
            await mm.load(state.model_id)
    return captured


@pytest.mark.asyncio
async def test_load_path_always_requires_every_allowed_gpu():
    kwargs = await _admission_kwargs({"split_policy": "always", "tensor_split": [1, 3]})
    assert kwargs["required_gpu_indices"] == [0, 1]
    # Reuses the configured ratios to size each GPU's share.
    assert kwargs["required_gpu_vram_mb"] == {0: 2_000, 1: 6_000}


@pytest.mark.asyncio
async def test_load_path_single_forbids_split_and_excluded_gpus():
    kwargs = await _admission_kwargs({"split_mode": "none", "disabled_gpus": [0]})
    assert kwargs["required_gpu_indices"] is None
    assert kwargs["allowed_gpus"] == [1]
    assert kwargs["allow_split"] is False


@pytest.mark.asyncio
async def test_load_path_when_needed_keeps_the_scheduler_free():
    kwargs = await _admission_kwargs({"split_mode": "layer"})
    assert kwargs["required_gpu_indices"] is None
    assert kwargs["allowed_gpus"] is None
    assert kwargs["allow_split"] is True


@pytest.mark.asyncio
async def test_load_path_preset_gpus_win_over_the_base_policy():
    kwargs = await _admission_kwargs({"split_policy": "always"}, runtime_gpus=[1])
    assert kwargs["required_gpu_indices"] == [1]
    assert kwargs["allowed_gpus"] is None


def test_config_schema_offers_split_policy():
    from ocabra.core import config_schema

    fields = {f.path: f for f in config_schema._LLAMA_FIELDS}
    field = fields[("llama_cpp", "split_policy")]
    assert field.type == "enum" and field.group == "gpus"
    assert field.label == "Si no cabe en una GPU"
    assert [value for value, _ in field.options] == ["single", "when_needed", "always"]


# ── Launch flags of existing configs (golden, computed before the change) ───

_LAUNCH_CASES = {
 "none_disabled0": ({"ctx_size": 262144, "batch_size": 2048, "flash_attn": True, "gpu_layers": 99, "kv_unified": True, "split_mode": "none", "ubatch_size": 512, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "cont_batching": True, "disabled_gpus": [0], "parallel_slots": 4, "cache_idle_slots": False, "startup_timeout_s": 300}, [1]),
 "none_disabled0_assigned_both": ({"ctx_size": 40960, "flash_attn": True, "gpu_layers": 99, "kv_unified": True, "split_mode": "none", "cache_type_k": "q8_0", "cache_type_v": "q8_0", "disabled_gpus": [0], "parallel_slots": 4}, [0, 1]),
 "no_split_mode_single": ({"ctx_size": 262144, "flash_attn": True, "gpu_layers": 99, "kv_unified": True, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "cont_batching": True, "parallel_slots": 4, "startup_timeout_s": 300}, [1]),
 "no_split_mode_tensor_split": ({"ctx_size": 524288, "main_gpu": 1, "batch_size": 2048, "flash_attn": True, "gpu_layers": 99, "kv_unified": True, "ubatch_size": 512, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "tensor_split": [1, 3], "cont_batching": True, "parallel_slots": 4, "split_strategy": "evenly"}, [0, 1]),
 "layer_tensor_split": ({"ctx_size": 40960, "main_gpu": 1, "flash_attn": True, "gpu_layers": 99, "kv_unified": True, "split_mode": "layer", "cache_type_k": "q8_0", "cache_type_v": "q8_0", "tensor_split": [1, 3], "cont_batching": True, "parallel_slots": 4, "split_strategy": "evenly", "cache_idle_slots": False}, [0, 1]),
 "tensor_mode": ({"ctx_size": 262144, "main_gpu": 1, "batch_size": 512, "flash_attn": True, "gpu_layers": 99, "kv_unified": True, "split_mode": "tensor", "ubatch_size": 128, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "tensor_split": [1, 3], "disabled_gpus": [], "parallel_slots": 4, "split_strategy": "evenly"}, [0, 1]),
 "embedding_none": ({"ctx_size": 40960, "main_gpu": None, "embedding": True, "flash_attn": True, "gpu_layers": 99, "kv_unified": True, "split_mode": "none", "cache_type_k": "q8_0", "cache_type_v": "q8_0", "tensor_split": None, "disabled_gpus": [], "parallel_slots": 4}, [1]),
 "cpu_moe": ({"ctx_size": 32768, "gpu_layers": 20, "n_cpu_moe": 8, "split_mode": "none", "parallel_slots": 2}, [1]),
}

_LAUNCH_GOLDEN = {
    'none_disabled0': (
        '--server-bin /bin/llama-server --model-id m --model-path <no encontrado: m> --host 127.0.0.1 --port 0 --ctx-size 262144 --batch-size 2048 --ubatch-size 512 --gpu-layers 99 --alias m --flash-attn --cache-type-k q8_0 --cache-type-v q8_0 --kv-unified --no-cache-idle-slots --split-mode none --parallel 4 --cont-batching',
        '1',
    ),
    'none_disabled0_assigned_both': (
        '--server-bin /bin/llama-server --model-id m --model-path <no encontrado: m> --host 127.0.0.1 --port 0 --ctx-size 40960 --batch-size 512 --ubatch-size 128 --gpu-layers 99 --alias m --flash-attn --cache-type-k q8_0 --cache-type-v q8_0 --kv-unified --split-mode none --parallel 4 --cont-batching',
        '1',
    ),
    'no_split_mode_single': (
        '--server-bin /bin/llama-server --model-id m --model-path <no encontrado: m> --host 127.0.0.1 --port 0 --ctx-size 262144 --batch-size 512 --ubatch-size 128 --gpu-layers 99 --alias m --flash-attn --cache-type-k q8_0 --cache-type-v q8_0 --kv-unified --parallel 4 --cont-batching',
        '1',
    ),
    'no_split_mode_tensor_split': (
        '--server-bin /bin/llama-server --model-id m --model-path <no encontrado: m> --host 127.0.0.1 --port 0 --ctx-size 524288 --batch-size 2048 --ubatch-size 512 --gpu-layers 99 --alias m --flash-attn --cache-type-k q8_0 --cache-type-v q8_0 --kv-unified --main-gpu 1 --tensor-split 1,3 --parallel 4 --cont-batching',
        '0,1',
    ),
    'layer_tensor_split': (
        '--server-bin /bin/llama-server --model-id m --model-path <no encontrado: m> --host 127.0.0.1 --port 0 --ctx-size 40960 --batch-size 512 --ubatch-size 128 --gpu-layers 99 --alias m --flash-attn --cache-type-k q8_0 --cache-type-v q8_0 --kv-unified --no-cache-idle-slots --main-gpu 1 --tensor-split 1,3 --split-mode layer --parallel 4 --cont-batching',
        '0,1',
    ),
    'tensor_mode': (
        '--server-bin /bin/llama-server --model-id m --model-path <no encontrado: m> --host 127.0.0.1 --port 0 --ctx-size 262144 --batch-size 512 --ubatch-size 128 --gpu-layers 99 --alias m --flash-attn --cache-type-k q8_0 --cache-type-v q8_0 --kv-unified --main-gpu 1 --tensor-split 1,3 --split-mode tensor --parallel 4 --cont-batching',
        '0,1',
    ),
    'embedding_none': (
        '--server-bin /bin/llama-server --model-id m --model-path <no encontrado: m> --host 127.0.0.1 --port 0 --ctx-size 40960 --batch-size 512 --ubatch-size 128 --gpu-layers 99 --alias m --flash-attn --embedding --cache-type-k q8_0 --cache-type-v q8_0 --kv-unified --split-mode none --parallel 4 --cont-batching',
        '1',
    ),
    'cpu_moe': (
        '--server-bin /bin/llama-server --model-id m --model-path <no encontrado: m> --host 127.0.0.1 --port 0 --ctx-size 32768 --batch-size 512 --ubatch-size 128 --gpu-layers 20 --alias m --split-mode none --n-cpu-moe 8 --parallel 2 --cont-batching',
        '1',
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(_LAUNCH_CASES))
async def test_existing_configs_keep_their_exact_launch(case, monkeypatch):
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend

    llama, gpus = _LAUNCH_CASES[case]
    backend = LlamaCppBackend()
    monkeypatch.setattr(backend, "_resolve_server_bin", lambda: "/bin/llama-server")
    out = await backend.preview_launch(
        "m", list(gpus), {"llama_cpp": {**llama, "model_path": "/nonexistent/m.gguf"}}
    )
    argv, visible = _LAUNCH_GOLDEN[case]
    assert " ".join(out["argv"]) == argv
    assert out["env"]["CUDA_VISIBLE_DEVICES"] == visible


@pytest.mark.asyncio
async def test_split_mode_none_on_several_visible_gpus_becomes_layer(monkeypatch):
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend

    backend = LlamaCppBackend()
    monkeypatch.setattr(backend, "_resolve_server_bin", lambda: "/bin/llama-server")
    out = await backend.preview_launch(
        "m", [0, 1], {"llama_cpp": {"split_mode": "none", "gpu_layers": 99,
                                   "model_path": "/nonexistent/m.gguf"}}
    )
    argv = out["argv"]
    assert argv[argv.index("--split-mode") + 1] == "layer"
    assert out["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"


# ── Smoke test endpoint ─────────────────────────────────────────────────────


def _smoke_request(*, loaded_after=True, caps=None, load_error=None, forward=None,
                   initially_loaded=False, busy=False):
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import ModelStatus

    state = SimpleNamespace(
        model_id="m", backend_model_id="m", backend_type="vllm",
        status=ModelStatus.LOADED if initially_loaded else ModelStatus.CONFIGURED,
        capabilities=caps or BackendCapabilities(chat=True),
    )

    async def load(_model_id):
        if load_error is not None:
            raise load_error
        state.status = ModelStatus.LOADED if loaded_after else ModelStatus.ERROR
        return state

    async def unload(_model_id, reason=None):
        state.status = ModelStatus.UNLOADED

    mm = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        load=AsyncMock(side_effect=load),
        unload=AsyncMock(side_effect=unload),
        is_busy=MagicMock(return_value=busy),
    )
    pool = SimpleNamespace(
        forward_request=forward or AsyncMock(
            return_value={"usage": {"completion_tokens": 4}, "choices": []}
        ),
        get_worker=MagicMock(return_value=None),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        model_manager=mm, worker_pool=pool, profile_registry=None,
    )))
    return request, mm, pool


@pytest.mark.asyncio
async def test_smoke_test_chat_success():
    from ocabra.api.internal.models import SmokeTestRequest, smoke_test_model

    request, mm, pool = _smoke_request()
    out = await smoke_test_model("m", request, SmokeTestRequest(), _user=None)
    assert out["loaded"] is True and out["error"] is None
    assert out["request_kind"] == "chat" and out["request_ok"] is True
    assert isinstance(out["load_ms"], int) and out["request_ms"] >= 1
    assert out["tokens_per_second"] > 0
    path, body = pool.forward_request.await_args.args[1:]
    assert path == "/v1/chat/completions"
    assert body["messages"][0]["content"] == "Di hola." and body["max_tokens"] == 8
    mm.unload.assert_not_awaited()
    assert set(out) == {"loaded", "load_ms", "request_kind", "request_ok", "request_ms",
                        "tokens_per_second", "error"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caps", "path"),
    [({"embeddings": True}, "/v1/embeddings"), ({"rerank": True}, "/rerank")],
)
async def test_smoke_test_routes_by_capability(caps, path):
    from ocabra.api.internal.models import smoke_test_model
    from ocabra.backends.base import BackendCapabilities

    request, _mm, pool = _smoke_request(caps=BackendCapabilities(**caps),
                                        forward=AsyncMock(return_value={"data": []}))
    out = await smoke_test_model("m", request, None, _user=None)
    assert out["request_ok"] is True and out["tokens_per_second"] is None
    assert pool.forward_request.await_args.args[1] == path


@pytest.mark.asyncio
async def test_smoke_test_skips_the_request_for_other_models():
    from ocabra.api.internal.models import smoke_test_model
    from ocabra.backends.base import BackendCapabilities

    request, _mm, pool = _smoke_request(caps=BackendCapabilities(image_generation=True))
    out = await smoke_test_model("m", request, None, _user=None)
    assert out["loaded"] is True
    assert out["request_kind"] is None and out["request_ok"] is None and out["error"] is None
    pool.forward_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_smoke_test_reports_a_failed_load_without_raising():
    from ocabra.api.internal.models import smoke_test_model
    from ocabra.core.scheduler import InsufficientVRAMError

    request, _mm, pool = _smoke_request(load_error=InsufficientVRAMError("Need 30000 MB"))
    out = await smoke_test_model("m", request, None, _user=None)
    assert out["loaded"] is False and "Need 30000 MB" in out["error"]
    assert out["request_ok"] is None
    pool.forward_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_smoke_test_reports_a_model_left_unloaded():
    from ocabra.api.internal.models import smoke_test_model

    request, _mm, _pool = _smoke_request(loaded_after=False)
    out = await smoke_test_model("m", request, None, _user=None)
    assert out["loaded"] is False and "error" in out["error"]


@pytest.mark.asyncio
async def test_smoke_test_reports_a_failed_request():
    import httpx

    from ocabra.api.internal.models import smoke_test_model

    response = httpx.Response(500, text="boom", request=httpx.Request("POST", "http://w"))
    failing = AsyncMock(side_effect=httpx.HTTPStatusError("x", request=response.request,
                                                          response=response))
    request, _mm, _pool = _smoke_request(forward=failing)
    out = await smoke_test_model("m", request, None, _user=None)
    assert out["loaded"] is True and out["request_ok"] is False
    assert "HTTP 500" in out["error"] and "boom" in out["error"]


@pytest.mark.asyncio
async def test_smoke_test_cold_unloads_first():
    from ocabra.api.internal.models import SmokeTestRequest, smoke_test_model

    request, mm, _pool = _smoke_request(initially_loaded=True)
    out = await smoke_test_model("m", request, SmokeTestRequest(cold=True), _user=None)
    mm.unload.assert_awaited_once()
    mm.load.assert_awaited_once()
    assert out["loaded"] is True and out["request_ok"] is True


@pytest.mark.asyncio
async def test_smoke_test_cold_does_not_unload_a_busy_model():
    from ocabra.api.internal.models import SmokeTestRequest, smoke_test_model

    request, mm, _pool = _smoke_request(initially_loaded=True, busy=True)
    out = await smoke_test_model("m", request, SmokeTestRequest(cold=True), _user=None)
    mm.unload.assert_not_awaited()
    assert out["error"]


@pytest.mark.asyncio
async def test_smoke_test_unknown_model_is_404():
    from fastapi import HTTPException

    from ocabra.api.internal.models import smoke_test_model

    request, mm, _pool = _smoke_request()
    mm.get_state = AsyncMock(return_value=None)
    with pytest.raises(HTTPException) as exc:
        await smoke_test_model("nope", request, None, _user=None)
    assert exc.value.status_code == 404


def test_smoke_test_is_registered_as_post_with_model_manager_role():
    from ocabra.main import app

    routes = [r for r in app.routes if getattr(r, "path", "") ==
              "/ocabra/models/{model_id:path}/smoke-test"]
    assert routes and "POST" in routes[0].methods


def test_silence_wav_is_one_second_of_16k_mono():
    import io
    import wave

    from ocabra.api.internal.models import _silence_wav

    with wave.open(io.BytesIO(_silence_wav())) as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getnframes()) == (1, 16000, 16000)


@pytest.mark.asyncio
async def test_preview_endpoint_honours_placement(monkeypatch):
    from ocabra.api.internal import profiles as api

    monkeypatch.setattr(
        "ocabra.core.runtime_preset_capacity.resolve_arch_and_weights",
        AsyncMock(return_value=ArchResolution(_ARCH, 10_000, "")),
    )
    profile = SimpleNamespace(profile_kind="execution", base_model_id="llamacpp_mks/m",
                              load_overrides=None, runtime_policy=None)
    registry = SimpleNamespace(get=AsyncMock(return_value=profile))
    state = SimpleNamespace(backend_type="llamacpp_mks", preferred_gpu=1, extra_config={})
    mm = SimpleNamespace(
        get_state=AsyncMock(return_value=state),
        gpu_manager=SimpleNamespace(get_all_states=AsyncMock(return_value=_GPUS)),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        profile_registry=registry, model_manager=mm,
    )))
    body = RuntimePresetCreate(preset_key="draft", label="Draft", context_tokens=32_768,
                               placement_policy="forced", preferred_gpus=[1],
                               placement={"gpu_layers": 20})
    out = await api.preview_runtime_preset("m", body, request, _user=None)
    assert out["cpu_mb"] > 0
    assert out["vram_breakdown_by_gpu"]["1"]["weights"] == 5_000
