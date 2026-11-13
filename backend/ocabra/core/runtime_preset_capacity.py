"""Resolve persisted runtime presets against current model/GPU capacity."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Any

from ocabra.config import settings
from ocabra.core import observed_vram
from ocabra.core import vram_planner as vp
from ocabra.core.model_manager_helpers import (
    _resolve_rs_rollback_snapshots,
    resolve_llama_cpp_option,
)
from ocabra.core.model_ref import is_llama_cpp_family
from ocabra.core.runtime_presets import build_runtime_key
from ocabra.core.vram_capacity import kv_dtype_for, resolve_arch_and_weights

# Backends whose multi-GPU placement is a real, proportional spread across
# the requested GPUs (Ollama's shared-pool mode, llama.cpp/bitnet's default
# --split-mode layer) rather than a sharding scheme this planner would have
# to model explicitly (vLLM/sglang tensor-parallel). For these, summing the
# requested GPUs' capacity with the same single-GPU formula is a faithful
# model — validated empirically for llama_cpp against a real asymmetric
# 3060+3090 load (2026-10-26).
_PROPORTIONAL_MULTI_GPU_BACKENDS = {"ollama", "llama_cpp", "llamacpp_mks", "bitnet"}


@dataclass(frozen=True)
class _CacheEntry:
    created_at: float
    rows: list[dict[str, Any]]


class RuntimeCapacityCache:
    """Short-lived, coalesced capacity cache shared by API and selection paths."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, _CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _key(profile: Any, state: Any, gpu_states: list[Any]) -> str:
        presets = tuple(
            (
                preset.preset_key,
                preset.context_mode,
                preset.context_tokens,
                preset.parallelism,
                preset.kv_cache_dtype,
                preset.placement_policy,
                tuple(preset.preferred_gpus or ()),
                repr(sorted(preset_placement(preset).items())),
                preset.enabled,
                preset.is_default,
                preset.priority,
            )
            for preset in profile.runtime_presets
        )
        gpus = tuple(sorted((int(gpu.index), int(gpu.total_vram_mb)) for gpu in gpu_states))
        return repr(
            (
                profile.profile_id,
                profile.base_model_id,
                profile.runtime_policy,
                presets,
                state.backend_type,
                state.backend_model_id,
                state.preferred_gpu,
                state.updated_at,
                gpus,
            )
        )

    async def resolve(
        self,
        profile: Any,
        state: Any,
        gpu_states: list[Any],
        *,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        key = self._key(profile, state, gpu_states)
        now = time.monotonic()
        cached = self._entries.get(key)
        if not force and cached and now - cached.created_at < self._ttl_seconds:
            return [dict(row) for row in cached.rows]

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            cached = self._entries.get(key)
            if not force and cached and now - cached.created_at < self._ttl_seconds:
                return [dict(row) for row in cached.rows]
            rows = await resolve_profile_runtime_capacities(profile, state, gpu_states)
            self._entries[key] = _CacheEntry(now, [dict(row) for row in rows])
            return rows

    def clear(self) -> None:
        self._entries.clear()


runtime_capacity_cache = RuntimeCapacityCache()


# Fallback for the driver/context share when the process list can't attribute
# usage (e.g. a busy GPU sampled without per-process info): NVML reserves
# ~330-400 MiB on these cards plus a few MiB of contexts.
_DEFAULT_UNATTRIBUTED_MB = 512
_MAX_PLAUSIBLE_UNATTRIBUTED_MB = 1024


def _unattributed_vram_mb(gpu: Any) -> int:
    """VRAM no model can ever get on this card, independent of what is loaded.

    With the NVML v1 API ``used`` already includes the driver reservation, so
    ``total - free - Σ(process usage)`` is what is neither free nor owned by a
    process: the reservation plus foreign contexts. It does not change when
    models come and go, unlike ``free`` itself.
    """
    try:
        total = int(gpu.total_vram_mb)
        free = int(gpu.free_vram_mb)
    except (AttributeError, TypeError, ValueError):
        return _DEFAULT_UNATTRIBUTED_MB
    processes = getattr(gpu, "processes", None) or []
    attributed = 0
    for proc in processes:
        try:
            attributed += int(
                getattr(proc, "used_vram_mb", None)
                or (proc.get("used_vram_mb") if isinstance(proc, dict) else 0)
                or 0
            )
        except (TypeError, ValueError):
            continue
    unattributed = total - free - attributed
    if unattributed < 0 or unattributed > _MAX_PLAUSIBLE_UNATTRIBUTED_MB:
        return _DEFAULT_UNATTRIBUTED_MB
    return unattributed


def _gpu_dict(gpu: Any) -> dict[str, int]:
    """Capacity a preset may plan against on this GPU.

    Same FIXED margins the load admission applies (``ModelManager`` →
    ``GPUManager.get_free_vram`` after evicting everything): the driver's
    unattributed share and ``vram_buffer_mb``. Planning against the raw total
    made ``fits=true`` mean "fits on paper": qwen3:32b·32k and
    nemotron-3-nano:30b on the 3090 resolved as fitting and then failed every
    load with ``insufficient_vram`` (need 23761/24059 MB, 23647 available).

    Deliberately NOT the current free VRAM: that depends on what happens to be
    loaded, and would turn presets unfit (→ every request rejected) whenever
    another model sits on the card. Validated presets keep using their
    measured capacity (``_apply_validated_context_override``) and the
    admission still admits those against physical free VRAM without buffer.
    """
    total = int(gpu.total_vram_mb)
    usable = total - _unattributed_vram_mb(gpu) - max(0, int(settings.vram_buffer_mb or 0))
    return {
        "index": int(gpu.index),
        "total_mb": max(0, usable),
    }


def _uses_unified_kv_pool(state: Any) -> bool:
    """Whether preset context denotes one aggregate llama.cpp KV pool.

    llama.cpp's legacy preset contract sized an equal context partition per
    slot.  The oCabra build uses ``--kv-unified`` instead: all physical slots
    draw from one pool and a single request may consume all of it.  Capacity
    planning still accounts for per-slot fixed/RS overhead, but the context
    exposed to selection is the aggregate pool, not ``pool / slots``.
    """

    if not is_llama_cpp_family(getattr(state, "backend_type", "")):
        return False
    raw = resolve_llama_cpp_option(state, "kv_unified", False)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)



# Share of an MoE layer's weights that are expert tensors (what --n-cpu-moe
# moves to RAM). An approximation: the real share depends on the model, and the
# GGUF metadata the planner reads does not say which tensors are experts.
_MOE_EXPERT_WEIGHT_SHARE = vp.MOE_EXPERT_WEIGHT_SHARE
_MOE_ESTIMATE_WARNING = "estimación aproximada: expertos MoE en CPU"
_SPLIT_POLICIES = {"single", "when_needed", "always"}


@dataclass(frozen=True)
class _Offload:
    """How much of a preset stays on GPU when part of it goes to CPU."""

    # Share of the layers on GPU; the KV cache of the other layers sits in RAM.
    gpu_fraction: float = 1.0
    gpu_weights_mb: float = 0.0
    cpu_weights_mb: float = 0.0
    moe_estimate: bool = False

    @property
    def active(self) -> bool:
        return self.gpu_fraction < 1.0 or self.cpu_weights_mb > 0


def preset_placement(preset: Any) -> dict[str, Any]:
    """The preset's own placement (``ProfileRuntimePreset.placement``), or {}."""
    raw = getattr(preset, "placement", None)
    if raw is None:
        return {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(exclude_none=True)
    return {k: v for k, v in raw.items() if v is not None} if isinstance(raw, dict) else {}


def _resolve_offload(arch: Any, weights_mb: float, placement: dict[str, Any]) -> _Offload | None:
    """GPU share of a preset whose placement keeps layers or experts on CPU.

    ``gpu_layers`` below the model's layer count leaves the rest of the layers
    (their weights and their KV cache) in RAM. ``n_cpu_moe`` keeps the experts
    of the first N layers in RAM, approximated as ``_MOE_EXPERT_WEIGHT_SHARE``
    of those layers' weights. llama.cpp offloads the LAST ``gpu_layers``
    layers, so only experts of layers that are on GPU leave it.
    """
    if arch is None or not placement:
        return None
    layers = max(1, int(getattr(arch, "layers", 0) or 0))
    raw_gpu_layers = placement.get("gpu_layers")
    gpu_layers = layers if raw_gpu_layers is None else max(0, min(layers, int(raw_gpu_layers)))
    raw_moe = placement.get("n_cpu_moe")
    n_cpu_moe = max(0, min(layers, int(raw_moe))) if raw_moe is not None else 0
    if gpu_layers >= layers and n_cpu_moe <= 0:
        return None
    fraction = gpu_layers / layers
    gpu_weights = weights_mb * fraction
    if n_cpu_moe > 0:
        cpu_layers = layers - gpu_layers
        moe_layers_on_gpu = max(0, n_cpu_moe - cpu_layers)
        gpu_weights -= weights_mb * moe_layers_on_gpu / layers * _MOE_EXPERT_WEIGHT_SHARE
    gpu_weights = max(0.0, gpu_weights)
    return _Offload(
        gpu_fraction=fraction,
        gpu_weights_mb=gpu_weights,
        cpu_weights_mb=max(0.0, weights_mb - gpu_weights),
        moe_estimate=n_cpu_moe > 0,
    )


def _gpu_arch(arch: Any, offload: _Offload | None) -> Any:
    """``arch`` with its per-token KV and recurrent state cut to the GPU share."""
    if arch is None or offload is None or offload.gpu_fraction >= 1.0:
        return arch
    elements = arch.kv_bytes_per_token(1.0)
    # 0 would mean "no override" (full KV); keep a token's worth so the
    # planner still sees a KV cost for a model with every layer on CPU.
    scaled = max(2, int(round(elements * offload.gpu_fraction)))
    return dataclasses.replace(
        arch,
        kv_elements_per_token_override=scaled,
        rs_cache_mb_per_slot=arch.rs_cache_mb_per_slot * offload.gpu_fraction,
    )


def _footprint(
    backend: str,
    arch: Any,
    weights_mb: float,
    *,
    effective: int,
    parallelism: int,
    kv_bytes: float,
    unified_pool: bool,
    gpus: list[dict[str, int]],
    gpu_memory_utilization: float,
    offload: _Offload | None = None,
    ratios: list[float] | None = None,
) -> dict[str, Any]:
    """Estimated VRAM of a resolved preset, total and per target GPU.

    For the model editor: what a load with this preset would occupy, split
    across its GPUs in proportion to their size (llama.cpp's even split) or
    to ``ratios`` (the preset's tensor_split) when given for every GPU.
    vLLM/SGLang reserve their memory fraction of each GPU up front.

    With ``offload`` the GPU figures only count what stays on GPU, and
    ``cpu_mb`` estimates the RAM taken by the rest.
    """
    if effective <= 0 or not gpus:
        return {}
    if backend in ("vllm", "sglang"):
        per_gpu = {
            int(g["index"]): int(int(g["total_mb"]) * gpu_memory_utilization) for g in gpus
        }
        return {
            "estimated_vram_mb": sum(per_gpu.values()),
            "vram_by_gpu": per_gpu,
            "vram_breakdown_by_gpu": {g: {"engine": mb} for g, mb in per_gpu.items()},
        }
    if arch is None:
        return {}
    tokens = effective if unified_pool else effective * max(1, parallelism)
    slots = max(1, parallelism)
    gpu_arch = _gpu_arch(arch, offload)
    gpu_weights = offload.gpu_weights_mb if offload is not None else weights_mb
    parts = {
        "weights": gpu_weights,
        "kv": vp.kv_vram_mb(gpu_arch, tokens, kv_bytes),
        "rs_cache": vp.rs_cache_vram_mb(gpu_arch, slots=slots),
        "fixed": vp.fixed_overhead_vram_mb(arch),
    }
    total = sum(parts.values()) + vp.DEFAULT_OVERHEAD_MB
    if ratios is not None and len(ratios) == len(gpus) and sum(ratios) > 0:
        sizes = [max(0.0, float(r)) for r in ratios]
    else:
        sizes = [max(1, int(g["total_mb"])) for g in gpus]
    breakdown: dict[int, dict[str, int]] = {}
    for pos, (g, size) in enumerate(zip(gpus, sizes, strict=True)):
        share = size / sum(sizes)
        gpu_parts = {k: int(v * share) for k, v in parts.items() if v}
        if pos == 0:
            # The base overhead (CUDA context, buffers) sits on the main GPU.
            gpu_parts["overhead"] = vp.DEFAULT_OVERHEAD_MB
        breakdown[int(g["index"])] = gpu_parts
    per_gpu = {g: sum(p.values()) for g, p in breakdown.items()}
    out: dict[str, Any] = {
        "estimated_vram_mb": int(total),
        "vram_by_gpu": per_gpu,
        "vram_breakdown_by_gpu": breakdown,
    }
    if offload is not None and offload.active:
        cpu_kv = vp.kv_vram_mb(arch, tokens, kv_bytes) - parts["kv"]
        cpu_rs = vp.rs_cache_vram_mb(arch, slots=slots) - parts["rs_cache"]
        cpu_mb = int(offload.cpu_weights_mb + max(0.0, cpu_kv) + max(0.0, cpu_rs))
        if cpu_mb > 0:
            out["cpu_mb"] = cpu_mb
    return out


def _plan_preset_capacity(
    backend: str,
    arch: Any,
    weights_mb: float,
    *,
    gpu_total_mb: int,
    preset: Any,
    effective_parallelism: int,
    gpu_memory_utilization: float,
    kv_dtype_bytes: float,
    rs_rollback_snapshots: int,
    unified_pool: bool,
    offload: _Offload | None = None,
) -> dict[str, Any]:
    """Plan one preset, translating unified aggregate pools for the planner.

    ``vram_planner`` models equal per-slot context because that remains useful
    for fixed slot overhead.  For unified KV we feed it ``pool / slots`` for
    memory accounting, then translate the result back to aggregate tokens.

    ``offload`` plans only what stays on GPU: the GPU share of the weights and
    a KV cost per token cut to the layers on GPU.
    """

    if offload is not None:
        arch = _gpu_arch(arch, offload)
        weights_mb = offload.gpu_weights_mb
    requested: int | str | None
    if preset.context_mode == "max":
        requested = "max"
    elif unified_pool:
        requested = math.ceil(int(preset.context_tokens or 0) / effective_parallelism)
    else:
        requested = preset.context_tokens
    plan = vp.plan_use_case(
        backend,
        arch,
        weights_mb,
        gpu_total_mb=gpu_total_mb,
        requested_context=requested,
        slots=effective_parallelism,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_dtype_bytes=kv_dtype_bytes,
        rs_rollback_snapshots=rs_rollback_snapshots,
    )
    if not unified_pool:
        return plan

    translated = dict(plan)
    aggregate_max = int(plan.get("max_context") or 0) * effective_parallelism
    aggregate_effective = int(plan.get("effective_context") or 0) * effective_parallelism
    if preset.context_mode != "max" and preset.context_tokens is not None:
        aggregate_effective = min(int(preset.context_tokens), aggregate_effective)
    translated["max_context"] = aggregate_max
    translated["effective_context"] = aggregate_effective
    return translated


def _apply_validated_context_override(
    *,
    preset: Any,
    policy: dict[str, Any],
    effective: int,
    hardware_max: int,
    warnings: list[str],
) -> tuple[int, int, list[str]]:
    """Trust an exact, explicitly benchmarked fixed pool for this preset."""

    measured = policy.get("validated_context_tokens")
    raw = measured.get(preset.preset_key) if isinstance(measured, dict) else None
    try:
        validated = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        validated = None
    if (
        validated is None
        or validated <= 0
        or preset.context_mode != "fixed"
        or int(preset.context_tokens or 0) != validated
    ):
        return effective, hardware_max, warnings
    return validated, max(hardware_max, validated), []


def resolve_effective_parallelism(state: Any, requested: int) -> tuple[int, str | None]:
    """Apply a model/runtime parallelism cap before sizing KV memory.

    Some runtimes accept a global parallel setting and still reduce particular
    architectures to one sequence. The cap lives in model ``extra_config`` so
    an Ollama upgrade can remove it without changing presets or code.
    """

    requested = max(1, int(requested))
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    capabilities = extra.get("runtime_capabilities")
    raw_cap = capabilities.get("max_parallelism") if isinstance(capabilities, dict) else None
    try:
        cap = max(1, int(raw_cap)) if raw_cap is not None else None
    except (TypeError, ValueError):
        cap = None
    if cap is None or requested <= cap:
        return requested, None
    effective = min(requested, cap)
    return (
        effective,
        f"el runtime limita el paralelismo solicitado p{requested} a p{effective}",
    )


def _ordered_candidates(
    preset: Any,
    state: Any,
    gpus: list[dict[str, int]],
) -> list[dict[str, int]]:
    by_index = {gpu["index"]: gpu for gpu in gpus}
    requested = list(preset.preferred_gpus or [])
    if preset.placement_policy in {"forced", "preferred"} and requested:
        return [by_index[index] for index in requested if index in by_index]
    preferred = getattr(state, "preferred_gpu", None)
    return sorted(
        gpus,
        key=lambda gpu: (
            0 if gpu["index"] == preferred else 1,
            -gpu["total_mb"],
            gpu["index"],
        ),
    )


def _preset_llama_section(
    state: Any,
    profile: Any,
    preset: Any,
    *,
    effective_parallelism: int,
    backend: str,
    placement: dict[str, Any],
) -> dict[str, Any]:
    """The ``llama_cpp`` section a single-GPU worker of this preset runs with,
    as far as the measured curve cares (see ``observed_vram.curve_key``)."""
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    section = copy.deepcopy(extra.get("llama_cpp") or {})
    overrides = getattr(profile, "load_overrides", None)
    if isinstance(overrides, dict) and isinstance(overrides.get("llama_cpp"), dict):
        section.update(copy.deepcopy(overrides["llama_cpp"]))
    section["parallel_slots"] = effective_parallelism
    cache_type = kv_dtype_for(backend, preset.kv_cache_dtype)[2]
    if cache_type:
        section["cache_type_k"] = cache_type
        section["cache_type_v"] = cache_type
    section.update({"split_mode": "none", "tensor_split": None})
    for key in ("gpu_layers", "n_cpu_moe"):
        if placement.get(key) is not None:
            section[key] = placement[key]
    return section


def _apply_measured_curve(
    plan: dict[str, Any],
    curve: observed_vram.VramCurve | None,
    *,
    capacity_mb: int,
    preset: Any,
) -> dict[str, Any]:
    """Let the measured curve raise what the formula says fits, never lower it.

    Pool tokens (unified KV) on both sides: the curve is keyed by llama.cpp's
    total ``--ctx-size``. Only within the measured range: past the largest
    measured context the formula keeps deciding.
    """
    if curve is None:
        return plan
    measured_max = curve.max_context_within(int(capacity_mb))
    formula_max = int(plan.get("max_context") or 0)
    if measured_max <= formula_max:
        return plan
    out = dict(plan)
    out["max_context"] = measured_max
    out["measured_curve"] = curve
    if preset.context_mode == "max" or preset.context_tokens is None:
        out["effective_context"] = measured_max
        out["warnings"] = []
        return out
    requested = int(preset.context_tokens)
    out["effective_context"] = min(requested, measured_max)
    out["warnings"] = (
        []
        if requested <= measured_max
        else [
            f"contexto {requested} no cabe con {plan.get('slots')} slots "
            f"(máx medido {measured_max}); reducido a {measured_max}"
        ]
    )
    return out


@dataclass
class _PlanContext:
    """Everything shared by every preset of one profile resolution."""

    profile: Any
    backend: str
    arch: Any
    weights_mb: float
    gpus: list[dict[str, int]]
    gpu_memory_utilization: float
    rs_rollback_snapshots: int
    unified_pool: bool
    model_max: int | None
    policy: dict[str, Any]
    policy_max: int | None
    placement_aware: bool


def _unresolved_row(
    preset: Any,
    *,
    model_max: int | None,
    effective_parallelism: int,
    target_gpus: list[int],
    warning: str,
) -> dict[str, Any]:
    return {
        "preset_key": preset.preset_key,
        "resolved": False,
        "requested_context": preset.context_tokens,
        "effective_context": None,
        "model_max": model_max,
        "hardware_max": None,
        "requested_parallelism": preset.parallelism,
        "effective_parallelism": effective_parallelism,
        "target_gpus": target_gpus,
        "fits": False,
        "runtime_key": None,
        "warnings": [warning],
    }


def _clamp_effective(
    ctx: _PlanContext,
    preset: Any,
    plan: dict[str, Any],
) -> tuple[int, int, list[str]]:
    """Validated override, native-context cap and policy cap of one plan."""
    effective = int(plan["effective_context"] or 0)
    hardware_max = int(plan["max_context"] or 0)
    warnings = list(plan["warnings"])
    effective, hardware_max, warnings = _apply_validated_context_override(
        preset=preset,
        policy=ctx.policy,
        effective=effective,
        hardware_max=hardware_max,
        warnings=warnings,
    )
    if ctx.unified_pool and ctx.model_max is not None and effective > ctx.model_max:
        if preset.context_mode != "max":
            warnings.append(
                f"pool {effective} supera el contexto nativo {ctx.model_max}; "
                f"reducido a {ctx.model_max}"
            )
        effective = ctx.model_max
    if ctx.policy_max is not None and effective > ctx.policy_max:
        if preset.context_mode != "max":
            warnings.append(
                f"contexto {effective} supera el límite validado {ctx.policy_max}; "
                f"reducido a {ctx.policy_max}"
            )
        effective = ctx.policy_max
    return effective, hardware_max, warnings


def _resolved_row(
    ctx: _PlanContext,
    preset: Any,
    *,
    effective: int,
    hardware_max: int,
    warnings: list[str],
    fits: bool,
    effective_parallelism: int,
    target_gpus: list[int],
    footprint_gpus: list[dict[str, int]],
    kv_bytes: float,
    placement: dict[str, Any],
    offload: _Offload | None,
    curve: observed_vram.VramCurve | None = None,
) -> dict[str, Any]:
    runtime_key = None
    if effective > 0:
        runtime_key = build_runtime_key(
            model_id=ctx.profile.base_model_id,
            backend_type=ctx.backend,
            effective_context=effective,
            parallelism=effective_parallelism,
            kv_cache_dtype=preset.kv_cache_dtype,
            placement_policy=preset.placement_policy,
            gpu_indices=tuple(target_gpus),
            load_flags=ctx.profile.load_overrides,
            placement=placement or None,
        )
    ratios = placement.get("tensor_split") if len(footprint_gpus) > 1 else None
    footprint = _footprint(
        ctx.backend,
        ctx.arch,
        ctx.weights_mb,
        effective=effective,
        parallelism=effective_parallelism,
        kv_bytes=kv_bytes,
        unified_pool=ctx.unified_pool,
        gpus=list(footprint_gpus),
        gpu_memory_utilization=ctx.gpu_memory_utilization,
        offload=offload,
        ratios=ratios if isinstance(ratios, list) else None,
    )
    if curve is not None and effective > 0 and len(footprint_gpus) == 1:
        measured_mb = curve.predict_mb(effective)
        gpu_index = int(footprint_gpus[0]["index"])
        footprint = {
            "estimated_vram_mb": measured_mb,
            "vram_by_gpu": {gpu_index: measured_mb},
            "vram_breakdown_by_gpu": {gpu_index: {"measured": measured_mb}},
            "vram_source": "measured",
        }
    return {
        "preset_key": preset.preset_key,
        "resolved": True,
        "requested_context": preset.context_tokens,
        "effective_context": effective,
        "model_max": ctx.model_max,
        "hardware_max": hardware_max,
        "policy_max": ctx.policy_max,
        "requested_parallelism": preset.parallelism,
        "effective_parallelism": effective_parallelism,
        "target_gpus": list(target_gpus),
        "fits": fits,
        "runtime_key": runtime_key,
        "warnings": warnings,
        **footprint,
    }


def _resolve_multi_gpu_row(
    ctx: _PlanContext,
    preset: Any,
    target_gpus: list[int],
    *,
    effective_parallelism: int,
    parallelism_warning: str | None,
    placement: dict[str, Any],
    offload: _Offload | None,
) -> dict[str, Any]:
    """A preset spread over ``target_gpus`` (in main-GPU-first order)."""
    # Multi-GPU spread: solo soportado hoy para backends con reparto
    # proporcional real (Ollama con un pool ``shared`` que exponga
    # todas las GPUs + OLLAMA_SCHED_SPREAD=1; llama.cpp/bitnet con su
    # --split-mode layer por defecto). vLLM/sglang con tensor_parallel
    # genuino siguen devolviendo unresolved porque su planner de
    # sharding no está integrado.
    if ctx.backend not in _PROPORTIONAL_MULTI_GPU_BACKENDS:
        return _unresolved_row(
            preset,
            model_max=ctx.model_max,
            effective_parallelism=effective_parallelism,
            target_gpus=target_gpus,
            warning="la capacidad multi-GPU requiere el planner de sharding del backend",
        )
    by_index = {gpu["index"]: gpu for gpu in ctx.gpus}
    selected_gpus = [by_index[idx] for idx in target_gpus if idx in by_index]
    if len(selected_gpus) != len(target_gpus):
        return _unresolved_row(
            preset,
            model_max=ctx.model_max,
            effective_parallelism=effective_parallelism,
            target_gpus=target_gpus,
            warning=f"preferred_gpus={target_gpus} no están todas presentes",
        )
    virtual_total = sum(int(gpu["total_mb"]) for gpu in selected_gpus)
    kv_bytes = kv_dtype_for(ctx.backend, preset.kv_cache_dtype)[0]
    plan = _plan_preset_capacity(
        ctx.backend,
        ctx.arch,
        ctx.weights_mb,
        gpu_total_mb=virtual_total,
        preset=preset,
        effective_parallelism=effective_parallelism,
        gpu_memory_utilization=ctx.gpu_memory_utilization,
        kv_dtype_bytes=kv_bytes,
        rs_rollback_snapshots=ctx.rs_rollback_snapshots,
        unified_pool=ctx.unified_pool,
        offload=offload,
    )
    effective, hardware_max, warnings = _clamp_effective(ctx, preset, plan)
    if effective <= 0 and not warnings:
        warnings.append("el modelo y su KV cache no caben ni sumando ambas GPUs")
    fits = effective > 0 and not warnings
    if parallelism_warning:
        warnings.append(parallelism_warning)
    if offload is not None and offload.moe_estimate:
        warnings.append(_MOE_ESTIMATE_WARNING)
    return _resolved_row(
        ctx,
        preset,
        effective=effective,
        hardware_max=hardware_max,
        warnings=warnings,
        fits=fits,
        effective_parallelism=effective_parallelism,
        target_gpus=target_gpus,
        footprint_gpus=selected_gpus,
        kv_bytes=kv_bytes,
        placement=placement,
        offload=offload,
    )


def _resolve_single_gpu_row(
    ctx: _PlanContext,
    preset: Any,
    state: Any,
    *,
    effective_parallelism: int,
    parallelism_warning: str | None,
    placement: dict[str, Any],
    offload: _Offload | None,
    curves: dict[int, observed_vram.VramCurve | None] | None = None,
) -> dict[str, Any]:
    """A preset on the one GPU of its candidates where it fits best."""
    candidates = _ordered_candidates(preset, state, ctx.gpus)
    kv_bytes = kv_dtype_for(ctx.backend, preset.kv_cache_dtype)[0]
    evaluated: list[tuple[dict[str, int], dict[str, Any]]] = []
    for gpu in candidates:
        plan = _plan_preset_capacity(
            ctx.backend,
            ctx.arch,
            ctx.weights_mb,
            gpu_total_mb=gpu["total_mb"],
            preset=preset,
            effective_parallelism=effective_parallelism,
            gpu_memory_utilization=ctx.gpu_memory_utilization,
            kv_dtype_bytes=kv_bytes,
            rs_rollback_snapshots=ctx.rs_rollback_snapshots,
            unified_pool=ctx.unified_pool,
            offload=offload,
        )
        plan = _apply_measured_curve(
            plan,
            (curves or {}).get(int(gpu["index"])),
            capacity_mb=int(gpu["total_mb"]),
            preset=preset,
        )
        evaluated.append((gpu, plan))

    if not evaluated:
        return _unresolved_row(
            preset,
            model_max=ctx.model_max,
            effective_parallelism=effective_parallelism,
            target_gpus=list(preset.preferred_gpus or []),
            warning="no hay una GPU elegible para el preset",
        )

    if preset.placement_policy == "preferred":
        chosen_gpu, plan = next(
            (
                (gpu, candidate_plan)
                for gpu, candidate_plan in evaluated
                if candidate_plan["effective_context"] > 0
            ),
            evaluated[0],
        )
    else:
        chosen_gpu, plan = max(
            evaluated,
            key=lambda item: item[1]["effective_context"],
        )
    effective, hardware_max, warnings = _clamp_effective(ctx, preset, plan)
    if effective <= 0 and not warnings:
        warnings.append("el modelo y su KV cache no caben en la GPU elegida")
    fits = effective > 0 and not warnings
    if parallelism_warning:
        warnings.append(parallelism_warning)
    if offload is not None and offload.moe_estimate:
        warnings.append(_MOE_ESTIMATE_WARNING)
    return _resolved_row(
        ctx,
        preset,
        effective=effective,
        hardware_max=hardware_max,
        warnings=warnings,
        fits=fits,
        effective_parallelism=effective_parallelism,
        target_gpus=[chosen_gpu["index"]],
        footprint_gpus=[chosen_gpu],
        kv_bytes=kv_bytes,
        placement=placement,
        offload=offload,
        curve=plan.get("measured_curve"),
    )


async def _measured_curves(
    ctx: _PlanContext,
    state: Any,
    preset: Any,
    *,
    effective_parallelism: int,
    placement: dict[str, Any],
    offload: _Offload | None,
    **_unused: Any,
) -> dict[int, observed_vram.VramCurve | None]:
    """Measured VRAM curve of this preset's configuration on each candidate GPU.

    llama.cpp with a unified KV pool and every layer on GPU only: that is what
    the readings are recorded for, in the same pool units.
    """
    model_id = getattr(state, "model_id", None)
    if not model_id or not ctx.placement_aware or not ctx.unified_pool:
        return {}
    if offload is not None and offload.active:
        return {}
    section = _preset_llama_section(
        state,
        ctx.profile,
        preset,
        effective_parallelism=effective_parallelism,
        backend=ctx.backend,
        placement=placement,
    )
    curves: dict[int, observed_vram.VramCurve | None] = {}
    for gpu in _ordered_candidates(preset, state, ctx.gpus):
        index = int(gpu["index"])
        curves[index] = await observed_vram.lookup_curve(
            str(model_id), key=observed_vram.curve_key(section, [index])
        )
    return curves


def _spread_order(preset: Any, gpus: list[dict[str, int]]) -> list[int]:
    """Every GPU, the preset's preferred ones first (they become main GPU)."""
    requested = [int(i) for i in (preset.preferred_gpus or [])]
    known = [int(gpu["index"]) for gpu in gpus]
    return [i for i in requested if i in known] + [i for i in known if i not in requested]


def _split_policy(placement: dict[str, Any]) -> str:
    raw = placement.get("split_policy")
    return raw if raw in _SPLIT_POLICIES else "single"


async def resolve_profile_runtime_capacities(
    profile: Any,
    state: Any,
    gpu_states: list[Any],
) -> list[dict[str, Any]]:
    """Return a no-side-effect capacity snapshot for every profile preset.

    Single-GPU placements are resolved exactly with the existing VRAM planner.
    Multi-GPU placements (``forced`` over several GPUs, or a preset placement
    with ``split_policy`` ``always``, or ``when_needed`` when no single GPU
    fits) sum the GPUs' capacity for backends with a proportional split
    (llama.cpp family, bitnet, Ollama) and stay unresolved for the others,
    where summing VRAM would advertise unsafe values.

    A preset ``placement`` with ``gpu_layers`` below the model's layers or
    with ``n_cpu_moe`` plans only the GPU share and reports ``cpu_mb``.
    """

    presets = sorted(
        profile.runtime_presets,
        key=lambda item: (item.priority, item.preset_key),
    )
    if not presets:
        return []

    resolution = await resolve_arch_and_weights(state)
    gpus = [_gpu_dict(gpu) for gpu in gpu_states]
    if resolution.arch is None:
        return [
            {
                "preset_key": preset.preset_key,
                "resolved": False,
                "requested_context": preset.context_tokens,
                "effective_context": None,
                "model_max": None,
                "hardware_max": None,
                "requested_parallelism": preset.parallelism,
                "effective_parallelism": resolve_effective_parallelism(state, preset.parallelism)[
                    0
                ],
                "target_gpus": [],
                "fits": False,
                "runtime_key": None,
                "warnings": [resolution.note or "arquitectura no disponible"],
            }
            for preset in presets
        ]

    backend = str(state.backend_type or "").lower()
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    gpu_memory_utilization = float(
        (extra.get("vllm") or {}).get("gpu_memory_utilization")
        or settings.vllm_gpu_memory_utilization
    )
    policy = profile.runtime_policy if isinstance(profile.runtime_policy, dict) else {}
    raw_policy_max = policy.get("max_context_tokens")
    try:
        policy_max = int(raw_policy_max) if raw_policy_max is not None else None
    except (TypeError, ValueError):
        policy_max = None
    if policy_max is not None and policy_max <= 0:
        policy_max = None
    ctx = _PlanContext(
        profile=profile,
        backend=backend,
        arch=resolution.arch,
        weights_mb=resolution.weights_mb,
        gpus=gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        # Set for self-speculative loads (--spec-type draft-mtp and friends) —
        # widens the hybrid rs_cache deduction and adds the draft model's own
        # KV+compute overhead. Same resolution model_manager_helpers.py uses at
        # actual load time; without it this display path (and the "fits"/
        # "hardware_max" it reports) would silently drift back to the
        # pre-2026-10-27 over-optimistic numbers that caused the OOM.
        rs_rollback_snapshots=_resolve_rs_rollback_snapshots(state),
        unified_pool=_uses_unified_kv_pool(state),
        model_max=resolution.arch.context_length or None,
        policy=policy,
        policy_max=policy_max,
        # The preset placement is llama.cpp's (--n-gpu-layers, --n-cpu-moe,
        # --split-mode): other backends ignore it at load, so it must not
        # change their plan either.
        placement_aware=is_llama_cpp_family(backend),
    )
    rows: list[dict[str, Any]] = []

    for preset in presets:
        effective_parallelism, parallelism_warning = resolve_effective_parallelism(
            state, preset.parallelism
        )
        placement = preset_placement(preset) if ctx.placement_aware else {}
        offload = _resolve_offload(ctx.arch, ctx.weights_mb, placement)
        split_policy = _split_policy(placement)
        common = {
            "effective_parallelism": effective_parallelism,
            "parallelism_warning": parallelism_warning,
            "placement": placement,
            "offload": offload,
        }
        requested_gpus = list(preset.preferred_gpus or [])

        spread: list[int] | None = None
        if preset.placement_policy == "forced" and len(requested_gpus) > 1:
            spread = requested_gpus
        elif split_policy == "always" and preset.placement_policy != "forced":
            order = _spread_order(preset, gpus)
            spread = order if len(order) > 1 else None

        if spread is not None:
            rows.append(_resolve_multi_gpu_row(ctx, preset, spread, **common))
            continue

        curves = await _measured_curves(ctx, state, preset, **common)
        row = _resolve_single_gpu_row(ctx, preset, state, curves=curves, **common)
        if (
            split_policy == "when_needed"
            and preset.placement_policy != "forced"
            and not row.get("fits")
            and len(gpus) > 1
        ):
            # No single GPU holds it: resolve the same preset over every GPU,
            # the very path a forced multi-GPU preset takes.
            multi = _resolve_multi_gpu_row(ctx, preset, _spread_order(preset, gpus), **common)
            if multi.get("fits"):
                row = multi
        rows.append(row)
    return rows
