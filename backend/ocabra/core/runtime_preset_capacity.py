"""Resolve persisted runtime presets against current model/GPU capacity."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

from ocabra.config import settings
from ocabra.core import vram_planner as vp
from ocabra.core.model_manager_helpers import (
    _resolve_rs_rollback_snapshots,
    resolve_llama_cpp_option,
)
from ocabra.core.model_ref import is_llama_cpp_family
from ocabra.core.runtime_presets import build_runtime_key
from ocabra.core.vram_capacity import KV_DTYPES, resolve_arch_and_weights

# Backends whose multi-GPU placement is a real, proportional spread across
# the requested GPUs (Ollama's shared-pool mode, llama.cpp/bitnet's default
# --split-mode layer) rather than a sharding scheme this planner would have
# to model explicitly (vLLM/sglang tensor-parallel). For these, summing the
# requested GPUs' capacity with the same single-GPU formula is a faithful
# model — validated empirically for llama_cpp against a real asymmetric
# 3060+3090 load (2026-10-26).
_PROPORTIONAL_MULTI_GPU_BACKENDS = {"ollama", "llama_cpp", "llama_cpp_mks", "bitnet"}


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
                preset.enabled,
                preset.is_default,
                preset.priority,
            )
            for preset in profile.runtime_presets
        )
        gpus = tuple(
            sorted((int(gpu.index), int(gpu.total_vram_mb)) for gpu in gpu_states)
        )
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


def _gpu_dict(gpu: Any) -> dict[str, int]:
    return {
        "index": int(gpu.index),
        "total_mb": int(gpu.total_vram_mb),
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
) -> dict[str, Any]:
    """Plan one preset, translating unified aggregate pools for the planner.

    ``vram_planner`` models equal per-slot context because that remains useful
    for fixed slot overhead.  For unified KV we feed it ``pool / slots`` for
    memory accounting, then translate the result back to aggregate tokens.
    """

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


async def resolve_profile_runtime_capacities(
    profile: Any,
    state: Any,
    gpu_states: list[Any],
) -> list[dict[str, Any]]:
    """Return a no-side-effect capacity snapshot for every profile preset.

    Single-GPU placements are resolved exactly with the existing VRAM planner.
    Explicit multi-GPU presets remain unresolved until the backend-specific
    sharding planner is available; summing VRAM would advertise unsafe values.
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
                "effective_parallelism": resolve_effective_parallelism(
                    state, preset.parallelism
                )[0],
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
    # Set for self-speculative loads (--spec-type draft-mtp and friends) —
    # widens the hybrid rs_cache deduction and adds the draft model's own
    # KV+compute overhead. Same resolution model_manager_helpers.py uses at
    # actual load time; without it this display path (and the "fits"/
    # "hardware_max" it reports) would silently drift back to the
    # pre-2026-10-27 over-optimistic numbers that caused the OOM.
    rs_rollback_snapshots = _resolve_rs_rollback_snapshots(state)
    unified_pool = _uses_unified_kv_pool(state)
    model_max = resolution.arch.context_length or None
    policy = profile.runtime_policy if isinstance(profile.runtime_policy, dict) else {}
    raw_policy_max = policy.get("max_context_tokens")
    try:
        policy_max = int(raw_policy_max) if raw_policy_max is not None else None
    except (TypeError, ValueError):
        policy_max = None
    if policy_max is not None and policy_max <= 0:
        policy_max = None
    rows: list[dict[str, Any]] = []

    for preset in presets:
        effective_parallelism, parallelism_warning = resolve_effective_parallelism(
            state, preset.parallelism
        )
        requested_gpus = list(preset.preferred_gpus or [])
        if preset.placement_policy == "forced" and len(requested_gpus) > 1:
            # Multi-GPU spread: solo soportado hoy para backends con reparto
            # proporcional real (Ollama con un pool ``shared`` que exponga
            # todas las GPUs + OLLAMA_SCHED_SPREAD=1; llama.cpp/bitnet con su
            # --split-mode layer por defecto). vLLM/sglang con tensor_parallel
            # genuino siguen devolviendo unresolved porque su planner de
            # sharding no está integrado.
            if backend not in _PROPORTIONAL_MULTI_GPU_BACKENDS:
                rows.append(
                    {
                        "preset_key": preset.preset_key,
                        "resolved": False,
                        "requested_context": preset.context_tokens,
                        "effective_context": None,
                        "model_max": model_max,
                        "hardware_max": None,
                        "requested_parallelism": preset.parallelism,
                        "effective_parallelism": effective_parallelism,
                        "target_gpus": requested_gpus,
                        "fits": False,
                        "runtime_key": None,
                        "warnings": [
                            "la capacidad multi-GPU requiere el planner de sharding del backend"
                        ],
                    }
                )
                continue
            by_index = {gpu["index"]: gpu for gpu in gpus}
            selected_gpus = [by_index[idx] for idx in requested_gpus if idx in by_index]
            if len(selected_gpus) != len(requested_gpus):
                rows.append(
                    {
                        "preset_key": preset.preset_key,
                        "resolved": False,
                        "requested_context": preset.context_tokens,
                        "effective_context": None,
                        "model_max": model_max,
                        "hardware_max": None,
                        "requested_parallelism": preset.parallelism,
                        "effective_parallelism": effective_parallelism,
                        "target_gpus": requested_gpus,
                        "fits": False,
                        "runtime_key": None,
                        "warnings": [
                            f"preferred_gpus={requested_gpus} no están todas presentes"
                        ],
                    }
                )
                continue
            virtual_gpu = {
                "index": -1,
                "total_mb": sum(int(gpu["total_mb"]) for gpu in selected_gpus),
            }
            kv_bytes = KV_DTYPES.get(str(preset.kv_cache_dtype).lower(), (2.0, None, "f16"))[0]
            plan = _plan_preset_capacity(
                backend,
                resolution.arch,
                resolution.weights_mb,
                gpu_total_mb=virtual_gpu["total_mb"],
                preset=preset,
                effective_parallelism=effective_parallelism,
                gpu_memory_utilization=gpu_memory_utilization,
                kv_dtype_bytes=kv_bytes,
                rs_rollback_snapshots=rs_rollback_snapshots,
                unified_pool=unified_pool,
            )
            effective = int(plan["effective_context"] or 0)
            hardware_max = int(plan["max_context"] or 0)
            warnings = list(plan["warnings"])
            effective, hardware_max, warnings = _apply_validated_context_override(
                preset=preset,
                policy=policy,
                effective=effective,
                hardware_max=hardware_max,
                warnings=warnings,
            )
            if unified_pool and model_max is not None and effective > model_max:
                if preset.context_mode != "max":
                    warnings.append(
                        f"pool {effective} supera el contexto nativo {model_max}; "
                        f"reducido a {model_max}"
                    )
                effective = model_max
            if policy_max is not None and effective > policy_max:
                if preset.context_mode != "max":
                    warnings.append(
                        f"contexto {effective} supera el límite validado {policy_max}; "
                        f"reducido a {policy_max}"
                    )
                effective = policy_max
            if effective <= 0 and not warnings:
                warnings.append(
                    "el modelo y su KV cache no caben ni sumando ambas GPUs"
                )
            fits = effective > 0 and not warnings
            if parallelism_warning:
                warnings.append(parallelism_warning)
            runtime_key = None
            if effective > 0:
                runtime_key = build_runtime_key(
                    model_id=profile.base_model_id,
                    backend_type=backend,
                    effective_context=effective,
                    parallelism=effective_parallelism,
                    kv_cache_dtype=preset.kv_cache_dtype,
                    placement_policy=preset.placement_policy,
                    gpu_indices=tuple(requested_gpus),
                    load_flags=profile.load_overrides,
                )
            rows.append(
                {
                    "preset_key": preset.preset_key,
                    "resolved": True,
                    "requested_context": preset.context_tokens,
                    "effective_context": effective,
                    "model_max": model_max,
                    "hardware_max": hardware_max,
                    "policy_max": policy_max,
                    "requested_parallelism": preset.parallelism,
                    "effective_parallelism": effective_parallelism,
                    "target_gpus": list(requested_gpus),
                    "fits": fits,
                    "runtime_key": runtime_key,
                    "warnings": warnings,
                }
            )
            continue

        candidates = _ordered_candidates(preset, state, gpus)
        kv_bytes = KV_DTYPES.get(str(preset.kv_cache_dtype).lower(), (2.0, None, "f16"))[0]
        evaluated: list[tuple[dict[str, int], dict[str, Any]]] = []
        for gpu in candidates:
            plan = _plan_preset_capacity(
                backend,
                resolution.arch,
                resolution.weights_mb,
                gpu_total_mb=gpu["total_mb"],
                preset=preset,
                effective_parallelism=effective_parallelism,
                gpu_memory_utilization=gpu_memory_utilization,
                kv_dtype_bytes=kv_bytes,
                rs_rollback_snapshots=rs_rollback_snapshots,
                unified_pool=unified_pool,
            )
            evaluated.append((gpu, plan))

        if not evaluated:
            rows.append(
                {
                    "preset_key": preset.preset_key,
                    "resolved": False,
                    "requested_context": preset.context_tokens,
                    "effective_context": None,
                    "model_max": model_max,
                    "hardware_max": None,
                    "requested_parallelism": preset.parallelism,
                    "effective_parallelism": effective_parallelism,
                    "target_gpus": requested_gpus,
                    "fits": False,
                    "runtime_key": None,
                    "warnings": ["no hay una GPU elegible para el preset"],
                }
            )
            continue

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
        effective = int(plan["effective_context"] or 0)
        hardware_max = int(plan["max_context"] or 0)
        warnings = list(plan["warnings"])
        effective, hardware_max, warnings = _apply_validated_context_override(
            preset=preset,
            policy=policy,
            effective=effective,
            hardware_max=hardware_max,
            warnings=warnings,
        )
        if unified_pool and model_max is not None and effective > model_max:
            if preset.context_mode != "max":
                warnings.append(
                    f"pool {effective} supera el contexto nativo {model_max}; "
                    f"reducido a {model_max}"
                )
            effective = model_max
        if policy_max is not None and effective > policy_max:
            if preset.context_mode != "max":
                warnings.append(
                    f"contexto {effective} supera el límite validado {policy_max}; "
                    f"reducido a {policy_max}"
                )
            effective = policy_max
        if effective <= 0 and not warnings:
            warnings.append("el modelo y su KV cache no caben en la GPU elegida")
        fits = effective > 0 and not warnings
        if parallelism_warning:
            warnings.append(parallelism_warning)
        runtime_key = None
        if effective > 0:
            runtime_key = build_runtime_key(
                model_id=profile.base_model_id,
                backend_type=backend,
                effective_context=effective,
                parallelism=effective_parallelism,
                kv_cache_dtype=preset.kv_cache_dtype,
                placement_policy=preset.placement_policy,
                gpu_indices=(chosen_gpu["index"],),
                load_flags=profile.load_overrides,
            )
        rows.append(
            {
                "preset_key": preset.preset_key,
                "resolved": True,
                "requested_context": preset.context_tokens,
                "effective_context": effective,
                "model_max": model_max,
                "hardware_max": hardware_max,
                "policy_max": policy_max,
                "requested_parallelism": preset.parallelism,
                "effective_parallelism": effective_parallelism,
                "target_gpus": [chosen_gpu["index"]],
                "fits": fits,
                "runtime_key": runtime_key,
                "warnings": warnings,
            }
        )
    return rows
