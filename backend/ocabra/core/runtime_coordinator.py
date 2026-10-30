"""Request-time coordination for profile runtime presets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import structlog

from ocabra.config import settings
from ocabra.core.model_manager_helpers import compute_worker_key

logger = structlog.get_logger(__name__)
from ocabra.core.runtime_preset_capacity import runtime_capacity_cache
from ocabra.core.runtime_presets import (
    ContextCapacityError,
    ResolvedRuntimePreset,
    select_runtime_preset,
)


@dataclass(frozen=True)
class RuntimePlan:
    worker_key: str
    load_overrides: dict[str, Any] | None
    preferred_gpu: int | None
    load_policy: str | None
    preset_key: str | None
    runtime_key: str | None
    effective_context: int | None
    required_tokens: int | None
    selection_reason: str | None


def _text_size(value: Any) -> tuple[int, int]:
    """Return (characters, words) from the textual portions of an OpenAI body."""

    if isinstance(value, str):
        return len(value), len(value.split())
    if isinstance(value, list):
        parts = [_text_size(item) for item in value]
        return sum(item[0] for item in parts), sum(item[1] for item in parts)
    if not isinstance(value, dict):
        return (0, 0)
    chars = words = 0
    for key, item in value.items():
        if key in {"image_url", "input_audio", "video_url"}:
            continue
        item_chars, item_words = _text_size(item)
        chars += item_chars
        words += item_words
    return chars, words


def estimate_request_token_budget(
    body: dict[str, Any], *, default_output_reservation: int | None = None
) -> int:
    """Conservatively estimate prompt plus reserved output tokens.

    Exact tokenization is model-specific and may require loading tokenizer code.
    The estimator uses both character and word bounds with headroom, then adds
    the client's complete output reservation. It intentionally rounds upward.

    ``default_output_reservation`` overrides the global
    ``settings.runtime_preset_default_output_reservation_tokens`` fallback —
    used by callers that know this profile's own
    ``runtime_policy.default_output_reservation_tokens`` (see
    ``plan_profile_runtime``). 2026-10-29: a single global 4096-token default
    for every model, reasoning or not, meant a client that didn't pass
    ``max_tokens`` against a heavy-reasoning profile (gemma4, qwen3.8) could
    burn its entire reservation on ``reasoning_content`` and get cut with
    ``finish_reason: "length"`` and empty ``content`` — the preset system
    had no way to know it needed more room than the global default assumed.
    """
    default_reservation = (
        default_output_reservation
        if default_output_reservation is not None
        else settings.runtime_preset_default_output_reservation_tokens
    )

    if isinstance(body.get("messages"), list):
        source: Any = body["messages"]
        structural_overhead = 16 * len(body["messages"])
    else:
        source = body.get("prompt", body.get("input", ""))
        structural_overhead = 8
    chars, words = _text_size(source)
    tools_chars, tools_words = _text_size(body.get("tools", []))
    chars += tools_chars
    words += tools_words
    prompt_estimate = max(math.ceil(chars / 3), math.ceil(words * 1.5))
    prompt_estimate = math.ceil(prompt_estimate * 1.15) + structural_overhead

    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    output_value = body.get(
        "max_completion_tokens",
        body.get(
            "max_tokens",
            options.get("num_predict", default_reservation),
        ),
    )
    try:
        reserved_output = max(1, int(output_value))
    except (TypeError, ValueError):
        reserved_output = default_reservation
    estimated = max(1, prompt_estimate + reserved_output)
    try:
        requested_context = max(0, int(options.get("num_ctx") or 0))
    except (TypeError, ValueError):
        requested_context = 0
    return max(estimated, requested_context)


def _requested_preset_key(body: dict[str, Any]) -> str | None:
    extension = body.get("ocabra")
    if not isinstance(extension, dict):
        return None
    value = extension.get("context_preset")
    return str(value).strip() if value else None


async def plan_profile_runtime(
    profile: Any,
    model_manager: Any,
    request_body: dict[str, Any] | None,
) -> RuntimePlan:
    """Resolve and select the runtime variant for one profile request."""

    enabled_presets = [
        preset
        for preset in getattr(profile, "runtime_presets", [])
        if preset.enabled
    ]
    if not enabled_presets:
        worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
        return RuntimePlan(
            worker_key=worker_key,
            load_overrides=profile.load_overrides,
            preferred_gpu=None,
            load_policy=None,
            preset_key=None,
            runtime_key=None,
            effective_context=None,
            required_tokens=None,
            selection_reason=None,
        )

    state = await model_manager.get_state(profile.base_model_id)
    if state is None:
        raise ContextCapacityError(f"base model '{profile.base_model_id}' is not configured")
    gpu_manager = model_manager.gpu_manager
    gpu_states = await gpu_manager.get_all_states() if gpu_manager is not None else []
    rows = await runtime_capacity_cache.resolve(profile, state, gpu_states)
    rows_by_key = {row["preset_key"]: row for row in rows}

    candidates: list[ResolvedRuntimePreset] = []
    preset_by_key = {preset.preset_key: preset for preset in enabled_presets}
    for preset in enabled_presets:
        row = rows_by_key.get(preset.preset_key)
        if not row or not row.get("resolved") or not row.get("fits"):
            continue
        candidates.append(
            ResolvedRuntimePreset(
                preset_key=preset.preset_key,
                effective_context=int(row["effective_context"]),
                parallelism=int(row.get("effective_parallelism") or preset.parallelism),
                kv_cache_dtype=preset.kv_cache_dtype,
                placement_policy=preset.placement_policy,
                gpu_indices=tuple(row.get("target_gpus") or ()),
                execution_policy=preset.execution_policy,
                enabled=preset.enabled,
                is_default=preset.is_default,
                priority=preset.priority,
                runtime_key=row.get("runtime_key"),
            )
        )

    if not candidates:
        logger.warning(
            "runtime_plan_no_candidates",
            profile_id=profile.profile_id,
            base_model_id=profile.base_model_id,
            enabled_preset_keys=[preset.preset_key for preset in enabled_presets],
            fits_by_key={row["preset_key"]: row.get("fits") for row in rows},
        )
        raise ContextCapacityError("profile has no enabled runtime preset that fits")

    body = request_body or {}
    policy = profile.runtime_policy if isinstance(profile.runtime_policy, dict) else {}
    raw_reservation = policy.get("default_output_reservation_tokens")
    try:
        default_output_reservation = (
            int(raw_reservation) if raw_reservation is not None else None
        )
    except (TypeError, ValueError):
        default_output_reservation = None
    required_tokens = estimate_request_token_budget(
        body, default_output_reservation=default_output_reservation
    )
    explicit_key = _requested_preset_key(body)
    loaded_keys: set[str] = set()
    for runtime_state in await model_manager.list_states(include_derived=True):
        if runtime_state.status.value != "loaded":
            continue
        extra = runtime_state.extra_config if isinstance(runtime_state.extra_config, dict) else {}
        metadata = extra.get("_runtime")
        if isinstance(metadata, dict) and metadata.get("runtime_key"):
            loaded_keys.add(str(metadata["runtime_key"]))

    if explicit_key:
        explicit = next((item for item in candidates if item.preset_key == explicit_key), None)
        if explicit is None:
            raise ContextCapacityError(
                f"runtime preset '{explicit_key}' is disabled, unavailable, or does not fit"
            )
        if explicit.effective_context < required_tokens:
            raise ContextCapacityError(
                f"runtime preset '{explicit_key}' cannot cover {required_tokens} tokens"
            )
        chosen = explicit
        reason = "explicit"
    else:
        selected = select_runtime_preset(
            candidates,
            required_tokens=required_tokens,
            loaded_runtime_keys=loaded_keys,
            selection=str(policy.get("selection", "prefer_loaded")),
        )
        chosen = selected.preset
        reason = selected.reason

    logger.info(
        "runtime_plan_selected",
        profile_id=profile.profile_id,
        base_model_id=profile.base_model_id,
        preset_key=chosen.preset_key,
        selection_reason=reason,
        required_tokens=required_tokens,
        effective_context=chosen.effective_context,
        parallelism=chosen.parallelism,
        gpu_indices=list(chosen.gpu_indices),
        candidate_presets=[item.preset_key for item in candidates],
        fits_by_preset={row["preset_key"]: bool(row.get("fits")) for row in rows},
        loaded_runtime_keys=sorted(loaded_keys),
        explicit=bool(explicit_key),
    )

    preset = preset_by_key[chosen.preset_key]
    overrides = dict(profile.load_overrides or {})
    overrides["use_case"] = {
        "context": chosen.effective_context,
        "slots": chosen.parallelism,
        "kv_dtype": preset.kv_cache_dtype,
    }
    overrides["_runtime"] = {
        "profile_id": profile.profile_id,
        "base_model_id": profile.base_model_id,
        "preset_key": preset.preset_key,
        "runtime_key": chosen.runtime_key,
        "effective_context": chosen.effective_context,
        "requested_parallelism": preset.parallelism,
        "parallelism": chosen.parallelism,
        "kv_cache_dtype": preset.kv_cache_dtype,
        "placement_policy": preset.placement_policy,
        "execution_policy": preset.execution_policy,
        # Multi-GPU hint para los backends con pool shared (Ollama). El
        # coordinator no sabe pools, pero sí sabe que este preset pide >1 GPU.
        # ``_select_pool`` del backend usa esta lista para elegir un pool que
        # exponga todas estas GPUs juntas (``multi_gpu=true`` + ``gpus`` ⊇).
        "gpu_indices": list(chosen.gpu_indices),
        "tracking_worker_key": compute_worker_key(
            profile.base_model_id,
            profile.load_overrides,
        ),
        "default_context": next(
            (
                item.effective_context
                for item in candidates
                if item.is_default
            ),
            chosen.effective_context,
        ),
        "restore_default_when_idle": bool(
            (profile.runtime_policy or {}).get("restore_default_when_idle", True)
            if isinstance(profile.runtime_policy, dict)
            else True
        ),
        "downgrade_grace_seconds": int(
            (profile.runtime_policy or {}).get("downgrade_grace_seconds", 600)
            if isinstance(profile.runtime_policy, dict)
            else 600
        ),
        "minimum_residency_seconds": int(
            (profile.runtime_policy or {}).get("minimum_residency_seconds", 120)
            if isinstance(profile.runtime_policy, dict)
            else 120
        ),
        "small_requests_extend_large_ttl": bool(
            (profile.runtime_policy or {}).get("small_requests_extend_large_ttl", False)
            if isinstance(profile.runtime_policy, dict)
            else False
        ),
    }
    worker_key = compute_worker_key(profile.base_model_id, overrides)
    preferred_gpu = chosen.gpu_indices[0] if len(chosen.gpu_indices) == 1 else None
    # load_policy lives on the base model (ModelConfig), not per-preset
    # (2026-10-30 — presets used to carry their own load_policy, which
    # unconditionally overrode the base model's: setting a model to "pin"
    # had no real effect once a request resolved through a profile, since
    # every preset defaulted to "warm"/"on_demand" and that value always
    # won). Only the DEFAULT preset for a profile inherits the base model's
    # residency policy; any other preset (a request that needed a bigger
    # context than usual) is always on_demand — a one-off excursion outside
    # the steady-state configuration shouldn't become pinned/warm residency.
    worker_load_policy = state.load_policy.value if chosen.is_default else "on_demand"
    return RuntimePlan(
        worker_key=worker_key,
        load_overrides=overrides,
        preferred_gpu=preferred_gpu,
        load_policy=worker_load_policy,
        preset_key=preset.preset_key,
        runtime_key=chosen.runtime_key,
        effective_context=chosen.effective_context,
        required_tokens=required_tokens,
        selection_reason=reason,
    )
