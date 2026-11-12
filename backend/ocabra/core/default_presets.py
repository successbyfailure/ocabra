"""Every LLM has at least one runtime preset ("normal").

With presets, a model's base config is never loaded on its own: every request
(and the manual "Load") picks a preset. A model without presets was the one
exception, which made the editor show the base as a fake rung. Since
2026-11-12 every LLM gets a ``normal`` preset built from what its base loads
today (same context, parallelism, KV precision and GPUs), so nothing changes
in how it runs, and the base becomes "what all rungs share".

Embedding and rerank models are left alone: their requests never go through
presets. Ollama has none either (one global server).

A preset the capacity planner judges as not fitting would make the profile
reject every request (the fit guard, see ``api/internal/profiles.py``), so a
seeded preset that does not fit is removed again and the model keeps loading
its base, with a warning in the log.
"""

from __future__ import annotations

from typing import Any

import structlog

from ocabra.config import settings
from ocabra.core.model_ref import is_llama_cpp_family

logger = structlog.get_logger(__name__)

PRESET_BACKENDS = {"llama_cpp", "llamacpp_mks", "bitnet", "vllm", "sglang"}
PRESET_KEY = "normal"


def _kv_label(cache_type: Any) -> str:
    value = str(cache_type or "").lower()
    if not value or value in {"f16", "fp16", "bf16", "auto"}:
        return "fp16"
    if value.startswith("fp8"):
        return "fp8"
    if value.startswith("q4"):
        return "q4"
    # q8_0, q5_x: q8 sizes at least as much as they use.
    return "q8"


def preset_from_base(state: Any, gpu_indices: list[int]) -> dict[str, Any] | None:
    """The ``normal`` preset equivalent to what ``state`` loads today."""
    backend = state.backend_type
    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    placement: dict[str, Any] = {"placement_policy": "auto", "preferred_gpus": None}
    if state.preferred_gpu is not None:
        placement = {"placement_policy": "preferred", "preferred_gpus": [state.preferred_gpu]}

    if is_llama_cpp_family(backend) or backend == "bitnet":
        section_name = "bitnet" if backend == "bitnet" else "llama_cpp"
        sec = extra.get(section_name) if isinstance(extra.get(section_name), dict) else {}
        ctx = int(sec.get("ctx_size") or (
            settings.bitnet_ctx_size if backend == "bitnet" else settings.llama_cpp_ctx_size
        ))
        slots = int(sec.get("parallel" if backend == "bitnet" else "parallel_slots") or 1)
        slots = max(1, min(64, slots))
        unified = sec.get("kv_unified") is True
        # Presets express the pool (kv_unified) or the per-slot context; the
        # base's ctx_size is always the total.
        context = ctx if unified else max(1, ctx // slots)
        disabled = {int(g) for g in sec.get("disabled_gpus") or [] if isinstance(g, int)}
        visible = [g for g in gpu_indices if g not in disabled]
        if disabled and visible:
            single = sec.get("split_mode") == "none" or len(visible) == 1
            if single:
                target = state.preferred_gpu if state.preferred_gpu in visible else visible[0]
                placement = {"placement_policy": "forced", "preferred_gpus": [target]}
            else:
                placement = {"placement_policy": "forced", "preferred_gpus": visible}
        return {
            "preset_key": PRESET_KEY,
            "label": "normal",
            "context_mode": "fixed",
            "context_tokens": context,
            "parallelism": slots,
            "kv_cache_dtype": _kv_label(sec.get("cache_type_k")),
            **placement,
            "execution_policy": "auto",
            "enabled": True,
            "is_default": True,
        }

    if backend in {"vllm", "sglang"}:
        sec = extra.get(backend) if isinstance(extra.get(backend), dict) else {}
        length = sec.get("max_model_len" if backend == "vllm" else "context_length")
        seqs = sec.get("max_num_seqs") if backend == "vllm" else None
        return {
            "preset_key": PRESET_KEY,
            "label": "normal",
            "context_mode": "fixed" if length else "max",
            "context_tokens": int(length) if length else None,
            "parallelism": max(1, min(64, int(seqs or 1))),
            "kv_cache_dtype": _kv_label(sec.get("kv_cache_dtype")),
            **placement,
            "execution_policy": "auto",
            "enabled": True,
            "is_default": True,
        }
    return None


async def ensure_llm_default_presets(
    registry: Any,
    model_manager: Any,
    *,
    only_model_id: str | None = None,
) -> dict[str, list[str]]:
    """Seed a ``normal`` preset on every LLM default profile without presets.

    Returns ``{"created": [...], "skipped": [...]}`` (profile ids). A seeded
    preset the planner judges as not fitting is removed again (skipped).
    """
    from ocabra.core.runtime_preset_capacity import runtime_capacity_cache
    from ocabra.database import AsyncSessionLocal

    created: list[str] = []
    skipped: list[str] = []
    gpu_manager = getattr(model_manager, "gpu_manager", None) or getattr(
        model_manager, "_gpu_manager", None
    )
    try:
        gpu_states = await gpu_manager.get_all_states() if gpu_manager is not None else []
    except Exception:  # noqa: BLE001
        gpu_states = []
    gpu_indices = [int(g.index) for g in gpu_states]

    profiles = list(getattr(registry, "_profiles", {}).values())
    for profile in profiles:
        if (
            getattr(profile, "profile_kind", "execution") != "execution"
            or getattr(profile, "category", None) != "llm"
            or not getattr(profile, "is_default", False)
            or profile.base_model_id is None
            or (only_model_id and profile.base_model_id != only_model_id)
            or getattr(profile, "runtime_presets", None)
        ):
            continue
        state = await model_manager.get_state(profile.base_model_id)
        if state is None or state.backend_type not in PRESET_BACKENDS:
            continue
        values = preset_from_base(state, gpu_indices)
        if values is None:
            continue
        try:
            async with AsyncSessionLocal() as session:
                await registry.create_runtime_preset(session, profile.profile_id, values)
                await session.commit()
            fresh = await registry.get(profile.profile_id)
            rows = await runtime_capacity_cache.resolve(fresh, state, gpu_states, force=True)
            row = next((r for r in rows if r.get("preset_key") == PRESET_KEY), None)
            if row is not None and row.get("resolved") and not row.get("fits"):
                async with AsyncSessionLocal() as session:
                    await registry.delete_runtime_preset(session, profile.profile_id, PRESET_KEY)
                    await session.commit()
                skipped.append(profile.profile_id)
                logger.warning(
                    "default_preset_does_not_fit",
                    profile_id=profile.profile_id,
                    requested_context=row.get("requested_context"),
                    effective_context=row.get("effective_context"),
                    hint="the model keeps loading its base config; create a preset that fits",
                )
                continue
            created.append(profile.profile_id)
            logger.info("default_preset_seeded", profile_id=profile.profile_id, preset=values)
        except Exception as exc:  # noqa: BLE001 - never block startup or registration
            skipped.append(profile.profile_id)
            logger.warning("default_preset_seed_failed", profile_id=profile.profile_id, error=str(exc))
    return {"created": created, "skipped": skipped}
