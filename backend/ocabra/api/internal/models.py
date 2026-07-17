import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.core import vram_planner as vp
from ocabra.core.model_ref import parse_model_ref
from ocabra.core.vram_capacity import resolve_arch_and_weights
from ocabra.db.model_config import (
    get_all_model_schedule_rows,
    get_model_schedule_rows,
    model_schedule_rows_to_payload,
    replace_model_schedules,
)
from ocabra.registry.ollama_registry import OllamaRegistry

router = APIRouter(tags=["models"])
_ollama_registry = OllamaRegistry()


class ModelPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    display_name: str | None = Field(default=None, alias="displayName")
    backend_type: str | None = Field(default=None, alias="backendType")
    load_policy: str | None = Field(default=None, alias="loadPolicy")
    auto_reload: bool | None = Field(default=None, alias="autoReload")
    preferred_gpu: int | None = Field(default=None, alias="preferredGpu")
    vram_estimate_mb: int | None = Field(default=None, alias="vramEstimateMb")
    extra_config: dict | None = Field(default=None, alias="extraConfig")
    schedules: list | None = Field(default=None, alias="schedules")


class AddModelRequest(BaseModel):
    model_id: str
    backend_type: str
    display_name: str | None = None
    load_policy: str = "on_demand"
    auto_reload: bool = False
    preferred_gpu: int | None = None
    extra_config: dict | None = None


class ModelMemoryEstimateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_gpu: int | None = None
    extra_config: dict | None = None
    run_probe: bool = False


@router.get(
    "/models",
    summary="List all models",
    description="Return every configured model with its runtime state, capabilities, and disk size. "
    "When federation is enabled, includes federation metadata showing which remote nodes "
    "have each model, and adds remote-only models as read-only entries.",
)
async def list_models(
    request: Request,
    user: UserContext = Depends(require_role("user")),
) -> list[dict]:
    """List all configured models and their runtime state.

    Non-admin users only see models they have access to via their groups.

    When federation is enabled, each model includes a ``federation`` section
    with a ``nodes`` list indicating where it is available. Remote-only models
    (not configured locally) appear as read-only entries.
    """
    mm = request.app.state.model_manager
    await _sync_ollama_inventory(mm)
    states = await mm.list_states()
    ollama_sizes = await _get_ollama_sizes_bytes()

    federation_manager = getattr(request.app.state, "federation_manager", None)
    remote_models: dict[str, list] = {}
    if federation_manager is not None:
        remote_models = federation_manager.get_remote_models()

    # Bulk-fetch every per-model schedule row in one query and group by model_id
    # so we don't trigger one query per model during serialization.
    from collections import defaultdict

    from ocabra.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        all_schedule_rows = await get_all_model_schedule_rows(session)
    rows_by_model: dict[str, list] = defaultdict(list)
    for row in all_schedule_rows:
        if row.model_id is not None:
            rows_by_model[row.model_id].append(row)
    schedules_by_model: dict[str, list[dict]] = {
        model_id: model_schedule_rows_to_payload(rows)
        for model_id, rows in rows_by_model.items()
    }

    profile_registry = getattr(request.app.state, "profile_registry", None)

    async def _is_accessible(model_id: str) -> bool:
        """True if the user can access the model (admin or any of its profiles in their set)."""
        if user.is_admin:
            return True
        if model_id in user.accessible_model_ids:
            return True
        if profile_registry is None:
            return False
        profiles = await profile_registry.list_by_model(model_id)
        return any(p.profile_id in user.accessible_model_ids for p in profiles)

    local_model_ids: set[str] = set()
    payloads = []
    for state in states:
        if not await _is_accessible(state.model_id):
            continue
        item = await _serialize_model_state(
            request,
            state,
            ollama_sizes,
            schedules=schedules_by_model.get(state.model_id, []),
        )
        local_model_ids.add(state.model_id)

        # Add federation metadata if enabled
        if federation_manager is not None:
            peers_for_model = remote_models.get(state.model_id, [])
            item["federation"] = {
                "local": True,
                "nodes": [
                    {"node_name": "local", "node_id": federation_manager.node_id},
                ]
                + [{"node_name": p.name, "node_id": p.peer_id} for p in peers_for_model],
            }

        payloads.append(item)

    # Add agent entries (plan: docs/tasks/agents-mcp-plan.md — Inventario).
    try:
        payloads.extend(await _list_agent_entries(user))
    except Exception as exc:  # noqa: BLE001
        import structlog

        structlog.get_logger(__name__).warning(
            "internal_models_agent_listing_failed", error=str(exc)
        )

    # Add remote-only models (not configured locally)
    if federation_manager is not None:
        for model_id, peers in remote_models.items():
            if model_id in local_model_ids:
                continue
            if not await _is_accessible(model_id):
                continue
            first_peer = peers[0]
            # Find model info from the first peer's model list
            model_info: dict = {}
            for m in first_peer.models:
                if m.get("model_id") == model_id:
                    model_info = m
                    break
            payloads.append(
                {
                    "model_id": model_id,
                    "backend_type": "remote",
                    "backend_model_id": model_id,
                    "display_name": model_id.split("/")[-1] if "/" in model_id else model_id,
                    "status": "remote",
                    "capabilities": {},
                    "profiles": model_info.get("profiles", []),
                    "disk_size_bytes": None,
                    "federation": {
                        "local": False,
                        "read_only": True,
                        "nodes": [{"node_name": p.name, "node_id": p.peer_id} for p in peers],
                    },
                }
            )

    return payloads


@router.get(
    "/models/storage",
    summary="Get models storage usage",
    description="Return filesystem usage (total, used, free bytes) for the configured models directory.",
)
async def get_models_storage(
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Return storage usage for the models directory filesystem."""
    models_dir = Path(settings.models_dir or "/data/models")
    try:
        stats = await asyncio.to_thread(os.statvfs, models_dir)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Unable to read models storage stats: {exc}"
        ) from exc

    total_bytes = int(stats.f_frsize * stats.f_blocks)
    free_bytes = int(stats.f_frsize * stats.f_bavail)
    used_bytes = max(0, total_bytes - free_bytes)
    return {
        "path": str(models_dir),
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "free_bytes": free_bytes,
    }


_KV_DTYPE_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "q8": 1.0, "q4": 0.5}


@router.get(
    "/models/{model_id:path}/capacity",
    summary="Model VRAM capacity plan",
    description=(
        "Plan a model's VRAM as a function of context and parallelism: the max "
        "context that fits per slot/concurrency on a GPU, plus the VRAM curve over "
        "context. Applies to KV-cache backends (Ollama, llama.cpp, vLLM, SGLang); "
        "non-LLM backends return applicable=false."
    ),
    responses={404: {"description": "Model not found"}},
)
async def get_model_capacity(
    model_id: str,
    request: Request,
    gpu: int | None = Query(default=None, description="Target GPU index (default: preferred/most-free)"),
    slots: str = Query(default="1,2,4", description="Comma list of slot/concurrency counts"),
    kv_dtype: str = Query(default="fp16", description="KV cache dtype: fp16|fp8|q4"),
    context: str | None = Query(default=None, description="Validate a desired context (int or 'max') at slots[0]"),
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """VRAM capacity/planning report for a model on a given GPU."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    backend = (state.backend_type or "").lower()
    res = await resolve_arch_and_weights(state)

    gpu_manager = request.app.state.gpu_manager
    gpu_states = await gpu_manager.get_all_states()
    gpus = [
        {"index": g.index, "total_mb": g.total_vram_mb, "free_mb": g.free_vram_mb}
        for g in gpu_states
    ]

    if gpu is not None:
        target_gpu = gpu
    elif state.preferred_gpu is not None:
        target_gpu = state.preferred_gpu
    elif gpu_states:
        target_gpu = max(gpu_states, key=lambda g: g.free_vram_mb).index
    else:
        target_gpu = 0

    kv_bytes = _KV_DTYPE_BYTES.get(kv_dtype.lower(), 2.0)
    slot_counts = tuple(int(s) for s in slots.split(",") if s.strip().isdigit()) or (1, 2, 4)

    report: dict = {
        "model_id": model_id,
        "backend_type": backend,
        "applicable": res.arch is not None,
        "weights_mb": round(res.weights_mb),
        "overhead_mb": vp.DEFAULT_OVERHEAD_MB,
        "kv_dtype": kv_dtype.lower(),
        "target_gpu": target_gpu,
        "gpus": gpus,
        "note": res.note,
    }
    if res.arch is None:
        return report

    arch = res.arch
    gsel = next((g for g in gpu_states if g.index == target_gpu), None)
    total_mb = gsel.total_vram_mb if gsel else 0
    free_mb = gsel.free_vram_mb if gsel else 0

    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    gpu_mem_util = float(
        (extra.get("vllm") or {}).get("gpu_memory_utilization")
        or settings.vllm_gpu_memory_utilization
    )

    report["arch"] = {
        "layers": arch.layers,
        "n_kv_heads": arch.n_kv_heads,
        "key_length": arch.key_length,
        "value_length": arch.value_length,
        "native_context": arch.context_length,
        "kv_bytes_per_token": arch.kv_bytes_per_token(kv_bytes),
        "kv_mb_per_1k_tokens": round(arch.kv_bytes_per_token(kv_bytes) * 1000 / (1024 * 1024), 1),
    }
    report["concurrency_label"] = "concurrency" if backend in vp._POOLED_BACKENDS else "slots"
    report["fits_free_now"] = free_mb >= res.weights_mb + vp.DEFAULT_OVERHEAD_MB
    report["capacity"] = vp.capacity_rows(
        backend, arch, res.weights_mb,
        gpu_total_mb=total_mb, slots=slot_counts,
        gpu_memory_utilization=gpu_mem_util, kv_dtype_bytes=kv_bytes,
    )
    report["vram_curve"] = vp.vram_curve(arch, res.weights_mb, kv_dtype_bytes=kv_bytes)

    if context is not None:
        requested: int | str = context if context.lower() in ("max", "auto") else int(context)
        plan = vp.plan_use_case(
            backend, arch, res.weights_mb,
            gpu_total_mb=total_mb, requested_context=requested, slots=slot_counts[0],
            gpu_memory_utilization=gpu_mem_util, kv_dtype_bytes=kv_bytes,
        )
        report["validation"] = {
            "requested_context": requested,
            "slots": slot_counts[0],
            "effective_context": plan["effective_context"],
            "max_context": plan["max_context"],
            "fits": not plan["warnings"],
            "warnings": plan["warnings"],
        }

    return report


@router.get(
    "/models/{model_id:path}",
    summary="Get model state",
    description="Return the full runtime state, capabilities, and disk size of a single model.",
    responses={404: {"description": "Model not found"}},
)
async def get_model(
    model_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Get state of a specific model."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    ollama_sizes = await _get_ollama_sizes_bytes() if state.backend_type == "ollama" else {}
    return await _serialize_model_state(request, state, ollama_sizes)


@router.post(
    "/models",
    summary="Register a new model",
    description="Create a model configuration entry. The model is not loaded until an explicit load request.",
    responses={400: {"description": "Invalid model configuration"}},
)
async def add_model(
    body: AddModelRequest,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Register a new model configuration."""
    mm = request.app.state.model_manager
    try:
        state = await mm.add_model(
            model_id=body.model_id,
            backend_type=body.backend_type,
            display_name=body.display_name,
            load_policy=body.load_policy,
            auto_reload=body.auto_reload,
            preferred_gpu=body.preferred_gpu,
            extra_config=body.extra_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return state.to_dict()


@router.post(
    "/models/{model_id:path}/load",
    summary="Load model onto GPU",
    description="Start the backend worker process and load the model weights into GPU VRAM.",
    responses={
        404: {"description": "Model not found"},
        409: {"description": "Model already loaded or insufficient VRAM"},
    },
)
async def load_model(
    model_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Load a model onto a GPU."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    from ocabra.core.model_manager import ModelStatus
    from ocabra.core.scheduler import InsufficientVRAMError

    if state.status == ModelStatus.LOADED:
        raise HTTPException(status_code=409, detail="Model is already loaded")
    try:
        updated = await mm.load(model_id)
    except InsufficientVRAMError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return updated.to_dict()


@router.post(
    "/models/{model_id:path}/unload",
    summary="Unload model from GPU",
    description="Stop the backend worker process and free the GPU VRAM occupied by this model.",
    responses={404: {"description": "Model not found"}},
)
async def unload_model(
    model_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Unload a model from GPU."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    await mm.unload(model_id, reason="manual")
    updated = await mm.get_state(model_id)
    return updated.to_dict()


@router.patch(
    "/models/{model_id:path}",
    summary="Update model configuration",
    description="Patch mutable fields such as load_policy, preferred_gpu, display_name, or extra_config.",
    responses={
        400: {"description": "Invalid configuration value"},
        404: {"description": "Model not found"},
    },
)
async def update_model(
    model_id: str,
    body: ModelPatch,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Update model configuration.

    The ``schedules`` field is persisted separately from ``update_config`` since
    eviction schedules live in their own table. Fields the lower-level
    ``update_config`` does not accept (``backend_type``, ``vram_estimate_mb``)
    are silently dropped — they exist in the API schema for forward
    compatibility but are not yet mutable post-registration.
    """
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    patch = body.model_dump(exclude_unset=True)

    schedules_payload = patch.pop("schedules", None)
    # Not yet supported by update_config; ignore for now.
    patch.pop("backend_type", None)
    patch.pop("vram_estimate_mb", None)

    if schedules_payload is not None:
        from ocabra.database import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                await replace_model_schedules(session, model_id, schedules_payload)
                await session.commit()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if patch:
        try:
            updated = await mm.update_config(model_id, patch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        updated = state

    ollama_sizes = (
        await _get_ollama_sizes_bytes() if updated.backend_type == "ollama" else {}
    )
    return await _serialize_model_state(request, updated, ollama_sizes)


@router.post(
    "/models/{model_id:path}/memory-estimate",
    summary="Estimate model memory requirements",
    description=(
        "Compute a VRAM estimate for the model under the given configuration. "
        "Uses heuristic sizing by default; set run_probe=true for a runtime validation (vLLM only)."
    ),
    responses={404: {"description": "Model not found"}},
)
async def estimate_model_memory(
    model_id: str,
    body: ModelMemoryEstimateRequest,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Estimate memory requirements for the current backend/config combination."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    extra_config = (
        body.extra_config if isinstance(body.extra_config, dict) else state.extra_config or {}
    )
    return await _build_model_memory_estimate(
        request=request,
        state=state,
        extra_config=extra_config,
        preferred_gpu=body.preferred_gpu,
        run_probe=body.run_probe,
    )


@router.get(
    "/models/{model_id:path}/speculative-candidates",
    summary="List speculative-decoding draft candidates",
    description=(
        "Return llama.cpp models whose tokenizer fingerprint "
        "(``vocab_size``, ``bos_id``, ``eos_id``) matches the given target "
        "model and that are therefore safe to use as a speculative-decoding "
        "draft. The target model itself is excluded."
    ),
    responses={404: {"description": "Model not found"}},
)
async def list_speculative_candidates(
    model_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> list[dict]:
    """List llama.cpp draft candidates compatible with ``model_id``.

    Args:
        model_id: Canonical model_id of the target (large) model.

    Returns:
        List of ``{model_id, display_name, vocab_size, bos_id, eos_id}``
        dictionaries, sorted by ``model_id``.
    """
    import sqlalchemy as sa

    from ocabra.database import AsyncSessionLocal
    from ocabra.db.model_config import ModelConfig

    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    async with AsyncSessionLocal() as session:
        target = (
            await session.execute(
                sa.select(ModelConfig).where(ModelConfig.model_id == model_id)
            )
        ).scalar_one_or_none()
        if target is None or target.vocab_size is None:
            return []

        rows = (
            (
                await session.execute(
                    sa.select(ModelConfig).where(
                        ModelConfig.backend_type == "llama_cpp",
                        ModelConfig.model_id != model_id,
                        ModelConfig.vocab_size == target.vocab_size,
                        ModelConfig.bos_id == target.bos_id,
                        ModelConfig.eos_id == target.eos_id,
                    )
                )
            )
            .scalars()
            .all()
        )

    return sorted(
        (
            {
                "model_id": row.model_id,
                "display_name": row.display_name or row.model_id,
                "vocab_size": row.vocab_size,
                "bos_id": row.bos_id,
                "eos_id": row.eos_id,
            }
            for row in rows
        ),
        key=lambda item: item["model_id"],
    )


@router.post(
    "/models/refresh-tokenizer-fingerprints",
    summary="Backfill llama.cpp tokenizer fingerprints",
    description=(
        "Re-parse every registered ``llama_cpp`` model GGUF and write "
        "``vocab_size``, ``bos_id`` and ``eos_id`` to its ``model_configs`` "
        "row. Newly registered models pick up the fingerprint at registration "
        "time; this endpoint is for backfilling rows that pre-date Sprint 17.4."
    ),
)
async def refresh_tokenizer_fingerprints(
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Backfill GGUF tokenizer fingerprints for existing llama.cpp models."""
    mm = request.app.state.model_manager
    return await mm.refresh_tokenizer_fingerprints()


@router.delete(
    "/models/{model_id:path}",
    summary="Delete a model",
    description=(
        "Remove the model configuration from the database and optionally delete "
        "the model files from disk. The model is unloaded first if currently loaded."
    ),
    responses={
        400: {"description": "Path traversal refused"},
        404: {"description": "Model not found"},
    },
)
async def delete_model(
    model_id: str,
    request: Request,
    delete_files: bool = Query(default=True),
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Remove a model configuration and its files from disk.

    Args:
        model_id: Canonical model id (backend/org/name).
        delete_files: If true (default), also delete the model files from disk.

    Returns:
        {"ok": true, "deleted_path": path_or_null}
    """
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    await mm.delete_model(model_id)

    deleted_path: str | None = None
    if delete_files:
        deleted_path = await _delete_model_files(model_id, state.backend_type)
        # Drop the LocalScanner cache so the user sees the deletion reflected
        # in /registry/local without having to wait for the TTL.
        from ocabra.api.internal.registry import invalidate_local_scan

        invalidate_local_scan()

    return {"ok": True, "deleted_path": deleted_path}


async def _delete_model_files(model_id: str, backend_type: str) -> str | None:
    """Delete model files from disk based on backend type."""
    import asyncio

    if backend_type == "ollama":
        # Ollama manages its own storage — use ollama rm
        try:
            parts = model_id.split("/", 1)
            ollama_name = parts[1] if len(parts) == 2 else model_id
            proc = await asyncio.create_subprocess_exec(
                "ollama",
                "rm",
                ollama_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception:
            pass
        return None

    # HuggingFace-backed models (vllm, diffusers, whisper, transformers, tensorrt_llm…)
    parts = model_id.split("/", 1)
    raw_model = parts[1] if len(parts) == 2 else model_id
    hf_dir_name = raw_model.replace("/", "--")
    models_dir = Path(settings.models_dir or "/data/models")
    candidate = models_dir / "huggingface" / hf_dir_name
    if candidate.exists():
        if not _is_path_within_base(candidate, models_dir / "huggingface"):
            raise HTTPException(
                status_code=400,
                detail="Refusing to delete a model path outside the configured models directory",
            )
        await asyncio.to_thread(shutil.rmtree, candidate)
        return str(candidate)
    return None


async def _get_ollama_sizes_bytes() -> dict[str, int]:
    try:
        details = await _ollama_registry.list_installed_details()
    except Exception:
        return {}
    size_map: dict[str, int] = {}
    for item in details:
        name = str(item.get("name") or "").strip().lower()
        if not name:
            continue
        size_map[name] = int(item.get("size") or 0)
    return size_map


async def _sync_ollama_inventory(model_manager) -> None:
    try:
        installed = await _ollama_registry.list_installed()
        loaded_details = await _ollama_registry.list_loaded_details()
    except Exception:
        return

    loaded: list[str] = []
    loaded_vram_mb: dict[str, int] = {}
    for item in loaded_details:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        loaded.append(name)
        loaded_vram_mb[name] = int((item.get("size_vram") or 0) / 1024 / 1024)

    await model_manager.sync_ollama_inventory(installed, loaded, loaded_vram_mb=loaded_vram_mb)


async def _resolve_disk_size_bytes(
    state,
    ollama_sizes: dict[str, int],
) -> int | None:
    backend_type, backend_model_id = parse_model_ref(state.model_id)
    if backend_type == "ollama":
        return ollama_sizes.get(backend_model_id.strip().lower())
    if backend_type == "tensorrt_llm":
        path = _resolve_extra_config_path(
            state.extra_config, "engine_dir"
        ) or _resolve_tensorrt_engine_path(backend_model_id)
        if path is None or not path.exists():
            return None
        return await asyncio.to_thread(_compute_path_size_bytes, path)

    path = _resolve_extra_config_path(
        state.extra_config, "model_path", "base_model_id"
    ) or _resolve_local_model_path(backend_model_id)
    if path is None or not path.exists():
        return None
    return await asyncio.to_thread(_compute_path_size_bytes, path)


async def _build_model_memory_estimate(
    request: Request,
    state,
    extra_config: dict[str, Any],
    preferred_gpu: int | None,
    run_probe: bool,
) -> dict[str, Any]:
    worker_pool = getattr(request.app.state, "worker_pool", None)
    gpu_manager = getattr(request.app.state, "gpu_manager", None)
    backend = await worker_pool.get_backend(state.backend_type) if worker_pool else None

    gpu_index = (
        preferred_gpu
        if preferred_gpu is not None
        else state.preferred_gpu
        if getattr(state, "preferred_gpu", None) is not None
        else settings.default_gpu_index
    )
    gpu_state = None
    if gpu_manager is not None:
        try:
            gpu_state = await gpu_manager.get_state(gpu_index)
        except (KeyError, IndexError):
            pass
    total_vram_mb = int(getattr(gpu_state, "total_vram_mb", 0) or 0) or None
    free_vram_mb = int(getattr(gpu_state, "free_vram_mb", 0) or 0) or None

    estimate = {
        "backend_type": state.backend_type,
        "gpu_index": gpu_index,
        "total_vram_mb": total_vram_mb,
        "free_vram_mb": free_vram_mb,
        "budget_vram_mb": None,
        "requested_context_length": _resolve_requested_context_length(state, extra_config),
        "estimated_weights_mb": None,
        "estimated_engine_mb_per_gpu": None,
        "estimated_kv_cache_mb": None,
        "estimated_max_context_length": None,
        "model_loading_memory_mb": None,
        "maximum_concurrency": None,
        "tensor_parallel_size": _resolve_tensor_parallel_size(state, extra_config),
        "fits_current_gpu": None,
        "engine_present": None,
        "source": "heuristic",
        "status": "ok",
        "warning": None,
        "notes": [],
    }

    if backend is None:
        estimate["status"] = "warning"
        estimate["warning"] = "No backend runtime available to calculate memory estimates."
        return estimate

    heuristic_mb = int(await backend.get_vram_estimate_mb(state.backend_model_id) or 0)

    if state.backend_type == "vllm":
        gpu_util = _resolve_vllm_gpu_memory_utilization(extra_config)
        budget_vram_mb = int((total_vram_mb or 0) * gpu_util) if total_vram_mb is not None else None
        estimate.update(
            {
                "budget_vram_mb": budget_vram_mb,
                "estimated_weights_mb": heuristic_mb or None,
                "estimated_kv_cache_mb": None,
                "fits_current_gpu": None,
            }
        )
        estimate["notes"] = [
            "La estimación rápida de vLLM aproxima el peso cargado y la reserva objetivo de VRAM.",
            "La memoria de KV cache y el contexto máximo real dependen del engine; usa el probe para una validación fiable.",
        ]
        if run_probe and hasattr(backend, "estimate_memory_profile"):
            profile = await backend.estimate_memory_profile(
                state.backend_model_id,
                gpu_index=gpu_index,
                extra_config=extra_config,
            )
            estimate.update(
                {
                    "source": "runtime_probe",
                    "model_loading_memory_mb": profile.get("model_loading_memory_mb"),
                    "estimated_kv_cache_mb": profile.get("available_kv_cache_mb")
                    or estimate["estimated_kv_cache_mb"],
                    "estimated_max_context_length": profile.get("estimated_max_model_len")
                    or profile.get("gpu_kv_cache_tokens"),
                    "maximum_concurrency": profile.get("maximum_concurrency"),
                }
            )
            if profile.get("requested_context_length") and not estimate["requested_context_length"]:
                estimate["requested_context_length"] = profile.get("requested_context_length")
            if profile.get("status") != "ok":
                estimate["status"] = "error"
                estimate["warning"] = profile.get("error") or "vLLM runtime probe failed."
                if profile.get("estimated_max_model_len"):
                    estimate["notes"].append(
                        f"El engine estima un máximo de {profile['estimated_max_model_len']} tokens con esta configuración."
                    )
            else:
                estimate["notes"].append(
                    "Probe real de vLLM completado; las cifras de KV cache y contexto vienen del engine."
                )
        return estimate

    if state.backend_type == "llama_cpp":
        # Sprint 17.2 — deterministic GGUF estimator.
        return _estimate_llama_cpp_memory(
            estimate=estimate,
            state=state,
            extra_config=extra_config,
            heuristic_mb=heuristic_mb,
            total_vram_mb=total_vram_mb,
        )

    if state.backend_type == "tensorrt_llm":
        engine_path = _resolve_extra_config_path(
            extra_config, "engine_dir"
        ) or _resolve_tensorrt_engine_path(state.backend_model_id)
        tp_size = _resolve_tensor_parallel_size(state, extra_config)
        estimate.update(
            {
                "estimated_engine_mb_per_gpu": heuristic_mb or None,
                "engine_present": bool(engine_path and engine_path.exists()),
                "fits_current_gpu": (
                    heuristic_mb <= total_vram_mb
                    if total_vram_mb is not None and heuristic_mb > 0
                    else None
                ),
                "tensor_parallel_size": tp_size,
            }
        )
        if not estimate["engine_present"]:
            estimate["status"] = "error"
            estimate["warning"] = "Engine directory not found for this TensorRT-LLM model."
        estimate["notes"] = [
            "La estimación de TensorRT-LLM usa el tamaño real del engine por GPU y el tp_size detectado en config.json.",
        ]
        return estimate

    estimate.update(
        {
            "estimated_weights_mb": heuristic_mb or None,
            "fits_current_gpu": (
                heuristic_mb <= total_vram_mb
                if total_vram_mb is not None and heuristic_mb > 0
                else None
            ),
            "notes": [
                "Estimación heurística basada en el tamaño de los artefactos locales del modelo."
            ],
        }
    )
    return estimate


def _estimate_llama_cpp_memory(
    *,
    estimate: dict[str, Any],
    state,
    extra_config: dict[str, Any],
    heuristic_mb: int,
    total_vram_mb: int | None,
) -> dict[str, Any]:
    """Sprint 17.2 — deterministic GGUF VRAM estimator for llama.cpp models.

    Falls back to the legacy heuristic when the GGUF file cannot be located or
    parsed (e.g. only-on-disk via Ollama, or remote model). The legacy probe
    remains the source for everything else.
    """
    from ocabra.core.llama_cpp_estimator import estimate_vram_safe
    from ocabra.schemas.backend_load import LlamaCppLoadConfig

    nested = extra_config.get("llama_cpp") if isinstance(extra_config, dict) else None
    payload = nested if isinstance(nested, dict) else {}
    try:
        load_config = LlamaCppLoadConfig.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 — surface as warning, keep estimate
        estimate.update(
            {
                "estimated_weights_mb": heuristic_mb or None,
                "fits_current_gpu": (
                    heuristic_mb <= total_vram_mb
                    if total_vram_mb is not None and heuristic_mb > 0
                    else None
                ),
                "status": "warning",
                "warning": f"Configuración llama.cpp inválida: {exc}",
                "notes": ["Estimación heurística (config inválida)."],
            }
        )
        return estimate

    gguf_path = _resolve_extra_config_path(extra_config, "model_path") or _resolve_local_model_path(
        state.backend_model_id
    )
    gguf_file: Path | None = None
    if gguf_path is not None:
        if gguf_path.is_file() and gguf_path.suffix == ".gguf":
            gguf_file = gguf_path
        elif gguf_path.is_dir():
            ggufs = sorted(gguf_path.rglob("*.gguf"))
            if ggufs:
                gguf_file = max(ggufs, key=lambda p: p.stat().st_size)

    breakdown = estimate_vram_safe(str(gguf_file), load_config) if gguf_file is not None else None

    if breakdown is None:
        # Couldn't locate / parse — degrade gracefully to the legacy heuristic.
        estimate.update(
            {
                "estimated_weights_mb": heuristic_mb or None,
                "fits_current_gpu": (
                    heuristic_mb <= total_vram_mb
                    if total_vram_mb is not None and heuristic_mb > 0
                    else None
                ),
                "notes": [
                    "Estimación heurística: GGUF no encontrado o no parseable.",
                ],
            }
        )
        return estimate

    bytes_per_mb = 1024 * 1024
    weights_mb = max(1, breakdown["model_bytes"] // bytes_per_mb)
    kv_mb = max(0, breakdown["kv_bytes"] // bytes_per_mb)
    compute_mb = max(0, breakdown["compute_buffer_bytes"] // bytes_per_mb)
    total_mb = max(1, breakdown["total_bytes"] // bytes_per_mb)

    notes = [
        "Estimación determinística desde el header GGUF (Sprint 17.2).",
        f"Desglose en MB — pesos: {weights_mb} · KV: {kv_mb} · compute: {compute_mb} · total: {total_mb}.",
    ]
    if load_config.cache_type_k or load_config.cache_type_v:
        notes.append(
            "KV cache cuantizado: K="
            f"{load_config.cache_type_k or 'f16'} · V={load_config.cache_type_v or 'f16'}."
        )

    estimate.update(
        {
            "estimated_weights_mb": weights_mb,
            "estimated_kv_cache_mb": kv_mb,
            "model_loading_memory_mb": total_mb,
            "fits_current_gpu": (total_mb <= total_vram_mb if total_vram_mb is not None else None),
            "source": "heuristic",
            "notes": notes,
        }
    )
    return estimate


async def _serialize_model_state(
    request: Request,
    state,
    ollama_sizes: dict[str, int],
    schedules: list[dict] | None = None,
) -> dict:
    item = state.to_dict()
    item["disk_size_bytes"] = await _resolve_disk_size_bytes(state, ollama_sizes)
    item["capabilities"] = await _resolve_capabilities_payload(request, state, item["capabilities"])

    # Attach profiles if the registry is available
    profile_registry = getattr(request.app.state, "profile_registry", None)
    if profile_registry is not None:
        from ocabra.schemas.profiles import ProfileOut

        profiles = await profile_registry.list_by_model(state.model_id)
        item["profiles"] = [ProfileOut.model_validate(p).model_dump(mode="json") for p in profiles]
    else:
        item["profiles"] = []

    # Per-model eviction schedules (UI shape: {id, days, start, end, enabled}).
    if schedules is None:
        from ocabra.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            rows = await get_model_schedule_rows(session, state.model_id)
        schedules = model_schedule_rows_to_payload(rows)
    item["schedules"] = schedules

    return item


async def _resolve_capabilities_payload(
    request: Request,
    state,
    current_payload: dict,
) -> dict:
    has_meaningful_caps = (
        any(
            bool(current_payload.get(key))
            for key in (
                "chat",
                "completion",
                "tools",
                "vision",
                "embeddings",
                "pooling",
                "rerank",
                "classification",
                "score",
                "reasoning",
                "streaming",
                "audio_transcription",
                "audio_input",
                "video_input",
                "tts",
                "music_generation",
                "image_generation",
            )
        )
        or int(current_payload.get("context_length") or 0) > 0
    )

    if has_meaningful_caps and not (
        state.backend_type == "tensorrt_llm"
        and int(current_payload.get("context_length") or 0) <= 0
        and isinstance(state.extra_config, dict)
        and state.extra_config.get("context_length")
    ):
        return current_payload

    worker_pool = getattr(request.app.state, "worker_pool", None)
    if worker_pool is None:
        return _apply_capability_fallbacks(state, current_payload)

    try:
        backend = await worker_pool.get_backend(state.backend_type)
        capabilities = await backend.get_capabilities(state.backend_model_id)
        payload = capabilities.to_dict()
    except Exception:
        payload = dict(current_payload)

    return _apply_capability_fallbacks(state, payload)


def _apply_capability_fallbacks(state, payload: dict) -> dict:
    merged = dict(payload)
    fallback_context_length = _resolve_context_length_fallback(state)
    current_context_length = int(merged.get("context_length") or 0)
    if fallback_context_length > 0 and (
        current_context_length <= 0
        or (
            state.backend_type == "tensorrt_llm"
            and fallback_context_length > current_context_length
        )
    ):
        merged["context_length"] = fallback_context_length
    return merged


def _resolve_vllm_gpu_memory_utilization(extra_config: dict[str, Any]) -> float:
    vllm_config = extra_config.get("vllm")
    if isinstance(vllm_config, dict):
        if vllm_config.get("gpu_memory_utilization") is not None:
            return float(vllm_config["gpu_memory_utilization"])
        if vllm_config.get("gpuMemoryUtilization") is not None:
            return float(vllm_config["gpuMemoryUtilization"])
    if extra_config.get("gpu_memory_utilization") is not None:
        return float(extra_config["gpu_memory_utilization"])
    if extra_config.get("gpuMemoryUtilization") is not None:
        return float(extra_config["gpuMemoryUtilization"])
    return float(settings.vllm_gpu_memory_utilization)


def _resolve_requested_context_length(state, extra_config: dict[str, Any]) -> int | None:
    for key in ("context_length", "max_model_len", "ctx_size"):
        for candidate in (key, _to_camel_key(key)):
            value = extra_config.get(candidate)
            if isinstance(value, int) and value > 0:
                return value
    for section in ("vllm", "sglang", "llama_cpp", "bitnet", "tensorrt_llm"):
        nested = extra_config.get(section)
        if isinstance(nested, dict):
            for key in ("context_length", "max_model_len", "ctx_size"):
                for candidate in (key, _to_camel_key(key)):
                    value = nested.get(candidate)
                    if isinstance(value, int) and value > 0:
                        return value
    fallback = _resolve_context_length_fallback(state)
    return fallback if fallback > 0 else None


def _resolve_tensor_parallel_size(state, extra_config: dict[str, Any]) -> int | None:
    for section in ("vllm", "sglang"):
        nested = extra_config.get(section)
        if isinstance(nested, dict):
            value = nested.get("tensor_parallel_size") or nested.get("tensorParallelSize")
            if isinstance(value, int) and value > 0:
                return value
    if state.backend_type == "tensorrt_llm":
        engine_path = _resolve_extra_config_path(
            extra_config, "engine_dir"
        ) or _resolve_tensorrt_engine_path(state.backend_model_id)
        if engine_path is not None and engine_path.exists():
            scan_dir = engine_path / "engine" if (engine_path / "engine").is_dir() else engine_path
            try:
                import json as _json

                cfg = _json.loads((scan_dir / "config.json").read_text())
                mapping = cfg.get("pretrained_config", {}).get("mapping", {})
                return int(mapping.get("tp_size") or mapping.get("world_size") or 1)
            except Exception:
                return 1
    return 1


def _to_camel_key(value: str) -> str:
    return "".join(
        part.capitalize() if index else part for index, part in enumerate(value.split("_"))
    )


def _resolve_context_length_fallback(state) -> int:
    extra_config = state.extra_config if isinstance(state.extra_config, dict) else {}

    for key in ("context_length", "max_model_len", "ctx_size"):
        value = extra_config.get(key)
        if isinstance(value, int) and value > 0:
            return value

    vllm_config = extra_config.get("vllm")
    if isinstance(vllm_config, dict):
        for key in ("max_model_len", "context_length"):
            value = vllm_config.get(key)
            if isinstance(value, int) and value > 0:
                return value

    backend_model_id = getattr(state, "backend_model_id", "")
    model_path = _resolve_extra_config_path(
        extra_config, "model_path", "base_model_id"
    ) or _resolve_local_model_path(backend_model_id)
    if model_path is None:
        return 0

    for file_name, keys in (
        (
            "config.json",
            (
                "max_position_embeddings",
                "max_seq_len",
                "model_max_length",
                "max_sequence_length",
                "seq_length",
                "n_positions",
                "context_length",
                "max_context_length",
            ),
        ),
        ("tokenizer_config.json", ("model_max_length",)),
    ):
        value = _read_first_positive_int(model_path / file_name, keys)
        if value > 0:
            return value

    return 0


def _resolve_local_model_path(model_id: str) -> Path | None:
    base = Path(settings.models_dir)
    direct = base / model_id
    if direct.exists():
        return direct

    if "::" in model_id:
        base_model_id, _variant = model_id.split("::", 1)
        base_direct = base / base_model_id
        if base_direct.exists():
            return base_direct

    # Hugging Face local layout: /data/models/huggingface/org--repo[--artifact-stem]
    if "::" in model_id:
        repo_id, variant_stem = model_id.split("::", 1)
        hf_dir_name = f"{repo_id.replace('/', '--')}--{variant_stem}"
    else:
        hf_dir_name = model_id.replace("/", "--")
    hf_layout = base / "huggingface" / hf_dir_name
    if hf_layout.exists():
        return hf_layout

    if "::" in model_id:
        repo_id, _variant_stem = model_id.split("::", 1)
        base_hf_layout = base / "huggingface" / repo_id.replace("/", "--")
        if base_hf_layout.exists():
            return base_hf_layout

    # GGUF / local file fallback for llama.cpp and bitnet style registrations.
    base_model_id = model_id.split("::", 1)[0]
    leaf = Path(base_model_id).name
    variant_stem = model_id.split("::", 1)[1] if "::" in model_id else ""
    candidate_stems = {leaf, Path(leaf).stem}
    if variant_stem:
        candidate_stems.add(variant_stem)
        candidate_stems.add(Path(variant_stem).stem)
    gguf_candidates = [
        path
        for path in base.rglob("*.gguf")
        if path.name == leaf
        or any(path.stem == stem or path.stem.endswith(stem) for stem in candidate_stems)
    ]
    if gguf_candidates:
        return max(gguf_candidates, key=lambda path: path.stat().st_mtime)

    # Optional HF cache layout fallback.
    hf_cache_dir = (settings.hf_cache_dir or "").strip()
    if hf_cache_dir:
        cache_root = (
            Path(hf_cache_dir) / "hub" / f"models--{model_id.split('::', 1)[0].replace('/', '--')}"
        )
        snapshots_dir = cache_root / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            refs_main = cache_root / "refs" / "main"
            if refs_main.exists():
                try:
                    commit = refs_main.read_text(encoding="utf-8").strip()
                except Exception:
                    commit = ""
                if commit:
                    candidate = snapshots_dir / commit
                    if candidate.exists():
                        return candidate
            candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)

    return None


def _resolve_extra_config_path(extra_config: dict | None, *keys: str) -> Path | None:
    if not isinstance(extra_config, dict):
        return None
    models_dir = Path(settings.models_dir or "/data/models")
    for key in keys:
        raw = extra_config.get(key)
        if not raw or not isinstance(raw, str):
            continue
        path = Path(raw)
        if path.exists() and _is_path_within_base(path, models_dir):
            return path
    return None


def _resolve_tensorrt_engine_path(engine_name: str) -> Path | None:
    if settings.tensorrt_llm_engines_dir:
        base = Path(settings.tensorrt_llm_engines_dir)
    else:
        models_container = settings.tensorrt_llm_docker_models_mount_container or "/data/models"
        base = Path(models_container) / "tensorrt_llm"

    candidate = base / engine_name
    if candidate.exists():
        return candidate
    engine_subdir = candidate / "engine"
    if engine_subdir.exists():
        return engine_subdir
    return None


def _compute_path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _is_path_within_base(path: Path, base: Path) -> bool:
    try:
        return path.resolve(strict=False).is_relative_to(base.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False


def _read_first_positive_int(path: Path, keys: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    for key in keys:
        value = payload.get(key)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return numeric
    return 0


async def _list_agent_entries(user: UserContext) -> list[dict]:
    """Return agent entries formatted for ``GET /ocabra/models``.

    See ``docs/tasks/agents-mcp-plan.md`` section "Inventario de modelos".
    Admins see every agent; other callers are filtered by ``group_id``.
    """
    import sqlalchemy as sa
    from sqlalchemy.orm import selectinload

    from ocabra.database import AsyncSessionLocal
    from ocabra.db.agents import Agent

    async with AsyncSessionLocal() as session:
        rows = (
            (
                await session.execute(
                    sa.select(Agent).options(selectinload(Agent.mcp_links)).order_by(Agent.slug)
                )
            )
            .scalars()
            .all()
        )

    group_set = set(user.group_ids or [])
    entries: list[dict] = []
    for row in rows:
        if not user.is_admin:
            if row.group_id is not None and str(row.group_id) not in group_set:
                continue
        entries.append(
            {
                "model_id": f"agent/{row.slug}",
                "backend_type": "ocabra-agent",
                "backend_model_id": f"agent/{row.slug}",
                "display_name": row.display_name,
                "description": row.description,
                "status": "configured",
                "capabilities": {"chat": True, "tools": True},
                "profiles": [],
                "disk_size_bytes": None,
                "ocabra": {
                    "kind": "agent",
                    "base_model_id": row.base_model_id,
                    "profile_id": row.profile_id,
                    "tool_choice_default": row.tool_choice_default,
                    "require_approval": row.require_approval,
                    "max_tool_hops": row.max_tool_hops,
                    "mcp_server_count": len(row.mcp_links or []),
                },
            }
        )
    return entries
