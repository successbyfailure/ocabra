import asyncio
import json
import os
import shutil
import time
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


class MaintenanceToggle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
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
    states = await _overlay_loaded_derived_state(mm, states)
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


@router.get(
    "/models/activity",
    summary="Per-model in-flight activity",
    description="In-flight request count and oldest-request age per loaded model, for live 'processing' and 'stuck' indicators.",
)
async def get_models_activity(
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Live in-flight counts + oldest age keyed by model_id."""
    mm = request.app.state.model_manager
    snapshot = mm.inflight_snapshot()
    ages = mm.active_request_ages()
    sources = mm.activity_sources()
    now_ts = time.time()
    requests_by_key: dict[str, list] = {}
    for req in mm.active_requests_snapshot():
        requests_by_key.setdefault(req.model_id, []).append(req)
    # ``list_states()`` (sin include_derived) devuelve solo los canónicos,
    # pero ``begin_request`` registra con el worker-key derivado del
    # preset activo (``ollama/qwen3.8:27b::<hash>``). Para que el badge
    # del card del modelo canónico cuente las peticiones de sus
    # derivados, suma todos los inflights cuyo ``model_id`` empieza por
    # ``canonical.model_id + "::"``.
    states = await mm.list_states()
    # Derivados cargados (``<canonical>::<hash>``). Nos interesan por dos
    # motivos: (1) agregarles las peticiones en vuelo al canonical, y
    # (2) leer el ``_runtime.parallelism`` del preset activo, que vive en
    # el derivado y no en el canonical state.
    all_states = await mm.list_states(include_derived=True)
    runtime_by_canonical: dict[str, int] = {}
    for s in all_states:
        if "::" not in s.model_id:
            continue
        base = s.model_id.rsplit("::", 1)[0]
        meta = s.extra_config.get("_runtime") if isinstance(s.extra_config, dict) else None
        if not isinstance(meta, dict):
            continue
        try:
            p = int(meta.get("parallelism") or 0)
        except (TypeError, ValueError):
            p = 0
        if p > 0:
            # Guardar el mayor por si hay varios derivados activos (distintos
            # presets) — es la capacidad real de slots concurrentes.
            runtime_by_canonical[base] = max(runtime_by_canonical.get(base, 0), p)
    derived_keys_by_canonical: dict[str, list[str]] = {}
    for key in snapshot:
        if "::" not in key:
            continue
        base = key.rsplit("::", 1)[0]
        derived_keys_by_canonical.setdefault(base, []).append(key)
    activity: dict[str, dict] = {}
    for state in states:
        # begin_request may key by the canonical id, the raw backend model
        # name, or a derived worker-key (``<canonical>::<hash>``) when the
        # active preset creates a dedicated runner.
        keys = (
            state.model_id,
            state.backend_model_id,
            *derived_keys_by_canonical.get(state.model_id, ()),
        )
        count = sum(snapshot.get(k, 0) for k in keys)
        if count > 0:
            oldest = max((ages.get(k, 0.0) for k in keys), default=0.0)
            source_counts: dict[str, int] = {}
            for key in keys:
                for source, source_count in sources.get(key, {}).items():
                    source_counts[source] = max(
                        source_counts.get(source, 0), source_count
                    )
            # Concurrency the backend will actually serve in parallel. Prefer
            # the ``_runtime.parallelism`` set by the coordinator (reflects
            # the active preset); fall back to the admission cap. Clients
            # render ``processing = min(count, slots)`` and
            # ``queued = max(0, count - slots)``.
            # Preferir el ``parallelism`` del derivado cargado (preset
            # activo) sobre el del canonical o el cap global.
            slots: int | None = runtime_by_canonical.get(state.model_id) or None
            if slots is None:
                runtime_meta = (
                    state.extra_config.get("_runtime")
                    if isinstance(state.extra_config, dict)
                    else None
                )
                if isinstance(runtime_meta, dict):
                    try:
                        slots = int(runtime_meta.get("parallelism") or 0) or None
                    except (TypeError, ValueError):
                        slots = None
            if slots is None:
                cap = int(getattr(settings, "max_inflight_per_model", 0) or 0)
                slots = cap if cap > 0 else None
            # Per-request progress for the "is this hung or still working"
            # indicator on the model/GPU card (2026-10-29). ``tokensPerSecond``
            # is None until at least one output token has streamed — division
            # by elapsed time before that is meaningless, not just zero.
            requests_detail: list[dict] = []
            for key in keys:
                if not requests_by_key.get(key):
                    continue
                # Same phase for every request piled up behind the same
                # worker_key — computed once per key, not per request.
                load_phase, waiting_on = mm.get_load_phase(key)
                for req in requests_by_key.get(key, ()):
                    elapsed_s = max(0.0, now_ts - req.started_at)
                    tokens_per_second = None
                    if req.output_tokens_estimate and elapsed_s > 0:
                        tokens_per_second = round(req.output_tokens_estimate / elapsed_s, 1)
                    if load_phase == "loading":
                        phase = "loading"
                    elif load_phase == "waiting_for_gpu":
                        phase = "waiting_for_gpu"
                    elif req.output_tokens_estimate is None:
                        phase = "prompt_eval"
                    else:
                        phase = "generating"
                    requests_detail.append({
                        "requestId": req.request_id,
                        "username": req.username,
                        "elapsedSeconds": round(elapsed_s, 1),
                        "inputTokensEstimate": req.input_tokens_estimate,
                        "outputTokensEstimate": req.output_tokens_estimate,
                        "tokensPerSecond": tokens_per_second,
                        "lastActivitySecondsAgo": round(
                            max(0.0, now_ts - (req.last_activity_at or req.started_at)), 1
                        ),
                        "phase": phase,
                        "waitingOnModelId": waiting_on,
                    })
            activity[state.model_id] = {
                "inFlight": count,
                "oldestSeconds": round(oldest, 1),
                "sources": source_counts,
                "slots": slots,
                "requests": requests_detail,
            }
    return {
        "activity": activity,
        "stuckThresholdSeconds": int(getattr(settings, "busy_timeout_seconds", 300)),
    }


@router.get(
    "/models/ollama-runtime",
    summary="Ollama runtime placement",
    description="Per-loaded-Ollama-model VRAM and GPU/CPU split from /api/ps (Ollama manages placement itself, so oCabra can't record a GPU index).",
)
async def get_ollama_runtime(
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """CPU/GPU split + VRAM + context per loaded Ollama model."""
    import httpx

    base = settings.ollama_base_url.rstrip("/")
    out: dict[str, dict] = {}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            data = (await client.get(f"{base}/api/ps")).json()
        for m in data.get("models", []):
            size = int(m.get("size", 0) or 0)
            vram = int(m.get("size_vram", 0) or 0)
            gpu_pct = round(vram / size * 100) if size else 0
            out[str(m.get("name") or m.get("model"))] = {
                "sizeMb": size // (1024 * 1024),
                "sizeVramMb": vram // (1024 * 1024),
                "gpuPct": gpu_pct,
                "cpuPct": max(0, 100 - gpu_pct),
                "contextLength": m.get("context_length"),
            }
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    return {"models": out}


@router.get(
    "/models/{model_id:path}/status",
    summary="Model load status",
    description=(
        "Lightweight load-status channel for oCabra-aware clients: current status "
        "and, when not loaded, the estimated wait to warm the model. Lets callers "
        "check/await a cold start out-of-band instead of parsing status frames from "
        "the OpenAI stream. OpenAI clients with an API key can read the same via "
        "GET /v1/models/{id} (ocabra extension)."
    ),
    responses={404: {"description": "Model not found"}},
)
async def get_model_status(
    model_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict:
    """Load status + expected wait for a model."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    status = state.status.value
    expected_wait = None
    if status != "loaded":
        expected_wait = await mm.get_expected_load_seconds(state.model_id)
    return {
        "modelId": state.model_id,
        "backendType": state.backend_type,
        "status": status,
        "loaded": status == "loaded",
        "expectedWaitSeconds": expected_wait,
        "gpu": state.current_gpu,
        "vramUsedMb": state.vram_used_mb,
        "errorMessage": state.error_message,
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

    # Without this hook a manually registered model would remain invisible in
    # ``/v1/models`` (clients resolve by profile_id, not by canonical model_id)
    # until the next restart, when ``ensure_default_profiles`` sweeps again.
    profile_registry = getattr(request.app.state, "profile_registry", None)
    if profile_registry is not None:
        await profile_registry.on_model_added(state.model_id)
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


@router.post(
    "/models/{model_id:path}/maintenance",
    summary="Pause or resume new requests for a model",
    description=(
        "While enabled, every new request for this model (and any of its derived "
        "``::<hash>`` worker keys) is rejected with 503 instead of being admitted — "
        "e.g. to free the GPU for manual benchmarking without real traffic contending "
        "for it. Requests already in flight when the window opens are not affected. "
        "In-memory only; does not survive an API restart."
    ),
    responses={404: {"description": "Model not found"}},
)
async def set_model_maintenance(
    model_id: str,
    body: MaintenanceToggle,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    mm.set_maintenance(model_id, body.enabled)
    return {"model_id": model_id, "maintenance": body.enabled}


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
        "llama.cpp models get a deterministic estimate from vram_planner.py "
        "(same architecture/KV/rs_cache/MTP-overhead model the runtime-presets "
        "panel uses, so both panels always agree); other backends fall back to "
        "a generic heuristic. Set run_probe=true for a runtime validation (vLLM only)."
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
        # Ollama manages its own storage in a separate service, so we must call
        # its HTTP API — the ``ollama`` CLI is NOT installed in the api
        # container (a previous ``ollama rm`` subprocess silently no-op'd and
        # the model reappeared on the next inventory sync).
        import httpx

        parts = model_id.split("/", 1)
        ollama_name = parts[1] if len(parts) == 2 else model_id
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.request(
                    "DELETE",
                    f"{settings.ollama_base_url}/api/delete",
                    json={"name": ollama_name},
                )
        except Exception as exc:  # noqa: BLE001 — surface connectivity failures
            raise HTTPException(
                status_code=502,
                detail=f"Could not reach Ollama to delete '{ollama_name}': {exc}",
            ) from exc
        # 404 = already gone; treat as success. Anything else is a real error.
        if resp.status_code not in (200, 404):
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Ollama refused to delete '{ollama_name}': "
                    f"{resp.status_code} {resp.text[:200]}"
                ),
            )
        return f"ollama:{ollama_name}"

    # HuggingFace-backed models (vllm, diffusers, whisper, transformers, tensorrt_llm…)
    #
    # Blobs live in up to three places on our layout:
    #   1) ``<models_dir>/huggingface/<repo>--<subdir>``: the download manager
    #      target for first-time pulls of pipeline artifacts.
    #   2) ``<hf_cache_dir>/hub/models--<repo>``: the standard HuggingFace
    #      Hub cache used by ``huggingface_hub`` / ``transformers`` when
    #      loading pretrained checkpoints on the fly (this is where the
    #      heavy safetensors typically end up).
    #   3) ``<hf_cache_dir>/models--<repo>``: a legacy layout that the
    #      snapshot_download path used to write to before v0.30-ish; still
    #      appears in older installations. Left in place for compatibility.
    #
    # Prior to 2026-09-17 only (1) was cleaned up, so deleting a vLLM
    # model from the UI left tens of GB of orphaned blobs in (2)/(3) — the
    # dashboard would report the model gone while ``df`` insisted the
    # disk was full. Now we clean up whichever locations exist.
    parts = model_id.split("/", 1)
    raw_model = parts[1] if len(parts) == 2 else model_id
    hf_dir_name = raw_model.replace("/", "--")
    hub_dir_name = f"models--{hf_dir_name}"
    models_dir = Path(settings.models_dir or "/data/models")
    hf_cache_dir = Path(settings.hf_cache_dir or "/data/hf_cache")

    deleted_paths: list[str] = []
    candidates: list[tuple[Path, Path]] = [
        # (path, base_dir_for_traversal_check)
        (models_dir / "huggingface" / hf_dir_name, models_dir / "huggingface"),
        (hf_cache_dir / "hub" / hub_dir_name, hf_cache_dir / "hub"),
        (hf_cache_dir / hub_dir_name, hf_cache_dir),
    ]
    for candidate, base in candidates:
        if not candidate.exists():
            continue
        if not _is_path_within_base(candidate, base):
            raise HTTPException(
                status_code=400,
                detail=f"Refusing to delete '{candidate}' outside its configured base",
            )
        await asyncio.to_thread(shutil.rmtree, candidate)
        deleted_paths.append(str(candidate))
    return ",".join(deleted_paths) if deleted_paths else None


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

    heuristic_mb = int(await backend.get_vram_estimate_mb(state.backend_model_id, extra_config=extra_config) or 0)

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
    """Deterministic GGUF VRAM estimator for llama.cpp models.

    Shares its architecture/KV/rs_cache/MTP-draft-overhead math with
    ``vram_planner.py`` — the same module the scheduler's pre-load admission
    check and the runtime-presets panel use — so this "Memoria y contexto"
    panel and the rest of the UI never disagree about the same model. Falls
    back to the legacy heuristic when the GGUF file cannot be located or
    parsed (e.g. only-on-disk via Ollama, or remote model) or when the
    configured ``gpu_layers`` is 0 (CPU-only, nothing to size on the GPU).
    """
    from ocabra.core.model_manager_helpers import (
        _LLAMA_CPP_DEFAULT_DRAFT_N_MAX,
        _LLAMA_CPP_SPEC_TYPES_NEEDING_RS_ROLLBACK,
    )
    from ocabra.schemas.backend_load import LlamaCppLoadConfig

    def _fallback_to_heuristic(notes: list[str], *, status: str = "ok", warning: str | None = None) -> dict[str, Any]:
        estimate.update(
            {
                "estimated_weights_mb": heuristic_mb or None,
                "fits_current_gpu": (
                    heuristic_mb <= total_vram_mb
                    if total_vram_mb is not None and heuristic_mb > 0
                    else None
                ),
                "status": status,
                "warning": warning,
                "notes": notes,
            }
        )
        return estimate

    nested = extra_config.get("llama_cpp") if isinstance(extra_config, dict) else None
    payload = nested if isinstance(nested, dict) else {}
    try:
        load_config = LlamaCppLoadConfig.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 — surface as warning, keep estimate
        return _fallback_to_heuristic(
            ["Estimación heurística (config inválida)."],
            status="warning",
            warning=f"Configuración llama.cpp inválida: {exc}",
        )

    gguf_path = _resolve_extra_config_path(extra_config, "model_path") or _resolve_local_model_path(
        state.backend_model_id
    )
    gguf_file: Path | None = None
    if gguf_path is not None:
        if gguf_path.is_file():
            # model_path points straight at the weights file — trust it
            # regardless of extension. Ollama's blob store names files by
            # content hash ("sha256-<hash>", no ".gguf" suffix) when
            # llama_cpp reuses weights pulled through Ollama.
            gguf_file = gguf_path
        elif gguf_path.is_dir():
            ggufs = sorted(gguf_path.rglob("*.gguf"))
            if ggufs:
                gguf_file = max(ggufs, key=lambda p: p.stat().st_size)

    arch = vp.arch_from_gguf(str(gguf_file)) if gguf_file is not None else None
    if arch is None or gguf_file is None:
        return _fallback_to_heuristic(["Estimación heurística: GGUF no encontrado o no parseable."])

    gpu_layers = load_config.gpu_layers if load_config.gpu_layers is not None else settings.llama_cpp_gpu_layers
    if gpu_layers <= 0:
        return _fallback_to_heuristic(["Estimación heurística: gpu_layers=0 (modelo en CPU)."])

    ctx_size = int(load_config.ctx_size or settings.llama_cpp_ctx_size)
    parallel_slots = max(1, load_config.parallel_slots or settings.llama_cpp_parallel_slots)
    kv_dtype_bytes = vp.kv_dtype_bytes_for_cache_type(load_config.cache_type_k or load_config.cache_type_v)

    rs_rollback_snapshots = 0
    if load_config.speculative is not None:
        spec_type = (load_config.speculative.spec_type or "").lower()
        if spec_type in _LLAMA_CPP_SPEC_TYPES_NEEDING_RS_ROLLBACK:
            rs_rollback_snapshots = (
                load_config.speculative.draft_n
                if load_config.speculative.draft_n is not None
                else _LLAMA_CPP_DEFAULT_DRAFT_N_MAX
            )

    file_size_mb = gguf_file.stat().st_size / (1024 * 1024)
    total_layers = max(1, arch.layers)
    offloaded = max(0, min(gpu_layers, total_layers))
    frac = offloaded / total_layers
    weights_mb = file_size_mb * frac
    kv_arch = (
        arch
        if offloaded == total_layers
        else vp.ModelArch(
            offloaded, arch.n_kv_heads, arch.key_length, arch.value_length,
            arch.hidden_size, arch.context_length, arch.weight_dtype_bytes,
            int(arch.kv_elements_per_token_override * frac) if arch.kv_elements_per_token_override else 0,
            arch.rs_cache_mb_per_slot * frac,
            arch.fixed_load_overhead_mb * frac,
        )
    )
    draft_overhead_mb = (
        vp.mtp_draft_overhead_mb(parallel_slots) if rs_rollback_snapshots > 0 else 0.0
    )
    rs_mb = vp.rs_cache_vram_mb(kv_arch, slots=parallel_slots, rollback_snapshots=rs_rollback_snapshots)
    fixed_mb = vp.fixed_overhead_vram_mb(kv_arch)
    kv_mb = vp.kv_vram_mb(kv_arch, max(1, ctx_size), kv_dtype_bytes)
    overhead_mb = vp.DEFAULT_OVERHEAD_MB + draft_overhead_mb
    total_mb = int(weights_mb + kv_mb + rs_mb + fixed_mb + overhead_mb)

    estimated_max_context_length = None
    if total_vram_mb is not None:
        estimated_max_context_length = vp.max_context_tokens(
            total_vram_mb,
            weights_mb,
            kv_arch,
            slots=parallel_slots,
            kv_dtype_bytes=kv_dtype_bytes,
            overhead_mb=overhead_mb,
            rs_rollback_snapshots=rs_rollback_snapshots,
        ) or None

    notes = [
        "Estimación determinística desde el header GGUF (arquitectura, KV cache, "
        "rs_cache híbrido y overhead de draft MTP, cuando aplica).",
        f"Desglose en MB — pesos: {int(weights_mb)} · KV: {int(kv_mb)}"
        + (f" · rs_cache: {int(rs_mb)}" if rs_mb else "")
        + (f" · overhead fijo: {int(fixed_mb)}" if fixed_mb else "")
        + (f" · overhead MTP: {int(draft_overhead_mb)}" if draft_overhead_mb else "")
        + f" · overhead base: {vp.DEFAULT_OVERHEAD_MB} · total: {total_mb}.",
    ]
    if load_config.cache_type_k or load_config.cache_type_v:
        notes.append(
            "KV cache cuantizado: K="
            f"{load_config.cache_type_k or 'f16'} · V={load_config.cache_type_v or 'f16'}."
        )
    if rs_rollback_snapshots > 0:
        notes.append(
            f"Decodificación especulativa autoreferente ({load_config.speculative.spec_type}): "
            f"reserva adicional de rs_cache y buffer de draft para {rs_rollback_snapshots} tokens."
        )

    estimate.update(
        {
            "estimated_weights_mb": int(weights_mb),
            "estimated_kv_cache_mb": int(kv_mb),
            "estimated_max_context_length": estimated_max_context_length,
            "model_loading_memory_mb": total_mb,
            "fits_current_gpu": (total_mb <= total_vram_mb if total_vram_mb is not None else None),
            "source": "gguf_header",
            "notes": notes,
        }
    )
    return estimate


async def _overlay_loaded_derived_state(mm, states: list) -> list:
    """Roll a loaded derived worker's live status onto its base catalog entry.

    Runtime presets load a per-request-size derived worker key
    (``base_model_id::<hash>``), not the base ``model_id`` itself — see
    ``ModelManager.add_model``/``_load_model``. ``list_states()`` (no
    ``include_derived``) only returns canonical/persisted entries, so the
    base entry's own ``status`` stays ``configured`` forever even while its
    derived worker is actively loaded and serving. The dashboard ("Modelos
    activos") and ``/v1/models``-adjacent UI read exactly this list, so a
    model could be resident and consuming real VRAM while the UI insists
    nothing is loaded. Reproduced live 2026-10-26: qwen3.8:27b showing
    "Sin modelos cargados" while GPU1 sat at 95% util / 17.7GB used.

    Overlays only the runtime-facing fields (status, current_gpu,
    vram_used_mb, loaded_at, last_request_at, worker_info) from the most
    recently loaded derived sibling; identity/config fields (model_id,
    display_name, capabilities, extra_config, ...) stay the base entry's own.
    """
    import dataclasses
    from datetime import UTC, datetime

    from ocabra.core.model_manager import ModelStatus

    min_dt = datetime.min.replace(tzinfo=UTC)
    all_states = await mm.list_states(include_derived=True)
    loaded_derived_by_base: dict[str, Any] = {}
    for candidate in all_states:
        if candidate.status != ModelStatus.LOADED:
            continue
        base_id, sep, _suffix = candidate.model_id.partition("::")
        if not sep:
            continue
        existing = loaded_derived_by_base.get(base_id)
        if existing is None or (candidate.loaded_at or min_dt) > (existing.loaded_at or min_dt):
            loaded_derived_by_base[base_id] = candidate

    overlaid = []
    for state in states:
        derived = loaded_derived_by_base.get(state.model_id) if state.status != ModelStatus.LOADED else None
        if derived is None:
            overlaid.append(state)
            continue
        overlaid.append(
            dataclasses.replace(
                state,
                status=derived.status,
                current_gpu=derived.current_gpu,
                vram_used_mb=derived.vram_used_mb,
                loaded_at=derived.loaded_at,
                last_request_at=derived.last_request_at or state.last_request_at,
                worker_info=derived.worker_info,
                # The derived worker's capabilities reflect what the live
                # process actually supports (probed from its real chat
                # template — see llama_cpp_backend._template_caps) — the
                # base entry's own cached capabilities are stale defaults
                # from before it was ever loaded.
                capabilities=derived.capabilities,
                error_message=None,
            )
        )
    return overlaid


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
                "video_upscaling",
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
    allowed_bases = [Path(settings.models_dir or "/data/models")]
    if settings.ollama_shared_models_dir:
        allowed_bases.append(Path(settings.ollama_shared_models_dir))
    # llama.cpp stores its load options — including model_path — nested
    # under extra_config["llama_cpp"], not at the top level like other
    # backends' extra_config. Check both so GGUF weights reused from
    # Ollama's blob store (hence the extra allowed base above, outside
    # models_dir) still resolve for size/memory-estimate lookups.
    nested = extra_config.get("llama_cpp")
    nested = nested if isinstance(nested, dict) else {}
    for key in keys:
        raw = extra_config.get(key) or nested.get(key)
        if not raw or not isinstance(raw, str):
            continue
        path = Path(raw)
        if path.exists() and any(_is_path_within_base(path, base) for base in allowed_bases):
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
