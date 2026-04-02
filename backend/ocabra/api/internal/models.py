import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from ocabra.config import settings
from ocabra.core.model_ref import parse_model_ref
from ocabra.registry.ollama_registry import OllamaRegistry

router = APIRouter(tags=["models"])
_ollama_registry = OllamaRegistry()


class ModelPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = None
    load_policy: str | None = None
    auto_reload: bool | None = None
    preferred_gpu: int | None = None
    extra_config: dict | None = None


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


@router.get("/models")
async def list_models(request: Request) -> list[dict]:
    """List all configured models and their runtime state."""
    mm = request.app.state.model_manager
    await _sync_ollama_inventory(mm)
    states = await mm.list_states()
    ollama_sizes = await _get_ollama_sizes_bytes()
    payloads = []
    for state in states:
        item = await _serialize_model_state(request, state, ollama_sizes)
        payloads.append(item)
    return payloads


@router.get("/models/storage")
async def get_models_storage() -> dict:
    """Return storage usage for the models directory filesystem."""
    models_dir = Path(settings.models_dir or "/data/models")
    try:
        stats = await asyncio.to_thread(os.statvfs, models_dir)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read models storage stats: {exc}") from exc

    total_bytes = int(stats.f_frsize * stats.f_blocks)
    free_bytes = int(stats.f_frsize * stats.f_bavail)
    used_bytes = max(0, total_bytes - free_bytes)
    return {
        "path": str(models_dir),
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "free_bytes": free_bytes,
    }


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> dict:
    """Get state of a specific model."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    ollama_sizes = await _get_ollama_sizes_bytes() if state.backend_type == "ollama" else {}
    return await _serialize_model_state(request, state, ollama_sizes)


@router.post("/models")
async def add_model(body: AddModelRequest, request: Request) -> dict:
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


@router.post("/models/{model_id:path}/load")
async def load_model(model_id: str, request: Request) -> dict:
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


@router.post("/models/{model_id:path}/unload")
async def unload_model(model_id: str, request: Request) -> dict:
    """Unload a model from GPU."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    await mm.unload(model_id, reason="manual")
    updated = await mm.get_state(model_id)
    return updated.to_dict()


@router.patch("/models/{model_id:path}")
async def update_model(model_id: str, body: ModelPatch, request: Request) -> dict:
    """Update model configuration."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        updated = await mm.update_config(model_id, patch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return updated.to_dict()


@router.post("/models/{model_id:path}/memory-estimate")
async def estimate_model_memory(
    model_id: str,
    body: ModelMemoryEstimateRequest,
    request: Request,
) -> dict:
    """Estimate memory requirements for the current backend/config combination."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    extra_config = body.extra_config if isinstance(body.extra_config, dict) else state.extra_config or {}
    return await _build_model_memory_estimate(
        request=request,
        state=state,
        extra_config=extra_config,
        preferred_gpu=body.preferred_gpu,
        run_probe=body.run_probe,
    )


@router.delete("/models/{model_id:path}")
async def delete_model(
    model_id: str,
    request: Request,
    delete_files: bool = Query(default=True),
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
                "ollama", "rm", ollama_name,
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
        path = _resolve_extra_config_path(state.extra_config, "engine_dir") or _resolve_tensorrt_engine_path(backend_model_id)
        if path is None or not path.exists():
            return None
        return await asyncio.to_thread(_compute_path_size_bytes, path)

    path = (
        _resolve_extra_config_path(state.extra_config, "model_path", "base_model_id")
        or _resolve_local_model_path(backend_model_id)
    )
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
    gpu_state = await gpu_manager.get_state(gpu_index) if gpu_manager is not None else None
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
                    "estimated_kv_cache_mb": profile.get("available_kv_cache_mb") or estimate["estimated_kv_cache_mb"],
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

    if state.backend_type == "tensorrt_llm":
        engine_path = _resolve_extra_config_path(extra_config, "engine_dir") or _resolve_tensorrt_engine_path(
            state.backend_model_id
        )
        tp_size = _resolve_tensor_parallel_size(state, extra_config)
        estimate.update(
            {
                "estimated_engine_mb_per_gpu": heuristic_mb or None,
                "engine_present": bool(engine_path and engine_path.exists()),
                "fits_current_gpu": (
                    heuristic_mb <= total_vram_mb if total_vram_mb is not None and heuristic_mb > 0 else None
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
                heuristic_mb <= total_vram_mb if total_vram_mb is not None and heuristic_mb > 0 else None
            ),
            "notes": ["Estimación heurística basada en el tamaño de los artefactos locales del modelo."],
        }
    )
    return estimate


async def _serialize_model_state(
    request: Request,
    state,
    ollama_sizes: dict[str, int],
) -> dict:
    item = state.to_dict()
    item["disk_size_bytes"] = await _resolve_disk_size_bytes(state, ollama_sizes)
    item["capabilities"] = await _resolve_capabilities_payload(request, state, item["capabilities"])
    return item


async def _resolve_capabilities_payload(
    request: Request,
    state,
    current_payload: dict,
) -> dict:
    has_meaningful_caps = any(
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
            "tts",
        )
    ) or int(current_payload.get("context_length") or 0) > 0

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
        engine_path = _resolve_extra_config_path(extra_config, "engine_dir") or _resolve_tensorrt_engine_path(
            state.backend_model_id
        )
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
        part.capitalize() if index else part
        for index, part in enumerate(value.split("_"))
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
    model_path = (
        _resolve_extra_config_path(extra_config, "model_path", "base_model_id")
        or _resolve_local_model_path(backend_model_id)
    )
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
        path for path in base.rglob("*.gguf")
        if path.name == leaf or any(path.stem == stem or path.stem.endswith(stem) for stem in candidate_stems)
    ]
    if gguf_candidates:
        return max(gguf_candidates, key=lambda path: path.stat().st_mtime)

    # Optional HF cache layout fallback.
    hf_cache_dir = (settings.hf_cache_dir or "").strip()
    if hf_cache_dir:
        cache_root = Path(hf_cache_dir) / "hub" / f"models--{model_id.split('::', 1)[0].replace('/', '--')}"
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
