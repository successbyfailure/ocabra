import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ocabra.config import settings
from ocabra.core.model_ref import parse_model_ref
from ocabra.registry.ollama_registry import OllamaRegistry

router = APIRouter(tags=["models"])
_ollama_registry = OllamaRegistry()


class ModelPatch(BaseModel):
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


@router.get("/models")
async def list_models(request: Request) -> list[dict]:
    """List all configured models and their runtime state."""
    mm = request.app.state.model_manager
    await _sync_ollama_inventory(mm)
    states = await mm.list_states()
    ollama_sizes = await _get_ollama_sizes_bytes()
    payloads = []
    for state in states:
        item = state.to_dict()
        item["disk_size_bytes"] = await _resolve_disk_size_bytes(state.model_id, ollama_sizes)
        payloads.append(item)
    return payloads


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> dict:
    """Get state of a specific model."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    item = state.to_dict()
    ollama_sizes = await _get_ollama_sizes_bytes() if state.backend_type == "ollama" else {}
    item["disk_size_bytes"] = await _resolve_disk_size_bytes(state.model_id, ollama_sizes)
    return item


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
    updated = await mm.update_config(model_id, patch)
    return updated.to_dict()


@router.delete("/models/{model_id:path}")
async def delete_model(model_id: str, request: Request) -> dict:
    """Remove a model configuration and optionally its files."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    await mm.delete_model(model_id)
    return {"ok": True}


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
        loaded = await _ollama_registry.list_loaded()
    except Exception:
        return
    await model_manager.sync_ollama_inventory(installed, loaded)


async def _resolve_disk_size_bytes(
    model_id: str,
    ollama_sizes: dict[str, int],
) -> int | None:
    backend_type, backend_model_id = parse_model_ref(model_id)
    if backend_type == "ollama":
        return ollama_sizes.get(backend_model_id.strip().lower())

    path = _resolve_local_model_path(backend_model_id)
    if path is None or not path.exists():
        return None
    return await asyncio.to_thread(_compute_path_size_bytes, path)


def _resolve_local_model_path(model_id: str) -> Path | None:
    base = Path(settings.models_dir)
    direct = base / model_id
    if direct.exists():
        return direct

    # Hugging Face local layout: /data/models/huggingface/org--repo[--artifact-stem]
    if "::" in model_id:
        repo_id, variant_stem = model_id.split("::", 1)
        hf_dir_name = f"{repo_id.replace('/', '--')}--{variant_stem}"
    else:
        hf_dir_name = model_id.replace("/", "--")
    hf_layout = base / "huggingface" / hf_dir_name
    if hf_layout.exists():
        return hf_layout

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


def _compute_path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total
