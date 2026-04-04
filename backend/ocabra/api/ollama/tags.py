"""
GET /api/tags — list installed models in Ollama-compatible format.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from ocabra.api._deps_auth import UserContext
from ocabra.config import settings
from ocabra.core.model_ref import build_model_ref
from ocabra.registry.ollama_registry import OllamaRegistry

from ._mapper import OllamaNameMapper
from ._shared import get_ollama_user

router = APIRouter()
_mapper = OllamaNameMapper()
_registry = OllamaRegistry()


@router.get("/tags", summary="List models")
async def list_tags(
    request: Request,
    user: UserContext = Depends(get_ollama_user),
) -> dict:
    """
    List all configured models in Ollama /api/tags format.

    Filters models by the caller's group membership unless the caller is an admin.

    Returns:
      {"models": [{"name": ..., "model": ..., "size": ..., "details": {...}}, ...]}
    """
    model_manager = request.app.state.model_manager
    all_states = await model_manager.list_states()

    if user.is_admin:
        filtered_states = all_states
    else:
        filtered_states = [s for s in all_states if s.model_id in user.accessible_model_ids]

    states = filtered_states
    by_id = {state.model_id: state for state in states}

    try:
        installed_details = await _registry.list_installed_details()
        loaded = set(await _registry.list_loaded())
    except Exception:
        installed_details = [{"name": _mapper.to_ollama(state.model_id), "size": 0, "modified_at": ""} for state in states]
        loaded = {str(item.get("name") or "") for item in installed_details}

    models: list[dict] = []
    for item in installed_details:
        ollama_name = str(item.get("name") or "")
        if not ollama_name:
            continue

        state = by_id.get(build_model_ref("ollama", ollama_name))
        if state is None:
            mapped_id = _mapper.to_internal(ollama_name)
            state = by_id.get(mapped_id)

        # Non-admin callers only see models that are in their accessible set
        # (reflected by `by_id` which is already filtered). Skip Ollama-native
        # models that have no corresponding oCabra state visible to this user.
        if not user.is_admin and state is None:
            continue

        canonical_id = state.model_id if state else build_model_ref("ollama", ollama_name)
        backend_model_id = state.backend_model_id if state else ollama_name

        remote_modified = str(item.get("modified_at") or "")
        modified_at = _to_iso_z(state.loaded_at if state else None) if state and state.loaded_at else (remote_modified or _to_iso_z(None))
        family = ollama_name.split(":", 1)[0]
        is_loaded = ollama_name in loaded
        size_bytes = int(item.get("size") or 0)
        if size_bytes <= 0:
            size_bytes = _estimate_size_bytes(backend_model_id, state.vram_used_mb if state else 0)
        models.append(
            {
                "name": ollama_name,
                "model": ollama_name,
                "modified_at": modified_at,
                "size": size_bytes,
                "digest": f"sha256:{hashlib.sha256(canonical_id.encode('utf-8')).hexdigest()}",
                "details": {
                    "parent_model": "",
                    "format": _infer_format(backend_model_id),
                    "family": family,
                    "families": [family],
                    "parameter_size": _infer_parameter_size(ollama_name),
                    "quantization_level": "F16",
                },
                "loaded": is_loaded,
            }
        )

    return {"models": models}


def _to_iso_z(value: datetime | None) -> str:
    dt = value or datetime.now(UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _estimate_size_bytes(model_id: str, vram_used_mb: int) -> int:
    model_dir = Path(settings.models_dir) / model_id
    if model_dir.exists():
        size = sum(
            path.stat().st_size
            for path in model_dir.rglob("*")
            if path.is_file() and path.suffix in {".safetensors", ".bin", ".gguf"}
        )
        if size > 0:
            return int(size)

    return int(max(0, vram_used_mb) * 1024 * 1024)


def _infer_format(model_id: str) -> str:
    lower = model_id.lower()
    if lower.endswith(".gguf"):
        return "gguf"
    if "diffusion" in lower:
        return "safetensors"
    return "safetensors"


def _infer_parameter_size(ollama_name: str) -> str:
    if ":" not in ollama_name:
        return "unknown"
    tag = ollama_name.split(":", 1)[1].strip()
    return tag.upper()
