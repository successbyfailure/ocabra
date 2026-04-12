"""
GET /api/tags — list installed models in Ollama-compatible format.

With the profile system enabled, this endpoint exposes **enabled profiles**
rather than raw model entries.

When federation is enabled and peers are online, remote models from federated
nodes are merged into the listing with deduplication.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    List all enabled profiles in Ollama /api/tags format.

    Filters profiles by the caller's group membership unless the caller is
    an admin. When federation is enabled, remote models from online peers
    are merged with deduplication.

    Returns:
      {"models": [{"name": ..., "model": ..., "size": ..., "details": {...}}, ...]}
    """
    model_manager = request.app.state.model_manager
    profile_registry = getattr(request.app.state, "profile_registry", None)
    federation_manager = getattr(request.app.state, "federation_manager", None)

    # If profile registry is available, list profiles; otherwise fall back to
    # the legacy model-based listing.
    if profile_registry is not None:
        return await _list_tags_from_profiles(
            model_manager,
            profile_registry,
            user,
            federation_manager=federation_manager,
        )

    return await _list_tags_legacy(model_manager, user)


async def _list_tags_from_profiles(
    model_manager,
    profile_registry,
    user: UserContext,
    *,
    federation_manager=None,
) -> dict:
    """Build Ollama-format tags from enabled profiles."""
    enabled_profiles = await profile_registry.list_enabled()

    models: list[dict] = []
    local_model_ids: set[str] = set()
    local_profile_ids: set[str] = set()
    entry_index_by_model_id: dict[str, int] = {}

    for profile in enabled_profiles:
        # Access control
        if not user.is_admin and profile.profile_id not in user.accessible_model_ids:
            continue

        base_state = await model_manager.get_state(profile.base_model_id)

        backend_model_id = base_state.backend_model_id if base_state else profile.base_model_id
        loaded_at = base_state.loaded_at if base_state and base_state.loaded_at else None
        modified_at = _to_iso_z(loaded_at)
        vram_mb = base_state.vram_used_mb if base_state else 0
        size_bytes = _estimate_size_bytes(backend_model_id, vram_mb)

        local_model_ids.add(profile.base_model_id)
        local_profile_ids.add(profile.profile_id)

        entry: dict[str, Any] = {
            "name": profile.profile_id,
            "model": profile.profile_id,
            "modified_at": modified_at,
            "size": size_bytes,
            "digest": "sha256:" + hashlib.sha256(
                profile.profile_id.encode("utf-8")
            ).hexdigest(),
            "details": {
                "parent_model": profile.base_model_id,
                "format": _infer_format(backend_model_id),
                "family": profile.category,
                "families": [profile.category],
                "parameter_size": "unknown",
                "quantization_level": "F16",
            },
            "loaded": bool(base_state and base_state.status.value == "loaded"),
        }

        entry_index_by_model_id[profile.base_model_id] = len(models)
        models.append(entry)

    # ── Federation: merge remote models ─────────────────────────
    if federation_manager is not None:
        remote_models = federation_manager.get_remote_models()
        for model_id, peers in remote_models.items():
            if model_id in local_model_ids:
                # Model exists locally — annotate with remote availability
                idx = entry_index_by_model_id.get(model_id)
                if idx is not None and idx < len(models):
                    models[idx]["federation"] = {
                        "remote": False,
                        "also_available_on": [
                            {"node_name": p.name, "node_id": p.peer_id}
                            for p in peers
                        ],
                    }
            else:
                # Check if any profile from remote overlaps with local profiles
                remote_profiles: set[str] = set()
                for peer in peers:
                    for m in peer.models:
                        if m.get("model_id") == model_id:
                            remote_profiles.update(m.get("profiles", []))
                if remote_profiles & local_profile_ids:
                    continue

                first_peer = peers[0]
                display_id = model_id
                for m in first_peer.models:
                    if m.get("model_id") == model_id:
                        model_profiles = m.get("profiles", [])
                        if model_profiles:
                            display_id = model_profiles[0]
                        break

                models.append(
                    {
                        "name": display_id,
                        "model": display_id,
                        "modified_at": _to_iso_z(None),
                        "size": 0,
                        "digest": "sha256:" + hashlib.sha256(
                            model_id.encode("utf-8")
                        ).hexdigest(),
                        "details": {
                            "parent_model": model_id,
                            "format": "unknown",
                            "family": "remote",
                            "families": ["remote"],
                            "parameter_size": "unknown",
                            "quantization_level": "unknown",
                        },
                        "loaded": True,
                        "federation": {
                            "remote": True,
                            "node_name": first_peer.name,
                            "node_id": first_peer.peer_id,
                            "available_on": [
                                {"node_name": p.name, "node_id": p.peer_id}
                                for p in peers
                            ],
                        },
                    }
                )

    return {"models": models}


async def _list_tags_legacy(model_manager, user: UserContext) -> dict:
    """Legacy listing when profile registry is not available."""
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
        installed_details = [
            {"name": _mapper.to_ollama(state.model_id), "size": 0, "modified_at": ""}
            for state in states
        ]
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

        if not user.is_admin and state is None:
            continue

        canonical_id = state.model_id if state else build_model_ref("ollama", ollama_name)
        backend_model_id = state.backend_model_id if state else ollama_name

        remote_modified = str(item.get("modified_at") or "")
        modified_at = (
            _to_iso_z(state.loaded_at if state else None)
            if state and state.loaded_at
            else (remote_modified or _to_iso_z(None))
        )
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
