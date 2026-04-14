"""
GET /v1/models — list available models in OpenAI format.

With the profile system enabled, this endpoint exposes **profiles** rather
than raw model entries.  Each profile is presented as an OpenAI-compatible
"model" object whose ``id`` is the ``profile_id``.

When federation is enabled and peers are online, remote models from federated
nodes are merged into the listing with deduplication.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from ocabra.api._deps_auth import UserContext

from ._deps import (
    _openai_error,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    resolve_profile,
)

router = APIRouter()


def _get_federation_manager(request: Request):
    """Return the FederationManager if enabled, or None."""
    return getattr(request.app.state, "federation_manager", None)


def _build_federation_nodes_metadata(
    peers: list,
) -> list[dict[str, Any]]:
    """Build a list of node metadata dicts from PeerState objects."""
    return [
        {
            "node_name": p.name,
            "node_id": p.peer_id,
        }
        for p in peers
    ]


@router.get("/models", summary="List models")
async def list_models(
    request: Request,
    user: UserContext = Depends(get_openai_user),
) -> dict:
    """
    List all enabled profiles as OpenAI-compatible model objects.

    Filters profiles by the caller's group membership unless the caller is
    an admin.  Includes an ``ocabra`` extension field with category, status,
    capabilities, and display name.

    When federation is enabled, remote models from online peers are merged
    into the listing. Duplicate models (same model_id available both locally
    and remotely) appear once with metadata about all available nodes.
    """
    from ocabra.core.model_manager import ModelStatus

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)
    enabled_profiles = await profile_registry.list_enabled()

    visible_statuses = {
        ModelStatus.LOADED,
        ModelStatus.CONFIGURED,
        ModelStatus.LOADING,
        ModelStatus.UNLOADED,
    }

    data = []
    now_ts = int(time.time())

    # Track local model_ids and profile_ids for deduplication with remote
    local_model_ids: set[str] = set()
    local_profile_ids: set[str] = set()
    # Map from model entry index to model_id for federation annotation
    entry_index_by_model_id: dict[str, int] = {}

    for profile in enabled_profiles:
        # Access control: non-admin users only see profiles in their set.
        # Check both profile_id and base_model_id since group_models stores
        # canonical model_ids while clients use profile_ids.
        if (
            not user.is_admin
            and profile.profile_id not in user.accessible_model_ids
            and profile.base_model_id not in user.accessible_model_ids
        ):
            continue

        base_state = await model_manager.get_state(profile.base_model_id)
        if base_state is None:
            continue
        if base_state.status not in visible_statuses:
            continue

        local_model_ids.add(profile.base_model_id)
        local_profile_ids.add(profile.profile_id)

        entry: dict[str, Any] = {
            "id": profile.profile_id,
            "object": "model",
            "created": now_ts,
            "owned_by": "ocabra",
            "ocabra": {
                "category": profile.category,
                "status": base_state.status.value,
                "capabilities": base_state.capabilities.to_dict(),
                "display_name": profile.display_name or base_state.display_name,
                "base_model_id": profile.base_model_id,
                "load_policy": base_state.load_policy.value,
                "gpu": base_state.current_gpu,
                "vram_used_mb": base_state.vram_used_mb,
            },
        }

        entry_index_by_model_id[profile.base_model_id] = len(data)
        data.append(entry)

    # ── Federation: merge remote models ─────────────────────────
    fm = _get_federation_manager(request)
    if fm is not None:
        remote_models = fm.get_remote_models()
        for model_id, peers in remote_models.items():
            if model_id in local_model_ids:
                # Model exists locally — annotate with remote availability
                idx = entry_index_by_model_id.get(model_id)
                if idx is not None and idx < len(data):
                    data[idx]["federation"] = {
                        "remote": False,
                        "also_available_on": _build_federation_nodes_metadata(peers),
                    }
            else:
                # Check if any profile_id from the remote model matches a local one
                remote_profiles = set()
                for peer in peers:
                    for m in peer.models:
                        if m.get("model_id") == model_id:
                            remote_profiles.update(m.get("profiles", []))
                if remote_profiles & local_profile_ids:
                    # Already listed under a local profile — skip
                    continue

                # Remote-only model: add it with federation metadata
                # Use the first profile_id if available, otherwise model_id
                first_peer = peers[0]
                display_id = model_id
                for m in first_peer.models:
                    if m.get("model_id") == model_id:
                        model_profiles = m.get("profiles", [])
                        if model_profiles:
                            display_id = model_profiles[0]
                        break

                data.append(
                    {
                        "id": display_id,
                        "object": "model",
                        "created": now_ts,
                        "owned_by": first_peer.name,
                        "federation": {
                            "remote": True,
                            "node_name": first_peer.name,
                            "node_id": first_peer.peer_id,
                            "available_on": _build_federation_nodes_metadata(peers),
                        },
                    }
                )

    return {"object": "list", "data": data}


@router.get("/models/{model_id:path}", summary="Retrieve a model")
async def get_model(
    model_id: str,
    request: Request,
    user: UserContext = Depends(get_openai_user),
) -> dict:
    """Retrieve a single model by profile_id (or legacy model_id with fallback)."""
    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    try:
        profile, state = await resolve_profile(
            model_id,
            model_manager,
            profile_registry,
            user=user,
        )
    except Exception as exc:
        raise _openai_error(
            f"The model '{model_id}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        ) from exc

    return {
        "id": profile.profile_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ocabra",
        "ocabra": {
            "category": profile.category,
            "status": state.status.value,
            "capabilities": state.capabilities.to_dict(),
            "display_name": profile.display_name or state.display_name,
            "base_model_id": profile.base_model_id,
            "load_policy": state.load_policy.value,
            "gpu": state.current_gpu,
            "vram_used_mb": state.vram_used_mb,
        },
    }
