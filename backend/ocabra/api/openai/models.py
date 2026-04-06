"""
GET /v1/models — list available models in OpenAI format.

With the profile system enabled, this endpoint exposes **profiles** rather
than raw model entries.  Each profile is presented as an OpenAI-compatible
"model" object whose ``id`` is the ``profile_id``.
"""

from __future__ import annotations

import time

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

    for profile in enabled_profiles:
        # Access control: non-admin users only see profiles in their set
        if not user.is_admin and profile.profile_id not in user.accessible_model_ids:
            continue

        base_state = await model_manager.get_state(profile.base_model_id)
        if base_state is None:
            continue
        if base_state.status not in visible_statuses:
            continue

        data.append(
            {
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
