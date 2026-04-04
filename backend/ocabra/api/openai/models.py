"""
GET /v1/models — list available models in OpenAI format.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from ocabra.api._deps_auth import UserContext

from ._deps import get_model_manager, get_openai_user, resolve_model

router = APIRouter()


@router.get("/models", summary="List models")
async def list_models(
    request: Request,
    user: UserContext = Depends(get_openai_user),
) -> dict:
    """
    List all configured/loaded models in OpenAI-compatible format.

    Filters models by the caller's group membership unless the caller is an admin.
    Includes an 'ocabra' extension field with status, capabilities, and GPU info.
    """
    from ocabra.core.model_manager import ModelStatus

    model_manager = get_model_manager(request)
    all_states = await model_manager.list_states()

    visible_statuses = {
        ModelStatus.LOADED,
        ModelStatus.CONFIGURED,
        ModelStatus.LOADING,
        ModelStatus.UNLOADED,
    }

    if user.is_admin:
        states = all_states
    else:
        states = [s for s in all_states if s.model_id in user.accessible_model_ids]

    data = []
    for state in states:
        if state.status not in visible_statuses:
            continue
        data.append({
            "id": state.model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ocabra",
            "ocabra": {
                "status": state.status.value,
                "capabilities": state.capabilities.to_dict(),
                "load_policy": state.load_policy.value,
                "gpu": state.current_gpu,
                "vram_used_mb": state.vram_used_mb,
                "display_name": state.display_name,
                "backend_model_id": state.backend_model_id,
            },
        })

    return {"object": "list", "data": data}


@router.get("/models/{model_id:path}", summary="Retrieve a model")
async def get_model(
    model_id: str,
    request: Request,
    user: UserContext = Depends(get_openai_user),
) -> dict:
    """Retrieve a single model by ID or by backend model name alias."""
    model_manager = get_model_manager(request)
    _, state = await resolve_model(model_manager, model_id, user=user)
    if not state:
        from ._deps import _openai_error

        raise _openai_error(
            f"The model '{model_id}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    return {
        "id": state.model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ocabra",
        "ocabra": {
            "status": state.status.value,
            "capabilities": state.capabilities.to_dict(),
            "load_policy": state.load_policy.value,
            "gpu": state.current_gpu,
            "vram_used_mb": state.vram_used_mb,
            "display_name": state.display_name,
            "backend_model_id": state.backend_model_id,
        },
    }
