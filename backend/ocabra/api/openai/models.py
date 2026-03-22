"""
GET /v1/models — list available models in OpenAI format.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

from ._deps import get_model_manager

router = APIRouter()


@router.get("/models", summary="List models")
async def list_models(request: Request) -> dict:
    """
    List all configured/loaded models in OpenAI-compatible format.
    Includes an 'ocabra' extension field with status, capabilities, and GPU info.
    """
    from ocabra.core.model_manager import ModelStatus

    model_manager = get_model_manager(request)
    states = await model_manager.list_states()

    visible_statuses = {
        ModelStatus.LOADED,
        ModelStatus.CONFIGURED,
        ModelStatus.LOADING,
        ModelStatus.UNLOADED,
    }
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
            },
        })

    return {"object": "list", "data": data}


@router.get("/models/{model_id:path}", summary="Retrieve a model")
async def get_model(model_id: str, request: Request) -> dict:
    """Retrieve a single model by ID."""
    from ocabra.core.model_manager import ModelStatus

    model_manager = get_model_manager(request)
    state = await model_manager.get_state(model_id)
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
        },
    }
