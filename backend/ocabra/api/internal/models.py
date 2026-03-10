from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["models"])


class ModelPatch(BaseModel):
    display_name: str | None = None
    load_policy: str | None = None
    auto_reload: bool | None = None
    preferred_gpu: int | None = None


class AddModelRequest(BaseModel):
    model_id: str
    backend_type: str
    display_name: str | None = None
    load_policy: str = "on_demand"
    auto_reload: bool = False
    preferred_gpu: int | None = None


@router.get("/models")
async def list_models(request: Request) -> list[dict]:
    """List all configured models and their runtime state."""
    mm = request.app.state.model_manager
    states = await mm.list_states()
    return [s.to_dict() for s in states]


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> dict:
    """Get state of a specific model."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return state.to_dict()


@router.post("/models")
async def add_model(body: AddModelRequest, request: Request) -> dict:
    """Register a new model configuration."""
    mm = request.app.state.model_manager
    state = await mm.add_model(
        model_id=body.model_id,
        backend_type=body.backend_type,
        display_name=body.display_name,
        load_policy=body.load_policy,
        auto_reload=body.auto_reload,
        preferred_gpu=body.preferred_gpu,
    )
    return state.to_dict()


@router.post("/models/{model_id:path}/load")
async def load_model(model_id: str, request: Request) -> dict:
    """Load a model onto a GPU."""
    mm = request.app.state.model_manager
    state = await mm.get_state(model_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    from ocabra.core.model_manager import ModelStatus
    if state.status == ModelStatus.LOADED:
        raise HTTPException(status_code=409, detail="Model is already loaded")
    updated = await mm.load(model_id)
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
