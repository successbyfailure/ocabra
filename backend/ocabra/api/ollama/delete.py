"""
DELETE /api/delete — remove a model configuration.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ocabra.registry.ollama_registry import OllamaRegistry

from ._mapper import OllamaNameMapper

router = APIRouter()
_mapper = OllamaNameMapper()
_registry = OllamaRegistry()


class DeleteRequest(BaseModel):
    name: str


@router.delete("/delete", summary="Delete a model")
async def delete_model(body: DeleteRequest, request: Request) -> dict:
    """
    Delete a model by Ollama name.

    Parameters:
      - name: Ollama model name to remove.

    Response:
      - {"status": "success"}
    """
    model_id = _mapper.to_internal(body.name)
    model_manager = request.app.state.model_manager

    state = await model_manager.get_state(model_id)
    if state is not None:
        await model_manager.delete_model(model_id)

    try:
        await _registry.delete(body.name)
    except Exception as exc:
        if state is None:
            raise HTTPException(status_code=404, detail={"error": f"model '{body.name}' not found"}) from exc

    return {"status": "success"}
