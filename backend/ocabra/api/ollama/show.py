"""
POST /api/show — return details for a model in Ollama format.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ._mapper import OllamaNameMapper, resolve_model

router = APIRouter()
_mapper = OllamaNameMapper()


class ShowRequest(BaseModel):
    name: str


@router.post("/show", summary="Show model details")
async def show_model(body: ShowRequest, request: Request) -> dict:
    """
    Return model details in Ollama /api/show response format.

    Parameters:
      - name: Ollama model name, e.g. llama3.2:3b

    Response:
      - modelfile, parameters, template, details, model_info
    """
    model_manager = request.app.state.model_manager
    model_id, state = await resolve_model(model_manager, body.name)
    if state is None:
        state = await model_manager.get_state(model_id)
    if state is None:
        raise HTTPException(status_code=404, detail={"error": f"model '{body.name}' not found"})

    ollama_name = state.backend_model_id if state.backend_type == "ollama" else _mapper.to_ollama(state.model_id)
    family = ollama_name.split(":", 1)[0]

    return {
        "license": "",
        "modelfile": f"FROM {ollama_name}\n",
        "parameters": "",
        "template": "{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "safetensors",
            "family": family,
            "families": [family],
            "parameter_size": ollama_name.split(":", 1)[1].upper() if ":" in ollama_name else "unknown",
            "quantization_level": "F16",
        },
        "model_info": {
            "general.architecture": family,
            "general.parameter_count": 0,
            "general.file_type": "safetensors",
            "ocabra.model_id": state.model_id,
            "ocabra.backend_type": state.backend_type,
            "ocabra.backend_model_id": state.backend_model_id,
            "ocabra.context_length": state.capabilities.context_length,
            "ocabra.capabilities": state.capabilities.to_dict(),
        },
    }
