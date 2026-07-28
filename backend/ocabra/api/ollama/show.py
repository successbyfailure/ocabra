"""
POST /api/show — return details for a model in Ollama format.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ocabra.api._deps_auth import UserContext

from ._mapper import OllamaNameMapper, resolve_model
from ._shared import get_ollama_user

router = APIRouter()
_mapper = OllamaNameMapper()


class ShowRequest(BaseModel):
    name: str


@router.post("/show", summary="Show model details")
async def show_model(
    body: ShowRequest,
    request: Request,
    user: UserContext = Depends(get_ollama_user),
) -> dict:
    """
    Return model details in Ollama /api/show response format.

    Parameters:
      - name: Ollama model name, e.g. llama3.2:3b

    Response:
      - modelfile, parameters, template, details, model_info
    """
    model_manager = request.app.state.model_manager
    model_id, state = await resolve_model(model_manager, body.name, user=user)
    if state is None:
        state = await model_manager.get_state(model_id)
    if state is None:
        raise HTTPException(status_code=404, detail={"error": f"model '{body.name}' not found"})

    ollama_name = state.backend_model_id if state.backend_type == "ollama" else _mapper.to_ollama(state.model_id)
    family = ollama_name.split(":", 1)[0]

    # Ollama exposes a top-level ``capabilities`` array (e.g.
    # ["completion", "vision", "tools", "thinking", "embedding"]) that clients
    # like OpenWebUI read to enable vision uploads, tool calling, etc. Build it
    # from oCabra's resolved capabilities so those features light up through the
    # Ollama-compat path too (otherwise a multimodal model like gemma4 shows up
    # as text-only).
    caps = state.capabilities
    capabilities: list[str] = []
    if getattr(caps, "completion", False) or getattr(caps, "chat", False):
        capabilities.append("completion")
    if getattr(caps, "embeddings", False):
        capabilities.append("embedding")
    if getattr(caps, "vision", False):
        capabilities.append("vision")
    if getattr(caps, "tools", False):
        capabilities.append("tools")
    if getattr(caps, "reasoning", False):
        capabilities.append("thinking")

    return {
        "license": "",
        "modelfile": f"FROM {ollama_name}\n",
        "parameters": "",
        "template": "{{ .Prompt }}",
        "capabilities": capabilities,
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
