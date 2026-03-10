"""
POST /api/embeddings — Ollama embeddings compatibility endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from ocabra.api.openai._deps import check_capability, ensure_loaded, get_model_manager

from ._mapper import OllamaNameMapper

router = APIRouter()
_mapper = OllamaNameMapper()


@router.post("/embeddings", summary="Create embeddings")
async def embeddings(request: Request) -> dict:
    """
    Create embeddings from text input.

    Parameters:
      - model: Ollama model name or internal model id
      - input: a string or a list of strings

    Response:
      - {"model": ..., "embeddings": [[...], ...]}
    """
    body = await request.json()
    ollama_model = str(body.get("model", ""))
    model_id = _mapper.to_internal(ollama_model)

    model_manager = get_model_manager(request)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "embeddings", "embeddings")

    worker_pool = request.app.state.worker_pool
    result = await worker_pool.forward_request(
        model_id,
        "/v1/embeddings",
        {
            "model": model_id,
            "input": body.get("input", ""),
        },
    )

    vectors = [item.get("embedding", []) for item in result.get("data", [])]
    return {
        "model": ollama_model,
        "embeddings": vectors,
    }
