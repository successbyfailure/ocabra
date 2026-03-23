"""
POST /api/embeddings — Ollama embeddings compatibility endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from ocabra.api.openai._deps import check_capability, ensure_loaded, get_model_manager

from ._mapper import resolve_model

router = APIRouter()


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

    model_manager = get_model_manager(request)
    model_id, _ = await resolve_model(model_manager, ollama_model)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "embeddings", "embeddings")

    worker_pool = request.app.state.worker_pool
    if state.backend_type == "ollama":
        result = await worker_pool.forward_request(
            model_id,
            "/api/embeddings",
            {
                "model": ollama_model,
                "prompt": body.get("input", ""),
            },
        )
        vectors = result.get("embedding", [])
        if vectors and isinstance(vectors[0], (int, float)):
            vectors = [vectors]
    else:
        result = await worker_pool.forward_request(
            model_id,
            "/v1/embeddings",
            {
                "model": state.backend_model_id,
                "input": body.get("input", ""),
            },
        )
        vectors = [item.get("embedding", []) for item in result.get("data", [])]

    return {
        "model": ollama_model,
        "embeddings": vectors,
    }
