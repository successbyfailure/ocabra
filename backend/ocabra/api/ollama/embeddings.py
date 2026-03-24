"""
POST /api/embeddings — Ollama legacy embeddings endpoint.
POST /api/embed      — Ollama v0.3+ embeddings endpoint (supports input arrays).
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from ocabra.api.openai._deps import check_capability, ensure_loaded, get_model_manager

from ._mapper import resolve_model

router = APIRouter()


async def _run_embeddings(request: Request, body: dict, legacy: bool) -> dict:
    """Shared logic for /api/embeddings and /api/embed."""
    ollama_model = str(body.get("model", ""))

    model_manager = get_model_manager(request)
    model_id, _ = await resolve_model(model_manager, ollama_model)
    state = await ensure_loaded(model_manager, model_id)
    check_capability(state, "embeddings", "embeddings")

    worker_pool = request.app.state.worker_pool
    raw_input = body.get("input") or body.get("prompt") or ""

    if state.backend_type == "ollama":
        if legacy:
            # /api/embeddings accepts a single prompt string
            result = await worker_pool.forward_request(
                model_id,
                "/api/embeddings",
                {"model": ollama_model, "prompt": raw_input if isinstance(raw_input, str) else raw_input[0] if raw_input else ""},
            )
            vectors = result.get("embedding", [])
            if vectors and isinstance(vectors[0], (int, float)):
                vectors = [vectors]
        else:
            # /api/embed accepts input as string or list
            result = await worker_pool.forward_request(
                model_id,
                "/api/embed",
                {"model": ollama_model, "input": raw_input},
            )
            vectors = result.get("embeddings", [])
    else:
        result = await worker_pool.forward_request(
            model_id,
            "/v1/embeddings",
            {"model": state.backend_model_id, "input": raw_input},
        )
        vectors = [item.get("embedding", []) for item in result.get("data", [])]

    return {"model": ollama_model, "embeddings": vectors}


@router.post("/embeddings", summary="Create embeddings (legacy)")
async def embeddings(request: Request) -> dict:
    """
    Create embeddings from text input (Ollama legacy format).

    Parameters:
      - model: Ollama model name or internal model id
      - input: a string (use /api/embed for array support)

    Response:
      - {"model": ..., "embeddings": [[...], ...]}
    """
    body = await request.json()
    return await _run_embeddings(request, body, legacy=True)


@router.post("/embed", summary="Create embeddings")
async def embed(request: Request) -> dict:
    """
    Create embeddings from text input (Ollama v0.3+ format).

    Parameters:
      - model: Ollama model name or internal model id
      - input: a string or a list of strings

    Response:
      - {"model": ..., "embeddings": [[...], ...]}
    """
    body = await request.json()
    return await _run_embeddings(request, body, legacy=False)
