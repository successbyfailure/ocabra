"""
POST /api/embeddings — Ollama legacy embeddings endpoint.
POST /api/embed      — Ollama v0.3+ embeddings endpoint (supports input arrays).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

logger = structlog.get_logger(__name__)

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai._deps import (
    check_capability,
    compute_worker_key,
    ensure_loaded,
    get_federation_manager,
    get_model_manager,
    get_profile_registry,
    merge_profile_defaults,
    resolve_profile,
)

from ._mapper import resolve_model
from ._shared import get_ollama_user

router = APIRouter()


async def _run_embeddings(
    request: Request,
    body: dict,
    legacy: bool,
    user: UserContext,
) -> dict:
    """Shared logic for /api/embeddings and /api/embed."""
    ollama_model = str(body.get("model", ""))

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    # --- Federation hook ---
    federation_manager = get_federation_manager(request)
    if federation_manager is not None:
        from ocabra.config import settings as _settings

        if _settings.federation_enabled:
            from ocabra.core.federation import resolve_federated

            target, peer = await resolve_federated(ollama_model, model_manager, federation_manager)
            if target == "remote":
                request.state.federation_remote_node_id = peer.peer_id
                # Determine which Ollama embed path to use
                embed_path = "/api/embed" if not legacy else "/api/embeddings"
                resp = await federation_manager.proxy_request(
                    peer, "POST", embed_path, body,
                )
                import json as _json

                try:
                    return _json.loads(resp.content)
                except Exception:
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type"),
                    )
    # --- End federation hook ---

    # Try profile resolution first
    try:
        profile, state = await resolve_profile(
            ollama_model,
            model_manager,
            profile_registry,
            user=user,
        )
        merged_body = merge_profile_defaults(profile, body)
        worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
    except HTTPException as profile_exc:
        # Fallback to legacy Ollama resolution for native Ollama models
        logger.debug(
            "embed_profile_miss",
            model=ollama_model,
            profile_status=profile_exc.status_code,
        )
        model_id, resolved_state = await resolve_model(model_manager, ollama_model, user=user)
        if resolved_state is None:
            logger.warning(
                "embed_model_not_found",
                model=ollama_model,
                resolved_id=model_id,
                is_anonymous=user.is_anonymous if user else None,
                accessible_count=len(user.accessible_model_ids) if user else None,
            )
            raise HTTPException(
                status_code=404,
                detail=f"Model '{ollama_model}' not found",
            ) from profile_exc
        state = await ensure_loaded(model_manager, model_id)
        merged_body = body
        worker_key = model_id
        profile = None

    check_capability(state, "embeddings", "embeddings")

    worker_pool = request.app.state.worker_pool
    raw_input = merged_body.get("input") or merged_body.get("prompt") or ""

    if state.backend_type == "ollama":
        # Try /api/embed first (Ollama v0.3+); fall back to /api/embeddings for older versions.
        # Both paths are normalised to return a list of embedding vectors.
        if not legacy:
            try:
                result = await worker_pool.forward_request(
                    worker_key,
                    "/api/embed",
                    {"model": ollama_model, "input": raw_input},
                )
                vectors = result.get("embeddings", [])
            except Exception:
                # Older Ollama (pre-0.3) doesn't have /api/embed — use legacy endpoint.
                legacy = True

        if legacy:
            # /api/embeddings: single prompt string only.
            prompt = (
                raw_input if isinstance(raw_input, str) else (raw_input[0] if raw_input else "")
            )
            result = await worker_pool.forward_request(
                worker_key,
                "/api/embeddings",
                {"model": ollama_model, "prompt": prompt},
            )
            vectors = result.get("embedding", [])
            if vectors and isinstance(vectors[0], (int, float)):
                vectors = [vectors]
    else:
        result = await worker_pool.forward_request(
            worker_key,
            "/v1/embeddings",
            {"model": state.backend_model_id, "input": raw_input},
        )
        vectors = [item.get("embedding", []) for item in result.get("data", [])]

    response: dict = {"model": ollama_model, "embeddings": vectors}
    # Preserve token-count fields so the stats middleware can record usage.
    for field in ("prompt_eval_count", "total_duration", "load_duration"):
        if field in result:
            response[field] = result[field]
    if "usage" in result:
        response["usage"] = result["usage"]
    return response


@router.post("/embeddings", summary="Create embeddings (legacy)")
async def embeddings(
    request: Request,
    user: UserContext = Depends(get_ollama_user),
) -> dict:
    """
    Create embeddings from text input (Ollama legacy format).

    Parameters:
      - model: profile_id or Ollama model name
      - input: a string (use /api/embed for array support)

    Response:
      - {"model": ..., "embeddings": [[...], ...]}
    """
    body = await request.json()
    return await _run_embeddings(request, body, legacy=True, user=user)


@router.post("/embed", summary="Create embeddings")
async def embed(
    request: Request,
    user: UserContext = Depends(get_ollama_user),
) -> dict:
    """
    Create embeddings from text input (Ollama v0.3+ format).

    Parameters:
      - model: profile_id or Ollama model name
      - input: a string or a list of strings

    Response:
      - {"model": ..., "embeddings": [[...], ...]}
    """
    body = await request.json()
    return await _run_embeddings(request, body, legacy=False, user=user)
