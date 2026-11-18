"""
POST /api/embeddings — Ollama legacy embeddings endpoint.
POST /api/embed      — Ollama v0.3+ embeddings endpoint (supports input arrays).
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai._deps import (
    EngineSuffixedModelName,
    check_capability,
    compute_worker_key,
    ensure_loaded,
    get_federation_manager,
    get_model_manager,
    get_profile_registry,
    merge_profile_defaults,
    read_json_object,
    resolve_profile,
    to_backend_body,
)
from ocabra.api.openai.embeddings import _truncate_and_renormalize
from ocabra.core.federation import note_federation_fallback
from ocabra.core.privacy import preview as privacy_preview
from ocabra.core.request_telemetry import mark_prefill_since

from ._mapper import resolve_model
from ._shared import dispatch_error_types, get_ollama_user, raise_dispatch_error
from .chat import _native_backend_model_name

logger = structlog.get_logger(__name__)

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

            # With the profile registry, so profile ids and routers resolve to
            # their base model before the local/remote decision (BUG-12).
            target, peer = await resolve_federated(
                ollama_model, model_manager, federation_manager, profile_registry, user=user
            )
            if target == "remote":
                import json as _json

                import structlog as _structlog

                from ocabra.core.federation import should_fallback_to_local

                _fed_logger = _structlog.get_logger(__name__)
                request.state.federation_remote_node_id = peer.peer_id
                # Determine which Ollama embed path to use
                embed_path = "/api/embed" if not legacy else "/api/embeddings"
                try:
                    resp = await federation_manager.proxy_request(
                        peer=peer,
                        path=embed_path,
                        body=body,
                        headers=dict(request.headers),
                    )
                except Exception as exc:
                    _fed_logger.warning(
                        "federation_peer_network_error_fallback_local",
                        peer=peer.name,
                        model_id=ollama_model,
                        error=str(exc),
                    )
                    request.state.federation_remote_node_id = None
                    note_federation_fallback(peer, exc)
                    resp = None
                if resp is not None and (
                    resp.status_code < 400 or not should_fallback_to_local(resp.status_code)
                ):
                    try:
                        return _json.loads(resp.content)
                    except Exception:
                        return Response(
                            content=resp.content,
                            status_code=resp.status_code,
                            media_type=resp.headers.get("content-type"),
                        )
                if resp is not None:
                    _fed_logger.warning(
                        "federation_peer_rejected_fallback_local",
                        peer=peer.name,
                        model_id=ollama_model,
                        status=resp.status_code,
                        body_preview=privacy_preview(resp.text),
                    )
                    request.state.federation_remote_node_id = None
                    note_federation_fallback(peer, resp.status_code)
                # Fall through to local processing.
    # --- End federation hook ---

    # Try profile resolution first. With the router resolver and the request
    # state, like /v1/embeddings: routers (``recomendado-embedding``) did not
    # resolve here (BUG-12).
    try:
        profile, state = await resolve_profile(
            ollama_model,
            model_manager,
            profile_registry,
            user=user,
            request_body=body,
            router_resolver=getattr(request.app.state, "router_resolver", None),
            request_state=request.state,
        )
        merged_body = merge_profile_defaults(profile, body)
        worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
    except EngineSuffixedModelName:
        # Nombre que lleva el motor dentro: no hay via legacy que
        # probar, y su mensaje dice los dos nombres validos.
        raise
    except HTTPException as profile_exc:
        # Only "no such profile" falls back to the legacy Ollama names; a 403,
        # a 503 or a capability error is the answer (before, any of them was
        # retried as a legacy name and came back as a misleading 404).
        if profile_exc.status_code != 404:
            raise
        logger.debug("embed_profile_miss", model=ollama_model)
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

    dimensions = merged_body.get("dimensions")
    if dimensions is not None and (
        isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1
    ):
        raise HTTPException(status_code=400, detail="'dimensions' must be a positive integer")

    worker_pool = request.app.state.worker_pool
    raw_input = merged_body.get("input") or merged_body.get("prompt") or ""
    forward_started = time.monotonic()

    try:
        if state.backend_type == "ollama":
            result, vectors = await _embed_with_ollama(
                worker_pool,
                worker_key,
                _native_backend_model_name(ollama_model, state.backend_model_id),
                merged_body,
                raw_input,
                legacy=legacy,
            )
        else:
            payload = {"model": state.backend_model_id, "input": raw_input}
            if dimensions is not None:
                payload["dimensions"] = dimensions
            result = await worker_pool.forward_request(
                worker_key, "/v1/embeddings", to_backend_body(state, payload)
            )
            vectors = [item.get("embedding", []) for item in result.get("data", [])]
    except dispatch_error_types() as exc:
        raise_dispatch_error(exc)

    # Un embedding es todo prefill: con este tiempo el middleware de stats
    # calcula los tokens/s que el worker no devuelve.
    mark_prefill_since(request.state, forward_started)
    if dimensions is not None:
        # Backends that ignore ``dimensions`` (llama-server) return the full
        # vector; truncate and renormalise as /v1/embeddings does (BUG-12).
        vectors = [
            _truncate_and_renormalize(v, dimensions)
            if isinstance(v, list) and len(v) > dimensions
            else v
            for v in vectors
        ]
    response: dict = {"model": ollama_model, "embeddings": vectors}
    if legacy:
        # Ollama's own /api/embeddings answers ``embedding`` (one vector);
        # ``embeddings`` stays for the clients written against oCabra.
        response["embedding"] = vectors[0] if vectors else []
    # Preserve token-count fields so the stats middleware can record usage.
    for field in ("prompt_eval_count", "total_duration", "load_duration"):
        if field in result:
            response[field] = result[field]
    if "usage" in result:
        response["usage"] = result["usage"]
    return response


async def _embed_with_ollama(
    worker_pool,
    worker_key: str,
    model: str,
    body: dict,
    raw_input,
    *,
    legacy: bool,
) -> tuple[dict, list]:
    """Embed on the Ollama daemon: ``/api/embed``, or ``/api/embeddings`` per input.

    Before, any failure of ``/api/embed`` (a timeout, a 400 for an input too
    long) fell back to the legacy endpoint, which embedded ``input[0]`` only:
    N inputs, one vector back, and nothing in the log (BUG-12). Now only a 404
    (an Ollama without ``/api/embed``) falls back, and the fallback embeds
    every input.
    """
    import httpx

    if not legacy:
        payload = {"model": model, "input": raw_input}
        for key in ("dimensions", "truncate", "options", "keep_alive"):
            if key in body:
                payload[key] = body[key]
        try:
            result = await worker_pool.forward_request(worker_key, "/api/embed", payload)
            return result, list(result.get("embeddings", []))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            logger.info("ollama_embed_endpoint_missing_using_legacy", model=model)

    inputs = raw_input if isinstance(raw_input, list) else [raw_input]
    vectors: list = []
    result: dict = {}
    for text in inputs:
        result = await worker_pool.forward_request(
            worker_key, "/api/embeddings", {"model": model, "prompt": text}
        )
        vector = result.get("embedding", [])
        vectors.append(vector)
    return result, vectors


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
    body = await read_json_object(request)
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
    body = await read_json_object(request)
    return await _run_embeddings(request, body, legacy=False, user=user)
