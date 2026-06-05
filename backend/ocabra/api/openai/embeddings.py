"""
POST /v1/embeddings — text embeddings endpoint.
"""

from __future__ import annotations

from typing import Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from ocabra.api._deps_auth import UserContext

from ._deps import (
    check_capability,
    compute_worker_key,
    get_federation_manager,
    get_model_manager,
    get_openai_user,
    get_profile_registry,
    merge_profile_defaults,
    raise_upstream_http_error,
    resolve_profile,
    to_backend_body,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/embeddings", summary="Create embeddings")
async def embeddings(
    request: Request,
    user: Annotated[UserContext, Depends(get_openai_user)],
) -> Any:
    """
    Create text embeddings. Proxies to the resolved model worker (backend-agnostic).
    Requires a model with capability embeddings=True.
    """
    body = await request.json()
    model_id: str = body.get("model", "")

    model_manager = get_model_manager(request)
    profile_registry = get_profile_registry(request)

    # --- Federation hook ---
    federation_manager = get_federation_manager(request)
    if federation_manager is not None:
        from ocabra.config import settings as _settings

        if _settings.federation_enabled:
            from ocabra.core.federation import resolve_federated

            target, peer = await resolve_federated(
                model_id, model_manager, federation_manager, profile_registry
            )
            if target == "remote":
                from ocabra.core.federation import should_fallback_to_local

                request.state.federation_remote_node_id = peer.peer_id
                try:
                    resp = await federation_manager.proxy_request(
                        peer=peer,
                        path=request.url.path,
                        body=body,
                        headers=dict(request.headers),
                    )
                except Exception as exc:
                    logger.warning(
                        "federation_peer_network_error_fallback_local",
                        peer=peer.name,
                        model_id=model_id,
                        error=str(exc),
                    )
                    request.state.federation_remote_node_id = None
                    resp = None
                if resp is not None and (
                    resp.status_code < 400
                    or not should_fallback_to_local(resp.status_code)
                ):
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type"),
                    )
                if resp is not None:
                    logger.warning(
                        "federation_peer_rejected_fallback_local",
                        peer=peer.name,
                        model_id=model_id,
                        status=resp.status_code,
                        body_preview=resp.text[:200],
                    )
                    request.state.federation_remote_node_id = None
                # Fall through to local processing.
    # --- End federation hook ---

    profile, state = await resolve_profile(
        model_id,
        model_manager,
        profile_registry,
        user=user,
    )
    check_capability(state, "embeddings", "embeddings")

    merged_body = merge_profile_defaults(profile, body)
    worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)

    worker_pool = request.app.state.worker_pool
    try:
        return await worker_pool.forward_request(
            worker_key,
            "/v1/embeddings",
            to_backend_body(state, merged_body),
        )
    except httpx.HTTPStatusError as exc:
        raise_upstream_http_error(exc)
