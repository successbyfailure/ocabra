"""Shared federation helpers for OpenAI-compatible endpoints.

All non-streaming endpoints follow the same flow when ``federation_enabled``:

1. ``resolve_federated(...)`` decides ``local`` vs ``remote`` based on which
   nodes have the model warm.
2. If ``remote``: proxy the request to the peer, then:

   * Peer returned a successful body → return it.
   * Peer returned a *non-recoverable* error → propagate the same status.
   * Peer returned a *recoverable* error or the network call failed → log and
     fall back to local processing.

The two helpers below capture that flow once. Endpoints just do::

    fed_resp = await try_proxy_json(request, model_id=..., body=...,
                                    federation_manager=..., model_manager=...,
                                    profile_registry=..., logger=logger)
    if fed_resp is not None:
        return fed_resp
    # ... local processing
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from fastapi import Request
from fastapi.responses import Response

# Sentinel — when an endpoint wants the helper to return a plain dict instead
# of a Response (e.g. so it can post-process with json.loads), it can opt out
# by passing ``return_raw=True``. Default is the safer Response wrap.


async def try_proxy_json(
    request: Request,
    *,
    model_id: str,
    body: dict[str, Any],
    federation_manager: Any,
    model_manager: Any,
    profile_registry: Any,
    logger: structlog.stdlib.BoundLogger,
) -> Response | None:
    """Proxy a JSON request to the best peer for *model_id*.

    Returns:
        * ``Response`` — the peer served the request (or returned a
          non-recoverable error worth propagating to the caller).
        * ``None`` — caller should fall back to local processing (peer
          unreachable, peer 401/403/404/5xx, federation disabled, or no
          remote peer has this model warm).
    """
    from ocabra.config import settings

    if federation_manager is None or not settings.federation_enabled:
        return None

    from ocabra.core.federation import resolve_federated, should_fallback_to_local

    target, peer = await resolve_federated(
        model_id, model_manager, federation_manager, profile_registry
    )
    if target != "remote" or peer is None:
        return None

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
            error_type=type(exc).__name__,
        )
        request.state.federation_remote_node_id = None
        return None

    if resp.status_code < 400 or not should_fallback_to_local(resp.status_code):
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    logger.warning(
        "federation_peer_rejected_fallback_local",
        peer=peer.name,
        model_id=model_id,
        status=resp.status_code,
        body_preview=resp.text[:200],
    )
    request.state.federation_remote_node_id = None
    return None


async def try_proxy_multipart(
    request: Request,
    *,
    model_id: str,
    files: dict[str, tuple[str, bytes, str]],
    data: dict[str, Any],
    federation_manager: Any,
    model_manager: Any,
    profile_registry: Any,
    logger: structlog.stdlib.BoundLogger,
) -> httpx.Response | None:
    """Proxy a multipart/form-data request to the best peer for *model_id*.

    The semantics mirror :func:`try_proxy_json` but the helper returns the raw
    ``httpx.Response`` so callers can pick their own response wrapping
    (PlainTextResponse for SRT/VTT, JSON re-parsing for image edits, etc.).
    Returns ``None`` when the caller should fall back to local processing.
    """
    from ocabra.config import settings

    if federation_manager is None or not settings.federation_enabled:
        return None

    from ocabra.core.federation import resolve_federated, should_fallback_to_local

    target, peer = await resolve_federated(
        model_id, model_manager, federation_manager, profile_registry
    )
    if target != "remote" or peer is None:
        return None

    request.state.federation_remote_node_id = peer.peer_id
    try:
        resp = await federation_manager.proxy_multipart(
            peer=peer,
            path=request.url.path,
            files=files,
            data=data,
            headers=dict(request.headers),
        )
    except Exception as exc:
        logger.warning(
            "federation_peer_network_error_fallback_local",
            peer=peer.name,
            model_id=model_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        request.state.federation_remote_node_id = None
        return None

    if resp.status_code < 400 or not should_fallback_to_local(resp.status_code):
        return resp

    logger.warning(
        "federation_peer_rejected_fallback_local",
        peer=peer.name,
        model_id=model_id,
        status=resp.status_code,
        body_preview=resp.text[:200],
    )
    request.state.federation_remote_node_id = None
    return None
