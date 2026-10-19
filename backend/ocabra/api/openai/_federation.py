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

    profile = None
    execution_policy = "auto"
    if profile_registry is not None:
        try:
            profile = await profile_registry.get(model_id)
            if profile is None:
                alias_target = await profile_registry.resolve_alias(model_id)
                profile = alias_target[0] if alias_target is not None else None
            if profile is not None and getattr(profile, "runtime_presets", []):
                from ocabra.core.runtime_coordinator import plan_profile_runtime

                plan = await plan_profile_runtime(profile, model_manager, body)
                selected = next(
                    (
                        preset
                        for preset in getattr(profile, "runtime_presets", [])
                        if preset.preset_key == plan.preset_key
                    ),
                    None,
                )
                if selected is not None:
                    execution_policy = selected.execution_policy
        except Exception:  # noqa: BLE001 -- federation remains best effort
            profile = None
            execution_policy = "auto"
    if execution_policy == "local":
        return None

    target, peer = await resolve_federated(
        model_id, model_manager, federation_manager, profile_registry
    )
    if execution_policy == "remote" and profile is not None:
        forced_peer = federation_manager.find_best_peer(profile.base_model_id)
        if forced_peer is not None:
            target, peer = "remote", forced_peer
    if target != "remote" or peer is None:
        return None

    # ``resolve_federated`` resolves public profile ids to their canonical
    # ``base_model_id`` before inspecting the peer inventory.  Prefer the
    # original public id when the selected peer explicitly advertises it in
    # the matching model's ``profiles`` list.  Access control on a peer can
    # grant a profile id without granting its canonical base id, so blindly
    # sending the canonical id would produce a misleading 404.  Fall back to
    # the canonical id only when the peer does not advertise the local alias.
    proxy_body = body
    if profile_registry is not None:
        try:
            profile = profile or await profile_registry.get(model_id)
            if profile is None:
                alias_target = await profile_registry.resolve_alias(model_id)
                profile = alias_target[0] if alias_target is not None else None
        except Exception:  # noqa: BLE001 -- registry lookup must not break inference
            profile = None
        if profile is not None:
            remote_model_id = profile.base_model_id
            for remote_model in getattr(peer, "models", []):
                if remote_model.get("model_id") != profile.base_model_id:
                    continue
                advertised_profiles = remote_model.get("profiles") or []
                if model_id in advertised_profiles:
                    remote_model_id = model_id
                break
            if remote_model_id != body.get("model"):
                proxy_body = {**body, "model": remote_model_id}

    request.state.federation_remote_node_id = peer.peer_id
    try:
        resp = await federation_manager.proxy_request(
            peer=peer,
            path=request.url.path,
            body=proxy_body,
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

    # Retry with the canonical ``base_model_id`` before giving up. Observed in
    # production (2026-10-18): a peer's heartbeat can advertise a profile_id
    # that its own in-memory profile registry hasn't ingested yet (profile
    # inserted directly in DB on the peer side without a restart). The first
    # proxy sends the profile id (because the peer advertised it), gets a
    # 404, and we fell back to local — defeating the whole point of
    # federation. If we haven't tried the canonical id yet, do so now.
    if (
        resp.status_code == 404
        and profile is not None
        and proxy_body.get("model") != profile.base_model_id
    ):
        canonical_body = {**body, "model": profile.base_model_id}
        try:
            retry_resp = await federation_manager.proxy_request(
                peer=peer,
                path=request.url.path,
                body=canonical_body,
                headers=dict(request.headers),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "federation_peer_canonical_retry_failed",
                peer=peer.name,
                model_id=model_id,
                error=str(exc),
            )
        else:
            if retry_resp.status_code < 400:
                logger.info(
                    "federation_peer_served_via_canonical_id",
                    peer=peer.name,
                    original_model_id=model_id,
                    canonical_model_id=profile.base_model_id,
                )
                return Response(
                    content=retry_resp.content,
                    status_code=retry_resp.status_code,
                    media_type=retry_resp.headers.get("content-type"),
                )
            # Still broken → fall through to the normal fallback path below.
            resp = retry_resp

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
