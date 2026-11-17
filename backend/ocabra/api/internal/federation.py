"""Federation API endpoints for peer management and heartbeat.

Provides:
- GET  /federation/heartbeat      — heartbeat response for remote peers
- GET  /federation/peers          — list all peers with online status
- POST /federation/peers          — add a new peer
- PATCH /federation/peers/{id}    — update a peer
- DELETE /federation/peers/{id}   — remove a peer
- POST /federation/peers/{id}/test — test connection to a peer
- POST /federation/peers/{id}/rotate-token — new tunnel token for a tunnel peer
- WS   /federation/tunnel           — reverse tunnel opened by a tunnel peer
- GET  /federation/node             — this node's identity and federation switches
- POST /federation/join-requests    — (public) a node asks to join us
- GET  /federation/join-requests/{id}/status — (public) the asker polls its request
- GET  /federation/join-requests    — pending requests (admin)
- POST /federation/join-requests/{id}/accept|deny
- GET/POST /federation/invites, DELETE /federation/invites/{id}
- POST /federation/join             — ask another node to let us join
- POST /federation/peers/{id}/join-check — re-check a pending outgoing request
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from starlette.responses import StreamingResponse

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.core.federation import PeerState, new_tunnel_token
from ocabra.schemas.federation import (
    FederationRules,
    FederationStrategyIn,
    HeartbeatGpu,
    HeartbeatLoad,
    HeartbeatModel,
    HeartbeatResponse,
    InviteCreate,
    JoinOutgoing,
    JoinRequestCreate,
    JoinRequestDecision,
    PeerCreate,
    PeerCreated,
    PeerOut,
    PeerTestResult,
    PeerUpdate,
    RemoteDownloadRequest,
    RemoteLoadRequest,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["federation"])


def _effective_rules_out(p: PeerState) -> dict | None:
    from ocabra.core.federation_rules import ALL_ALLOWED, effective_rules

    rules = effective_rules(p)
    return None if rules == ALL_ALLOWED else rules.to_dict()


def _peer_out(p: PeerState, cls: type[PeerOut] = PeerOut, **extra) -> PeerOut:
    tunnel = None
    if p.tunnel_connected:
        session = p.tunnel_session
        tunnel = {
            "remote_node_id": session.remote_info.get("node_id"),
            "remote_node_name": session.remote_info.get("node_name"),
            "remote_version": session.remote_info.get("version"),
            "connected_at": session.connected_at,
            "active_streams": session.active_streams,
            "requests_sent": session.requests_sent,
            "requests_served": session.requests_served,
        }
    dialer = p.dialer
    return cls(
        id=p.peer_id,
        name=p.name,
        url=p.url,
        access_level=p.access_level,
        enabled=p.enabled,
        online=p.online,
        last_heartbeat=p.last_heartbeat,
        models=p.models,
        gpus=p.gpus,
        load=p.load,
        connection_mode=p.connection_mode,
        mode=p.mode,
        notes=p.notes,
        remote_node_id=p.remote_node_id,
        local_access_level=p.local_access_level,
        join_state=p.join_state,
        join_verification_code=p.join_verification_code,
        privacy_mode=p.privacy_mode,
        rules=p.rules,
        effective_rules=_effective_rules_out(p),
        their_rules=p.their_rules,
        dial_state=dialer.state if dialer is not None else None,
        last_error=dialer.last_error if dialer is not None else None,
        requests_proxied=p.requests_proxied,
        last_proxied_at=p.last_proxied_at,
        tunnel_connected=p.tunnel_connected,
        tunnel=tunnel,
        **extra,
    )


def _get_federation_manager(request: Request):
    """Extract FederationManager from app state, raise 503 if not enabled."""
    fm = getattr(request.app.state, "federation_manager", None)
    if fm is None:
        raise HTTPException(
            status_code=503,
            detail="Federation is not enabled on this node",
        )
    return fm


def _capability_flags(state) -> dict[str, bool] | None:
    """A model's yes/no capabilities for the heartbeat, or None if unknown.

    The capabilities dict also carries numbers (``context_length``…): sent
    as-is they failed validation and, since one failure aborted the loop,
    every heartbeat from 0.97.0 to 0.98.4 advertised no models at all. Never
    let this lose the model list again.
    """
    from ocabra.core.model_manager import resolve_display_capabilities

    try:
        caps, _source = resolve_display_capabilities(state)
    except Exception:  # noqa: BLE001 — capabilities are a hint
        return None
    return {k: v for k, v in caps.items() if isinstance(v, bool)}


@router.get(
    "/federation/heartbeat",
    summary="Federation heartbeat",
    description=(
        "Returns this node's current state including GPU info, loaded models, "
        "and load metrics. Used by remote peers for health checking."
    ),
    response_model=HeartbeatResponse,
)
async def heartbeat(
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> HeartbeatResponse:
    """Return this node's heartbeat information for federation peers.

    Includes GPU states, the models we offer with their profiles (loaded ones,
    plus cold ones we would load for the caller — its «rules» decide), and
    current load metrics (active requests and average GPU utilization).
    """
    fm = _get_federation_manager(request)
    gpu_manager = getattr(request.app.state, "gpu_manager", None)
    model_manager = getattr(request.app.state, "model_manager", None)
    profile_registry = getattr(request.app.state, "profile_registry", None)

    # GPUs
    gpus: list[HeartbeatGpu] = []
    if gpu_manager is not None:
        try:
            gpu_states = await gpu_manager.get_all_states()
            for gs in gpu_states:
                gpus.append(
                    HeartbeatGpu(
                        index=gs.index,
                        name=gs.name,
                        total_vram_mb=gs.total_vram_mb,
                        free_vram_mb=gs.free_vram_mb,
                    )
                )
        except Exception as exc:
            logger.warning("federation_heartbeat_gpu_error", error=str(exc))

    from ocabra.core.federation_rules import ALL_ALLOWED

    rules = fm.rules_for_user(getattr(_user, "user_id", None))

    # Models
    models: list[HeartbeatModel] = []
    if model_manager is not None:
        try:
            from ocabra.core.model_manager import ModelStatus

            model_states = await model_manager.list_states(include_derived=True)
            for ms in model_states:
                if ms.status != ModelStatus.LOADED:
                    continue
                extra = ms.extra_config if isinstance(ms.extra_config, dict) else {}
                runtime = extra.get("_runtime")
                runtime = runtime if isinstance(runtime, dict) else {}
                public_model_id = str(runtime.get("base_model_id") or ms.model_id)
                profile_ids: list[str] = []
                if runtime.get("profile_id"):
                    profile_ids = [str(runtime["profile_id"])]
                elif profile_registry is not None:
                    try:
                        profiles = await profile_registry.list_by_model(public_model_id)
                        profile_ids = [p.profile_id for p in profiles if p.enabled]
                    except Exception:
                        pass
                models.append(
                    HeartbeatModel(
                        model_id=public_model_id,
                        status=ms.status.value,
                        profiles=profile_ids,
                        runtime_preset=runtime.get("preset_key"),
                        effective_context=runtime.get("effective_context"),
                        parallelism=runtime.get("parallelism"),
                        capabilities=_capability_flags(ms),
                    )
                )
            # Cold models too: the caller's federation adds them up with its
            # own, and a request for one makes us load it. Not when the
            # caller may only use what is warm here («rules»).
            if rules is None or rules.allow_loads:
                advertised = {m.model_id for m in models}
                for ms in model_states:
                    if "::" in ms.model_id or ms.model_id in advertised:
                        continue
                    if ms.status not in (
                        ModelStatus.CONFIGURED,
                        ModelStatus.UNLOADED,
                        ModelStatus.LOADING,
                    ):
                        continue
                    profile_ids = []
                    if profile_registry is not None:
                        try:
                            profiles = await profile_registry.list_by_model(ms.model_id)
                            profile_ids = [p.profile_id for p in profiles if p.enabled]
                        except Exception:
                            pass
                    models.append(
                        HeartbeatModel(
                            model_id=ms.model_id,
                            status=ms.status.value,
                            profiles=profile_ids,
                            capabilities=_capability_flags(ms),
                        )
                    )
        except Exception as exc:
            logger.warning("federation_heartbeat_models_error", error=str(exc))

    # Load
    active_requests = 0
    gpu_util_avg = 0.0
    if gpu_manager is not None:
        try:
            gpu_states = await gpu_manager.get_all_states()
            if gpu_states:
                gpu_util_avg = sum(gs.utilization_pct for gs in gpu_states) / len(gpu_states)
        except Exception:
            pass
    if model_manager is not None:
        try:
            active_requests = getattr(model_manager, "active_request_count", 0)
            if callable(active_requests):
                active_requests = active_requests()
        except Exception:
            active_requests = 0

    import asyncio as _asyncio

    from ocabra.core.cuda_probe import probe_cuda

    cuda = await _asyncio.to_thread(probe_cuda)

    from ocabra.core import privacy

    # «Mi servidor, mis reglas»: the caller sees the rules we apply to it, and
    # only the models offered to it (so it doesn't send what we'd refuse).
    rules_out = None
    if rules is not None and rules != ALL_ALLOWED:
        rules_out = rules.to_dict()
        if rules.offered_models is not None:
            offered = set(rules.offered_models)
            models = [m for m in models if m.model_id in offered or offered & set(m.profiles or [])]

    return HeartbeatResponse(
        cuda=cuda,
        privacy_mode=privacy.mode(),
        rules=rules_out,
        node_id=fm.node_id,
        node_name=fm.node_name,
        version=settings.app_version,
        uptime_seconds=round(fm.uptime_seconds, 1),
        gpus=gpus,
        models=models,
        load=HeartbeatLoad(
            active_requests=active_requests,
            gpu_utilization_avg_pct=round(gpu_util_avg, 1),
        ),
    )


@router.get(
    "/federation/peers",
    summary="List federation peers",
    description="Return all configured federation peers with their online status and cached state.",
)
async def list_peers(
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> list[PeerOut]:
    """List all federation peers with their current runtime state."""
    fm = _get_federation_manager(request)
    return [_peer_out(p) for p in fm.get_all_peers()]


@router.post(
    "/federation/peers",
    summary="Add federation peer",
    description=(
        "Register a new federation peer. In 'http' mode the API key is encrypted "
        "before storage. In 'tunnel' mode the response includes a one-time "
        "tunnel_token the remote node uses to dial this node."
    ),
    responses={400: {"description": "Invalid peer configuration"}},
)
async def add_peer(
    body: PeerCreate,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> PeerCreated:
    """Add a new federation peer.

    Args:
        body: name, connection_mode, access_level, notes; url and api_key for
            'http' peers.

    Returns:
        The newly created peer. ``tunnel_token`` is set (and only ever shown
        here) for tunnel peers.
    """
    fm = _get_federation_manager(request)
    token = new_tunnel_token() if body.connection_mode == "tunnel_in" else None
    try:
        state = await fm.add_peer(
            name=body.name,
            url=body.url if body.connection_mode == "http" else "",
            api_key=body.api_key if body.connection_mode == "http" else "",
            access_level=body.access_level,
            connection_mode=body.connection_mode,
            notes=body.notes,
            tunnel_token=token,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _peer_out(state, PeerCreated, tunnel_token=token)


@router.patch(
    "/federation/peers/{peer_id}",
    summary="Update federation peer",
    description="Update fields of an existing federation peer.",
    responses={404: {"description": "Peer not found"}},
)
async def update_peer(
    peer_id: str,
    body: PeerUpdate,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> PeerOut:
    """Update an existing federation peer.

    Args:
        peer_id: UUID of the peer to update.
        body: Fields to update (name, url, api_key, access_level, enabled).

    Returns:
        The updated peer state.
    """
    fm = _get_federation_manager(request)
    patch = body.model_dump(exclude_unset=True)
    if "rules" in patch:
        from ocabra.core.federation_rules import clean_rules

        patch["rules"] = clean_rules(patch["rules"]) if patch["rules"] else None
    state = await fm.update_peer(peer_id, **patch)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Peer '{peer_id}' not found")
    return _peer_out(state)


@router.post(
    "/federation/peers/{peer_id}/rotate-token",
    summary="Rotate tunnel token",
    description=(
        "Issue a new tunnel token for a tunnel-mode peer. The old token stops "
        "working immediately and the live tunnel, if any, is closed."
    ),
    responses={
        400: {"description": "Peer is not in tunnel mode"},
        404: {"description": "Peer not found"},
    },
)
async def rotate_tunnel_token(
    peer_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> PeerCreated:
    """Rotate the tunnel token of a tunnel peer.

    Returns:
        The peer with the new ``tunnel_token`` (shown only here).
    """
    fm = _get_federation_manager(request)
    peer = next((p for p in fm.get_all_peers() if p.peer_id == peer_id), None)
    if peer is None:
        raise HTTPException(status_code=404, detail=f"Peer '{peer_id}' not found")
    if peer.connection_mode != "tunnel_in":
        raise HTTPException(status_code=400, detail="Peer is not an incoming tunnel peer")
    token = new_tunnel_token()
    state = await fm.update_peer(peer_id, tunnel_token=token)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Peer '{peer_id}' not found")
    return _peer_out(state, PeerCreated, tunnel_token=token)


@router.websocket("/federation/tunnel")
async def federation_tunnel(websocket: WebSocket) -> None:
    """Reverse tunnel endpoint: a tunnel-mode peer dials in and serves our requests.

    The peer authenticates with the token in its first (HELLO) frame — not with
    a user session or API key. Once accepted, requests this node routes to that
    peer travel over this socket until it closes. Protocol in
    ``ocabra/core/federation_tunnel.py``.
    """
    from ocabra.core.federation_tunnel import StarletteChannel, TunnelConnection, accept_tunnel

    fm = getattr(websocket.app.state, "federation_manager", None)
    if fm is None:
        await websocket.close(code=1013, reason="federation disabled")
        return
    await websocket.accept()
    channel = StarletteChannel(websocket)
    accepted = await accept_tunnel(channel, authenticate=fm.authenticate_tunnel)
    if accepted is None:
        client = websocket.client.host if websocket.client else None
        logger.warning("federation_tunnel_auth_failed", client=client)
        return
    peer, remote_info = accepted
    session = TunnelConnection(
        channel, dialer=False, remote_info=remote_info, server=fm.local_server_for(peer)
    )
    await fm.attach_tunnel(peer, session)
    try:
        await session.run()
    finally:
        fm.detach_tunnel(peer, session)


@router.delete(
    "/federation/peers/{peer_id}",
    summary="Remove federation peer",
    description="Remove a federation peer from the database and runtime state.",
    responses={404: {"description": "Peer not found"}},
)
async def remove_peer(
    peer_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Remove a federation peer.

    Args:
        peer_id: UUID of the peer to remove.

    Returns:
        Confirmation dict with ok=True.
    """
    fm = _get_federation_manager(request)
    removed = await fm.remove_peer(peer_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Peer '{peer_id}' not found")
    return {"ok": True}


@router.post(
    "/federation/peers/{peer_id}/test",
    summary="Test peer connection",
    description="Test connectivity to a peer by sending a single heartbeat request.",
    response_model=PeerTestResult,
    responses={404: {"description": "Peer not found"}},
)
async def test_peer(
    peer_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> PeerTestResult:
    """Test connectivity to a federation peer with a single heartbeat attempt.

    Args:
        peer_id: UUID of the peer to test.

    Returns:
        Test result with success status, latency, and any error message.
    """
    fm = _get_federation_manager(request)
    result = await fm.test_peer_connection(peer_id)
    if result.get("error") == "Peer not found":
        raise HTTPException(status_code=404, detail=f"Peer '{peer_id}' not found")
    return PeerTestResult(**result)


# ── Phase 5: Remote operations (access_level == "full") ─────────


async def _get_full_access_peer(peer_id: str, request: Request) -> PeerState:
    """Get a peer with full access level, or raise 403/404.

    Args:
        peer_id: UUID string of the peer.
        request: FastAPI request (used to extract FederationManager).

    Returns:
        PeerState with access_level == "full".

    Raises:
        HTTPException 404: Peer not found or offline.
        HTTPException 403: Peer does not have full access level.
    """
    fm = _get_federation_manager(request)
    peers = fm.get_all_peers()
    peer = next((p for p in peers if p.peer_id == peer_id), None)
    if peer is None:
        raise HTTPException(status_code=404, detail=f"Peer '{peer_id}' not found")
    if peer.access_level != "full":
        raise HTTPException(
            status_code=403,
            detail=f"Peer '{peer.name}' does not have 'full' access level",
        )
    if not peer.online:
        raise HTTPException(
            status_code=502,
            detail=f"Peer '{peer.name}' is currently offline",
        )
    return peer


@router.post(
    "/federation/peers/{peer_id}/models/{model_id:path}/load",
    summary="Remote model load",
    description=(
        "Proxy a model load request to a remote peer. "
        "Requires the peer to have access_level='full'."
    ),
    responses={
        403: {"description": "Peer does not have full access"},
        404: {"description": "Peer not found"},
        502: {"description": "Peer is offline or returned an error"},
    },
)
async def remote_load_model(
    peer_id: str,
    model_id: str,
    request: Request,
    body: RemoteLoadRequest | None = None,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Load a model on a remote peer.

    Args:
        peer_id: UUID of the target peer.
        model_id: Model identifier to load on the peer.
        body: Optional load configuration (preferred_gpu, extra_config).

    Returns:
        The peer's response as a JSON dict.
    """
    peer = await _get_full_access_peer(peer_id, request)
    fm = _get_federation_manager(request)
    payload = body.model_dump(exclude_none=True) if body else {}
    try:
        resp = await fm.proxy_request(
            peer=peer,
            path=f"/ocabra/models/{model_id}/load",
            body=payload,
            headers=dict(request.headers),
        )
        return resp.json()
    except Exception as exc:
        logger.warning(
            "remote_load_failed",
            peer=peer.name,
            model_id=model_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to load model on peer '{peer.name}': {exc}",
        ) from exc


@router.post(
    "/federation/peers/{peer_id}/models/{model_id:path}/unload",
    summary="Remote model unload",
    description=(
        "Proxy a model unload request to a remote peer. "
        "Requires the peer to have access_level='full'."
    ),
    responses={
        403: {"description": "Peer does not have full access"},
        404: {"description": "Peer not found"},
        502: {"description": "Peer is offline or returned an error"},
    },
)
async def remote_unload_model(
    peer_id: str,
    model_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Unload a model on a remote peer.

    Args:
        peer_id: UUID of the target peer.
        model_id: Model identifier to unload on the peer.

    Returns:
        The peer's response as a JSON dict.
    """
    peer = await _get_full_access_peer(peer_id, request)
    fm = _get_federation_manager(request)
    try:
        resp = await fm.proxy_request(
            peer=peer,
            path=f"/ocabra/models/{model_id}/unload",
            body={},
            headers=dict(request.headers),
        )
        return resp.json()
    except Exception as exc:
        logger.warning(
            "remote_unload_failed",
            peer=peer.name,
            model_id=model_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to unload model on peer '{peer.name}': {exc}",
        ) from exc


@router.post(
    "/federation/peers/{peer_id}/downloads",
    summary="Remote model download",
    description=(
        "Start a model download on a remote peer (same body as POST /ocabra/downloads "
        "there). Returns the peer's download job; follow it with "
        "GET /federation/peers/{peer_id}/downloads/{job_id}. Requires access_level='full'."
    ),
    responses={
        403: {"description": "Peer does not have full access"},
        404: {"description": "Peer not found"},
        502: {"description": "Peer is offline or returned an error"},
    },
)
async def remote_download(
    peer_id: str,
    body: RemoteDownloadRequest,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Start a download on a remote peer.

    It used to relay the response as an SSE stream, but the peer's
    ``POST /ocabra/downloads`` answers with the job as JSON at once: callers
    reading ``data:`` lines saw nothing (and retried, starting it twice).

    Returns:
        The peer's download job (``job_id``, ``status``, ``progress_pct``…).
    """
    peer = await _get_full_access_peer(peer_id, request)
    fm = _get_federation_manager(request)
    try:
        resp = await fm.proxy_request(
            peer=peer,
            path="/ocabra/downloads",
            body=body.model_dump(exclude_none=True),
            headers=dict(request.headers),
        )
    except Exception as exc:
        logger.warning("remote_download_failed", peer=peer.name, error=str(exc))
        raise HTTPException(
            status_code=502, detail=f"Failed to start download on peer '{peer.name}': {exc}"
        ) from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Peer '{peer.name}': {resp.text[:300]}")
    return resp.json()


@router.get(
    "/federation/peers/{peer_id}/downloads/{job_id}",
    summary="Remote download status",
    description="Status of a download job on a remote peer. Requires access_level='full'.",
    responses={
        403: {"description": "Peer does not have full access"},
        404: {"description": "Peer or job not found"},
        502: {"description": "Peer is offline or returned an error"},
    },
)
async def remote_download_status(
    peer_id: str,
    job_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Return a remote download job.

    Returns:
        The peer's download job.
    """
    peer = await _get_full_access_peer(peer_id, request)
    fm = _get_federation_manager(request)
    try:
        resp = await fm.proxy_get(peer, f"/ocabra/downloads/{job_id}", dict(request.headers))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Peer '{peer.name}': {exc}") from exc
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Download job not found on the peer")
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Peer '{peer.name}': {resp.text[:300]}")
    return resp.json()


@router.get(
    "/federation/peers/{peer_id}/gpus",
    summary="Remote GPU monitoring",
    description=(
        "Proxy a GPU status request to a remote peer. "
        "Requires the peer to have access_level='full'."
    ),
    responses={
        403: {"description": "Peer does not have full access"},
        404: {"description": "Peer not found"},
        502: {"description": "Peer is offline or returned an error"},
    },
)
async def remote_gpus(
    peer_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> list[dict]:
    """Get GPU status from a remote peer.

    Args:
        peer_id: UUID of the target peer.

    Returns:
        List of GPU status dicts from the peer.
    """
    peer = await _get_full_access_peer(peer_id, request)
    fm = _get_federation_manager(request)
    try:
        # GET: it used to POST here, which the peer's GET-only route answers 405.
        resp = await fm.proxy_get(peer, "/ocabra/gpus", dict(request.headers))
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning(
            "remote_gpus_failed",
            peer=peer.name,
            error=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to get GPUs from peer '{peer.name}': {exc}",
        ) from exc


# ── Node identity ───────────────────────────────────────────────


@router.get(
    "/federation/node",
    summary="This node's federation identity",
    description=(
        "Node id/name, public URL, and the federation switches. Works with "
        "federation disabled (the UI uses it to offer turning it on)."
    ),
)
async def node_info(
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Return this node's identity and federation settings.

    Returns:
        ``{enabled, node_id, node_name, public_url, accept_join_requests, version,
        pending_join_requests}``.
    """
    fm = getattr(request.app.state, "federation_manager", None)
    pending = 0
    if fm is not None and fm.join_service is not None:
        try:
            pending = len(await fm.join_service.list_requests())
        except Exception:  # noqa: BLE001
            pending = 0
    return {
        "enabled": fm is not None,
        "node_id": fm.node_id if fm is not None else (settings.federation_node_id or None),
        "node_name": fm.node_name if fm is not None else (settings.federation_node_name or None),
        "public_url": getattr(settings, "public_url", None) or None,
        "accept_join_requests": settings.federation_accept_join_requests,
        "version": settings.app_version,
        "pending_join_requests": pending,
    }


# ── Join requests (hub side) ────────────────────────────────────


def _join_service(request: Request):
    fm = _get_federation_manager(request)
    if fm.join_service is None:
        raise HTTPException(status_code=503, detail="Federation is starting")
    return fm.join_service


def _client_ip(request: Request) -> str | None:
    """Client address for join requests (shown to the admin, rate limits).

    Caddy resolves it from the proxy chain and sets ``X-Real-IP`` (overwriting
    any value the client sent). The first ``X-Forwarded-For`` entry is NOT
    used: nginx-style proxies append to whatever the client sent, so that entry
    is forgeable and would let anyone dodge the per-IP limits.
    """
    real = request.headers.get("x-real-ip", "").strip()
    if real:
        return real[:64]
    return request.client.host if request.client else None


def _join_http_error(exc: Exception) -> HTTPException:
    from ocabra.core.federation_join import JoinError

    if isinstance(exc, JoinError):
        return HTTPException(status_code=exc.status, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/federation/join-requests",
    summary="Ask to join this node's federation",
    description=(
        "Public: a node asks to join. Without a valid invite_code this node must "
        "have join requests enabled and an admin must accept it; with one it is "
        "accepted on the spot and the response carries the tunnel token. Rate "
        "limited per IP."
    ),
    responses={
        403: {"description": "Join requests disabled or invalid invite"},
        409: {"description": "Node already federated"},
        429: {"description": "Too many requests"},
    },
)
async def create_join_request(body: JoinRequestCreate, request: Request) -> dict:
    """Register a join request from another node.

    Returns:
        ``{request_id, poll_secret, verification_code, status, mode, hub}`` and,
        when accepted by invite, ``tunnel_token`` and ``peer_name``.
    """
    svc = _join_service(request)
    try:
        return await svc.create_request(body.model_dump(), source_ip=_client_ip(request))
    except Exception as exc:
        raise _join_http_error(exc) from exc


@router.get(
    "/federation/join-requests/{request_id}/status",
    summary="Join request status (for the requesting node)",
    description=(
        "Public, authenticated by the X-Join-Secret header returned when the "
        "request was made. Once accepted the response includes tunnel_token "
        "until the node first connects."
    ),
    responses={404: {"description": "Unknown request or wrong secret"}},
)
async def join_request_status(request_id: str, request: Request) -> dict:
    """Return the status of a join request to the node that made it.

    Returns:
        ``{request_id, status, mode, verification_code, hub[, tunnel_token]}``.
    """
    svc = _join_service(request)
    secret = request.headers.get("x-join-secret", "")
    if not secret:
        raise HTTPException(status_code=404, detail="Solicitud no encontrada")
    try:
        return await svc.request_status(request_id, secret)
    except Exception as exc:
        raise _join_http_error(exc) from exc


@router.get(
    "/federation/join-requests",
    summary="List join requests",
    description="Pending join requests (all recent ones with include_decided=true).",
)
async def list_join_requests(
    request: Request,
    include_decided: bool = False,
    _user: UserContext = Depends(require_role("system_admin")),
) -> list[dict]:
    """List join requests.

    Args:
        include_decided: Also return accepted/denied/expired ones.

    Returns:
        Requests with remote node info, message, source IP and verification code.
    """
    return await _join_service(request).list_requests(include_decided=include_decided)


@router.post(
    "/federation/join-requests/{request_id}/accept",
    summary="Accept a join request",
    description=(
        "Accept a pending request: creates the peer (incoming tunnel) and a local "
        "service user federation:<name> that runs its requests here."
    ),
    responses={404: {"description": "Not found"}, 409: {"description": "Not pending"}},
)
async def accept_join_request(
    request_id: str,
    request: Request,
    body: JoinRequestDecision | None = None,
    _user: UserContext = Depends(require_role("system_admin")),
) -> PeerOut:
    """Accept a join request.

    Args:
        body: Optional peer ``name`` (defaults to the node's name) and ``notes``.

    Returns:
        The new peer (offline until the node picks up its token and dials in).
    """
    svc = _join_service(request)
    try:
        peer = await svc.accept(
            request_id,
            name=body.name if body else None,
            notes=body.notes if body else None,
        )
    except Exception as exc:
        raise _join_http_error(exc) from exc
    return _peer_out(peer)


@router.post(
    "/federation/join-requests/{request_id}/deny",
    summary="Deny a join request",
    description="Deny a pending join request. The node sees 'denied' on its next check.",
    responses={404: {"description": "Not found"}, 409: {"description": "Not pending"}},
)
async def deny_join_request(
    request_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Deny a join request.

    Returns:
        ``{"ok": True}``.
    """
    try:
        await _join_service(request).deny(request_id)
    except Exception as exc:
        raise _join_http_error(exc) from exc
    return {"ok": True}


# ── Invites ─────────────────────────────────────────────────────


@router.get(
    "/federation/invites",
    summary="List invites",
    description="Recent invites with their uses and whether they are still usable.",
)
async def list_invites(
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> list[dict]:
    """List invites (codes are never shown again).

    Returns:
        ``[{id, label, mode, max_uses, uses, expires_at, revoked, usable, created_at}]``.
    """
    return await _join_service(request).list_invites()


@router.post(
    "/federation/invites",
    summary="Create invite",
    description=(
        "Create a pre-approved join code: a node joining with it is accepted "
        "without manual approval. The code is shown only in this response."
    ),
)
async def create_invite(
    body: InviteCreate,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Create an invite.

    Returns:
        The invite plus ``code`` (shown once) and ``join_url`` (this node's public URL).
    """
    try:
        out = await _join_service(request).create_invite(
            label=body.label,
            mode=body.mode,
            max_uses=body.max_uses,
            expires_in_hours=body.expires_in_hours,
        )
    except Exception as exc:
        raise _join_http_error(exc) from exc
    out["join_url"] = getattr(settings, "public_url", None) or None
    return out


@router.delete(
    "/federation/invites/{invite_id}",
    summary="Revoke invite",
    description="Revoke an invite so it can no longer be used.",
    responses={404: {"description": "Not found"}},
)
async def revoke_invite(
    invite_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Revoke an invite.

    Returns:
        ``{"ok": True}``.
    """
    try:
        await _join_service(request).revoke_invite(invite_id)
    except Exception as exc:
        raise _join_http_error(exc) from exc
    return {"ok": True}


# ── Joining another node (requester side) ───────────────────────


async def _ensure_federation_started(request: Request):
    """Turn federation on (persisted) if it is off; return the manager."""
    fm = getattr(request.app.state, "federation_manager", None)
    if fm is not None:
        return fm
    from ocabra.core.federation import FederationManager
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.server_config import save_override

    settings.federation_enabled = True
    async with AsyncSessionLocal() as session:
        await save_override(session, "federation_enabled", True)
        await session.commit()
    fm = FederationManager(settings, AsyncSessionLocal)
    fm.gpu_manager = getattr(request.app.state, "gpu_manager", None)
    await fm.start()
    request.app.state.federation_manager = fm
    logger.info("federation_enabled_by_join")
    return fm


@router.post(
    "/federation/join",
    summary="Join another node's federation",
    description=(
        "Ask the node at hub_url to let this node join (turns federation on here "
        "if it was off). Only the other node needs a reachable URL: once accepted "
        "this node dials it and both use each other's models over that tunnel. "
        "With an invite_code it connects right away; otherwise it stays pending "
        "until the other admin accepts."
    ),
    responses={
        409: {"description": "Already linked to that node"},
        502: {"description": "The other node could not be reached or refused"},
    },
)
async def join_federation(
    body: JoinOutgoing,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> PeerOut:
    """Send a join request to another node.

    Returns:
        The new peer: ``join_state='pending'`` with ``join_verification_code``, or
        connecting already when accepted by invite.
    """
    fm = await _ensure_federation_started(request)
    try:
        peer = await fm.join_service.join(
            body.hub_url,
            message=body.message,
            invite_code=body.invite_code,
            name=body.name,
            mode=body.mode,
        )
    except Exception as exc:
        raise _join_http_error(exc) from exc
    return _peer_out(peer)


@router.post(
    "/federation/peers/{peer_id}/join-check",
    summary="Re-check a pending join request",
    description="Ask the other node now whether our pending join request was decided.",
    responses={404: {"description": "Peer not found"}},
)
async def join_check(
    peer_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> PeerOut:
    """Poll the hub once for a pending outgoing join request.

    Returns:
        The peer with its updated ``join_state``.
    """
    fm = _get_federation_manager(request)
    peer = next((p for p in fm.get_all_peers() if p.peer_id == peer_id), None)
    if peer is None:
        raise HTTPException(status_code=404, detail=f"Peer '{peer_id}' not found")
    if peer.join_state == "pending" and fm.join_service is not None:
        await fm.join_service.check_outgoing(peer)
    return _peer_out(peer)


# ── Remote panel: our UI showing another node ───────────────────

_PANEL_PASS_HEADERS = ("content-type", "accept")
_PANEL_RETURN_HEADERS = ("content-type", "content-disposition", "cache-control")


@router.api_route(
    "/federation/peers/{peer_id}/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    summary="Remote panel proxy",
    description=(
        "Forward a request to a federated node's API through the federation "
        "link, so this node's UI can show and manage it. Requires full access "
        "to the peer; the peer itself refuses users, groups, keys, its "
        "federation and config changes."
    ),
    responses={
        400: {"description": "Path outside the node's API"},
        403: {"description": "Peer does not have full access"},
        502: {"description": "Peer is offline or unreachable"},
    },
)
async def remote_panel_proxy(
    peer_id: str,
    path: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
):
    """Proxy one request of the remote panel to *peer_id*.

    Returns:
        The peer's response (status, content type and body, streamed).
    """
    from starlette.background import BackgroundTask

    peer = await _get_full_access_peer(peer_id, request)
    fm = _get_federation_manager(request)
    target = "/" + path.lstrip("/")
    if not target.startswith(("/ocabra/", "/v1/", "/api/")) and target not in (
        "/health",
        "/ready",
        "/version",
    ):
        raise HTTPException(status_code=400, detail="Path outside the node's API")
    if request.url.query:
        target = f"{target}?{request.url.query}"
    headers = {k: v for k, v in request.headers.items() if k.lower() in _PANEL_PASS_HEADERS}
    try:
        resp = await fm.proxy_raw(peer, request.method, target, await request.body(), headers)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Peer '{peer.name}' unreachable: {exc}"
        ) from exc
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() in _PANEL_RETURN_HEADERS}
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=out_headers,
        background=BackgroundTask(resp.aclose),
    )


# ── «Mi servidor, mis reglas» ────────────────────────────────────


@router.get(
    "/federation/rules",
    summary="Node rules for peers in rules mode",
    description=(
        "«Mi servidor, mis reglas»: what peers in `rules` mode may do on this node. "
        "`base` is where rules mode starts from, `defaults` this node's changes to it, "
        "and `effective` the result before each peer's own adjustments."
    ),
)
async def get_rules(
    request: Request, _user: UserContext = Depends(require_role("system_admin"))
) -> dict:
    """Node defaults for rules mode.

    Returns:
        ``{"base": {...}, "defaults": {...}, "effective": {...}}``.
    """
    from ocabra.core.federation_rules import RULES_MODE_BASE, PeerRules

    fm = _get_federation_manager(request)
    return {
        "base": RULES_MODE_BASE,
        "defaults": fm.default_rules,
        "effective": PeerRules.from_dict({**RULES_MODE_BASE, **fm.default_rules}).to_dict(),
    }


@router.put(
    "/federation/rules",
    summary="Set node rules for peers in rules mode",
    description="Replace this node's default rules (fields left out fall back to the base).",
)
async def set_rules(
    body: FederationRules,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Set the node defaults.

    Returns:
        The same shape as ``GET /federation/rules``.
    """
    fm = _get_federation_manager(request)
    await fm.set_default_rules(body.model_dump(exclude_unset=True))
    return await get_rules(request, _user)


@router.get(
    "/federation/peers/{peer_id}/rules-usage",
    summary="What a peer in rules mode has used",
    description="Requests in the last hour and Wh this month served to the peer, against its limits.",
    responses={404: {"description": "Peer not found"}},
)
async def get_rules_usage(
    peer_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Usage of a peer against its rules.

    Returns:
        ``{"requests_last_hour", "max_requests_per_hour", "wh_this_month", "max_wh_per_month"}``.
    """
    from ocabra.core import federation_rules as fr

    fm = _get_federation_manager(request)
    peer = next((p for p in fm.get_all_peers() if p.peer_id == peer_id), None)
    if peer is None:
        raise HTTPException(status_code=404, detail=f"Peer '{peer_id}' not found")
    rules = fm.rules_for(peer)
    return {
        "requests_last_hour": fr.requests_last_hour(peer.peer_id),
        "max_requests_per_hour": rules.max_requests_per_hour,
        "wh_this_month": round(await fr.energy_this_month(fm, peer), 2),
        "max_wh_per_month": rules.max_wh_per_month,
    }


# ── Per-model strategies (core/federation_policy.py) ─────────────


def _strategy_nodes(fm, model_id: str) -> list[dict]:
    """Every peer we know, with how it has *model_id* (loaded / cold / no)."""
    offered = {peer.peer_id: entry for peer, entry in fm.get_remote_catalog().get(model_id, [])}
    out = []
    for peer in fm.get_all_peers():
        entry = offered.get(peer.peer_id)
        status = None
        if entry is not None:
            status = "loaded" if str(entry.get("status", "")).lower() == "loaded" else "cold"
        out.append(
            {"name": peer.name, "peer_id": peer.peer_id, "online": peer.online, "model": status}
        )
    return out


@router.get(
    "/federation/strategies",
    summary="Per-model federation strategies",
    description=(
        "Where each model's requests are served: `auto` (warm beats cold, local bias), "
        "`local_only`, or `prefer` an ordered list of nodes. Models not listed are `auto`. "
        "With `?model_id=` returns that model's strategy and every node with its status "
        "for it (`loaded`, `cold` or null)."
    ),
)
async def get_strategies(
    request: Request,
    model_id: str | None = None,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """List strategies, or describe one model's.

    Returns:
        ``{"strategies": {model_id: {...}}}``, or for ``model_id``:
        ``{"model_id", "strategy", "nodes", "fallback", "local", "peers": [...]}``.
    """
    from ocabra.core import federation_policy as fp

    fm = _get_federation_manager(request)
    if model_id is None:
        return {"strategies": {k: v.to_dict() for k, v in fp.POLICIES.items()}}
    mm = getattr(request.app.state, "model_manager", None)
    local = mm is not None and await mm.get_state(model_id) is not None
    return {
        "model_id": model_id,
        **fp.policy_for(model_id).to_dict(),
        "local": local,
        "peers": _strategy_nodes(fm, model_id),
    }


@router.put(
    "/federation/strategies",
    summary="Set a model's federation strategy",
    description=(
        "`prefer` needs `nodes` (peer names or ids, or `local`), tried in order; the first "
        "online one that has the model serves it, loading it if cold. `fallback: false` "
        "answers 503 when none can, instead of going on as `auto`. `auto` clears it."
    ),
    responses={400: {"description": "Invalid strategy"}},
)
async def set_strategy(
    body: FederationStrategyIn,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Store a model's strategy.

    Returns:
        The same shape as ``GET /federation/strategies?model_id=``.
    """
    from ocabra.core import federation_policy as fp

    fm = _get_federation_manager(request)
    try:
        policy = fp.clean_policy(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await fm.set_policy(body.model_id, policy)
    return await get_strategies(request, body.model_id, _user)
