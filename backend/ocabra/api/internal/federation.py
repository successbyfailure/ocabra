"""Federation API endpoints for peer management and heartbeat.

Provides:
- GET  /federation/heartbeat      — heartbeat response for remote peers
- GET  /federation/peers          — list all peers with online status
- POST /federation/peers          — add a new peer
- PATCH /federation/peers/{id}    — update a peer
- DELETE /federation/peers/{id}   — remove a peer
- POST /federation/peers/{id}/test — test connection to a peer
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from starlette.responses import StreamingResponse

from ocabra.core.federation import PeerState
from ocabra.schemas.federation import (
    HeartbeatGpu,
    HeartbeatLoad,
    HeartbeatModel,
    HeartbeatResponse,
    PeerCreate,
    PeerOut,
    PeerTestResult,
    PeerUpdate,
    RemoteDownloadRequest,
    RemoteLoadRequest,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["federation"])


def _get_federation_manager(request: Request):
    """Extract FederationManager from app state, raise 503 if not enabled."""
    fm = getattr(request.app.state, "federation_manager", None)
    if fm is None:
        raise HTTPException(
            status_code=503,
            detail="Federation is not enabled on this node",
        )
    return fm


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

    Includes GPU states, loaded models with their profiles, and current
    load metrics (active requests and average GPU utilization).
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

    # Models
    models: list[HeartbeatModel] = []
    if model_manager is not None:
        try:
            from ocabra.core.model_manager import ModelStatus

            model_states = await model_manager.list_states()
            for ms in model_states:
                if ms.status != ModelStatus.LOADED:
                    continue
                profile_ids: list[str] = []
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

    return HeartbeatResponse(
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
    peers = fm.get_all_peers()
    return [
        PeerOut(
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
        )
        for p in peers
    ]


@router.post(
    "/federation/peers",
    summary="Add federation peer",
    description="Register a new federation peer. The API key is encrypted before storage.",
    responses={400: {"description": "Invalid peer configuration"}},
)
async def add_peer(
    body: PeerCreate,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> PeerOut:
    """Add a new federation peer.

    Args:
        body: Peer creation request with name, url, api_key, and access_level.

    Returns:
        The newly created peer with its initial state.
    """
    fm = _get_federation_manager(request)
    try:
        state = await fm.add_peer(
            name=body.name,
            url=body.url,
            api_key=body.api_key,
            access_level=body.access_level,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PeerOut(
        id=state.peer_id,
        name=state.name,
        url=state.url,
        access_level=state.access_level,
        enabled=state.enabled,
        online=state.online,
        last_heartbeat=state.last_heartbeat,
        models=state.models,
        gpus=state.gpus,
        load=state.load,
    )


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
    state = await fm.update_peer(peer_id, **patch)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Peer '{peer_id}' not found")
    return PeerOut(
        id=state.peer_id,
        name=state.name,
        url=state.url,
        access_level=state.access_level,
        enabled=state.enabled,
        online=state.online,
        last_heartbeat=state.last_heartbeat,
        models=state.models,
        gpus=state.gpus,
        load=state.load,
    )


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


async def _get_full_access_peer(
    peer_id: str, request: Request
) -> PeerState:
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
    "/federation/peers/{peer_id}/models/{model_id}/load",
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
    "/federation/peers/{peer_id}/models/{model_id}/unload",
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
        "Trigger a model download on a remote peer and stream back SSE progress. "
        "Requires the peer to have access_level='full'."
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
) -> StreamingResponse:
    """Trigger a model download on a remote peer and stream progress via SSE.

    Args:
        peer_id: UUID of the target peer.
        body: Download request (source, model_ref, artifact, register_config).

    Returns:
        StreamingResponse with SSE events relaying the peer's download progress.
    """
    peer = await _get_full_access_peer(peer_id, request)
    fm = _get_federation_manager(request)
    payload = body.model_dump(exclude_none=True)

    async def _stream():
        try:
            async for chunk in fm.proxy_stream(
                peer=peer,
                path="/ocabra/downloads",
                body=payload,
                headers=dict(request.headers),
            ):
                yield chunk
        except Exception as exc:
            logger.warning(
                "remote_download_stream_error",
                peer=peer.name,
                error=str(exc),
            )
            yield f"data: {{\"error\": \"{exc}\"}}\n\n".encode()

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
        resp = await fm.proxy_request(
            peer=peer,
            path="/ocabra/gpus",
            body={},
            headers=dict(request.headers),
        )
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
