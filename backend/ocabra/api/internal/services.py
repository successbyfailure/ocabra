from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from ocabra.api._deps_auth import UserContext, require_role

router = APIRouter(tags=["services"])


class ServiceRuntimePatch(BaseModel):
    runtime_loaded: bool
    active_model_ref: str | None = None
    detail: str | None = None


class ServicePatch(BaseModel):
    enabled: bool


class EnsureVramRequest(BaseModel):
    vram_needed_mb: int
    # Holds off the normal auto_reload watcher on any evicted WARM model for this
    # many seconds, so it doesn't reload itself mid-generation and re-fill the GPU.
    # 0 = no hold (default pressure-eviction behaviour: may reload within ~30s).
    suppress_reload_seconds: int = 0


@router.get(
    "/services",
    summary="List all services",
    description="Return the state of all registered interactive services (ComfyUI, A1111, Hunyuan, etc.).",
)
async def list_services(
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict]:
    sm = request.app.state.service_manager
    states = await sm.list_states()
    return [state.to_dict() for state in states]


@router.get(
    "/services/{service_id}",
    summary="Get service state",
    description="Return the full state of a single interactive service.",
    responses={404: {"description": "Service not found"}},
)
async def get_service(
    service_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Get current state for one generation service."""
    sm = request.app.state.service_manager
    state = await sm.get_state(service_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")
    return state.to_dict()


@router.patch(
    "/services/{service_id}",
    summary="Enable or disable a service",
    description="Toggle the enabled flag for a service. Disabled services are excluded from scheduling.",
    responses={404: {"description": "Service not found"}},
)
async def patch_service(
    service_id: str,
    body: ServicePatch,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Enable or disable a generation service in oCabra."""
    sm = request.app.state.service_manager
    try:
        state = await sm.set_enabled(service_id, enabled=body.enabled)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.to_dict()


@router.post(
    "/services/{service_id}/refresh",
    summary="Refresh service state",
    description="Run a health check and runtime probe to update the service state.",
    responses={404: {"description": "Service not found"}},
)
async def refresh_service(
    service_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    sm = request.app.state.service_manager
    try:
        state = await sm.refresh(service_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.to_dict()


@router.patch(
    "/services/{service_id}/runtime",
    summary="Update service runtime state",
    description="Mark a service as having its runtime/weights loaded or unloaded, and optionally set the active model reference.",
    responses={
        404: {"description": "Service not found"},
        409: {"description": "Conflicting runtime state"},
    },
)
async def patch_service_runtime(
    service_id: str,
    body: ServiceRuntimePatch,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    sm = request.app.state.service_manager
    try:
        state = await sm.mark_runtime(
            service_id,
            runtime_loaded=body.runtime_loaded,
            active_model_ref=body.active_model_ref,
            detail=body.detail,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return state.to_dict()


@router.post(
    "/services/{service_id}/touch",
    summary="Touch service activity",
    description="Update last_activity_at to reset the idle eviction timer. Called by the gateway proxy on each proxied request.",
    responses={404: {"description": "Service not found"}},
)
async def touch_service(
    service_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    """Mark a service as active (updates last_activity_at to reset idle timer).

    Called by the gateway proxy on each proxied request to prevent idle eviction
    while users are actively using the service.
    """
    sm = request.app.state.service_manager
    try:
        state = await sm.touch(service_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.to_dict()


@router.post(
    "/services/{service_id}/start",
    summary="Start a service",
    description="Start the Docker container for an interactive service.",
    responses={
        404: {"description": "Service not found"},
        409: {"description": "Service is already running"},
        502: {"description": "Failed to start the service container"},
    },
)
async def start_service(
    service_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    sm = request.app.state.service_manager
    try:
        state = await sm.start_service(service_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return state.to_dict()


@router.get(
    "/services/{service_id}/generations",
    summary="List service generations",
    description="Return recent generation events for a service, ordered newest first.",
)
async def get_service_generations(
    service_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict]:
    """Return recent generation events for a service, newest first."""
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.stats import ServiceGenerationStat
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ServiceGenerationStat)
            .where(ServiceGenerationStat.service_id == service_id)
            .order_by(ServiceGenerationStat.started_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

    return [
        {
            "id": str(row.id),
            "service_id": row.service_id,
            "service_type": row.service_type,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "duration_ms": row.duration_ms,
            "gpu_index": row.gpu_index,
            "vram_peak_mb": row.vram_peak_mb,
            "evicted": row.evicted,
        }
        for row in rows
    ]


@router.post(
    "/services/{service_id}/ensure_vram",
    summary="Ensure free VRAM on a service's preferred GPU",
    description=(
        "Evict on-demand/warm inference models (and other evictable services) on "
        "this service's preferred GPU until vram_needed_mb is free. Never evicts "
        "PIN-policy models. Call this right before loading a heavy pipeline in a "
        "service that manages its own VRAM outside of oCabra's scheduler (Hunyuan, "
        "TRELLIS.2, ...) — those services never trigger eviction on their own."
    ),
    responses={404: {"description": "Service not found"}},
)
async def ensure_service_vram(
    service_id: str,
    body: EnsureVramRequest,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    sm = request.app.state.service_manager
    state = await sm.get_state(service_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")

    from ocabra.core.scheduler import InsufficientVRAMError

    mm = request.app.state.model_manager
    try:
        gpu_indices = await mm.ensure_vram_free(
            requesting_id=f"service:{service_id}",
            vram_needed_mb=body.vram_needed_mb,
            preferred_gpu=state.preferred_gpu,
            suppress_reload_seconds=body.suppress_reload_seconds,
        )
        # Reserve the freed VRAM for the duration of the service's own generation
        # (released on /unload, on the next health check once runtime_loaded goes
        # false, or if the container becomes unreachable) — otherwise an unrelated
        # inference load can land on this GPU mid-generation and starve it.
        await sm.reserve_gpu_vram(service_id, body.vram_needed_mb)
        return {"ok": True, "gpu_indices": gpu_indices}
    except InsufficientVRAMError as exc:
        # Eviction couldn't free the FULL amount requested — but the caller's own
        # design (see TRELLIS.2/server_app.py's _ensure_ocabra_vram) is to attempt
        # the load anyway; it may still fit in whatever got freed, or fail with a
        # clean CUDA OOM. Either way it's about to use this GPU, so still reserve —
        # skipping the reservation here would leave a real, in-progress generation
        # completely unprotected against a second unrelated load racing in right
        # behind it (reproduced 2026-09-29: ensure_vram partial-failed, no lock was
        # taken, the generation ran with zero protection).
        await sm.reserve_gpu_vram(service_id, body.vram_needed_mb)
        return {"ok": False, "detail": str(exc)}


@router.post(
    "/services/{service_id}/unload",
    summary="Unload service runtime",
    description="Unload the runtime/weights from a service to free GPU VRAM.",
    responses={
        404: {"description": "Service not found"},
        409: {"description": "Service is not in an unloadable state"},
        502: {"description": "Unload request to the service failed"},
    },
)
async def unload_service(
    service_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict:
    sm = request.app.state.service_manager
    try:
        state = await sm.unload(service_id, reason="manual")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return state.to_dict()
