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
