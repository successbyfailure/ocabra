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


@router.get("/services")
async def list_services(
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict]:
    sm = request.app.state.service_manager
    states = await sm.list_states()
    return [state.to_dict() for state in states]


@router.get("/services/{service_id}")
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


@router.patch("/services/{service_id}")
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


@router.post("/services/{service_id}/refresh")
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


@router.patch("/services/{service_id}/runtime")
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


@router.post("/services/{service_id}/touch")
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


@router.post("/services/{service_id}/start")
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


@router.get("/services/{service_id}/generations")
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


@router.post("/services/{service_id}/unload")
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
