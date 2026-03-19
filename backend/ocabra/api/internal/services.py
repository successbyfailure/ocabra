from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["services"])


class ServiceRuntimePatch(BaseModel):
    runtime_loaded: bool
    active_model_ref: str | None = None
    detail: str | None = None


@router.get("/services")
async def list_services(request: Request) -> list[dict]:
    sm = request.app.state.service_manager
    states = await sm.list_states()
    return [state.to_dict() for state in states]


@router.get("/services/{service_id}")
async def get_service(service_id: str, request: Request) -> dict:
    sm = request.app.state.service_manager
    state = await sm.get_state(service_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")
    return state.to_dict()


@router.post("/services/{service_id}/refresh")
async def refresh_service(service_id: str, request: Request) -> dict:
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
    return state.to_dict()


@router.post("/services/{service_id}/start")
async def start_service(service_id: str, request: Request) -> dict:
    sm = request.app.state.service_manager
    try:
        state = await sm.start_service(service_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return state.to_dict()


@router.post("/services/{service_id}/unload")
async def unload_service(service_id: str, request: Request) -> dict:
    sm = request.app.state.service_manager
    try:
        state = await sm.unload(service_id, reason="manual")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return state.to_dict()
