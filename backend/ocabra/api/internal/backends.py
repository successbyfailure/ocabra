"""Internal API — modular backends (Bloque 15 Fase 1).

Endpoints:

* ``GET    /ocabra/backends``                         list all known backends
* ``GET    /ocabra/backends/{backend_type}``          detailed state
* ``POST   /ocabra/backends/{backend_type}/install``  SSE stream of progress
* ``POST   /ocabra/backends/{backend_type}/uninstall``
* ``GET    /ocabra/backends/{backend_type}/logs``     install/uninstall logs

All endpoints require the ``model_manager`` role or higher.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.core.backend_installer import (
    BackendAlreadyInstallingError,
    BackendInstaller,
    BackendModuleState,
)

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["backends"])


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------


class InstallRequest(BaseModel):
    method: Literal["source", "oci"] = Field(
        default="source",
        description="Install method — 'source' creates a venv and runs pip; 'oci' pulls a pre-built image (Fase 3).",
    )


class BackendStateResponse(BaseModel):
    """Public shape for ``BackendModuleState`` — strictly additive on the API."""

    backend_type: str
    display_name: str
    description: str
    tags: list[str]
    install_status: str
    installed_version: str | None = None
    installed_at: str | None = None
    install_source: str | None = None
    estimated_size_mb: int = 0
    actual_size_mb: int | None = None
    error: str | None = None
    install_progress: float | None = None
    install_detail: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_installer(request: Request) -> BackendInstaller:
    installer: BackendInstaller | None = getattr(
        request.app.state, "backend_installer", None
    )
    if installer is None:
        raise HTTPException(status_code=503, detail="Backend installer not ready")
    return installer


def _state_to_dict(state: BackendModuleState) -> dict[str, Any]:
    raw = asdict(state)
    # Normalise types for JSON transport.
    installed_at = raw.get("installed_at")
    if isinstance(installed_at, datetime):
        raw["installed_at"] = installed_at.isoformat()
    install_status = raw.get("install_status")
    if install_status is not None and not isinstance(install_status, str):
        raw["install_status"] = str(install_status)
    return raw


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/backends",
    response_model=list[BackendStateResponse],
    summary="List modular backends",
    description=(
        "Return every known inference backend together with its install status. "
        "Built-in backends (always available) are included with `install_source='built-in'`."
    ),
)
async def list_backends(
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> list[dict[str, Any]]:
    installer = _get_installer(request)
    return [_state_to_dict(s) for s in installer.list_states()]


@router.get(
    "/backends/{backend_type}",
    response_model=BackendStateResponse,
    summary="Get backend state",
    description="Return detailed install status for a single backend.",
    responses={404: {"description": "Backend not registered"}},
)
async def get_backend(
    backend_type: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, Any]:
    installer = _get_installer(request)
    try:
        state = installer.get_state(backend_type)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Backend '{backend_type}' not registered"
        ) from exc
    return _state_to_dict(state)


@router.post(
    "/backends/{backend_type}/install",
    summary="Install a backend",
    description=(
        "Install a modular backend. The response is a Server-Sent Events stream with "
        "a `data:` line per progress update. The stream terminates when the backend "
        "reaches the `installed` or `error` status."
    ),
    responses={
        404: {"description": "Backend not registered"},
        409: {"description": "Install already in progress"},
        400: {"description": "Invalid install method for this backend"},
        501: {"description": "OCI install not implemented yet (Fase 3)"},
    },
)
async def install_backend(
    backend_type: str,
    payload: InstallRequest,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> StreamingResponse:
    installer = _get_installer(request)

    # Fail fast on unknown backend / invalid method so the caller gets a real
    # HTTP error instead of an SSE stream that closes immediately.
    try:
        installer.get_state(backend_type)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Backend '{backend_type}' not registered"
        ) from exc

    if payload.method == "oci":
        raise HTTPException(
            status_code=501,
            detail="OCI-based install is not yet implemented (planned for Fase 3).",
        )

    async def _event_stream():
        try:
            async for state in installer.install(backend_type, method=payload.method):
                data = json.dumps(_state_to_dict(state))
                yield f"data: {data}\n\n"
        except BackendAlreadyInstallingError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        except NotImplementedError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "backend_install_stream_failed", backend_type=backend_type
            )
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/backends/{backend_type}/uninstall",
    response_model=BackendStateResponse,
    summary="Uninstall a backend",
    description=(
        "Uninstall a modular backend. Fails with 409 if any model is currently loaded "
        "against this backend. Built-in backends (e.g. `ollama`) cannot be uninstalled."
    ),
    responses={
        404: {"description": "Backend not registered"},
        409: {"description": "Backend has loaded models or is built-in"},
    },
)
async def uninstall_backend(
    backend_type: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, Any]:
    installer = _get_installer(request)
    try:
        state = await installer.uninstall(backend_type)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Backend '{backend_type}' not registered"
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _state_to_dict(state)


@router.get(
    "/backends/{backend_type}/logs",
    summary="Fetch install logs",
    description=(
        "Return the captured stdout/stderr lines from the most recent install or "
        "uninstall operation for this backend. Oldest lines are dropped once the "
        "buffer is full."
    ),
    responses={404: {"description": "Backend not registered"}},
)
async def get_backend_logs(
    backend_type: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, Any]:
    installer = _get_installer(request)
    try:
        lines = installer.get_logs(backend_type)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Backend '{backend_type}' not registered"
        ) from exc
    return {"backend_type": backend_type, "lines": lines}
