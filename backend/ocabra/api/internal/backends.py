"""Internal API — modular backends (Bloque 15 Fase 1).

Endpoints:

* ``GET    /ocabra/backends``                         list all known backends
* ``GET    /ocabra/backends/{backend_type}``          detailed state
* ``POST   /ocabra/backends/{backend_type}/install``  SSE stream of progress
* ``POST   /ocabra/backends/{backend_type}/uninstall``
* ``GET    /ocabra/backends/{backend_type}/logs``     install/uninstall logs
* ``GET    /ocabra/backends/{backend_type}/published`` prebuilt image on offer

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
from ocabra.core.model_manager_helpers import strip_worker_suffix

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["backends"])


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------


class InstallRequest(BaseModel):
    method: Literal["source", "oci", "auto"] = Field(
        default="auto",
        description=(
            "Install method — 'oci' downloads the prebuilt backend published in the "
            "OCI repository (same versions as the server that published it); 'source' "
            "creates a venv and runs pip / builds; 'auto' uses the prebuilt image when "
            "there is one and falls back to source. Installing over an installed "
            "backend updates it (the old folder is kept until the new one is in place)."
        ),
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
    oci_ref: str | None = None
    oci_digest: str | None = None
    packages: dict[str, str] = Field(default_factory=dict)
    declared_packages: list[str] = Field(default_factory=list)
    git_ref: str | None = None
    estimated_size_mb: int = 0
    actual_size_mb: int | None = None
    error: str | None = None
    install_progress: float | None = None
    install_detail: str | None = None
    models_loaded: int = Field(
        default=0,
        description="Modelos de este backend cargados ahora mismo (cada modelo cuenta una vez "
        "aunque tenga varios workers de peldaño cargados).",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_installer(request: Request) -> BackendInstaller:
    installer: BackendInstaller | None = getattr(
        request.app.state, "backend_installer", None
    )
    if installer is None:
        raise HTTPException(
            status_code=503,
            detail="Backend installer not ready",
            headers={"Retry-After": "5"},
        )
    return installer


async def _loaded_by_backend(request: Request) -> dict[str, int]:
    """Modelos cargados por backend. Cuenta los ids canónicos: un modelo con
    presets se carga como worker derivado (``<id>::<hash>``) y no debe contar
    dos veces ni dejar de contar."""
    mm = getattr(request.app.state, "model_manager", None)
    if mm is None:
        return {}
    try:
        states = await mm.list_states(include_derived=True)
    except Exception:  # noqa: BLE001 — el listado de backends no debe romperse por esto
        return {}
    seen: dict[str, set[str]] = {}
    for st in states:
        if getattr(st.status, "value", st.status) != "loaded":
            continue
        # Only the preset hash goes: a GGUF ``repo::file`` is its own model.
        seen.setdefault(st.backend_type, set()).add(strip_worker_suffix(str(st.model_id)))
    return {backend: len(ids) for backend, ids in seen.items()}


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
    loaded = await _loaded_by_backend(request)
    return [
        {**_state_to_dict(s), "models_loaded": loaded.get(s.backend_type, 0)}
        for s in installer.list_states()
    ]


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
    loaded = await _loaded_by_backend(request)
    return {**_state_to_dict(state), "models_loaded": loaded.get(backend_type, 0)}


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
        409: {
            "description": "Install already in progress, or models of this backend "
            "are loaded (unload them to update it)"
        },
        400: {"description": "Invalid install method for this backend"},
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

    blocker = installer.reinstall_blocker(backend_type)
    if blocker:
        raise HTTPException(status_code=409, detail=blocker)

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


@router.get(
    "/backends/{backend_type}/install",
    summary="Install a backend (GET alias for EventSource)",
    description=(
        "GET variant of install that accepts `method` as a query parameter. Exists "
        "because the browser `EventSource` API only issues GET requests. Same SSE "
        "behaviour as POST."
    ),
    responses={
        404: {"description": "Backend not registered"},
        409: {
            "description": "Install already in progress, or models of this backend "
            "are loaded (unload them to update it)"
        },
        400: {"description": "Invalid install method for this backend"},
    },
)
async def install_backend_get(
    backend_type: str,
    request: Request,
    method: str = "auto",
    _user: UserContext = Depends(require_role("model_manager")),
) -> StreamingResponse:
    return await install_backend(
        backend_type,
        InstallRequest(method=method),
        request,
        _user,
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


@router.get(
    "/backends/{backend_type}/published",
    summary="Prebuilt image on offer for a backend",
    description=(
        "What the OCI repository (`BACKEND_OCI_REPOSITORY`) publishes for this backend: "
        "reference, digest, size, version and key package versions, so it can be "
        "compared with what is installed. `update_available` is true when the backend "
        "was installed from an older image, or from source with different package "
        "versions. Cached for 10 minutes unless `refresh=true`."
    ),
    responses={
        404: {"description": "Backend not registered"},
        502: {"description": "The registry could not be reached"},
    },
)
async def get_backend_published(
    backend_type: str,
    request: Request,
    refresh: bool = False,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, Any]:
    """Published prebuilt image for *backend_type*.

    Returns:
        ``{"published": {...} | null, "update_available": bool}``.
    """
    installer = _get_installer(request)
    try:
        state = installer.get_state(backend_type)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Backend '{backend_type}' not registered"
        ) from exc
    try:
        published = await installer.published(backend_type, refresh=refresh)
    except Exception as exc:  # noqa: BLE001 — surface as a gateway problem
        raise HTTPException(status_code=502, detail=f"Registry: {exc}") from exc
    update = False
    if published and str(state.install_status) == "installed":
        if state.oci_digest:
            update = state.oci_digest != published["digest"]
        elif state.install_source == "source":
            update = bool(published["packages"]) and published["packages"] != state.packages
    return {"published": published, "update_available": update}
