"""Sessions admin endpoints — GET / kill live Realtime sessions.

Bloque 20 — Etapa 4.

Backs the "Sessions" admin page in the UI. Two operations:

    GET  /ocabra/sessions          — enumerate active sessions with
                                     workers_held, durations, activity.
    POST /ocabra/sessions/{id}/kill — force-close a session (e.g. WebSocket
                                      zombie the auto-sweeper hasn't hit yet).

Both restricted to ``system_admin`` because seeing / disconnecting other
users' Realtime sessions is a privileged operation.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.core.session_registry import SessionRegistry

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["sessions"])


class SessionInfoOut(BaseModel):
    session_id: str
    user_id: uuid.UUID | None
    api_key_name: str | None
    workers_held: list[str]
    started_at: str
    last_activity_at: str
    is_paused: bool
    is_zombie: bool


class KillResponse(BaseModel):
    session_id: str
    killed: bool


def _get_registry(request: Request) -> SessionRegistry:
    registry: SessionRegistry | None = getattr(
        request.app.state, "session_registry", None
    )
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Session registry not ready",
        )
    return registry


@router.get(
    "/sessions",
    response_model=list[SessionInfoOut],
    summary="List live Realtime sessions",
    description=(
        "Returns every session the SessionRegistry currently holds. The "
        "``is_paused`` flag is a soft signal (workers may be shared with "
        "concurrent traffic); ``is_zombie`` means the session missed enough "
        "heartbeats that the auto-sweeper will unregister it on the next tick."
    ),
)
async def list_sessions(
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> list[SessionInfoOut]:
    registry = _get_registry(request)
    pause_threshold = int(settings.session_pause_threshold_s)
    max_idle = int(settings.session_max_idle_s)
    out: list[SessionInfoOut] = []
    for info in registry.list_sessions():
        out.append(
            SessionInfoOut(
                session_id=info.session_id,
                user_id=info.user_id,
                api_key_name=info.api_key_name,
                workers_held=list(info.workers_held),
                started_at=info.started_at.isoformat(),
                last_activity_at=info.last_activity_at.isoformat(),
                is_paused=info.is_paused(pause_threshold),
                is_zombie=info.is_zombie(max_idle),
            )
        )
    return out


@router.post(
    "/sessions/{session_id}/kill",
    response_model=KillResponse,
    summary="Force-close a Realtime session",
    description=(
        "Unregisters the session from the registry, releasing its workers "
        "for evict/routing decisions. Does NOT close the underlying "
        "WebSocket — the client sees a normal disconnect when their next "
        "chunk finds no session state on the server side."
    ),
    responses={404: {"description": "Session not found"}},
)
async def kill_session(
    session_id: str,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> KillResponse:
    registry = _get_registry(request)
    if registry.get(session_id) is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    await registry.unregister(session_id, reason="admin_kill")
    return KillResponse(session_id=session_id, killed=True)
