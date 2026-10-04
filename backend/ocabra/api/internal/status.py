"""Lightweight server-load status endpoint.

Provides a single snapshot with enough context for clients to explain what
the server is doing during cold starts:

- ``loads.queue_depth`` — total waiters behind the backend-load gate.
- ``loads.active`` — how many backend loads are running right now.
- ``loads.in_progress`` — canonical model ids currently being loaded.
- ``workers.loaded_count`` — how many models are currently resident.

Clients poll this while streaming stays quiet (e.g. between SSE keepalives)
or before firing a request when they want to render an expected wait.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.core.model_manager import ModelStatus

router = APIRouter(tags=["status"])


@router.get(
    "/status",
    summary="Server load snapshot",
    description=(
        "Returns the current queue depth of pending model loads, the "
        "actively loading model ids, and a count of resident workers. "
        "Intended for the Playground badge and for oCabra-aware SDKs "
        "that want to show a meaningful cold-start message."
    ),
)
async def server_status(
    request: Request,
    _user: UserContext = Depends(require_role("user")),
) -> dict[str, Any]:
    mm = request.app.state.model_manager
    in_progress = sorted(getattr(mm, "_load_in_progress", set()) or ())
    queue_waiters = int(getattr(mm, "_load_queue_waiters", 0) or 0)
    active = len(in_progress)

    loaded_ids: list[str] = []
    in_flight_total = 0
    try:
        for state in list(getattr(mm, "_states", {}).values()):
            if state.status == ModelStatus.LOADED:
                loaded_ids.append(state.model_id)
        in_flight_total = sum(int(v or 0) for v in (mm._in_flight or {}).values())
    except Exception:  # noqa: BLE001 — status endpoint must never 500
        pass

    # Requests parked in _wait_for_service_gpu_and_retry_load (see _deps.py):
    # they already failed a load attempt with InsufficientVRAMError caused by
    # an external service's GPU reservation (Hunyuan, TRELLIS.2, ...) and are
    # polling for it to clear instead of failing outright. Invisible to every
    # other counter above — those only see the backend-load gate, not this
    # separate retry loop — so without this a queued request looks like
    # nothing is happening for up to model_load_wait_timeout_s.
    service_gpu_waits: list[dict[str, str]] = []
    try:
        service_gpu_waits = [
            {"modelId": model_id, "blockedBy": blocker}
            for model_id, blocker in dict(getattr(mm, "_service_gpu_waiters", {}) or {}).items()
        ]
    except Exception:  # noqa: BLE001
        pass

    return {
        "loads": {
            "queue_depth": queue_waiters + active,
            "waiting": queue_waiters,
            "active": active,
            "in_progress": in_progress,
            "waiting_for_service_gpu": service_gpu_waits,
        },
        "workers": {
            "loaded_count": len(loaded_ids),
            "loaded_ids": loaded_ids,
            "in_flight_requests": in_flight_total,
        },
    }
