"""oCabra updating itself: /ocabra/system/update."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ocabra.api._deps_auth import UserContext, require_role
from ocabra.config import settings
from ocabra.core.self_update import SelfUpdater
from ocabra.schemas.system_update import SelfUpdateApply, SelfUpdatePolicy

router = APIRouter(tags=["system"])


def _updater(request: Request) -> SelfUpdater:
    updater = getattr(request.app.state, "self_updater", None)
    if updater is None:
        updater = SelfUpdater(request.app)
        request.app.state.self_updater = updater
    return updater


@router.get("/system/update", summary="Version, newest release and update policy")
async def get_update_status(
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """State of oCabra's self-update on this node.

    Response: ``current`` (running version), ``latest`` (newest release in the
    registry, cached up to 10 min), ``update_available``, ``policy``
    (``off``/``notify``/``auto``), ``idle_minutes``, ``phase``
    (``idle``/``waiting_idle``/``updating``/``failed``), ``busy`` (what keeps
    an automatic update waiting), ``deployment`` (``kind`` —
    ``compose``/``local_build``/``node``/``manual`` —, ``can_apply`` and the
    reason when it cannot) and ``updater`` (state and last log lines of the
    updater container, if one ran).
    """
    updater = _updater(request)
    await updater.check()
    return await updater.snapshot()


@router.post("/system/update/check", summary="Look for a new release now")
async def check_update(
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Ask the registry for the newest release, ignoring the cache. Same
    response as ``GET /system/update``."""
    updater = _updater(request)
    await updater.check(force=True)
    return await updater.snapshot()


@router.post(
    "/system/update/apply",
    summary="Update this node to a release",
    responses={409: {"description": "Cannot update now (busy, not a managed deployment...)"}},
)
async def apply_update(
    body: SelfUpdateApply,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Start the updater: fast-forwards the checkout to ``vX.Y.Z``, pulls the
    release images, recreates api and frontend and rolls back if the new API
    does not come up with that version. The API restarts during it (~1 min).

    Body: ``version`` (default: newest) and ``force`` (skip the idle check).
    409 with the reason when it cannot start. Response: the update state.
    """
    version = body.version.lstrip("v") if body.version else None
    try:
        return await _updater(request).apply(version, force=body.force)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.patch("/system/update/policy", summary="Set this node's update policy")
async def set_update_policy(
    body: SelfUpdatePolicy,
    request: Request,
    _user: UserContext = Depends(require_role("system_admin")),
) -> dict:
    """Body: ``policy`` (``off``, ``notify`` or ``auto``) and/or
    ``idleMinutes`` (how long ``auto`` waits with no activity). Persisted in
    the server config. Response: the update state."""
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.server_config import save_override

    async with AsyncSessionLocal() as session:
        if body.policy is not None:
            settings.self_update_policy = body.policy
            await save_override(session, "self_update_policy", body.policy)
        if body.idle_minutes is not None:
            settings.self_update_idle_minutes = body.idle_minutes
            await save_override(session, "self_update_idle_minutes", body.idle_minutes)
        await session.commit()
    return await _updater(request).snapshot()
