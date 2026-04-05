"""FastAPI dependencies for authentication and authorization."""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field

import structlog
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select

logger = structlog.get_logger(__name__)

# Role hierarchy: higher value = more permissions.
ROLE_HIERARCHY: dict[str, int] = {
    "user": 0,
    "model_manager": 1,
    "system_admin": 2,
}


@dataclass
class UserContext:
    """Resolved identity of the caller for the current request.

    Attributes:
        user_id: UUID string of the authenticated user, or ``None`` for anonymous.
        username: Display name, or ``None`` for anonymous.
        role: One of ``"user"``, ``"model_manager"``, ``"system_admin"``.
        group_ids: List of group UUID strings the user belongs to.
        accessible_model_ids: Set of model IDs accessible to the user.
            An empty set means **all models** (applies only to admins).
        is_anonymous: True when the caller has no credentials.
        is_admin: Shortcut for ``role == "system_admin"``.
    """

    user_id: str | None
    username: str | None
    role: str
    group_ids: list[str] = field(default_factory=list)
    accessible_model_ids: set[str] = field(default_factory=set)
    is_anonymous: bool = False
    key_group_id: str | None = None  # group_id from the API key used for this request

    @property
    def is_admin(self) -> bool:
        return self.role == "system_admin"


# ── Internal helpers ──────────────────────────────────────────────────────────


async def _build_anonymous_context(session) -> UserContext:
    """Build a UserContext for an unauthenticated caller.

    Anonymous users implicitly belong to the ``default`` group.  If the default
    group has at least one model assigned, only those models are accessible.
    If the default group is empty (no models explicitly assigned), all models
    are accessible — preserving the pre-auth open-access behaviour for
    deployments that have disabled API-key requirements without configuring
    model restrictions.
    """
    from ocabra.db.auth import Group, GroupModel

    # Single query: fetch default group id and all its model_ids at once.
    rows = await session.execute(
        select(Group.id, GroupModel.model_id)
        .outerjoin(GroupModel, GroupModel.group_id == Group.id)
        .where(Group.is_default.is_(True))
    )
    default_group_id: _uuid.UUID | None = None
    model_ids: set[str] = set()
    for group_id, model_id in rows:
        default_group_id = group_id
        if model_id is not None:
            model_ids.add(model_id)

    group_ids = [str(default_group_id)] if default_group_id else []

    # Empty set means "all models" (unrestricted) — used when the default
    # group has no explicit model assignments.
    return UserContext(
        user_id=None,
        username=None,
        role="user",
        group_ids=group_ids,
        accessible_model_ids=model_ids,  # empty = no models accessible (default group has none assigned)
        is_anonymous=True,
    )


async def _resolve_jwt_cookie(token: str, session) -> UserContext | None:
    """Try to resolve a UserContext from a JWT cookie value.

    Returns ``None`` if the token is invalid (caller should treat as no-auth).
    """
    from ocabra.core.auth_manager import AuthError, decode_access_token
    from ocabra.db.auth import User, UserGroup

    try:
        payload = decode_access_token(token)
    except AuthError as exc:
        logger.debug("auth_jwt_invalid", reason=str(exc))
        return None

    user_id: str = payload.get("sub", "")
    if not user_id:
        return None

    # Check Redis revocation list.
    try:
        from ocabra.redis_client import get_redis

        redis = await get_redis()
        jti = payload.get("jti", "")
        if jti and await redis.exists(f"jwt_revoked:{jti}"):
            logger.debug("auth_jwt_revoked", jti=jti)
            return None
    except Exception as exc:
        logger.warning("auth_redis_revocation_check_failed", error=str(exc))

    # Load user from DB.
    result = await session.execute(select(User).where(User.id == user_id))
    user: User | None = result.scalar_one_or_none()
    if user is None or not user.is_active:
        return None

    group_result = await session.execute(
        select(UserGroup.group_id).where(UserGroup.user_id == user.id)
    )
    group_ids = [str(gid) for gid in group_result.scalars().all()]

    accessible = await _fetch_accessible_models(user.role, group_ids, session)

    return UserContext(
        user_id=str(user.id),
        username=user.username,
        role=user.role,
        group_ids=group_ids,
        accessible_model_ids=accessible,
        is_anonymous=False,
    )


async def _resolve_api_key(raw_key: str, session) -> UserContext | None:
    """Try to resolve a UserContext from a raw API key value.

    Updates ``last_used_at`` on the key row on success.

    Returns ``None`` if the key is invalid, revoked, or expired.
    """
    from datetime import UTC, datetime

    from ocabra.core.auth_manager import hash_api_key
    from ocabra.db.auth import ApiKey, User, UserGroup
    from sqlalchemy.orm import joinedload

    key_hash = hash_api_key(raw_key)
    now = datetime.now(UTC)

    # Single query: load ApiKey with its User in one round-trip.
    result = await session.execute(
        select(ApiKey)
        .options(joinedload(ApiKey.user))
        .where(
            ApiKey.key_hash == key_hash,
            ApiKey.is_revoked.is_(False),
            (ApiKey.expires_at.is_(None)) | (ApiKey.expires_at > now),
        )
    )
    api_key: ApiKey | None = result.scalar_one_or_none()
    if api_key is None:
        return None

    user: User | None = api_key.user
    if user is None or not user.is_active:
        return None

    # Update last_used_at (fire-and-forget; don't fail request on error).
    try:
        api_key.last_used_at = now
        await session.commit()
    except Exception as exc:
        logger.warning("auth_api_key_last_used_update_failed", error=str(exc))
        await session.rollback()

    group_result = await session.execute(
        select(UserGroup.group_id).where(UserGroup.user_id == user.id)
    )
    group_ids = [str(gid) for gid in group_result.scalars().all()]

    accessible = await _fetch_accessible_models(user.role, group_ids, session)

    key_group_id = str(api_key.group_id) if api_key.group_id is not None else None

    return UserContext(
        user_id=str(user.id),
        username=user.username,
        role=user.role,
        group_ids=group_ids,
        accessible_model_ids=accessible,
        is_anonymous=False,
        key_group_id=key_group_id,
    )


async def _fetch_accessible_models(role: str, group_ids: list[str], session) -> set[str]:
    """Return the set of accessible model IDs.

    Admins receive an empty set; callers must check ``user.is_admin`` to grant
    unrestricted access — an empty set for non-admins means *no models accessible*.
    Regular users receive the union of model IDs across their groups.
    """
    if role == "system_admin":
        return set()

    if not group_ids:
        return set()

    from ocabra.db.auth import GroupModel

    parsed_ids = []
    for gid in group_ids:
        try:
            parsed_ids.append(_uuid.UUID(gid))
        except ValueError:
            pass

    if not parsed_ids:
        return set()

    result = await session.execute(
        select(GroupModel.model_id).where(GroupModel.group_id.in_(parsed_ids))
    )
    return set(result.scalars().all())


# ── Public dependencies ───────────────────────────────────────────────────────


async def get_current_user(request: Request) -> UserContext:
    """Resolve the calling user from the current request.

    Resolution order:
    1. ``X-Gateway-Token`` header (internal gateway service-to-service calls).
    2. Cookie ``ocabra_session`` (dashboard / browser).
    3. ``Authorization: Bearer`` header — tries JWT first, then API key.
    4. Anonymous context if auth is not required by server settings.

    Raises:
        HTTPException 401: If credentials are present but invalid, or if no
            credentials are provided when the server requires them.
    """
    from ocabra.config import settings
    from ocabra.database import AsyncSessionLocal

    # 0. Gateway service token (internal calls — no DB needed)
    gw_token = request.headers.get("X-Gateway-Token", "")
    if gw_token and settings.gateway_service_token and gw_token == settings.gateway_service_token:
        ctx = UserContext(
            user_id=None,
            username="__gateway__",
            role="model_manager",
            group_ids=[],
            accessible_model_ids=set(),
            is_anonymous=False,
        )
        request.state.auth_user = ctx
        return ctx

    async with AsyncSessionLocal() as session:
        # 1. JWT cookie
        cookie_token = request.cookies.get("ocabra_session")
        if cookie_token:
            ctx = await _resolve_jwt_cookie(cookie_token, session)
            if ctx is not None:
                request.state.auth_user = ctx
                return ctx
            # Cookie present but invalid → 401 (don't fall through to anonymous).
            raise HTTPException(status_code=401, detail="Session expired or invalid")

        # 2. Authorization header (Bearer API key)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            raw_key = auth_header[len("Bearer "):]
            ctx = await _resolve_api_key(raw_key, session)
            if ctx is not None:
                request.state.auth_user = ctx
                return ctx
            # Header present but key invalid → 401.
            raise HTTPException(status_code=401, detail="Invalid or expired API key")

        # 3. No credentials — check whether anonymous access is allowed.
        # Anonymous mode is allowed when BOTH require_api_key flags are false,
        # or when the request path is not under the protected API prefixes.
        path = request.url.path
        requires_key = (
            (settings.require_api_key_openai and path.startswith("/v1/"))
            or (settings.require_api_key_ollama and path.startswith("/api/"))
        )
        if requires_key:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        ctx = await _build_anonymous_context(session)
        request.state.auth_user = ctx
        return ctx


def require_role(min_role: str):
    """Dependency factory that enforces a minimum role.

    Usage::

        @router.get("/admin-only")
        async def admin_endpoint(user: UserContext = Depends(require_role("system_admin"))):
            ...

    Args:
        min_role: Minimum role required.  One of ``"user"``,
            ``"model_manager"``, or ``"system_admin"``.

    Returns:
        A FastAPI dependency that resolves to a :class:`UserContext`.

    Raises:
        HTTPException 401: If the caller is anonymous when a role is required.
        HTTPException 403: If the caller's role is below *min_role*.
    """

    async def _dependency(
        user: UserContext = Depends(get_current_user),
    ) -> UserContext:
        if user.is_anonymous:
            raise HTTPException(status_code=401, detail="Authentication required")
        caller_level = ROLE_HIERARCHY.get(user.role, -1)
        required_level = ROLE_HIERARCHY.get(min_role, 0)
        if caller_level < required_level:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{min_role}' or higher required",
            )
        return user

    return _dependency


async def get_user_accessible_models(user: UserContext, session) -> set[str]:
    """Return the set of accessible model IDs for *user*.

    Admins receive an empty set which conventionally means *all models*.
    The result is already cached on the :class:`UserContext` object; this
    function exists as a named entry-point for other modules that need to
    re-query or override the cached value.

    Args:
        user: The resolved :class:`UserContext` for the caller.
        session: An active :class:`sqlalchemy.ext.asyncio.AsyncSession`.

    Returns:
        Set of model ID strings, or empty set for admins (= all access).
    """
    if user.is_admin:
        return set()
    return await _fetch_accessible_models(user.role, user.group_ids, session)
