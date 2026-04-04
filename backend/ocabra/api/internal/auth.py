"""Auth endpoints: login, logout, me, refresh, password change, own API keys."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from ocabra.api._deps_auth import UserContext, get_current_user, require_role

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["auth"])


# ── Request / response schemas ────────────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str
    password: str
    remember: bool = False


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class CreateApiKeyRequest(BaseModel):
    name: str
    expires_in_days: int | None = None


# ── Helper ────────────────────────────────────────────────────────────────────


def _set_session_cookie(response: Response, token: str, remember: bool) -> None:
    """Attach the ``ocabra_session`` cookie to *response*."""
    from ocabra.config import settings

    if remember:
        max_age = settings.jwt_remember_days * 86400
    else:
        max_age = settings.jwt_ttl_hours * 3600

    response.set_cookie(
        key="ocabra_session",
        value=token,
        httponly=True,
        samesite="lax",
        secure=settings.use_https,
        max_age=max_age,
        path="/",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/auth/login")
async def login(body: LoginRequest, response: Response) -> dict:
    """Authenticate a user and issue a session cookie.

    Args:
        body: ``{ username, password, remember }``

    Returns:
        ``{ user: { id, username, email, role, created_at } }``

    Raises:
        HTTP 401: If credentials are invalid or the account is inactive.
    """
    from ocabra.core.auth_manager import create_access_token, verify_password
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.username == body.username)
        )
        user: User | None = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(
        user_id=str(user.id),
        role=user.role,
        remember=body.remember,
    )
    _set_session_cookie(response, token, remember=body.remember)

    logger.info("auth_login_success", username=user.username, role=user.role)

    return {
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "created_at": user.created_at.isoformat(),
        }
    }


@router.post("/auth/logout")
async def logout(
    response: Response,
    request: Request,
) -> dict:
    """Invalidate the current session.

    Deletes the ``ocabra_session`` cookie.  If a valid JWT is present its
    ``jti`` is added to the Redis revocation list with a TTL equal to the
    token's remaining lifetime.

    Returns:
        ``{ "ok": true }``
    """
    from ocabra.core.auth_manager import AuthError, decode_access_token

    cookie_token = request.cookies.get("ocabra_session")
    if cookie_token:
        try:
            payload = decode_access_token(cookie_token)
            jti = payload.get("jti")
            exp = payload.get("exp")
            if jti and exp:
                ttl = int(exp - datetime.now(UTC).timestamp())
                if ttl > 0:
                    try:
                        from ocabra.redis_client import get_redis

                        redis = await get_redis()
                        await redis.setex(f"jwt_revoked:{jti}", ttl, "1")
                    except Exception as exc:
                        logger.warning("auth_logout_redis_revoke_failed", error=str(exc))
        except AuthError:
            pass  # Expired/invalid token — nothing to revoke.

    response.delete_cookie("ocabra_session", path="/")
    return {"ok": True}


@router.get("/auth/me")
async def me(
    user: Annotated[UserContext, Depends(require_role("user"))],
) -> dict:
    """Return the currently authenticated user's profile.

    Returns:
        ``{ id, username, email, role, created_at }``

    Raises:
        HTTP 401: If the caller is not authenticated.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user.user_id))
        db_user: User | None = result.scalar_one_or_none()

    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": str(db_user.id),
        "username": db_user.username,
        "email": db_user.email,
        "role": db_user.role,
        "created_at": db_user.created_at.isoformat(),
    }


@router.put("/auth/password")
async def change_password(
    body: PasswordChangeRequest,
    user: Annotated[UserContext, Depends(require_role("user"))],
) -> dict:
    """Change the authenticated user's own password.

    Args:
        body: ``{ current_password, new_password }``

    Returns:
        ``{ "ok": true }``

    Raises:
        HTTP 400: If *current_password* is incorrect.
        HTTP 401: If the caller is not authenticated.
    """
    from ocabra.core.auth_manager import hash_password, verify_password
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user.user_id))
        db_user: User | None = result.scalar_one_or_none()
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")

        if not verify_password(body.current_password, db_user.hashed_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        db_user.hashed_password = hash_password(body.new_password)
        db_user.updated_at = datetime.now(UTC)
        await session.commit()

    logger.info("auth_password_changed", username=user.username)
    return {"ok": True}


@router.get("/auth/keys")
async def list_api_keys(
    user: Annotated[UserContext, Depends(require_role("user"))],
) -> list[dict]:
    """List the authenticated user's own API keys.

    Returns:
        List of ``{ id, name, key_prefix, expires_at, last_used_at, is_revoked, created_at }``

    Raises:
        HTTP 401: If the caller is not authenticated.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import ApiKey
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ApiKey)
            .where(ApiKey.user_id == user.user_id)
            .order_by(ApiKey.created_at.desc())
        )
        keys = result.scalars().all()

    return [
        {
            "id": str(k.id),
            "name": k.name,
            "key_prefix": k.key_prefix,
            "expires_at": k.expires_at.isoformat() if k.expires_at else None,
            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            "is_revoked": k.is_revoked,
            "created_at": k.created_at.isoformat(),
        }
        for k in keys
    ]


@router.post("/auth/keys", status_code=201)
async def create_api_key(
    body: CreateApiKeyRequest,
    user: Annotated[UserContext, Depends(require_role("user"))],
) -> dict:
    """Create a new API key for the authenticated user.

    The ``key`` field in the response is the only time the raw value is
    returned.  It is not stored and cannot be recovered.

    Args:
        body: ``{ name, expires_in_days }`` — ``expires_in_days=null`` means no expiry.

    Returns:
        ``{ id, name, key_prefix, expires_at, key }``

    Raises:
        HTTP 401: If the caller is not authenticated.
    """
    from ocabra.core.auth_manager import generate_api_key
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import ApiKey

    now = datetime.now(UTC)
    expires_at = None
    if body.expires_in_days is not None:
        expires_at = now + timedelta(days=body.expires_in_days)

    raw_key, key_hash, prefix = generate_api_key()

    async with AsyncSessionLocal() as session:
        api_key = ApiKey(
            user_id=user.user_id,
            name=body.name,
            key_hash=key_hash,
            key_prefix=prefix,
            expires_at=expires_at,
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    logger.info("auth_api_key_created", username=user.username, key_prefix=prefix)

    return {
        "id": str(api_key.id),
        "name": api_key.name,
        "key_prefix": api_key.key_prefix,
        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
        "key": raw_key,
    }


@router.delete("/auth/keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: str,
    user: Annotated[UserContext, Depends(require_role("user"))],
) -> None:
    """Revoke one of the authenticated user's own API keys.

    Args:
        key_id: UUID of the API key to revoke.

    Raises:
        HTTP 401: If the caller is not authenticated.
        HTTP 404: If the key does not exist or does not belong to the caller.
    """
    import uuid

    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import ApiKey
    from sqlalchemy import select

    try:
        parsed_key_id = uuid.UUID(key_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Key not found") from exc

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ApiKey).where(
                ApiKey.id == parsed_key_id,
                ApiKey.user_id == user.user_id,
            )
        )
        api_key: ApiKey | None = result.scalar_one_or_none()
        if api_key is None:
            raise HTTPException(status_code=404, detail="Key not found")

        api_key.is_revoked = True
        await session.commit()

    logger.info("auth_api_key_revoked", username=user.username, key_id=key_id)
