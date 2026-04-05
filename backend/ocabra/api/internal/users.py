"""User management endpoints (system_admin only, except self-service password)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ocabra.api._deps_auth import UserContext, require_role

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["users"])


# ── Request / response schemas ────────────────────────────────────────────────


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str
    email: str | None = None


class PatchUserRequest(BaseModel):
    role: str | None = None
    is_active: bool | None = None
    email: str | None = None


class ResetPasswordRequest(BaseModel):
    new_password: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _user_dict(user) -> dict:
    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat(),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/users")
async def list_users(
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> list[dict]:
    """List all users.

    Requires: system_admin

    Returns:
        List of ``{ id, username, email, role, is_active, created_at }``
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).order_by(User.created_at))
        users = result.scalars().all()

    return [_user_dict(u) for u in users]


@router.post("/users", status_code=201)
async def create_user(
    body: CreateUserRequest,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> dict:
    """Create a new user.

    Requires: system_admin

    Args:
        body: ``{ username, password, role, email }``

    Returns:
        ``{ id, username, email, role, is_active, created_at }``

    Raises:
        HTTP 409: If username already exists.
    """
    from ocabra.core.auth_manager import hash_password
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            select(User).where(User.username == body.username)
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail="Username already exists")

        new_user = User(
            username=body.username,
            hashed_password=hash_password(body.password),
            role=body.role,
            email=body.email,
            is_active=True,
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

    logger.info("user_created", username=new_user.username, role=new_user.role, by=caller.username)
    return _user_dict(new_user)


@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> dict:
    """Get a user by ID.

    Requires: system_admin

    Args:
        user_id: UUID of the user to retrieve.

    Returns:
        ``{ id, username, email, role, is_active, created_at }``

    Raises:
        HTTP 404: If the user does not exist.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == parsed_id))
        user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return _user_dict(user)


@router.patch("/users/{user_id}")
async def patch_user(
    user_id: str,
    body: PatchUserRequest,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> dict:
    """Update a user's role, active status, or email.

    Requires: system_admin

    Args:
        user_id: UUID of the user to update.
        body: ``{ role?, is_active?, email? }``

    Returns:
        Updated ``{ id, username, email, role, is_active, created_at }``

    Raises:
        HTTP 400: If attempting to deactivate or demote own account.
        HTTP 404: If the user does not exist.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc

    is_self = caller.user_id is not None and str(parsed_id) == caller.user_id

    if is_self:
        if body.is_active is False:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
        if body.role is not None and body.role != caller.role:
            raise HTTPException(status_code=400, detail="Cannot change your own role")

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == parsed_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        if body.role is not None:
            user.role = body.role
        if body.is_active is not None:
            user.is_active = body.is_active
        if body.email is not None:
            user.email = body.email

        user.updated_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(user)

    logger.info("user_patched", target_user_id=user_id, by=caller.username)
    return _user_dict(user)


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: str,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> None:
    """Delete a user.

    Requires: system_admin

    Args:
        user_id: UUID of the user to delete.

    Raises:
        HTTP 400: If attempting to delete own account.
        HTTP 404: If the user does not exist.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc

    if caller.user_id is not None and str(parsed_id) == caller.user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == parsed_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        await session.delete(user)
        await session.commit()

    logger.info("user_deleted", target_user_id=user_id, by=caller.username)


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    body: ResetPasswordRequest,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> dict:
    """Reset a user's password.

    Requires: system_admin

    Args:
        user_id: UUID of the user whose password will be reset.
        body: ``{ new_password }``

    Returns:
        ``{ message: "password reset" }``

    Raises:
        HTTP 404: If the user does not exist.
    """
    from ocabra.core.auth_manager import hash_password
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import User
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == parsed_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        user.hashed_password = hash_password(body.new_password)
        user.updated_at = datetime.now(UTC)
        await session.commit()

    logger.info("user_password_reset", target_user_id=user_id, by=caller.username)
    return {"message": "password reset"}


@router.get("/users/{user_id}/keys")
async def list_user_api_keys(
    user_id: str,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> list[dict]:
    """List all API keys for a user (without the raw key value).

    Requires: system_admin

    Args:
        user_id: UUID of the target user.

    Returns:
        List of ``{ id, name, key_prefix, expires_at, last_used_at, is_revoked, created_at }``

    Raises:
        HTTP 404: If the user does not exist.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import ApiKey, User
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc

    async with AsyncSessionLocal() as session:
        user_check = await session.execute(select(User).where(User.id == parsed_id))
        if user_check.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="User not found")

        result = await session.execute(
            select(ApiKey)
            .where(ApiKey.user_id == parsed_id)
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


@router.delete("/users/{user_id}/keys/{key_id}", status_code=204)
async def revoke_user_api_key(
    user_id: str,
    key_id: str,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> None:
    """Revoke a specific API key belonging to a user.

    Requires: system_admin

    Args:
        user_id: UUID of the target user.
        key_id: UUID of the API key to revoke.

    Raises:
        HTTP 404: If the user or key does not exist or the key does not belong to the user.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import ApiKey, User
    from sqlalchemy import select

    try:
        parsed_user_id = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc

    try:
        parsed_key_id = uuid.UUID(key_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Key not found") from exc

    async with AsyncSessionLocal() as session:
        user_check = await session.execute(select(User).where(User.id == parsed_user_id))
        if user_check.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="User not found")

        result = await session.execute(
            select(ApiKey).where(
                ApiKey.id == parsed_key_id,
                ApiKey.user_id == parsed_user_id,
            )
        )
        api_key = result.scalar_one_or_none()
        if api_key is None:
            raise HTTPException(status_code=404, detail="Key not found")

        api_key.is_revoked = True
        await session.commit()

    logger.info("user_api_key_revoked", target_user_id=user_id, key_id=key_id, by=caller.username)


class CreateUserApiKeyRequest(BaseModel):
    name: str
    expires_in_days: int | None = None
    group_id: str | None = None


@router.post("/users/{user_id}/keys", status_code=201)
async def create_user_api_key(
    user_id: str,
    body: CreateUserApiKeyRequest,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> dict:
    """Create an API key for a specific user (admin only).

    Requires: system_admin

    Args:
        user_id: UUID of the target user.
        body: { name, expires_in_days?, group_id? }

    Returns:
        { id, name, key_prefix, key, expires_at, created_at, group_id }
        The raw key is shown only once.

    Raises:
        HTTP 404: If the user does not exist.
    """
    import secrets
    from ocabra.core.auth_manager import hash_api_key
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import ApiKey, User
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc

    async with AsyncSessionLocal() as session:
        user_check = await session.execute(select(User).where(User.id == parsed_id))
        if user_check.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="User not found")

        raw_key = f"sk-ocabra-{secrets.token_urlsafe(32)}"
        prefix = raw_key[:16]
        key_hash = hash_api_key(raw_key)

        expires_at = None
        if body.expires_in_days:
            from datetime import timedelta
            expires_at = datetime.now(UTC) + timedelta(days=body.expires_in_days)

        parsed_group_id = None
        if body.group_id:
            try:
                parsed_group_id = uuid.UUID(body.group_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid group_id") from exc

        api_key = ApiKey(
            user_id=parsed_id,
            name=body.name,
            key_hash=key_hash,
            key_prefix=prefix,
            expires_at=expires_at,
            group_id=parsed_group_id,
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    logger.info("user_api_key_created_by_admin", target_user_id=user_id, key_name=body.name, by=caller.username)
    return {
        "id": str(api_key.id),
        "name": api_key.name,
        "keyPrefix": api_key.key_prefix,
        "key": raw_key,
        "expiresAt": api_key.expires_at.isoformat() if api_key.expires_at else None,
        "createdAt": api_key.created_at.isoformat(),
        "groupId": str(api_key.group_id) if api_key.group_id else None,
    }
