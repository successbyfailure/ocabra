"""Group management endpoints (system_admin only)."""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ocabra.api._deps_auth import UserContext, require_role

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["groups"])


# ── Request / response schemas ────────────────────────────────────────────────


class CreateGroupRequest(BaseModel):
    name: str
    description: str = ""


class PatchGroupRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class AddGroupModelRequest(BaseModel):
    model_id: str


class AddGroupMemberRequest(BaseModel):
    user_id: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _group_dict(group, member_count: int = 0, model_count: int = 0) -> dict:
    return {
        "id": str(group.id),
        "name": group.name,
        "description": group.description,
        "is_default": group.is_default,
        "member_count": member_count,
        "model_count": model_count,
        "created_at": group.created_at.isoformat(),
    }


def _group_dict_simple(group) -> dict:
    return {
        "id": str(group.id),
        "name": group.name,
        "description": group.description,
        "is_default": group.is_default,
        "created_at": group.created_at.isoformat(),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/groups")
async def list_groups(
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> list[dict]:
    """List all groups with member and model counts.

    Requires: system_admin

    Returns:
        List of ``{ id, name, description, is_default, member_count, model_count, created_at }``
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import Group, GroupModel, UserGroup
    from sqlalchemy import func, select

    async with AsyncSessionLocal() as session:
        groups_result = await session.execute(select(Group).order_by(Group.created_at))
        groups = groups_result.scalars().all()

        member_counts_result = await session.execute(
            select(UserGroup.group_id, func.count(UserGroup.user_id).label("cnt"))
            .group_by(UserGroup.group_id)
        )
        member_counts = {row.group_id: row.cnt for row in member_counts_result}

        model_counts_result = await session.execute(
            select(GroupModel.group_id, func.count(GroupModel.model_id).label("cnt"))
            .group_by(GroupModel.group_id)
        )
        model_counts = {row.group_id: row.cnt for row in model_counts_result}

    return [
        _group_dict(g, member_counts.get(g.id, 0), model_counts.get(g.id, 0))
        for g in groups
    ]


@router.post("/groups", status_code=201)
async def create_group(
    body: CreateGroupRequest,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> dict:
    """Create a new group.

    Requires: system_admin

    Args:
        body: ``{ name, description }``

    Returns:
        ``{ id, name, description, is_default, created_at }``

    Raises:
        HTTP 409: If a group with that name already exists.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import Group
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(Group).where(Group.name == body.name))
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail="Group name already exists")

        new_group = Group(name=body.name, description=body.description, is_default=False)
        session.add(new_group)
        await session.commit()
        await session.refresh(new_group)

    logger.info("group_created", group_name=new_group.name, by=caller.username)
    return _group_dict_simple(new_group)


@router.patch("/groups/{group_id}")
async def patch_group(
    group_id: str,
    body: PatchGroupRequest,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> dict:
    """Update a group's name or description.

    Requires: system_admin

    Args:
        group_id: UUID of the group to update.
        body: ``{ name?, description? }``

    Returns:
        Updated ``{ id, name, description, is_default, created_at }``

    Raises:
        HTTP 400: If attempting to rename the default group.
        HTTP 404: If the group does not exist.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import Group
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(group_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Group not found") from exc

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Group).where(Group.id == parsed_id))
        group = result.scalar_one_or_none()
        if group is None:
            raise HTTPException(status_code=404, detail="Group not found")

        if group.is_default and body.name is not None and body.name != group.name:
            raise HTTPException(status_code=400, detail="Cannot rename the default group")

        if body.name is not None:
            group.name = body.name
        if body.description is not None:
            group.description = body.description

        await session.commit()
        await session.refresh(group)

    logger.info("group_patched", group_id=group_id, by=caller.username)
    return _group_dict_simple(group)


@router.delete("/groups/{group_id}", status_code=204)
async def delete_group(
    group_id: str,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> None:
    """Delete a group.

    Requires: system_admin

    Args:
        group_id: UUID of the group to delete.

    Raises:
        HTTP 400: If the group is the default group.
        HTTP 404: If the group does not exist.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import Group
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(group_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Group not found") from exc

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Group).where(Group.id == parsed_id))
        group = result.scalar_one_or_none()
        if group is None:
            raise HTTPException(status_code=404, detail="Group not found")

        if group.is_default:
            raise HTTPException(status_code=400, detail="Cannot delete the default group")

        await session.delete(group)
        await session.commit()

    logger.info("group_deleted", group_id=group_id, by=caller.username)


# ── Group models ──────────────────────────────────────────────────────────────


@router.get("/groups/{group_id}/models")
async def list_group_models(
    group_id: str,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> list[dict]:
    """List all model IDs assigned to a group.

    Requires: system_admin

    Args:
        group_id: UUID of the target group.

    Returns:
        List of ``{ model_id }``

    Raises:
        HTTP 404: If the group does not exist.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import Group, GroupModel
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(group_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Group not found") from exc

    async with AsyncSessionLocal() as session:
        group_check = await session.execute(select(Group).where(Group.id == parsed_id))
        if group_check.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Group not found")

        result = await session.execute(
            select(GroupModel).where(GroupModel.group_id == parsed_id)
        )
        entries = result.scalars().all()

    return [{"model_id": e.model_id} for e in entries]


@router.post("/groups/{group_id}/models", status_code=201)
async def add_group_model(
    group_id: str,
    body: AddGroupModelRequest,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> dict:
    """Add a model to a group.

    Requires: system_admin

    Args:
        group_id: UUID of the target group.
        body: ``{ model_id }``

    Returns:
        ``{ group_id, model_id }``

    Raises:
        HTTP 404: If the group does not exist.
        HTTP 409: If the model is already assigned to the group.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import Group, GroupModel
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(group_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Group not found") from exc

    async with AsyncSessionLocal() as session:
        group_check = await session.execute(select(Group).where(Group.id == parsed_id))
        if group_check.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Group not found")

        existing = await session.execute(
            select(GroupModel).where(
                GroupModel.group_id == parsed_id,
                GroupModel.model_id == body.model_id,
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail="Model already assigned to this group")

        entry = GroupModel(group_id=parsed_id, model_id=body.model_id)
        session.add(entry)
        await session.commit()

    logger.info("group_model_added", group_id=group_id, model_id=body.model_id, by=caller.username)
    return {"group_id": group_id, "model_id": body.model_id}


@router.delete("/groups/{group_id}/models/{model_id:path}", status_code=204)
async def remove_group_model(
    group_id: str,
    model_id: str,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> None:
    """Remove a model from a group.

    Requires: system_admin

    Args:
        group_id: UUID of the target group.
        model_id: Model ID to remove (may contain slashes).

    Raises:
        HTTP 404: If the group or the model assignment does not exist.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import Group, GroupModel
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(group_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Group not found") from exc

    async with AsyncSessionLocal() as session:
        group_check = await session.execute(select(Group).where(Group.id == parsed_id))
        if group_check.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Group not found")

        result = await session.execute(
            select(GroupModel).where(
                GroupModel.group_id == parsed_id,
                GroupModel.model_id == model_id,
            )
        )
        entry = result.scalar_one_or_none()
        if entry is None:
            raise HTTPException(status_code=404, detail="Model not assigned to this group")

        await session.delete(entry)
        await session.commit()

    logger.info("group_model_removed", group_id=group_id, model_id=model_id, by=caller.username)


# ── Group members ─────────────────────────────────────────────────────────────


@router.get("/groups/{group_id}/members")
async def list_group_members(
    group_id: str,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> list[dict]:
    """List all members of a group.

    Requires: system_admin

    Args:
        group_id: UUID of the target group.

    Returns:
        List of ``{ user_id, username, role }``

    Raises:
        HTTP 404: If the group does not exist.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import Group, User, UserGroup
    from sqlalchemy import select

    try:
        parsed_id = uuid.UUID(group_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Group not found") from exc

    async with AsyncSessionLocal() as session:
        group_check = await session.execute(select(Group).where(Group.id == parsed_id))
        if group_check.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Group not found")

        result = await session.execute(
            select(User)
            .join(UserGroup, UserGroup.user_id == User.id)
            .where(UserGroup.group_id == parsed_id)
            .order_by(User.username)
        )
        members = result.scalars().all()

    return [{"user_id": str(m.id), "username": m.username, "role": m.role} for m in members]


@router.post("/groups/{group_id}/members", status_code=201)
async def add_group_member(
    group_id: str,
    body: AddGroupMemberRequest,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> dict:
    """Add a user to a group.

    Requires: system_admin

    Args:
        group_id: UUID of the target group.
        body: ``{ user_id }``

    Returns:
        ``{ group_id, user_id }``

    Raises:
        HTTP 400: If the group is the default group (membership is implicit).
        HTTP 404: If the group or user does not exist.
        HTTP 409: If the user is already a member.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import Group, User, UserGroup
    from sqlalchemy import select

    try:
        parsed_group_id = uuid.UUID(group_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Group not found") from exc

    try:
        parsed_user_id = uuid.UUID(body.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc

    async with AsyncSessionLocal() as session:
        group_result = await session.execute(select(Group).where(Group.id == parsed_group_id))
        group = group_result.scalar_one_or_none()
        if group is None:
            raise HTTPException(status_code=404, detail="Group not found")

        if group.is_default:
            raise HTTPException(
                status_code=400,
                detail="Default group membership is implicit; cannot manage members explicitly",
            )

        user_check = await session.execute(select(User).where(User.id == parsed_user_id))
        if user_check.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="User not found")

        existing = await session.execute(
            select(UserGroup).where(
                UserGroup.group_id == parsed_group_id,
                UserGroup.user_id == parsed_user_id,
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail="User is already a member of this group")

        entry = UserGroup(group_id=parsed_group_id, user_id=parsed_user_id)
        session.add(entry)
        await session.commit()

    logger.info("group_member_added", group_id=group_id, user_id=body.user_id, by=caller.username)
    return {"group_id": group_id, "user_id": body.user_id}


@router.delete("/groups/{group_id}/members/{user_id}", status_code=204)
async def remove_group_member(
    group_id: str,
    user_id: str,
    caller: Annotated[UserContext, Depends(require_role("system_admin"))],
) -> None:
    """Remove a user from a group.

    Requires: system_admin

    Args:
        group_id: UUID of the target group.
        user_id: UUID of the user to remove.

    Raises:
        HTTP 400: If the group is the default group.
        HTTP 404: If the group or the membership does not exist.
    """
    from ocabra.database import AsyncSessionLocal
    from ocabra.db.auth import Group, UserGroup
    from sqlalchemy import select

    try:
        parsed_group_id = uuid.UUID(group_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Group not found") from exc

    try:
        parsed_user_id = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc

    async with AsyncSessionLocal() as session:
        group_result = await session.execute(select(Group).where(Group.id == parsed_group_id))
        group = group_result.scalar_one_or_none()
        if group is None:
            raise HTTPException(status_code=404, detail="Group not found")

        if group.is_default:
            raise HTTPException(
                status_code=400,
                detail="Default group membership is implicit; cannot manage members explicitly",
            )

        result = await session.execute(
            select(UserGroup).where(
                UserGroup.group_id == parsed_group_id,
                UserGroup.user_id == parsed_user_id,
            )
        )
        entry = result.scalar_one_or_none()
        if entry is None:
            raise HTTPException(status_code=404, detail="User is not a member of this group")

        await session.delete(entry)
        await session.commit()

    logger.info("group_member_removed", group_id=group_id, user_id=user_id, by=caller.username)
