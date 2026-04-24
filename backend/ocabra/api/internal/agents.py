"""Internal REST endpoints for Agent CRUD + dry-run.

Plan: docs/tasks/agents-mcp-plan.md — "API REST nueva" / Agents.

Role matrix:

* ``GET /ocabra/agents``       → ``user`` (filtered by group membership)
* ``GET /ocabra/agents/{slug}``→ ``user`` (subject to group membership)
* ``POST``/``PATCH``/``DELETE`` → ``model_manager``
* ``POST /ocabra/agents/{slug}/test`` → ``model_manager``
"""

from __future__ import annotations

import uuid as _uuid
from typing import Any
from uuid import UUID

import sqlalchemy as sa
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import selectinload

from ocabra.agents.mcp_registry import MCPRegistry, get_registry
from ocabra.api._deps_auth import UserContext, require_role
from ocabra.database import AsyncSessionLocal
from ocabra.db.agents import Agent, AgentMCPServer
from ocabra.db.mcp import MCPServer
from ocabra.schemas.agents import (
    AgentCreate,
    AgentMCPServerLinkOut,
    AgentOut,
    AgentTestResult,
    AgentUpdate,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["agents"])


# ── Helpers ──────────────────────────────────────────────────


def _get_registry(request: Request) -> MCPRegistry | None:
    return getattr(request.app.state, "mcp_registry", None) or get_registry()


def _serialize(row: Agent) -> AgentOut:
    links = [
        AgentMCPServerLinkOut(
            mcp_server_id=link.mcp_server_id,
            allowed_tools=list(link.allowed_tools) if link.allowed_tools else None,
        )
        for link in (row.mcp_links or [])
    ]
    return AgentOut(
        id=row.id,
        slug=row.slug,
        display_name=row.display_name,
        description=row.description,
        base_model_id=row.base_model_id,
        profile_id=row.profile_id,
        system_prompt=row.system_prompt,
        tool_choice_default=row.tool_choice_default,
        max_tool_hops=row.max_tool_hops,
        tool_timeout_seconds=row.tool_timeout_seconds,
        require_approval=row.require_approval,
        request_defaults=row.request_defaults,
        group_id=row.group_id,
        mcp_servers=links,
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _is_agent_accessible(agent: Agent, user: UserContext) -> bool:
    """True when the caller may see/use this agent (admin or group match)."""
    if user.is_admin:
        return True
    if agent.group_id is None:
        # A group_id=NULL agent is considered public (available to any caller).
        return True
    agent_group = str(agent.group_id)
    return agent_group in (user.group_ids or [])


async def _load_agent_or_404(session, slug: str) -> Agent:
    row = (
        await session.execute(
            sa.select(Agent)
            .where(Agent.slug == slug)
            .options(selectinload(Agent.mcp_links))
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Agent '{slug}' not found")
    return row


async def _validate_mcp_links(
    session, links: list, existing_agent_id: UUID | None = None
) -> list[tuple[UUID, list[str] | None]]:
    """Resolve MCP server references, raise 400 if any are missing.

    Returns a normalised list of ``(mcp_server_id, allowed_tools)`` tuples.
    Deduplicates on ``mcp_server_id``.
    """
    del existing_agent_id  # reserved for future validation; silence ruff
    if not links:
        return []
    ids = {link.mcp_server_id for link in links}
    found = {
        row_id
        for (row_id,) in (
            await session.execute(sa.select(MCPServer.id).where(MCPServer.id.in_(ids)))
        ).all()
    }
    missing = ids - found
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown MCP server ids: {sorted(str(i) for i in missing)}",
        )
    seen: set[UUID] = set()
    resolved: list[tuple[UUID, list[str] | None]] = []
    for link in links:
        if link.mcp_server_id in seen:
            continue
        seen.add(link.mcp_server_id)
        resolved.append(
            (
                link.mcp_server_id,
                list(link.allowed_tools) if link.allowed_tools is not None else None,
            )
        )
    return resolved


# ── CRUD ─────────────────────────────────────────────────────


@router.get(
    "/agents",
    summary="List agents accessible to the caller",
    description=(
        "Returns every agent visible to the caller.  For non-admin users the "
        "listing is filtered to agents whose ``group_id`` matches one of the "
        "caller's groups (agents with ``group_id=NULL`` are considered public)."
    ),
    response_model=list[AgentOut],
)
async def list_agents(
    user: UserContext = Depends(require_role("user")),
) -> list[AgentOut]:
    async with AsyncSessionLocal() as session:
        rows = (
            (
                await session.execute(
                    sa.select(Agent)
                    .options(selectinload(Agent.mcp_links))
                    .order_by(Agent.slug)
                )
            )
            .scalars()
            .all()
        )
    return [_serialize(row) for row in rows if _is_agent_accessible(row, user)]


@router.post(
    "/agents",
    summary="Create an agent",
    status_code=status.HTTP_201_CREATED,
    response_model=AgentOut,
    responses={
        400: {"description": "Validation error or duplicate slug / unknown MCP server id"},
    },
)
async def create_agent(
    body: AgentCreate,
    user: UserContext = Depends(require_role("model_manager")),
) -> AgentOut:
    """Create a new agent.  ``slug`` must be unique across all agents."""
    async with AsyncSessionLocal() as session:
        existing = (
            await session.execute(sa.select(Agent.id).where(Agent.slug == body.slug))
        ).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Slug '{body.slug}' is already taken",
            )
        resolved_links = await _validate_mcp_links(session, body.mcp_servers)

        row = Agent(
            slug=body.slug,
            display_name=body.display_name,
            description=body.description,
            base_model_id=body.base_model_id,
            profile_id=body.profile_id,
            system_prompt=body.system_prompt,
            tool_choice_default=body.tool_choice_default,
            max_tool_hops=body.max_tool_hops,
            tool_timeout_seconds=body.tool_timeout_seconds,
            require_approval=body.require_approval,
            request_defaults=body.request_defaults,
            group_id=body.group_id,
            created_by=_uuid.UUID(user.user_id) if user.user_id else None,
        )
        session.add(row)
        await session.flush()  # ensure row.id is populated
        for mcp_id, allowed in resolved_links:
            session.add(
                AgentMCPServer(
                    agent_id=row.id,
                    mcp_server_id=mcp_id,
                    allowed_tools=allowed,
                )
            )
        await session.commit()
        # Re-fetch with the link collection populated.
        row = await _load_agent_or_404(session, body.slug)
    return _serialize(row)


@router.get(
    "/agents/{slug}",
    summary="Retrieve an agent",
    response_model=AgentOut,
    responses={
        403: {"description": "Caller does not have access to this agent"},
        404: {"description": "Agent not found"},
    },
)
async def get_agent(
    slug: str,
    user: UserContext = Depends(require_role("user")),
) -> AgentOut:
    async with AsyncSessionLocal() as session:
        row = await _load_agent_or_404(session, slug)
    if not _is_agent_accessible(row, user):
        raise HTTPException(status_code=403, detail="Forbidden")
    return _serialize(row)


@router.patch(
    "/agents/{slug}",
    summary="Update an agent",
    response_model=AgentOut,
    responses={
        400: {"description": "Validation error"},
        404: {"description": "Agent not found"},
    },
)
async def update_agent(
    slug: str,
    body: AgentUpdate,
    _user: UserContext = Depends(require_role("model_manager")),
) -> AgentOut:
    patch = body.model_dump(exclude_unset=True)
    async with AsyncSessionLocal() as session:
        row = await _load_agent_or_404(session, slug)
        for field_name in (
            "display_name",
            "description",
            "system_prompt",
            "tool_choice_default",
            "max_tool_hops",
            "tool_timeout_seconds",
            "require_approval",
            "request_defaults",
            "group_id",
        ):
            if field_name in patch:
                setattr(row, field_name, patch[field_name])

        # base_model_id / profile_id: must still satisfy the XOR constraint.
        new_base = patch.get("base_model_id", row.base_model_id)
        new_profile = patch.get("profile_id", row.profile_id)
        has_base = bool((new_base or "").strip()) if new_base is not None else bool(row.base_model_id)
        has_profile = (
            bool((new_profile or "").strip()) if new_profile is not None else bool(row.profile_id)
        )
        # If caller explicitly set one side, clear the other.
        if "base_model_id" in patch and patch["base_model_id"]:
            row.base_model_id = patch["base_model_id"]
            row.profile_id = None
            has_profile = False
            has_base = True
        if "profile_id" in patch and patch["profile_id"]:
            row.profile_id = patch["profile_id"]
            row.base_model_id = None
            has_base = False
            has_profile = True
        if has_base == has_profile:
            raise HTTPException(
                status_code=400,
                detail="Agent must have exactly one of base_model_id or profile_id",
            )

        if "mcp_servers" in patch and patch["mcp_servers"] is not None:
            # Replace the full link set.
            resolved = await _validate_mcp_links(session, body.mcp_servers or [])
            await session.execute(
                sa.delete(AgentMCPServer).where(AgentMCPServer.agent_id == row.id)
            )
            for mcp_id, allowed in resolved:
                session.add(
                    AgentMCPServer(
                        agent_id=row.id,
                        mcp_server_id=mcp_id,
                        allowed_tools=allowed,
                    )
                )

        await session.commit()
        row = await _load_agent_or_404(session, slug)
    return _serialize(row)


@router.delete(
    "/agents/{slug}",
    summary="Delete an agent",
    responses={404: {"description": "Agent not found"}},
)
async def delete_agent(
    slug: str,
    _user: UserContext = Depends(require_role("model_manager")),
) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        row = await _load_agent_or_404(session, slug)
        await session.delete(row)
        await session.commit()
    return {"ok": True, "slug": slug}


@router.post(
    "/agents/{slug}/test",
    summary="Dry-run an agent",
    response_model=AgentTestResult,
    responses={404: {"description": "Agent not found"}},
)
async def test_agent(
    slug: str,
    request: Request,
    _user: UserContext = Depends(require_role("model_manager")),
) -> AgentTestResult:
    """Validate that the base model is loadable and every MCP server is healthy."""
    registry = _get_registry(request)

    async with AsyncSessionLocal() as session:
        agent = await _load_agent_or_404(session, slug)
        link_rows = (
            (
                await session.execute(
                    sa.select(AgentMCPServer, MCPServer)
                    .join(MCPServer, MCPServer.id == AgentMCPServer.mcp_server_id)
                    .where(AgentMCPServer.agent_id == agent.id)
                )
            )
            .all()
        )

    # Base model existence is best-effort: we look it up against the model manager
    # if available.  For profile-based agents we check the profile registry.
    base_model_ok = True
    base_model_error: str | None = None
    model_manager = getattr(request.app.state, "model_manager", None)
    profile_registry = getattr(request.app.state, "profile_registry", None)
    if agent.base_model_id and model_manager is not None:
        try:
            base_state = await model_manager.get_state(agent.base_model_id)
            if base_state is None:
                base_model_ok = False
                base_model_error = f"base_model '{agent.base_model_id}' not registered"
        except Exception as exc:  # noqa: BLE001
            base_model_ok = False
            base_model_error = str(exc)
    elif agent.profile_id and profile_registry is not None:
        try:
            profile = await profile_registry.get(agent.profile_id)
            if profile is None:
                base_model_ok = False
                base_model_error = f"profile '{agent.profile_id}' not registered"
        except Exception as exc:  # noqa: BLE001
            base_model_ok = False
            base_model_error = str(exc)

    servers_report: list[dict[str, Any]] = []
    for link, server in link_rows:
        entry: dict[str, Any] = {
            "mcp_server_id": str(server.id),
            "alias": server.alias,
            "allowed_tools": link.allowed_tools,
            "healthy": False,
            "error": None,
            "tools_count": 0,
        }
        if registry is None:
            entry["error"] = "mcp_registry unavailable"
            servers_report.append(entry)
            continue
        try:
            tools = await registry.refresh(server.alias)
            entry["healthy"] = True
            entry["tools_count"] = len(tools)
        except KeyError:
            entry["error"] = f"alias '{server.alias}' not loaded"
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
        servers_report.append(entry)

    all_healthy = all(r["healthy"] for r in servers_report)
    return AgentTestResult(
        ok=bool(base_model_ok and (not servers_report or all_healthy)),
        base_model_ok=base_model_ok,
        base_model_error=base_model_error,
        servers=servers_report,
    )
