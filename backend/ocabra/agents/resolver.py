"""Resolve ``model="agent/<slug>"`` requests to a concrete :class:`AgentSpec`.

Plan: docs/tasks/agents-mcp-plan.md — Fase 2 / "resolver.py".

The resolver loads the agent row, its MCP server bindings, and the per-binding
``allowed_tools`` overrides in a single eager-loaded query so the executor does
not need any further DB roundtrips during the tool-loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ocabra.api._deps_auth import UserContext
from ocabra.db.agents import Agent, AgentMCPServer
from ocabra.db.mcp import MCPServer

AGENT_PREFIX = "agent/"


@dataclass
class AgentMCPBindingSpec:
    """Resolved view of one (agent, mcp_server) link.

    ``server_alias``/``server_id`` come from the joined :class:`MCPServer`.
    ``allowed_tools`` is the agent-scoped override (``None`` = inherit
    server-level).  ``server_allowed_tools`` is the server-scoped allowlist.
    The intersection happens in the executor.
    """

    server_id: UUID
    server_alias: str
    server_group_id: UUID | None
    allowed_tools: list[str] | None
    server_allowed_tools: list[str] | None


@dataclass
class AgentSpec:
    """Eagerly-loaded snapshot of an agent ready for the executor."""

    id: UUID
    slug: str
    display_name: str
    description: str | None
    base_model_id: str | None
    profile_id: str | None
    system_prompt: str
    tool_choice_default: str
    max_tool_hops: int
    tool_timeout_seconds: int
    require_approval: str
    request_defaults: dict[str, Any] | None
    group_id: UUID | None
    subagent_slugs: list[str] = field(default_factory=list)
    subagent_names: dict[str, str] = field(default_factory=dict)
    subagent_descriptions: dict[str, str | None] = field(default_factory=dict)
    bindings: list[AgentMCPBindingSpec] = field(default_factory=list)


def is_agent_model(model_id: str | None) -> bool:
    return bool(model_id) and str(model_id).startswith(AGENT_PREFIX)


def extract_slug(model_id: str) -> str:
    return model_id[len(AGENT_PREFIX) :].strip()


def _is_accessible(agent_group_id: UUID | None, user: UserContext) -> bool:
    """Mirror the access rule from ``api/internal/agents.py``.

    NULL group_id ⇒ public; otherwise the user must be admin or belong to the
    agent's group.
    """
    if user.is_admin:
        return True
    if agent_group_id is None:
        return True
    return str(agent_group_id) in (user.group_ids or [])


async def resolve_agent(
    model_id: str,
    db_session: AsyncSession,
    *,
    user: UserContext | None = None,
) -> AgentSpec | None:
    """Return an :class:`AgentSpec` for ``model="agent/<slug>"``, else ``None``.

    Returns ``None`` (rather than raising) when:

    * ``model_id`` does not have the ``agent/`` prefix.
    * The slug does not exist.
    * The caller does not have access to the agent (the executor caller maps
      this to a 404 to avoid leaking existence).
    """
    if not is_agent_model(model_id):
        return None
    slug = extract_slug(model_id)
    if not slug:
        return None

    row = (
        await db_session.execute(
            sa.select(Agent).where(Agent.slug == slug).options(selectinload(Agent.mcp_links))
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    if user is not None and not _is_accessible(row.group_id, user):
        return None

    server_ids = [link.mcp_server_id for link in (row.mcp_links or [])]
    server_lookup: dict[UUID, MCPServer] = {}
    if server_ids:
        servers = (
            (await db_session.execute(sa.select(MCPServer).where(MCPServer.id.in_(server_ids))))
            .scalars()
            .all()
        )
        server_lookup = {s.id: s for s in servers}

    bindings: list[AgentMCPBindingSpec] = []
    for link in row.mcp_links or []:
        server = server_lookup.get(link.mcp_server_id)
        if server is None:
            continue
        bindings.append(
            AgentMCPBindingSpec(
                server_id=server.id,
                server_alias=server.alias,
                server_group_id=server.group_id,
                allowed_tools=list(link.allowed_tools) if link.allowed_tools is not None else None,
                server_allowed_tools=list(server.allowed_tools)
                if server.allowed_tools is not None
                else None,
            )
        )

    spec = AgentSpec(
        id=row.id,
        slug=row.slug,
        display_name=row.display_name,
        description=row.description,
        base_model_id=row.base_model_id,
        profile_id=row.profile_id,
        system_prompt=row.system_prompt or "",
        tool_choice_default=row.tool_choice_default,
        max_tool_hops=row.max_tool_hops,
        tool_timeout_seconds=row.tool_timeout_seconds,
        require_approval=row.require_approval,
        request_defaults=row.request_defaults,
        group_id=row.group_id,
        bindings=bindings,
    )

    if row.subagent_slugs:
        wanted_slugs = [child_slug for child_slug in row.subagent_slugs if child_slug != row.slug]
        if wanted_slugs:
            child_rows = (
                (
                    await db_session.execute(
                        sa.select(Agent.slug, Agent.display_name, Agent.description, Agent.group_id)
                        .where(Agent.slug.in_(wanted_slugs))
                    )
                )
                .all()
            )
            visible: dict[str, tuple[str, str | None]] = {}
            for child_slug, display_name, description, group_id in child_rows:
                if user is not None and not _is_accessible(group_id, user):
                    continue
                visible[str(child_slug)] = (str(display_name), description)
            spec.subagent_slugs = [child_slug for child_slug in wanted_slugs if child_slug in visible]
            spec.subagent_names = {
                child_slug: visible[child_slug][0] for child_slug in spec.subagent_slugs
            }
            spec.subagent_descriptions = {
                child_slug: visible[child_slug][1] for child_slug in spec.subagent_slugs
            }

    return spec
# without pulling in :class:`AgentMCPServer`.
__all__ = [
    "AGENT_PREFIX",
    "AgentMCPBindingSpec",
    "AgentSpec",
    "extract_slug",
    "is_agent_model",
    "resolve_agent",
    "AgentMCPServer",
]
