"""SQLAlchemy ORM models for agents and the agent⇄mcp_server link table.

Plan: docs/tasks/agents-mcp-plan.md — section "Modelo de datos" / agents.

An agent has exactly one of ``base_model_id`` or ``profile_id`` (enforced by
DB ``CHECK`` constraint ``ck_agents_exactly_one_base``).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ocabra.database import Base

if TYPE_CHECKING:
    from ocabra.db.mcp import MCPServer


class Agent(Base):
    """An agent = base model/profile + system prompt + MCP tools + loop params."""

    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    slug: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    base_model_id: Mapped[str | None] = mapped_column(
        String(512),
        ForeignKey("model_configs.model_id", ondelete="CASCADE"),
        nullable=True,
    )
    profile_id: Mapped[str | None] = mapped_column(
        String(512),
        ForeignKey("model_profiles.profile_id", ondelete="CASCADE"),
        nullable=True,
    )

    system_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")

    tool_choice_default: Mapped[str] = mapped_column(
        String(16), nullable=False, default="auto"
    )
    max_tool_hops: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    tool_timeout_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, default=60
    )
    require_approval: Mapped[str] = mapped_column(
        String(16), nullable=False, default="never"
    )

    request_defaults: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    group_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    mcp_links: Mapped[list[AgentMCPServer]] = relationship(
        "AgentMCPServer",
        back_populates="agent",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "(base_model_id IS NOT NULL) <> (profile_id IS NOT NULL)",
            name="ck_agents_exactly_one_base",
        ),
    )


class AgentMCPServer(Base):
    """Many-to-many link between :class:`Agent` and :class:`MCPServer`.

    ``allowed_tools`` is an agent-scoped allowlist that is intersected with the
    server-level ``MCPServer.allowed_tools`` and with the per-request header
    ``x-ocabra-allowed-tools`` at invocation time.  ``None`` means
    "inherit the server-level allowlist".
    """

    __tablename__ = "agent_mcp_servers"

    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    mcp_server_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        primary_key=True,
    )
    allowed_tools: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    agent: Mapped[Agent] = relationship("Agent", back_populates="mcp_links")
    mcp_server: Mapped[MCPServer] = relationship(
        "MCPServer", back_populates="agent_links"
    )
