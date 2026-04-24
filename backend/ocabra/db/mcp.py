"""SQLAlchemy ORM models for MCP servers.

Plan: docs/tasks/agents-mcp-plan.md — section "Modelo de datos" / mcp_servers.

Secret columns (``env_encrypted``, ``auth_value_encrypted``) store Fernet
ciphertext (text, base64).  Plaintext secrets must never be persisted.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ocabra.database import Base

if TYPE_CHECKING:
    from ocabra.db.agents import AgentMCPServer


class MCPServer(Base):
    """Registered MCP server (http, sse, or stdio transport)."""

    __tablename__ = "mcp_servers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    alias: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    transport: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    # HTTP / SSE
    url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # stdio
    command: Mapped[str | None] = mapped_column(Text, nullable=True)
    args: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    env_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Auth
    auth_type: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    auth_value_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    oauth_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ACL
    allowed_tools: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    group_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("groups.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Runtime cache
    tools_cache: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    tools_cache_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    health_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unknown"
    )

    # Audit
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

    agent_links: Mapped[list[AgentMCPServer]] = relationship(
        "AgentMCPServer",
        back_populates="mcp_server",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
