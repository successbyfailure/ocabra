"""Pydantic schemas for MCP server CRUD endpoints.

Plan: docs/tasks/agents-mcp-plan.md — "API REST nueva" / MCP servers.

Secrets handling:

- ``MCPServerCreate`` / ``MCPServerUpdate`` accept the plaintext ``auth_value``
  and ``env`` values.  They are encrypted with Fernet before being written to
  the DB by the router layer.
- ``MCPServerOut`` is always redacted — ``auth_value`` and ``env`` are never
  returned.  Presence of a secret is indicated via ``has_auth`` and ``has_env``
  booleans so UIs can show a "set"/"unset" badge.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── Shared aliases ───────────────────────────────────────────

Transport = Literal["http", "sse", "stdio"]
AuthType = Literal["none", "api_key", "bearer", "basic", "oauth2"]
HealthStatus = Literal["unknown", "healthy", "unhealthy"]


# ── MCP tool (read-only) ─────────────────────────────────────


class MCPToolOut(BaseModel):
    """A single tool exposed by an MCP server."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


# ── Auth payload ─────────────────────────────────────────────


class MCPAuthPayload(BaseModel):
    """Plaintext auth payload supplied on create/update.

    Persisted encrypted (Fernet).  Interpretation:

    - ``api_key`` / ``bearer`` / ``basic``: ``value`` is the credential, and
      ``header_name`` (default ``Authorization``) is the HTTP header to use.
    - ``none``: this payload must be ``None``.
    - ``oauth2``: reserved for a future revision (configured via ``oauth_config``).
    """

    model_config = ConfigDict(extra="forbid")

    header_name: str | None = Field(default=None, max_length=128)
    value: str = Field(..., min_length=1)

    @field_validator("header_name")
    @classmethod
    def _strip_header(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


# ── Create / Update / Out ────────────────────────────────────


class _MCPServerBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    display_name: str = Field(..., min_length=1, max_length=256)
    description: str | None = None
    transport: Transport
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    auth_type: AuthType = "none"
    auth_value: MCPAuthPayload | None = None
    oauth_config: dict[str, Any] | None = None
    allowed_tools: list[str] | None = None
    group_id: UUID | None = None

    @model_validator(mode="after")
    def _check_transport_fields(self) -> _MCPServerBase:
        if self.transport in ("http", "sse"):
            if not (self.url or "").strip():
                raise ValueError("transport=http|sse requires `url`")
        if self.transport == "stdio":
            if not (self.command or "").strip():
                raise ValueError("transport=stdio requires `command`")
        if self.auth_type == "none" and self.auth_value is not None:
            raise ValueError("auth_type='none' must not include auth_value")
        if self.auth_type in ("api_key", "bearer", "basic") and self.auth_value is None:
            raise ValueError(f"auth_type='{self.auth_type}' requires auth_value")
        return self


class MCPServerCreate(_MCPServerBase):
    pass


class MCPServerUpdate(BaseModel):
    """All fields optional; semantic validation runs server-side on merge."""

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = None
    transport: Transport | None = None
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    auth_type: AuthType | None = None
    # ``None`` means "unchanged".  To explicitly clear the auth_value, send an
    # update with ``auth_type='none'``.
    auth_value: MCPAuthPayload | None = None
    oauth_config: dict[str, Any] | None = None
    allowed_tools: list[str] | None = None
    group_id: UUID | None = None


class MCPServerOut(BaseModel):
    """Redacted view of an MCP server (never exposes secrets)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    alias: str
    display_name: str
    description: str | None = None
    transport: Transport
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    # Redacted flags — always False if no secret is stored.
    has_env: bool = False
    has_auth: bool = False
    auth_type: AuthType = "none"
    oauth_config: dict[str, Any] | None = None
    allowed_tools: list[str] | None = None
    group_id: UUID | None = None
    tools_cache: list[MCPToolOut] | None = None
    tools_cache_updated_at: datetime | None = None
    last_error: str | None = None
    health_status: HealthStatus = "unknown"
    created_by: UUID | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MCPServerTestResult(BaseModel):
    """Result of a one-shot connection test (``POST /mcp-servers/{id}/test``)."""

    healthy: bool
    tools_count: int = 0
    error: str | None = None
