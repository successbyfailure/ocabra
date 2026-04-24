"""Pydantic schemas for agent CRUD endpoints.

Plan: docs/tasks/agents-mcp-plan.md — "API REST nueva" / Agents.

Exactly one of ``base_model_id`` / ``profile_id`` must be set on create.
``AgentMCPServerLink.allowed_tools`` is an agent-scoped tool allowlist that is
intersected with the server-level one at invocation time (see plan, section
"Seguridad — checklist obligatoria").
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

ToolChoiceDefault = Literal["auto", "required", "none"]
RequireApproval = Literal["never", "always"]


class AgentMCPServerLink(BaseModel):
    """Link entry describing which MCP server an agent can use and with which tools."""

    model_config = ConfigDict(extra="forbid")

    mcp_server_id: UUID
    # None = inherit server-level allowlist; [] = disable everything on this server.
    allowed_tools: list[str] | None = None


class AgentMCPServerLinkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    mcp_server_id: UUID
    allowed_tools: list[str] | None = None


class _AgentBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    display_name: str = Field(..., min_length=1, max_length=256)
    description: str | None = None
    base_model_id: str | None = None
    profile_id: str | None = None
    system_prompt: str = ""
    tool_choice_default: ToolChoiceDefault = "auto"
    max_tool_hops: int = Field(default=8, ge=0, le=64)
    tool_timeout_seconds: int = Field(default=60, ge=1, le=3600)
    require_approval: RequireApproval = "never"
    request_defaults: dict[str, Any] | None = None
    group_id: UUID | None = None
    mcp_servers: list[AgentMCPServerLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_exactly_one_base(self) -> _AgentBase:
        has_base = bool(self.base_model_id and self.base_model_id.strip())
        has_profile = bool(self.profile_id and self.profile_id.strip())
        if has_base == has_profile:
            raise ValueError("exactly one of `base_model_id` or `profile_id` is required")
        return self


class AgentCreate(_AgentBase):
    pass


class AgentUpdate(BaseModel):
    """All fields optional.  ``mcp_servers`` replaces the full link set when provided."""

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = None
    base_model_id: str | None = None
    profile_id: str | None = None
    system_prompt: str | None = None
    tool_choice_default: ToolChoiceDefault | None = None
    max_tool_hops: int | None = Field(default=None, ge=0, le=64)
    tool_timeout_seconds: int | None = Field(default=None, ge=1, le=3600)
    require_approval: RequireApproval | None = None
    request_defaults: dict[str, Any] | None = None
    group_id: UUID | None = None
    mcp_servers: list[AgentMCPServerLink] | None = None


class AgentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    slug: str
    display_name: str
    description: str | None = None
    base_model_id: str | None = None
    profile_id: str | None = None
    system_prompt: str
    tool_choice_default: ToolChoiceDefault
    max_tool_hops: int
    tool_timeout_seconds: int
    require_approval: RequireApproval
    request_defaults: dict[str, Any] | None = None
    group_id: UUID | None = None
    mcp_servers: list[AgentMCPServerLinkOut] = Field(default_factory=list)
    created_by: UUID | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AgentTestResult(BaseModel):
    """Dry-run result of ``POST /agents/{slug}/test``.

    ``ok`` is True when every check passes (base model exists & loadable, every
    MCP server healthy and exposes the allow-listed tools).
    """

    ok: bool
    base_model_ok: bool
    base_model_error: str | None = None
    servers: list[dict[str, Any]] = Field(default_factory=list)
