"""Tests for :func:`ocabra.agents.resolver.resolve_agent`."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from ocabra.agents.resolver import (
    extract_slug,
    is_agent_model,
    resolve_agent,
)
from ocabra.db.agents import Agent, AgentMCPServer
from ocabra.db.mcp import MCPServer
from tests.agents.conftest import make_user_context


def test_is_agent_model():
    assert is_agent_model("agent/foo")
    assert not is_agent_model("foo/bar")
    assert not is_agent_model("")
    assert not is_agent_model(None)


def test_extract_slug():
    assert extract_slug("agent/research-bot") == "research-bot"
    assert extract_slug("agent/") == ""


class _AsyncSession:
    """Minimal async-session test double exposing only ``execute``."""

    def __init__(self, side_effects):
        self._side_effects = list(side_effects)
        self.calls = []

    async def execute(self, stmt):
        self.calls.append(stmt)
        result = self._side_effects.pop(0)
        return result


def _agent_row(group_id=None, mcp_links=None) -> Agent:
    a = Agent(
        slug="bot",
        display_name="Bot",
        base_model_id="vllm/x",
        profile_id=None,
        system_prompt="hello",
        tool_choice_default="auto",
        max_tool_hops=3,
        tool_timeout_seconds=10,
        require_approval="never",
        group_id=group_id,
    )
    a.id = uuid.uuid4()
    a.mcp_links = mcp_links or []
    return a


def _server_row(alias: str, allowed=None, group_id=None) -> MCPServer:
    s = MCPServer(
        alias=alias,
        display_name=alias,
        transport="http",
        url="http://x",
        auth_type="none",
        allowed_tools=allowed,
        group_id=group_id,
    )
    s.id = uuid.uuid4()
    return s


def _scalar(value):
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def _scalars_all(values):
    r = MagicMock()
    r.scalars.return_value.all.return_value = values
    return r


@pytest.mark.asyncio
async def test_returns_none_for_non_agent_id():
    session = _AsyncSession([])
    spec = await resolve_agent("vllm/foo", session)
    assert spec is None


@pytest.mark.asyncio
async def test_returns_none_for_unknown_slug():
    session = _AsyncSession([_scalar(None)])
    spec = await resolve_agent("agent/nope", session)
    assert spec is None


@pytest.mark.asyncio
async def test_returns_spec_for_existing_slug():
    server = _server_row("fs", allowed=["read"])
    link = AgentMCPServer(
        agent_id=uuid.uuid4(), mcp_server_id=server.id, allowed_tools=["read", "write"]
    )
    link.mcp_server_id = server.id
    agent = _agent_row(mcp_links=[link])
    session = _AsyncSession([_scalar(agent), _scalars_all([server])])
    spec = await resolve_agent("agent/bot", session)
    assert spec is not None
    assert spec.slug == "bot"
    assert len(spec.bindings) == 1
    binding = spec.bindings[0]
    assert binding.server_alias == "fs"
    assert binding.allowed_tools == ["read", "write"]
    assert binding.server_allowed_tools == ["read"]


@pytest.mark.asyncio
async def test_filters_by_user_group():
    other_group = uuid.uuid4()
    agent = _agent_row(group_id=other_group)
    session = _AsyncSession([_scalar(agent)])
    user = make_user_context(role="user", group_ids=[str(uuid.uuid4())])
    spec = await resolve_agent("agent/bot", session, user=user)
    assert spec is None  # user not in group


@pytest.mark.asyncio
async def test_admin_can_see_any_group():
    other_group = uuid.uuid4()
    agent = _agent_row(group_id=other_group)
    session = _AsyncSession([_scalar(agent), _scalars_all([])])
    user = make_user_context(role="system_admin")
    spec = await resolve_agent("agent/bot", session, user=user)
    assert spec is not None
