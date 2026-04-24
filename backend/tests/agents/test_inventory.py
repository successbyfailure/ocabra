"""Tests for the agent entries added to ``/v1/models`` and ``/ocabra/models``.

The listings helpers (``_list_agent_model_entries`` on the OpenAI path and
``_list_agent_entries`` on the internal path) are exercised directly against
a patched ``AsyncSessionLocal``.  Running the full routes would pull in the
whole lifespan, which is overkill for what we're checking here.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from ocabra.api.internal.models import _list_agent_entries
from ocabra.api.openai.models import _list_agent_model_entries
from ocabra.db.agents import Agent
from tests.agents.conftest import (
    FakeSessionFactory,
    make_user_context,
    scalars_all,
)


def _make_agent(**overrides) -> Agent:
    row = Agent(
        slug=overrides.get("slug", "bot"),
        display_name=overrides.get("display_name", "Bot"),
        description=overrides.get("description", None),
        base_model_id=overrides.get("base_model_id", "vllm/model"),
        profile_id=overrides.get("profile_id", None),
        system_prompt=overrides.get("system_prompt", ""),
        tool_choice_default=overrides.get("tool_choice_default", "auto"),
        max_tool_hops=overrides.get("max_tool_hops", 8),
        tool_timeout_seconds=overrides.get("tool_timeout_seconds", 60),
        require_approval=overrides.get("require_approval", "never"),
        request_defaults=overrides.get("request_defaults", None),
        group_id=overrides.get("group_id", None),
        created_by=overrides.get("created_by", None),
    )
    row.id = overrides.get("id", uuid.uuid4())
    row.mcp_links = overrides.get("mcp_links", [])
    return row


@pytest.mark.asyncio
async def test_openai_inventory_lists_agents_with_prefix():
    user = make_user_context(role="system_admin")
    bot = _make_agent(slug="research", display_name="Research Bot")

    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [scalars_all([bot])]

    factory.configure(wire)

    with patch("ocabra.database.AsyncSessionLocal", new=factory):
        entries = await _list_agent_model_entries(user, now_ts=1234567890)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "agent/research"
    assert entry["owned_by"] == "ocabra-agent"
    assert entry["ocabra"]["kind"] == "agent"


@pytest.mark.asyncio
async def test_openai_inventory_filters_by_group_for_non_admin():
    my_group = uuid.uuid4()
    foreign_group = uuid.uuid4()
    user = make_user_context(role="user", group_ids=[str(my_group)])

    public_bot = _make_agent(slug="public", group_id=None)
    my_bot = _make_agent(slug="mine", group_id=my_group)
    foreign_bot = _make_agent(slug="theirs", group_id=foreign_group)

    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [scalars_all([public_bot, my_bot, foreign_bot])]

    factory.configure(wire)

    with patch("ocabra.database.AsyncSessionLocal", new=factory):
        entries = await _list_agent_model_entries(user, now_ts=1234567890)

    ids = {e["id"] for e in entries}
    assert ids == {"agent/public", "agent/mine"}


@pytest.mark.asyncio
async def test_internal_inventory_returns_ocabra_agent_backend_type():
    user = make_user_context(role="system_admin")
    bot = _make_agent(slug="research")

    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [scalars_all([bot])]

    factory.configure(wire)

    with patch("ocabra.database.AsyncSessionLocal", new=factory):
        entries = await _list_agent_entries(user)

    assert len(entries) == 1
    e = entries[0]
    assert e["model_id"] == "agent/research"
    assert e["backend_type"] == "ocabra-agent"
    assert e["ocabra"]["kind"] == "agent"
