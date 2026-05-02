"""Tests for the agent-scoped stats endpoints (Stream B / Fase 2)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api.internal.stats_agents import router as stats_router
from tests.agents.conftest import (
    FakeSessionFactory,
    make_user_context,
    override_user,
)


def _make_app(user_ctx) -> FastAPI:
    app = FastAPI()
    app.include_router(stats_router, prefix="/ocabra")
    override_user(app, user_ctx)
    return app


def test_by_agent_returns_empty_when_no_agents_visible():
    app = _make_app(make_user_context(role="user", group_ids=["group-a"]))
    factory = FakeSessionFactory()

    def wire(session):
        # First execute: agent listing → empty.
        result_agents = MagicMock()
        result_agents.all.return_value = []
        session.execute.side_effect = [result_agents]

    factory.configure(wire)
    with patch("ocabra.api.internal.stats_agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get("/ocabra/stats/by-agent?range=24h")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"by_agent": [], "by_tool": []}


def test_by_agent_filters_by_group_for_non_admin():
    user_group = "00000000-0000-0000-0000-000000000001"
    other_group = "00000000-0000-0000-0000-000000000002"
    app = _make_app(make_user_context(role="user", group_ids=[user_group]))
    factory = FakeSessionFactory()

    visible_id = uuid.uuid4()
    hidden_id = uuid.uuid4()

    def wire(session):
        result_agents = MagicMock()
        result_agents.all.return_value = [
            (visible_id, "visible", "Visible Agent", uuid.UUID(user_group)),
            (hidden_id, "hidden", "Hidden Agent", uuid.UUID(other_group)),
        ]
        # request_stats: one root (parent_request_id=NULL) + 1 hop (token sum)
        result_requests = MagicMock()
        result_requests.all.return_value = [
            (visible_id, 100, 10, 5, 200, None),  # root row
            (visible_id, 50, 7, 3, 200, uuid.uuid4()),  # hop row
        ]
        result_tc_count = MagicMock()
        result_tc_count.all.return_value = [(visible_id, 2)]
        result_tools = MagicMock()
        result_tools.all.return_value = [
            ("fs", "read", 30, "ok"),
            ("fs", "read", 60, "ok"),
        ]
        session.execute.side_effect = [
            result_agents,
            result_requests,
            result_tc_count,
            result_tools,
        ]

    factory.configure(wire)
    with patch("ocabra.api.internal.stats_agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get("/ocabra/stats/by-agent")
    assert resp.status_code == 200
    body = resp.json()
    # Only the visible agent appears
    assert len(body["by_agent"]) == 1
    row = body["by_agent"][0]
    assert row["slug"] == "visible"
    assert row["request_count"] == 1  # only the root row counts
    assert row["tool_call_count"] == 2
    # Tokens summed across root + hop = (10+5) + (7+3) = 25
    assert row["total_tokens"] == 25
    # Tools aggregated
    assert body["by_tool"][0]["mcp_server_alias"] == "fs"
    assert body["by_tool"][0]["invocations"] == 2


def test_tool_calls_filters_by_visible_agent():
    user_group = "00000000-0000-0000-0000-000000000001"
    other_group = "00000000-0000-0000-0000-000000000002"
    app = _make_app(make_user_context(role="user", group_ids=[user_group]))
    factory = FakeSessionFactory()

    visible_id = uuid.uuid4()
    hidden_id = uuid.uuid4()
    now = datetime.now(UTC)

    def wire(session):
        result_meta = MagicMock()
        result_meta.all.return_value = [
            (visible_id, "visible", uuid.UUID(user_group)),
            (hidden_id, "hidden", uuid.UUID(other_group)),
        ]
        result_calls = MagicMock()
        result_calls.all.return_value = [
            (
                uuid.uuid4(),
                now - timedelta(minutes=2),
                visible_id,
                "fs",
                "read",
                "ok",
                42,
                0,
                None,
                {"path": "/x"},
            ),
        ]
        session.execute.side_effect = [result_meta, result_calls]

    factory.configure(wire)
    with patch("ocabra.api.internal.stats_agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get("/ocabra/stats/tool-calls?limit=20")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["tool_calls"]) == 1
    assert body["tool_calls"][0]["agent_slug"] == "visible"
    assert body["tool_calls"][0]["status"] == "ok"


def test_tool_calls_unknown_agent_id_returns_empty():
    app = _make_app(make_user_context(role="system_admin"))
    factory = FakeSessionFactory()

    def wire(session):
        # Caller is admin → meta lists at least one agent
        result_meta = MagicMock()
        result_meta.all.return_value = []
        session.execute.side_effect = [result_meta]

    factory.configure(wire)
    with patch("ocabra.api.internal.stats_agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get(f"/ocabra/stats/tool-calls?agent_id={uuid.uuid4()}")
    assert resp.status_code == 200
    assert resp.json() == {"tool_calls": []}
