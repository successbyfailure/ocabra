"""Role-matrix and filtering tests for ``/ocabra/agents``."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.agents.mcp_registry import MCPRegistry, set_registry
from ocabra.api.internal.agents import router as agents_router
from ocabra.db.agents import Agent
from tests.agents.conftest import (
    FakeSessionFactory,
    make_user_context,
    override_user,
    scalar_result,
    scalars_all,
)


def _make_app(user_ctx) -> FastAPI:
    app = FastAPI()
    app.include_router(agents_router, prefix="/ocabra")
    override_user(app, user_ctx)
    reg = MCPRegistry(session_factory=None, fernet_secret="x" * 32)
    app.state.mcp_registry = reg
    set_registry(reg)
    return app


def _make_agent(**overrides) -> Agent:
    row = Agent(
        slug=overrides.get("slug", "bot"),
        display_name=overrides.get("display_name", "Bot"),
        description=overrides.get("description", None),
        base_model_id=overrides.get("base_model_id", "vllm/some-model"),
        profile_id=overrides.get("profile_id", None),
        system_prompt=overrides.get("system_prompt", "You are helpful."),
        tool_choice_default=overrides.get("tool_choice_default", "auto"),
        max_tool_hops=overrides.get("max_tool_hops", 8),
        tool_timeout_seconds=overrides.get("tool_timeout_seconds", 60),
        require_approval=overrides.get("require_approval", "never"),
        request_defaults=overrides.get("request_defaults", None),
        group_id=overrides.get("group_id", None),
        created_by=overrides.get("created_by", None),
    )
    row.id = overrides.get("id", uuid.uuid4())
    # selectinload() would populate this; our in-memory Agent needs it explicitly.
    row.mcp_links = overrides.get("mcp_links", [])
    return row


# ── Role matrix ──────────────────────────────────────────────


def test_list_accessible_to_user():
    """GET /ocabra/agents is available to any authenticated user."""
    app = _make_app(make_user_context(role="user"))
    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [scalars_all([])]

    factory.configure(wire)
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get("/ocabra/agents")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_forbidden_for_user():
    app = _make_app(make_user_context(role="user"))
    factory = FakeSessionFactory()
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/agents",
            json={
                "slug": "bot",
                "display_name": "Bot",
                "base_model_id": "vllm/some-model",
            },
        )
    assert resp.status_code == 403


def test_create_ok_for_model_manager():
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()

    created = _make_agent()

    load_after_flush_call_count = {"n": 0}

    def wire(session):
        def _on_execute(*_args, **_kwargs):
            # 1: duplicate slug check → None
            # 2: final re-load for response (selectinload)
            call_idx = load_after_flush_call_count["n"]
            load_after_flush_call_count["n"] += 1
            if call_idx == 0:
                return scalar_result(None)
            return scalar_result(created)

        session.execute.side_effect = _on_execute

    factory.configure(wire)
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/agents",
            json={
                "slug": "bot",
                "display_name": "Bot",
                "base_model_id": "vllm/some-model",
                "system_prompt": "hi",
            },
        )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["slug"] == "bot"
    assert body["display_name"] == "Bot"


def test_create_requires_exactly_one_base():
    """Pydantic must reject an agent with both base_model_id and profile_id."""
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/agents",
            json={
                "slug": "bot",
                "display_name": "Bot",
                "base_model_id": "vllm/m1",
                "profile_id": "p1",
            },
        )
    assert resp.status_code == 422


def test_create_rejects_missing_base():
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/agents",
            json={"slug": "bot", "display_name": "Bot"},
        )
    assert resp.status_code == 422


def test_delete_requires_model_manager():
    app = _make_app(make_user_context(role="user"))
    factory = FakeSessionFactory()
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.delete("/ocabra/agents/bot")
    assert resp.status_code == 403


def test_delete_ok_for_model_manager():
    app = _make_app(make_user_context(role="model_manager"))
    agent = _make_agent()
    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [scalar_result(agent)]

    factory.configure(wire)
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.delete(f"/ocabra/agents/{agent.slug}")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "slug": agent.slug}


# ── Group filtering ─────────────────────────────────────────


def test_list_filters_by_group_for_user():
    """Non-admin users must not see agents from groups they don't belong to."""
    my_group = str(uuid.uuid4())
    other_group = str(uuid.uuid4())
    app = _make_app(make_user_context(role="user", group_ids=[my_group]))

    public_agent = _make_agent(slug="public", group_id=None)
    my_agent = _make_agent(slug="mine", group_id=uuid.UUID(my_group))
    other_agent = _make_agent(slug="theirs", group_id=uuid.UUID(other_group))

    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [
            scalars_all([public_agent, my_agent, other_agent])
        ]

    factory.configure(wire)
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get("/ocabra/agents")
    assert resp.status_code == 200
    slugs = {row["slug"] for row in resp.json()}
    assert slugs == {"public", "mine"}


def test_list_shows_all_to_admin():
    my_group = str(uuid.uuid4())
    other_group = str(uuid.uuid4())
    app = _make_app(make_user_context(role="system_admin"))

    public_agent = _make_agent(slug="public", group_id=None)
    my_agent = _make_agent(slug="mine", group_id=uuid.UUID(my_group))
    other_agent = _make_agent(slug="theirs", group_id=uuid.UUID(other_group))

    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [
            scalars_all([public_agent, my_agent, other_agent])
        ]

    factory.configure(wire)
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get("/ocabra/agents")
    assert resp.status_code == 200
    slugs = {row["slug"] for row in resp.json()}
    assert slugs == {"public", "mine", "theirs"}


def test_get_returns_403_for_foreign_group():
    other_group = str(uuid.uuid4())
    app = _make_app(make_user_context(role="user", group_ids=[]))
    foreign = _make_agent(slug="theirs", group_id=uuid.UUID(other_group))

    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [scalar_result(foreign)]

    factory.configure(wire)
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get("/ocabra/agents/theirs")
    assert resp.status_code == 403


def test_test_endpoint_requires_model_manager():
    app = _make_app(make_user_context(role="user"))
    factory = FakeSessionFactory()
    with patch("ocabra.api.internal.agents.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.post("/ocabra/agents/bot/test")
    assert resp.status_code == 403
