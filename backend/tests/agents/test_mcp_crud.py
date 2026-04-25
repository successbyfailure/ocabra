"""Role-matrix and basic CRUD behaviour tests for ``/ocabra/mcp-servers``.

The router is exercised against an in-process FastAPI app while
``ocabra.database.AsyncSessionLocal`` is replaced with an ``AsyncMock``.  The
test's job is to script the SQL results returned to the router in the order
the router calls ``session.execute`` — see each test's ``side_effect`` list.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.agents.mcp_registry import MCPRegistry, set_registry
from ocabra.api.internal.mcp_servers import router as mcp_router
from ocabra.db.mcp import MCPServer
from tests.agents.conftest import (
    FakeSessionFactory,
    make_user_context,
    override_user,
    scalar_result,
    scalars_all,
)


def _make_app(user_ctx) -> FastAPI:
    app = FastAPI()
    app.include_router(mcp_router, prefix="/ocabra")
    override_user(app, user_ctx)
    # Attach a real (but empty) registry to app.state so the router's
    # ``_get_registry`` helper does not raise 503.
    reg = MCPRegistry(session_factory=None, fernet_secret="x" * 32)
    app.state.mcp_registry = reg
    set_registry(reg)
    return app


def _make_mcp_row(**overrides) -> MCPServer:
    row = MCPServer(
        alias=overrides.get("alias", "gh"),
        display_name=overrides.get("display_name", "GitHub"),
        description=overrides.get("description", None),
        transport=overrides.get("transport", "http"),
        url=overrides.get("url", "https://example.test/mcp"),
        command=overrides.get("command", None),
        args=overrides.get("args", None),
        env_encrypted=overrides.get("env_encrypted", None),
        auth_type=overrides.get("auth_type", "none"),
        auth_value_encrypted=overrides.get("auth_value_encrypted", None),
        oauth_config=overrides.get("oauth_config", None),
        allowed_tools=overrides.get("allowed_tools", None),
        group_id=overrides.get("group_id", None),
        tools_cache=overrides.get("tools_cache", None),
        tools_cache_updated_at=overrides.get("tools_cache_updated_at", None),
        last_error=overrides.get("last_error", None),
        health_status=overrides.get("health_status", "unknown"),
        created_by=overrides.get("created_by", None),
    )
    # The ORM wouldn't set an id until after flush; we need one for tests.
    row.id = overrides.get("id", uuid.uuid4())
    return row


# ── Role matrix ──────────────────────────────────────────────


def test_list_requires_model_manager_role():
    app = _make_app(make_user_context(role="user"))
    factory = FakeSessionFactory()
    with patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get("/ocabra/mcp-servers")
    assert resp.status_code == 403


def test_list_ok_for_model_manager():
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [scalars_all([])]

    factory.configure(wire)
    with patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get("/ocabra/mcp-servers")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_http_ok_for_model_manager():
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()

    created_row = _make_mcp_row()

    def wire(session):
        session.execute.side_effect = [
            scalar_result(None),  # duplicate alias check → no existing
        ]

        async def _refresh(obj):
            obj.id = created_row.id
            obj.created_at = created_row.created_at
            obj.updated_at = created_row.updated_at

        session.refresh.side_effect = _refresh

    factory.configure(wire)

    # Avoid actually calling registry.register on a fake URL.
    with (
        patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory),
        patch.object(app.state.mcp_registry, "register", new=AsyncMock(return_value=None)),
    ):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/mcp-servers",
            json={
                "alias": "gh",
                "display_name": "GitHub",
                "transport": "http",
                "url": "https://example.test/mcp",
                "auth_type": "bearer",
                "auth_value": {"value": "sk-test"},
            },
        )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["alias"] == "gh"
    # The response must redact the secret.
    assert body["has_auth"] is True
    assert "auth_value" not in body


def test_create_stdio_requires_system_admin():
    """transport=stdio with a mere model_manager must 403."""
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()
    with patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/mcp-servers",
            json={
                "alias": "fs",
                "display_name": "Local FS",
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-filesystem"],
            },
        )
    assert resp.status_code == 403


def test_create_stdio_ok_for_admin():
    app = _make_app(make_user_context(role="system_admin"))
    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [
            scalar_result(None),  # duplicate alias check → no existing
        ]

        async def _refresh(obj):
            obj.id = uuid.uuid4()

        session.refresh.side_effect = _refresh

    factory.configure(wire)

    with (
        patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory),
        patch.object(app.state.mcp_registry, "register", new=AsyncMock(return_value=None)),
    ):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/mcp-servers",
            json={
                "alias": "fs",
                "display_name": "Local FS",
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-filesystem"],
            },
        )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["transport"] == "stdio"
    assert body["has_auth"] is False


def test_create_duplicate_alias_400():
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()

    def wire(session):
        session.execute.side_effect = [
            scalar_result(uuid.uuid4()),  # duplicate alias
        ]

    factory.configure(wire)
    with patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/mcp-servers",
            json={
                "alias": "gh",
                "display_name": "GitHub",
                "transport": "http",
                "url": "https://example.test",
            },
        )
    assert resp.status_code == 400


def test_delete_blocks_when_in_use_without_force():
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()
    row = _make_mcp_row()

    def wire(session):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 3  # 3 agents reference it
        session.execute.side_effect = [
            scalar_result(row),  # load server
            count_result,  # in-use check
        ]

    factory.configure(wire)
    with patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.delete(f"/ocabra/mcp-servers/{row.id}")
    assert resp.status_code == 409


def test_delete_force_ok():
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()
    row = _make_mcp_row()

    def wire(session):
        count_result = MagicMock()
        count_result.scalar_one.return_value = 2
        session.execute.side_effect = [
            scalar_result(row),  # load server
            count_result,  # in-use check
        ]

    factory.configure(wire)
    with (
        patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory),
        patch.object(app.state.mcp_registry, "unregister", new=AsyncMock(return_value=None)),
    ):
        client = TestClient(app)
        resp = client.delete(f"/ocabra/mcp-servers/{row.id}?force=true")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_delete_stdio_requires_admin():
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()
    row = _make_mcp_row(transport="stdio", command="uvx", url=None)

    def wire(session):
        session.execute.side_effect = [scalar_result(row)]

    factory.configure(wire)
    with patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.delete(f"/ocabra/mcp-servers/{row.id}?force=true")
    assert resp.status_code == 403


def test_refresh_persists_cache():
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()
    row = _make_mcp_row()

    def wire(session):
        session.execute.side_effect = [
            scalar_result(row),  # load server
            MagicMock(),  # update() executed by registry.refresh(...)
        ]

    factory.configure(wire)

    from ocabra.agents.mcp_client import MCPTool

    async def _fake_refresh(alias, *, session=None):
        # Simulate what registry.refresh would do (minus the update query).
        if session is not None:
            # The real registry would issue an UPDATE; we already scripted that
            # via the second side_effect above.
            await session.execute(MagicMock())
        return [MCPTool(name="t1", description="d", input_schema={})]

    with (
        patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory),
        patch.object(app.state.mcp_registry, "refresh", new=_fake_refresh),
    ):
        client = TestClient(app)
        resp = client.post(f"/ocabra/mcp-servers/{row.id}/refresh")
    assert resp.status_code == 200
    body = resp.json()
    assert body and body[0]["name"] == "t1"


def test_get_returns_redacted():
    app = _make_app(make_user_context(role="model_manager"))
    factory = FakeSessionFactory()
    row = _make_mcp_row(
        auth_value_encrypted="ciphertext",
        env_encrypted="cipherenv",
        auth_type="api_key",
    )

    def wire(session):
        session.execute.side_effect = [scalar_result(row)]

    factory.configure(wire)
    with patch("ocabra.api.internal.mcp_servers.AsyncSessionLocal", new=factory):
        client = TestClient(app)
        resp = client.get(f"/ocabra/mcp-servers/{row.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_auth"] is True
    assert body["has_env"] is True
    assert "auth_value" not in body
    assert "env" not in body
    # And there is no accidental ciphertext leak either.
    dumped = resp.text
    assert "ciphertext" not in dumped
    assert "cipherenv" not in dumped
