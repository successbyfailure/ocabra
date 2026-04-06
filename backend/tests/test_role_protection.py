"""Tests for role-based endpoint protection.

Verifies that:
- Unauthenticated requests to /ocabra/* internal endpoints return 401.
- role="user" can access user-level endpoints but not admin/model_manager ones.
- role="model_manager" can access model operations but not config/admin endpoints.
- role="system_admin" can access all endpoints including config and user management.

These tests use a minimal FastAPI app per test group with mocked auth dependencies
so no real database or Redis is required.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user, require_role

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_user_context(role: str, user_id: str | None = None) -> UserContext:
    return UserContext(
        user_id=user_id or str(uuid.uuid4()),
        username=f"test-{role}",
        role=role,
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )


def _make_anon_context() -> UserContext:
    return UserContext(
        user_id=None,
        username=None,
        role="user",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=True,
    )


def _app_with_role_endpoint(min_role: str, path: str = "/protected") -> FastAPI:
    """Build a minimal app with a single endpoint protected by require_role."""
    app = FastAPI()

    @app.get(path)
    async def _endpoint(user: UserContext = Depends(require_role(min_role))):
        return {"role": user.role}

    return app


def _override_auth(app: FastAPI, user_ctx: UserContext) -> None:
    """Override auth dependency to return a fixed UserContext."""
    async def _fake_user():
        return user_ctx
    app.dependency_overrides[get_current_user] = _fake_user


# ── Unauthenticated requests ──────────────────────────────────────────────────


def test_unauthenticated_request_to_internal_returns_401():
    """Anonymous callers hitting a require_role-protected endpoint get 401."""
    app = _app_with_role_endpoint("user")
    anon_ctx = _make_anon_context()
    _override_auth(app, anon_ctx)

    client = TestClient(app)
    resp = client.get("/protected")

    assert resp.status_code == 401


# ── User role ─────────────────────────────────────────────────────────────────


def test_user_role_can_load_model():
    """A user-role caller can access a min_role='user' endpoint."""
    app = _app_with_role_endpoint("user", "/load-model")
    user_ctx = _make_user_context("user")
    _override_auth(app, user_ctx)

    client = TestClient(app)
    resp = client.get("/load-model")

    assert resp.status_code == 200
    assert resp.json()["role"] == "user"


def test_user_role_cannot_delete_model():
    """A user-role caller cannot access a min_role='model_manager' endpoint (403)."""
    app = _app_with_role_endpoint("model_manager", "/delete-model")
    user_ctx = _make_user_context("user")
    _override_auth(app, user_ctx)

    client = TestClient(app)
    resp = client.get("/delete-model")

    assert resp.status_code == 403


# ── Model manager role ────────────────────────────────────────────────────────


def test_model_manager_can_delete_model():
    """A model_manager-role caller can access a min_role='model_manager' endpoint."""
    app = _app_with_role_endpoint("model_manager", "/delete-model")
    mm_ctx = _make_user_context("model_manager")
    _override_auth(app, mm_ctx)

    client = TestClient(app)
    resp = client.get("/delete-model")

    assert resp.status_code == 200
    assert resp.json()["role"] == "model_manager"


def test_model_manager_cannot_access_config():
    """A model_manager-role caller cannot access a min_role='system_admin' endpoint (403)."""
    app = _app_with_role_endpoint("system_admin", "/config")
    mm_ctx = _make_user_context("model_manager")
    _override_auth(app, mm_ctx)

    client = TestClient(app)
    resp = client.get("/config")

    assert resp.status_code == 403


# ── System admin role ─────────────────────────────────────────────────────────


def test_system_admin_can_access_config():
    """A system_admin-role caller can access a min_role='system_admin' endpoint."""
    app = _app_with_role_endpoint("system_admin", "/config")
    admin_ctx = _make_user_context("system_admin")
    _override_auth(app, admin_ctx)

    client = TestClient(app)
    resp = client.get("/config")

    assert resp.status_code == 200
    assert resp.json()["role"] == "system_admin"


def test_system_admin_can_manage_users():
    """A system_admin-role caller can access user management endpoints."""
    from ocabra.api.internal.users import router as users_router

    app = FastAPI()
    app.include_router(users_router, prefix="/ocabra")

    admin_ctx = _make_user_context("system_admin")
    _override_auth(app, admin_ctx)

    users_list_result = MagicMock()
    users_list_result.scalars.return_value.all.return_value = []

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute.return_value = users_list_result

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.get("/ocabra/users")

    assert resp.status_code == 200


def test_user_cannot_access_user_management():
    """A user-role caller cannot access the /ocabra/users endpoint (403)."""
    from ocabra.api.internal.users import router as users_router

    app = FastAPI()
    app.include_router(users_router, prefix="/ocabra")

    user_ctx = _make_user_context("user")
    _override_auth(app, user_ctx)

    client = TestClient(app)
    resp = client.get("/ocabra/users")

    assert resp.status_code == 403


# ── Role hierarchy completeness ───────────────────────────────────────────────


def test_role_hierarchy_ordering():
    """ROLE_HIERARCHY values must be strictly ordered: user < model_manager < system_admin."""
    from ocabra.api._deps_auth import ROLE_HIERARCHY

    assert ROLE_HIERARCHY["user"] < ROLE_HIERARCHY["model_manager"]
    assert ROLE_HIERARCHY["model_manager"] < ROLE_HIERARCHY["system_admin"]


def test_system_admin_satisfies_all_minimum_roles():
    """system_admin passes require_role checks for all defined role levels."""
    from ocabra.api._deps_auth import ROLE_HIERARCHY

    admin_level = ROLE_HIERARCHY["system_admin"]
    for role, level in ROLE_HIERARCHY.items():
        assert admin_level >= level, f"system_admin should pass require_role('{role}')"


# ── require_role factory ──────────────────────────────────────────────────────


def test_require_role_unknown_min_role_treated_as_zero():
    """An unknown min_role defaults to required_level=0, so any authenticated user passes."""
    app = _app_with_role_endpoint("nonexistent_role", "/weird")
    user_ctx = _make_user_context("user")
    _override_auth(app, user_ctx)

    client = TestClient(app)
    resp = client.get("/weird")

    # level for "user" is 0, required level for "nonexistent_role" is 0 → passes
    assert resp.status_code == 200


def test_require_role_rejects_anonymous_even_at_user_level():
    """require_role('user') must reject anonymous callers with 401, not 403."""
    app = _app_with_role_endpoint("user", "/user-only")
    anon_ctx = _make_anon_context()
    _override_auth(app, anon_ctx)

    client = TestClient(app)
    resp = client.get("/user-only")

    assert resp.status_code == 401
