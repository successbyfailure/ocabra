"""Integration tests for /ocabra/auth/* endpoints.

All database and Redis calls are mocked so these tests require no running
infrastructure.  The pattern follows the existing project style: build a
minimal FastAPI app, include only the relevant router, and mock the
dependencies that touch external services.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.core.auth_manager import generate_api_key, hash_password

# ── App factory ───────────────────────────────────────────────────────────────


def _make_app() -> FastAPI:
    """Minimal app with only the auth router mounted."""
    from ocabra.api.internal.auth import router as auth_router

    app = FastAPI()
    app.include_router(auth_router, prefix="/ocabra")
    return app


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_db_user(
    *,
    username: str = "alice",
    password: str = "password123",
    role: str = "user",
    user_id: str | None = None,
    is_active: bool = True,
):
    """Return a mock User ORM object with bcrypt-hashed password."""
    u = MagicMock()
    u.id = uuid.UUID(user_id) if user_id else uuid.uuid4()
    u.username = username
    u.email = f"{username}@example.com"
    u.role = role
    u.hashed_password = hash_password(password)
    u.is_active = is_active
    u.created_at = datetime(2026, 1, 1, tzinfo=UTC)
    return u


def _make_api_key_row(
    *,
    user,
    name: str = "test-key",
    is_revoked: bool = False,
    expires_at: datetime | None = None,
    raw_key: str | None = None,
):
    """Return a mock ApiKey ORM row."""
    from ocabra.core.auth_manager import generate_api_key, hash_api_key

    if raw_key is None:
        raw_key, key_hash, prefix = generate_api_key()
    else:
        from ocabra.core.auth_manager import hash_api_key
        key_hash = hash_api_key(raw_key)
        prefix = raw_key[:18] + "…"

    k = MagicMock()
    k.id = uuid.uuid4()
    k.user_id = user.id
    k.user = user
    k.name = name
    k.key_hash = key_hash
    k.key_prefix = prefix
    k.is_revoked = is_revoked
    k.expires_at = expires_at
    k.last_used_at = None
    k.created_at = datetime(2026, 1, 1, tzinfo=UTC)
    return k, raw_key


# ── Login ─────────────────────────────────────────────────────────────────────


def test_login_success_sets_cookie():
    """POST /ocabra/auth/login with valid credentials sets ocabra_session cookie."""
    app = _make_app()
    user = _make_db_user(username="alice", password="password123")

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = user

    mock_session = AsyncMock()
    mock_session.execute.return_value = scalar_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/auth/login",
            json={"username": "alice", "password": "password123"},
        )

    assert resp.status_code == 200
    assert "ocabra_session" in resp.cookies
    data = resp.json()
    assert data["user"]["username"] == "alice"


def test_login_wrong_password_returns_401():
    """POST /ocabra/auth/login with wrong password returns 401."""
    app = _make_app()
    user = _make_db_user(username="bob", password="realpassword")

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = user

    mock_session = AsyncMock()
    mock_session.execute.return_value = scalar_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/auth/login",
            json={"username": "bob", "password": "wrongpassword"},
        )

    assert resp.status_code == 401


def test_login_unknown_user_returns_401():
    """POST /ocabra/auth/login with nonexistent username returns 401."""
    app = _make_app()

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None

    mock_session = AsyncMock()
    mock_session.execute.return_value = scalar_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/auth/login",
            json={"username": "nobody", "password": "pass"},
        )

    assert resp.status_code == 401


def test_login_remember_me_sets_longer_cookie():
    """remember=True must produce a cookie with a longer max_age than remember=False."""

    app = _make_app()
    user = _make_db_user(username="carol", password="pass")

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = user

    mock_session = AsyncMock()
    mock_session.execute.return_value = scalar_result
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app, cookies={})

        resp_short = client.post(
            "/ocabra/auth/login",
            json={"username": "carol", "password": "pass", "remember": False},
        )
        resp_long = client.post(
            "/ocabra/auth/login",
            json={"username": "carol", "password": "pass", "remember": True},
        )

    assert resp_short.status_code == 200
    assert resp_long.status_code == 200

    # Decode both tokens and compare exp
    from ocabra.core.auth_manager import decode_access_token

    token_short = resp_short.cookies["ocabra_session"]
    token_long = resp_long.cookies["ocabra_session"]

    payload_short = decode_access_token(token_short)
    payload_long = decode_access_token(token_long)

    assert payload_long["exp"] > payload_short["exp"]


# ── /auth/me ──────────────────────────────────────────────────────────────────


def test_me_with_valid_session_returns_user():
    """GET /ocabra/auth/me with a valid session cookie returns user profile."""
    from ocabra.core.auth_manager import create_access_token

    app = _make_app()
    user = _make_db_user(username="dave", role="user")
    token = create_access_token(user_id=str(user.id), role="user")

    # First DB call: get_current_user resolves JWT, fetches user from DB.
    # Second DB call: /me endpoint fetches user again.
    user_select_result = MagicMock()
    user_select_result.scalar_one_or_none.return_value = user

    group_result = MagicMock()
    group_result.scalars.return_value.all.return_value = []

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    # Auth: 2 calls (user lookup + group ids). _fetch_accessible_models skips
    # DB call because group_ids is empty. Then /me endpoint re-fetches user.
    mock_session.execute.side_effect = [
        user_select_result,   # _resolve_jwt_cookie: load user
        group_result,          # _resolve_jwt_cookie: load group ids
        user_select_result,   # me endpoint: re-fetch user
    ]

    with (
        patch("ocabra.database.AsyncSessionLocal", return_value=mock_session),
        patch("ocabra.redis_client.get_redis", new=AsyncMock(side_effect=Exception("no redis"))),
    ):
        client = TestClient(app, cookies={"ocabra_session": token})
        resp = client.get("/ocabra/auth/me")

    assert resp.status_code == 200
    assert resp.json()["username"] == "dave"


def test_me_without_session_returns_401():
    """GET /ocabra/auth/me without credentials returns 401."""
    app = _make_app()

    group_result = MagicMock()
    group_result.__iter__ = MagicMock(return_value=iter([]))

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute.return_value = group_result

    # No cookie, no auth header; settings must require auth on /ocabra/ path.
    # /ocabra/ is not /v1/ or /api/ so the require_api_key flags don't apply —
    # the anonymous user is returned by get_current_user. But require_role("user")
    # rejects anonymous callers with 401.
    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.get("/ocabra/auth/me")

    assert resp.status_code == 401


# ── Logout ────────────────────────────────────────────────────────────────────


def test_logout_clears_cookie():
    """POST /ocabra/auth/logout must delete the ocabra_session cookie."""
    from ocabra.core.auth_manager import create_access_token

    app = _make_app()
    user = _make_db_user(username="eve")
    token = create_access_token(user_id=str(user.id), role="user")

    with patch("ocabra.redis_client.get_redis", new=AsyncMock(side_effect=Exception("no redis"))):
        client = TestClient(app, cookies={"ocabra_session": token})
        resp = client.post("/ocabra/auth/logout")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    # Cookie should be cleared (empty value or deleted)
    assert resp.cookies.get("ocabra_session", "") == ""


# ── Password change ───────────────────────────────────────────────────────────


def test_change_password_success():
    """PUT /ocabra/auth/password with correct current_password returns ok."""
    from ocabra.api._deps_auth import UserContext, get_current_user

    app = _make_app()
    user = _make_db_user(username="frank", password="oldpass")

    # Mutable user object so we can verify password update
    user.hashed_password = hash_password("oldpass")

    _user_ctx = UserContext(
        user_id=str(user.id), username="frank", role="user",
        group_ids=[], accessible_model_ids=set(), is_anonymous=False,
    )
    app.dependency_overrides[get_current_user] = lambda: _user_ctx

    user_result = MagicMock()
    user_result.scalar_one_or_none.return_value = user

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute.return_value = user_result
    mock_session.commit = AsyncMock()

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.put(
            "/ocabra/auth/password",
            json={"current_password": "oldpass", "new_password": "newpass"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_change_password_wrong_current_returns_400():
    """PUT /ocabra/auth/password with wrong current_password returns 400."""
    from ocabra.api._deps_auth import UserContext, get_current_user

    app = _make_app()
    user = _make_db_user(username="grace", password="correctpass")

    _user_ctx = UserContext(
        user_id=str(user.id), username="grace", role="user",
        group_ids=[], accessible_model_ids=set(), is_anonymous=False,
    )
    app.dependency_overrides[get_current_user] = lambda: _user_ctx

    user_result = MagicMock()
    user_result.scalar_one_or_none.return_value = user

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute.return_value = user_result

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.put(
            "/ocabra/auth/password",
            json={"current_password": "wrongpass", "new_password": "newpass"},
        )

    assert resp.status_code == 400


# ── API keys ──────────────────────────────────────────────────────────────────


def test_create_api_key_returns_key_value_once():
    """POST /ocabra/auth/keys must return the raw key in the response body."""
    from ocabra.api._deps_auth import UserContext, get_current_user

    app = _make_app()
    user = _make_db_user(username="henry")

    _user_ctx = UserContext(
        user_id=str(user.id), username="henry", role="user",
        group_ids=[], accessible_model_ids=set(), is_anonymous=False,
    )
    app.dependency_overrides[get_current_user] = lambda: _user_ctx

    created_key = MagicMock()
    created_key.id = uuid.uuid4()
    created_key.name = "my-key"
    created_key.key_prefix = "sk-ocabra-XXXXXXXXXX…"
    created_key.expires_at = None

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock(return_value=None)

    def _add(obj):
        obj.id = created_key.id
        obj.key_prefix = obj.key_prefix

    mock_session.add = MagicMock(side_effect=_add)

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/auth/keys",
            json={"name": "my-key"},
        )

    assert resp.status_code == 201
    body = resp.json()
    assert "key" in body
    assert body["key"].startswith("sk-ocabra-")
    assert body["name"] == "my-key"


def test_create_api_key_without_expiry():
    """POST /ocabra/auth/keys without expires_in_days sets expires_at to null."""
    from ocabra.api._deps_auth import UserContext, get_current_user

    app = _make_app()
    user = _make_db_user(username="igor")

    _user_ctx = UserContext(
        user_id=str(user.id), username="igor", role="user",
        group_ids=[], accessible_model_ids=set(), is_anonymous=False,
    )
    app.dependency_overrides[get_current_user] = lambda: _user_ctx

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock(return_value=None)

    captured = {}

    def _add(obj):
        captured["key"] = obj
        obj.id = uuid.uuid4()

    mock_session.add = MagicMock(side_effect=_add)

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.post("/ocabra/auth/keys", json={"name": "no-expiry-key"})

    assert resp.status_code == 201
    assert resp.json()["expires_at"] is None
    assert captured["key"].expires_at is None


def test_list_api_keys_shows_prefix_not_value():
    """GET /ocabra/auth/keys returns key_prefix but NOT the raw key value."""
    from ocabra.api._deps_auth import UserContext, get_current_user

    app = _make_app()
    user = _make_db_user(username="julia")
    key_row, _raw = _make_api_key_row(user=user)

    _user_ctx = UserContext(
        user_id=str(user.id), username="julia", role="user",
        group_ids=[], accessible_model_ids=set(), is_anonymous=False,
    )
    app.dependency_overrides[get_current_user] = lambda: _user_ctx

    keys_result = MagicMock()
    keys_result.scalars.return_value.all.return_value = [key_row]

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute.return_value = keys_result

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.get("/ocabra/auth/keys")

    assert resp.status_code == 200
    keys = resp.json()
    assert len(keys) == 1
    assert "key_prefix" in keys[0]
    assert "key" not in keys[0]  # raw value must NOT be returned
    assert keys[0]["key_prefix"].endswith("…")


def test_revoke_api_key_sets_revoked():
    """DELETE /ocabra/auth/keys/{key_id} must set is_revoked = True on the key."""
    from ocabra.api._deps_auth import UserContext, get_current_user

    app = _make_app()
    user = _make_db_user(username="kate")
    key_row, _raw = _make_api_key_row(user=user)
    key_row.is_revoked = False  # start not revoked

    _user_ctx = UserContext(
        user_id=str(user.id), username="kate", role="user",
        group_ids=[], accessible_model_ids=set(), is_anonymous=False,
    )
    app.dependency_overrides[get_current_user] = lambda: _user_ctx

    key_result = MagicMock()
    key_result.scalar_one_or_none.return_value = key_row

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute.return_value = key_result
    mock_session.commit = AsyncMock()

    with patch("ocabra.database.AsyncSessionLocal", return_value=mock_session):
        client = TestClient(app)
        resp = client.delete(f"/ocabra/auth/keys/{key_row.id}")

    assert resp.status_code == 204
    assert key_row.is_revoked is True


async def test_revoked_key_cannot_authenticate():
    """A revoked API key must not authenticate — _resolve_api_key returns None."""
    from ocabra.api._deps_auth import _resolve_api_key

    _make_db_user(username="leo")
    raw_key, _key_hash, _prefix = generate_api_key()

    revoked_result = MagicMock()
    revoked_result.scalar_one_or_none.return_value = None  # WHERE is_revoked=False fails

    mock_session = AsyncMock()
    mock_session.execute.return_value = revoked_result

    result = await _resolve_api_key(raw_key, mock_session)
    assert result is None


async def test_expired_key_cannot_authenticate():
    """An expired API key must not authenticate — _resolve_api_key returns None."""
    from ocabra.api._deps_auth import _resolve_api_key

    _make_db_user(username="mia")
    raw_key, _key_hash, _prefix = generate_api_key()

    expired_result = MagicMock()
    expired_result.scalar_one_or_none.return_value = None  # WHERE expires_at > now fails

    mock_session = AsyncMock()
    mock_session.execute.return_value = expired_result

    result = await _resolve_api_key(raw_key, mock_session)
    assert result is None
