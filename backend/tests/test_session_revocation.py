"""SEG-19 / SEG-20: sessions after a password change, keys minting keys, Secure cookie."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ocabra.api import _deps_auth as auth
from ocabra.core.auth_manager import create_access_token, decode_access_token, hash_password


def test_a_session_from_before_the_password_change_is_rejected():
    changed = datetime.now(UTC)
    before = {"iat": int(changed.timestamp()) - 60}
    same_second = {"iat": int(changed.timestamp())}

    assert auth.session_predates(before, changed)
    assert not auth.session_predates(same_second, changed)  # the fresh one
    assert not auth.session_predates(before, None)  # never changed
    assert auth.session_predates({}, changed)  # no iat: cannot prove it is newer


@pytest.mark.asyncio
async def test_the_cookie_resolver_applies_the_cutoff():
    user = MagicMock(id=uuid.uuid4(), is_active=True, role="user", username="eve")
    user.sessions_valid_after = datetime.now(UTC) + timedelta(minutes=5)
    token = create_access_token(str(user.id), "user")
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    session = AsyncMock()
    session.execute.return_value = result

    redis = AsyncMock()
    redis.exists.return_value = 0
    with patch("ocabra.redis_client.get_redis", AsyncMock(return_value=redis)):
        assert await auth._resolve_jwt_cookie(token, session) is None
        user.sessions_valid_after = None
        assert await auth._resolve_jwt_cookie(token, session) is not None


def _ctx(**kw):
    values = dict(user_id=str(uuid.uuid4()), username="eve", role="user")
    values.update(kw)
    return auth.UserContext(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize("days", [None, 0, 91, 3650])
async def test_an_api_key_cannot_create_keys_that_do_not_expire_soon(days):
    from ocabra.api.internal.auth import CreateApiKeyRequest, create_api_key

    with pytest.raises(HTTPException) as exc:
        await create_api_key(
            CreateApiKeyRequest(name="x", expires_in_days=days), _ctx(via_api_key=True)
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_an_api_key_can_create_expiring_keys_like_coder_does():
    """Coder creates one key per workspace, with 30 days, using its own key.
    v0.109.0 refused every key created with a key and broke it."""
    from ocabra.api.internal.auth import CreateApiKeyRequest, create_api_key

    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.add = MagicMock()

    async def _refresh(row):
        row.id = uuid.uuid4()

    session.refresh = _refresh
    with patch("ocabra.database.AsyncSessionLocal", return_value=session):
        out = await create_api_key(
            CreateApiKeyRequest(name="coder-ws-owner-1", expires_in_days=30),
            _ctx(via_api_key=True),
        )
    assert out["key"].startswith("sk-ocabra-") and out["expires_at"]


@pytest.mark.asyncio
async def test_an_admin_api_key_cannot_create_keys_for_users():
    from ocabra.api.internal.users import CreateUserApiKeyRequest, create_user_api_key

    with pytest.raises(HTTPException) as exc:
        await create_user_api_key(
            str(uuid.uuid4()),
            CreateUserApiKeyRequest(name="x"),
            _ctx(role="system_admin", via_api_key=True),
        )
    assert exc.value.status_code == 403


def _app_as(ctx):
    from ocabra.api.internal.auth import router

    app = FastAPI()
    app.include_router(router, prefix="/ocabra")
    app.dependency_overrides[auth.get_current_user] = lambda: ctx
    return app


def _session_with(user):
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.execute.return_value = result
    return session


def test_changing_the_password_signs_out_elsewhere_and_keeps_this_session():
    user = SimpleNamespace(
        id=uuid.uuid4(), role="user", username="frank",
        hashed_password=hash_password("old-password"), sessions_valid_after=None,
        updated_at=None,
    )  # fmt: skip
    ctx = _ctx(user_id=str(user.id), username="frank")
    old = create_access_token(str(user.id), "user", remember=True)

    with patch("ocabra.database.AsyncSessionLocal", return_value=_session_with(user)):
        client = TestClient(_app_as(ctx))
        client.cookies.set("ocabra_session", old)
        resp = client.put(
            "/ocabra/auth/password",
            json={"current_password": "old-password", "new_password": "a-better-pass"},
        )

    assert resp.status_code == 200
    assert user.sessions_valid_after is not None
    fresh = resp.cookies.get("ocabra_session")
    assert fresh and fresh != old
    payload = decode_access_token(fresh)
    assert not auth.session_predates(payload, user.sessions_valid_after)
    assert payload["exp"] - payload["iat"] > 2 * 86400  # kept "remember me"


def test_a_key_changing_the_password_gets_no_cookie():
    user = SimpleNamespace(
        id=uuid.uuid4(), role="user", username="frank",
        hashed_password=hash_password("old-password"), sessions_valid_after=None,
        updated_at=None,
    )  # fmt: skip
    with patch("ocabra.database.AsyncSessionLocal", return_value=_session_with(user)):
        resp = TestClient(_app_as(_ctx(user_id=str(user.id), via_api_key=True))).put(
            "/ocabra/auth/password",
            json={"current_password": "old-password", "new_password": "a-better-pass"},
        )
    assert resp.status_code == 200
    assert "ocabra_session" not in resp.cookies
    assert user.sessions_valid_after is not None


@pytest.mark.parametrize(
    ("headers", "secure"),
    [({"x-forwarded-proto": "https"}, True), ({"x-forwarded-proto": "http"}, False), ({}, False)],
)
def test_the_cookie_is_secure_when_the_browser_used_https(headers, secure):
    from fastapi import Response

    from ocabra.api.internal.auth import _set_session_cookie

    request = SimpleNamespace(headers=headers, url=SimpleNamespace(scheme="http"))
    response = Response()
    _set_session_cookie(response, "t", remember=False, request=request)
    assert ("secure" in response.headers["set-cookie"].lower()) is secure
