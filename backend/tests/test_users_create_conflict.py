"""Conflict semantics for POST /ocabra/users.

Both ``username`` and ``email`` carry UNIQUE constraints in the ``users`` table.
The endpoint must return 409 (not 500) in three cases:

* the caller-supplied username collides with an existing row,
* the caller-supplied email collides with an existing row,
* a race condition lets a duplicate slip past the pre-check and the database
  raises ``IntegrityError`` on commit/flush.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from ocabra.api._deps_auth import UserContext, get_current_user


def _admin_ctx() -> UserContext:
    return UserContext(
        user_id=str(uuid.uuid4()),
        username="admin-test",
        role="system_admin",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )


def _make_app() -> FastAPI:
    from ocabra.api.internal.users import router as users_router

    app = FastAPI()
    app.include_router(users_router, prefix="/ocabra")
    app.dependency_overrides[get_current_user] = lambda: _admin_ctx()
    return app


def _existing_user(*, username: str = "alice", email: str | None = "alice@example.com"):
    u = MagicMock()
    u.id = uuid.uuid4()
    u.username = username
    u.email = email
    return u


def _session_with_lookup(found_user):
    """A mock session whose execute() returns *found_user* via scalar_one_or_none."""
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = found_user

    session = AsyncMock()
    session.execute = AsyncMock(return_value=scalar_result)
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def test_duplicate_username_returns_409_with_username_taken_code():
    app = _make_app()
    session = _session_with_lookup(_existing_user(username="alice", email="other@x.com"))

    with patch("ocabra.database.AsyncSessionLocal", return_value=session):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/users",
            json={"username": "alice", "password": "pw123456", "role": "user", "email": "n@x.com"},
        )

    assert resp.status_code == 409
    body = resp.json()
    assert body["detail"]["error"] == "username_taken"


def test_duplicate_email_returns_409_with_email_taken_code():
    app = _make_app()
    # Pre-check returns a row whose username DOES NOT match the request — the
    # collision is on the email column.
    session = _session_with_lookup(
        _existing_user(username="bob", email="alice@example.com")
    )

    with patch("ocabra.database.AsyncSessionLocal", return_value=session):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/users",
            json={
                "username": "alice",
                "password": "pw123456",
                "role": "user",
                "email": "alice@example.com",
            },
        )

    assert resp.status_code == 409
    body = resp.json()
    assert body["detail"]["error"] == "email_taken"


def test_integrity_error_on_flush_translates_to_409():
    """Race: pre-check finds nothing, but the unique index trips on flush.

    This is the original bug — without the IntegrityError catch the endpoint
    returns 500. After the fix we degrade to a clean 409.
    """
    app = _make_app()

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None  # pre-check passes

    session = AsyncMock()
    session.execute = AsyncMock(return_value=scalar_result)
    session.add = MagicMock()
    session.rollback = AsyncMock()

    # flush() raises IntegrityError mimicking what asyncpg returns on a
    # duplicate username.
    orig = Exception("duplicate key value violates unique constraint \"users_username_key\"")
    session.flush = AsyncMock(side_effect=IntegrityError("INSERT INTO users", {}, orig))
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    with patch("ocabra.database.AsyncSessionLocal", return_value=session):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/users",
            json={
                "username": "race-winner",
                "password": "pw123456",
                "role": "user",
                "email": "rw@x.com",
            },
        )

    assert resp.status_code == 409
    body = resp.json()
    assert body["detail"]["error"] == "username_taken"
    session.rollback.assert_awaited_once()


def test_integrity_error_on_email_translates_to_email_taken():
    """Same race as above, but the unique constraint is on ``email``."""
    app = _make_app()

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None

    session = AsyncMock()
    session.execute = AsyncMock(return_value=scalar_result)
    session.add = MagicMock()
    session.rollback = AsyncMock()
    orig = Exception("duplicate key value violates unique constraint \"users_email_key\"")
    session.flush = AsyncMock(side_effect=IntegrityError("INSERT INTO users", {}, orig))
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    with patch("ocabra.database.AsyncSessionLocal", return_value=session):
        client = TestClient(app)
        resp = client.post(
            "/ocabra/users",
            json={
                "username": "carol",
                "password": "pw123456",
                "role": "user",
                "email": "shared@x.com",
            },
        )

    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "email_taken"
