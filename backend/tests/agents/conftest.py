"""Shared helpers for the agents + MCP test suite.

These tests do **not** spin up a real database.  Each test mounts only the
router under test on a bare :class:`FastAPI` app, overrides
``get_current_user`` with a canned :class:`UserContext`, and patches
``ocabra.database.AsyncSessionLocal`` to return an :class:`AsyncMock` that the
test wires up with the SQL execution results the router expects.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

from fastapi import FastAPI

from ocabra.api._deps_auth import UserContext, get_current_user


def make_user_context(
    *,
    role: str = "user",
    user_id: str | None = None,
    group_ids: list[str] | None = None,
) -> UserContext:
    """Return a plain :class:`UserContext` usable as a dependency override."""
    return UserContext(
        user_id=user_id or str(uuid.uuid4()),
        username=f"{role}-test",
        role=role,
        group_ids=list(group_ids or []),
        accessible_model_ids=set(),
        is_anonymous=False,
    )


def override_user(app: FastAPI, ctx: UserContext) -> None:
    """Install *ctx* as the current user for the app."""
    app.dependency_overrides[get_current_user] = lambda: ctx


class FakeSessionFactory:
    """Drop-in replacement for ``AsyncSessionLocal``.

    Each call to the factory returns a fresh :class:`AsyncMock` wrapped in an
    async context manager.  Tests configure the mock via ``configure()``.
    """

    def __init__(self) -> None:
        self._configurators: list = []

    def configure(self, fn) -> None:
        """Register a callable that is invoked on each produced session mock."""
        self._configurators.append(fn)

    def __call__(self):
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.delete = AsyncMock()
        session.add = MagicMock()
        session.rollback = AsyncMock()
        for fn in self._configurators:
            fn(session)

        @asynccontextmanager
        async def _ctx():
            yield session

        return _ctx()


def scalar_result(value):
    """Return a MagicMock with ``.scalar_one_or_none()`` returning *value*."""
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    r.scalar_one.return_value = value if value is not None else 0
    return r


def scalars_all(values: list):
    """Return a MagicMock whose ``.scalars().all()`` returns *values*."""
    r = MagicMock()
    r.scalars.return_value.all.return_value = values
    return r


def row_tuples(values: list):
    """Return a MagicMock whose ``.all()`` returns *values* (list of tuples)."""
    r = MagicMock()
    r.all.return_value = values
    return r


def new_uuid() -> UUID:
    return uuid.uuid4()
