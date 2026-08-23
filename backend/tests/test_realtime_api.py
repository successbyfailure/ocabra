"""Authentication and attribution tests for the OpenAI Realtime WebSocket."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai.realtime import _authenticate_ws


def _context(**overrides) -> UserContext:
    values = {
        "user_id": "11111111-1111-1111-1111-111111111111",
        "username": "makespace",
        "role": "system_admin",
        "api_key_name": "realtime-client",
    }
    values.update(overrides)
    return UserContext(**values)


def _websocket(*, authorization: str = "", api_key: str = "", cookie: str = ""):
    return SimpleNamespace(
        headers={"authorization": authorization} if authorization else {},
        query_params={"api_key": api_key} if api_key else {},
        cookies={"ocabra_session": cookie} if cookie else {},
    )


def _session_factory() -> MagicMock:
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=object())
    context.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=context)


@pytest.mark.asyncio
async def test_authenticate_ws_returns_api_key_identity() -> None:
    expected = _context()
    websocket = _websocket(authorization="Bearer sk-ocabra-test")
    with (
        patch("ocabra.database.AsyncSessionLocal", _session_factory()),
        patch(
            "ocabra.api._deps_auth._resolve_api_key",
            new=AsyncMock(return_value=expected),
        ) as resolve,
    ):
        result = await _authenticate_ws(websocket)

    assert result is expected
    resolve.assert_awaited_once()
    assert resolve.await_args.args[0] == "sk-ocabra-test"


@pytest.mark.asyncio
async def test_authenticate_ws_returns_cookie_identity() -> None:
    expected = _context(api_key_name=None)
    websocket = _websocket(cookie="jwt-cookie")
    with (
        patch("ocabra.database.AsyncSessionLocal", _session_factory()),
        patch(
            "ocabra.api._deps_auth._resolve_jwt_cookie",
            new=AsyncMock(return_value=expected),
        ) as resolve,
    ):
        result = await _authenticate_ws(websocket)

    assert result is expected
    resolve.assert_awaited_once()
    assert resolve.await_args.args[0] == "jwt-cookie"


@pytest.mark.asyncio
async def test_authenticate_ws_builds_anonymous_identity_when_allowed() -> None:
    expected = _context(
        user_id=None,
        username=None,
        role="user",
        api_key_name=None,
        is_anonymous=True,
    )
    with (
        patch("ocabra.config.settings.require_api_key_openai", False),
        patch("ocabra.database.AsyncSessionLocal", _session_factory()),
        patch(
            "ocabra.api._deps_auth._build_anonymous_context",
            new=AsyncMock(return_value=expected),
        ),
    ):
        result = await _authenticate_ws(_websocket())

    assert result is expected


@pytest.mark.asyncio
async def test_authenticate_ws_rejects_missing_credentials_when_required() -> None:
    with patch("ocabra.config.settings.require_api_key_openai", True):
        assert await _authenticate_ws(_websocket()) is None
