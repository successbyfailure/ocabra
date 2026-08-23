"""Authentication and attribution tests for the OpenAI Realtime WebSocket."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai.realtime import _authenticate_ws, realtime_ws


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


@pytest.mark.asyncio
async def test_realtime_ws_records_completed_session() -> None:
    user = _context()
    websocket = SimpleNamespace(
        state=SimpleNamespace(),
        app=SimpleNamespace(
            state=SimpleNamespace(
                worker_pool=object(),
                model_manager=object(),
                profile_registry=SimpleNamespace(get=AsyncMock(return_value=None)),
            )
        ),
        client=SimpleNamespace(host="127.0.0.1"),
        headers={},
        accept=AsyncMock(),
    )

    class FakeRealtimeSession:
        def __init__(self, **kwargs) -> None:
            self._session_id = "sess_test"
            self.transcription_only = False
            self._response_task = None
            self._partial_task = None
            self._background_load_task = None

        async def run(self) -> None:
            raise WebSocketDisconnect(code=1000)

    recorder = AsyncMock()
    with (
        patch(
            "ocabra.api.openai.realtime._authenticate_ws",
            new=AsyncMock(return_value=user),
        ),
        patch("ocabra.api.openai.realtime.RealtimeSession", FakeRealtimeSession),
        patch("ocabra.api.openai.realtime.record_realtime_request", new=recorder),
    ):
        await realtime_ws(
            websocket,
            model="vllm/test-model",
            intent="",
            conversation_id="",
            num_speakers=0,
        )

    websocket.accept.assert_awaited_once()
    assert websocket.state.auth_user is user
    recorded = recorder.await_args.kwargs
    assert recorded["session_id"] == "sess_test"
    assert recorded["request_kind"] == "realtime_session"
    assert recorded["status_code"] == 101
    assert recorded["model_id"] == "vllm/test-model"
