"""SEG-09: who may act as another user.

The companion services' token (``X-Gateway-Token``: gateway, TRELLIS) reaches
several containers. With ``X-Internal-User-Id`` it used to let its holder act
as any user, admins included. Only the batch processor needs that, from inside
this process, so impersonation now takes a per-process token instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from ocabra.api import _deps_auth as auth


def _request(headers: dict[str, str]):
    return SimpleNamespace(
        headers=headers, cookies={}, state=SimpleNamespace(), url=SimpleNamespace(path="/v1/x")
    )


@pytest.fixture
def service_token(monkeypatch):
    from ocabra.config import settings

    monkeypatch.setattr(settings, "gateway_service_token", "svc-token")
    return "svc-token"


@pytest.fixture
def admin_in_db(monkeypatch):
    admin = auth.UserContext(user_id="a1", username="root-admin", role="system_admin")
    lookup = AsyncMock(return_value=admin)
    monkeypatch.setattr(auth, "_impersonated_context", lookup)
    return lookup


@pytest.mark.asyncio
async def test_the_service_token_cannot_act_as_another_user(service_token, admin_in_db):
    ctx = await auth.get_current_user(
        _request({"X-Gateway-Token": service_token, "X-Internal-User-Id": "a1"})
    )

    assert ctx.role == "model_manager" and ctx.username == "__gateway__"
    admin_in_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_in_process_token_acts_as_the_batch_owner(admin_in_db):
    ctx = await auth.get_current_user(
        _request({auth.INTERNAL_TOKEN_HEADER: auth.INTERNAL_TOKEN, "X-Internal-User-Id": "a1"})
    )

    assert ctx.user_id == "a1"


@pytest.mark.asyncio
async def test_a_missing_batch_owner_is_refused_not_upgraded(monkeypatch):
    """Before, a failed impersonation fell back to the model_manager service
    identity: a batch of a deleted user ran with more rights than its owner."""
    monkeypatch.setattr(auth, "_impersonated_context", AsyncMock(return_value=None))

    with pytest.raises(HTTPException) as exc:
        await auth.get_current_user(
            _request({auth.INTERNAL_TOKEN_HEADER: auth.INTERNAL_TOKEN, "X-Internal-User-Id": "gone"})
        )

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_a_wrong_in_process_token_gets_nothing(monkeypatch, admin_in_db):
    from ocabra.config import settings

    monkeypatch.setattr(settings, "require_api_key_openai", True)

    with pytest.raises(HTTPException) as exc:
        await auth.get_current_user(
            _request({auth.INTERNAL_TOKEN_HEADER: "guess", "X-Internal-User-Id": "a1"})
        )

    assert exc.value.status_code == 401
    admin_in_db.assert_not_awaited()


def test_only_the_in_process_token_marks_a_request_as_a_batch(service_token):
    from ocabra.api.openai._deps import is_batch_origin_request

    batch = {"X-Internal-Request-Origin": "batch"}

    assert is_batch_origin_request(
        _request({**batch, auth.INTERNAL_TOKEN_HEADER: auth.INTERNAL_TOKEN})
    )
    assert not is_batch_origin_request(_request({**batch, "X-Gateway-Token": service_token}))


def test_the_batch_processor_uses_the_in_process_token():
    import inspect

    from ocabra.core import batch_processor as bp

    source = inspect.getsource(bp)
    assert '"X-Ocabra-Internal-Token": self._token' in source
    assert '"X-Gateway-Token": self._token' not in source
