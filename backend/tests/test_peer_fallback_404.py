"""A model only a peer has, whose peer failed, answered 404 "does not exist"."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from starlette.exceptions import HTTPException


def _request(info):
    return SimpleNamespace(state=SimpleNamespace(federation_info=info), headers={})


@pytest.mark.asyncio
async def test_a_404_after_falling_back_from_a_peer_is_a_retryable_503():
    from ocabra.main import _peer_fallback_not_found_handler

    resp = await _peer_fallback_not_found_handler(
        _request({"fallback": True, "fallback_peer": "pod-1", "fallback_detail": "502"}),
        HTTPException(status_code=404, detail="The model 'x' does not exist."),
    )
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "15"
    assert body["detail"]["error"]["code"] == "peer_unavailable"
    assert "pod-1" in body["detail"]["error"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("info", "status"),
    [({}, 404), ({"decision": "local"}, 404), ({"fallback": True}, 400)],
)
async def test_other_errors_are_untouched(info, status):
    from ocabra.main import _peer_fallback_not_found_handler

    resp = await _peer_fallback_not_found_handler(
        _request(info), HTTPException(status_code=status, detail="x")
    )
    assert resp.status_code == status
