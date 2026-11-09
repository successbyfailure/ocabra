"""``/ocabra/stats/recent`` lo ve cualquier usuario, pero la IP y el user
agent de las peticiones solo los admin (decision del 2026-11-09)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocabra.api._deps_auth import UserContext
from ocabra.api.internal import stats as stats_api

ROW = {
    "id": "x",
    "modelId": "m",
    "username": "otra-persona",
    "apiKeyName": "su-clave",
    "clientAddr": "10.0.0.7",
    "userAgent": "opencode/1.0",
}


def _user(role: str) -> UserContext:
    return UserContext(
        user_id="u",
        username="u",
        role=role,
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "sees_network"),
    [("user", False), ("model_manager", False), ("system_admin", True)],
)
async def test_network_fields_only_for_admins(monkeypatch, role, sees_network):
    async def fake_recent(limit):
        return {"requests": [dict(ROW)]}

    monkeypatch.setattr("ocabra.stats.aggregator.get_recent_requests", fake_recent)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    result = await stats_api.recent_requests(request, limit=20, user=_user(role))
    [row] = result["requests"]
    assert row["username"] == "otra-persona"
    assert row["apiKeyName"] == "su-clave"
    if sees_network:
        assert (row["clientAddr"], row["userAgent"]) == ("10.0.0.7", "opencode/1.0")
    else:
        assert (row["clientAddr"], row["userAgent"]) == (None, None)
