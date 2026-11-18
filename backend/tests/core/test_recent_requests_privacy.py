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


@pytest.mark.asyncio
async def test_in_flight_rows_say_what_kind_of_request_is_running(monkeypatch):
    """Logs pinta las peticiones en curso: sin el tipo solo se veían guiones."""
    import time

    from ocabra.core.model_manager import ActiveRequest

    async def fake_recent(limit):
        return {"requests": []}

    monkeypatch.setattr("ocabra.stats.aggregator.get_recent_requests", fake_recent)
    active = ActiveRequest(request_id="r1", model_id="llamacpp_mks/m", started_at=time.time() - 3, request_kind="chat", username="ana")

    async def list_states(include_derived=False):
        return []

    mm = SimpleNamespace(
        active_requests_snapshot=lambda: [active],
        list_states=list_states,
        _canonical_runtime_model_id=lambda mid: mid,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model_manager=mm)))
    result = await stats_api.recent_requests(request, limit=20, user=_user("system_admin"))
    [row] = [r for r in result["requests"] if r.get("inFlight")]
    assert row["requestKind"] == "chat"
    assert row["username"] == "ana"


@pytest.mark.asyncio
async def test_in_flight_queue_uses_the_real_slot_count(monkeypatch):
    """BUG-31: the queue split read only ``_runtime.parallelism`` (presets).
    A llama.cpp model with ``parallel_slots: 4`` and no presets counted as one
    slot, so three of the four requests being processed showed as queued."""
    import time

    from ocabra.core.model_manager import ActiveRequest

    async def fake_recent(limit):
        return {"requests": []}

    monkeypatch.setattr("ocabra.stats.aggregator.get_recent_requests", fake_recent)
    now = time.time()
    active = [
        ActiveRequest(request_id=f"r{i}", model_id="llamacpp_mks/m", started_at=now - 10 + i)
        for i in range(5)
    ]
    state = SimpleNamespace(
        model_id="llamacpp_mks/m",
        backend_type="llamacpp_mks",
        extra_config={"llama_cpp": {"parallel_slots": 4}},
    )

    async def list_states(include_derived=False):
        return [state]

    mm = SimpleNamespace(
        active_requests_snapshot=lambda: active,
        list_states=list_states,
        _canonical_runtime_model_id=lambda mid: mid,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model_manager=mm)))
    result = await stats_api.recent_requests(request, limit=20, user=_user("system_admin"))
    rows = [r for r in result["requests"] if r.get("inFlight")]
    assert {r["slots"] for r in rows} == {4}
    assert sum(1 for r in rows if r["queued"]) == 1
