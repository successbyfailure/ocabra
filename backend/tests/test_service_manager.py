from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from ocabra.core.service_manager import ServiceManager


@pytest.mark.asyncio
async def test_service_manager_refresh_sets_service_alive(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    states: dict[str, dict] = {}

    async def fake_publish(channel: str, data: dict) -> None:
        events.append((channel, data))

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = ttl
        states[key] = data

    async def fake_get(self, url: str, *args, **kwargs):
        _ = args, kwargs
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("GET", url))

    monkeypatch.setattr("ocabra.core.service_manager.publish", fake_publish)
    monkeypatch.setattr("ocabra.core.service_manager.set_key", fake_set_key)
    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    manager = ServiceManager()
    state = await manager.refresh("comfyui")

    assert state.service_alive is True
    assert state.status == "idle"
    assert events[-1][0] == "service:events"
    assert states["service:state:comfyui"]["service_alive"] is True


@pytest.mark.asyncio
async def test_service_manager_touch_updates_runtime(monkeypatch) -> None:
    async def fake_publish(channel: str, data: dict) -> None:
        _ = channel, data

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = key, data, ttl

    monkeypatch.setattr("ocabra.core.service_manager.publish", fake_publish)
    monkeypatch.setattr("ocabra.core.service_manager.set_key", fake_set_key)

    manager = ServiceManager()
    state = await manager.get_state("a1111")
    assert state is not None
    state.service_alive = True

    updated = await manager.touch(
        "a1111",
        runtime_loaded=True,
        active_model_ref="sdxl-base",
        detail="proxy-activity",
    )

    assert updated.runtime_loaded is True
    assert updated.active_model_ref == "sdxl-base"
    assert updated.status == "active"
    assert updated.last_activity_at is not None


@pytest.mark.asyncio
async def test_service_manager_idle_unload(monkeypatch) -> None:
    async def fake_publish(channel: str, data: dict) -> None:
        _ = channel, data

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = key, data, ttl

    calls: list[tuple[str, str]] = []

    async def fake_request(self, method: str, url: str, **kwargs):
        calls.append((method, url))
        return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))

    monkeypatch.setattr("ocabra.core.service_manager.publish", fake_publish)
    monkeypatch.setattr("ocabra.core.service_manager.set_key", fake_set_key)
    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    manager = ServiceManager()
    state = await manager.get_state("comfyui")
    assert state is not None
    state.service_alive = True
    state.runtime_loaded = True
    state.last_activity_at = datetime.now(timezone.utc) - timedelta(
        seconds=state.idle_unload_after_seconds + 5
    )

    await manager.check_idle_unloads()

    assert calls
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/free")
    assert state.runtime_loaded is False
    assert state.status == "idle"
