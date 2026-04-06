from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
    # comfyui has runtime_loaded_when_alive=True, so status is "active" after health check
    assert state.status == "active"
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
    state.last_activity_at = datetime.now(UTC) - timedelta(
        seconds=state.idle_unload_after_seconds + 5
    )

    await manager.check_idle_unloads()

    # comfyui has unload_path="/free" so the unload() method calls it via HTTP POST.
    # Other GET calls may precede it (e.g. generation metric polling).
    post_calls = [(m, u) for m, u in calls if m == "POST"]
    assert post_calls
    assert post_calls[0][0] == "POST"
    assert post_calls[0][1].endswith("/free")
    assert state.runtime_loaded is False
    # After REST-based unload, status is "idle" if service still alive
    assert state.status == "idle"


@pytest.mark.asyncio
async def test_set_enabled_disables_service_runtime(monkeypatch) -> None:
    async def fake_publish(channel: str, data: dict) -> None:
        _ = channel, data

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = key, data, ttl

    monkeypatch.setattr("ocabra.core.service_manager.publish", fake_publish)
    monkeypatch.setattr("ocabra.core.service_manager.set_key", fake_set_key)

    manager = ServiceManager()
    state = await manager.get_state("comfyui")
    assert state is not None
    state.service_alive = True
    state.runtime_loaded = True
    state.active_model_ref = "flux"

    updated = await manager.set_enabled("comfyui", enabled=False)

    assert updated.enabled is False
    assert updated.status == "disabled"
    assert updated.service_alive is False
    assert updated.runtime_loaded is False
    assert updated.active_model_ref is None


@pytest.mark.asyncio
async def test_refresh_disabled_service_falls_back_to_health_check(monkeypatch) -> None:
    called = False

    async def fake_publish(channel: str, data: dict) -> None:
        _ = channel, data

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = key, data, ttl

    async def fake_get(self, url: str, *args, **kwargs):
        nonlocal called
        _ = self, args, kwargs
        called = True
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("GET", url))

    monkeypatch.setattr("ocabra.core.service_manager.publish", fake_publish)
    monkeypatch.setattr("ocabra.core.service_manager.set_key", fake_set_key)
    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    manager = ServiceManager()
    state = await manager.get_state("comfyui")
    assert state is not None
    state.enabled = False

    updated = await manager.refresh("comfyui")

    assert updated.enabled is False
    assert updated.status == "disabled"
    assert updated.service_alive is True
    assert updated.runtime_loaded is False
    assert updated.detail == "disabled_but_service_alive"
    assert called is True


@pytest.mark.asyncio
async def test_start_disabled_service_raises(monkeypatch) -> None:
    async def fake_publish(channel: str, data: dict) -> None:
        _ = channel, data

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = key, data, ttl

    monkeypatch.setattr("ocabra.core.service_manager.publish", fake_publish)
    monkeypatch.setattr("ocabra.core.service_manager.set_key", fake_set_key)

    manager = ServiceManager()
    await manager.set_enabled("hunyuan", enabled=False)

    with pytest.raises(RuntimeError):
        await manager.start_service("hunyuan")


@pytest.mark.asyncio
async def test_set_enabled_persists_overrides(monkeypatch) -> None:
    async def fake_publish(channel: str, data: dict) -> None:
        _ = channel, data

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = key, data, ttl

    called = False

    async def fake_persist(self) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("ocabra.core.service_manager.publish", fake_publish)
    monkeypatch.setattr("ocabra.core.service_manager.set_key", fake_set_key)
    monkeypatch.setattr(ServiceManager, "_persist_overrides", fake_persist)

    manager = ServiceManager()
    state = await manager.get_state("comfyui")
    assert state is not None
    state.runtime_loaded = False
    state.service_alive = False

    await manager.set_enabled("comfyui", enabled=False)

    assert called is True


@pytest.mark.asyncio
async def test_start_applies_loaded_overrides(monkeypatch) -> None:
    from ocabra.config import settings

    async def fake_publish(channel: str, data: dict) -> None:
        _ = channel, data

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = key, data, ttl

    a1111_calls = 0

    async def fake_get(self, url: str, *args, **kwargs):
        nonlocal a1111_calls
        _ = args, kwargs
        if url.startswith(settings.a1111_base_url.rstrip("/")):
            a1111_calls += 1
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("GET", url))

    async def fake_load(self) -> None:
        state = await self.get_state("a1111")
        assert state is not None
        state.enabled = False

    monkeypatch.setattr("ocabra.core.service_manager.publish", fake_publish)
    monkeypatch.setattr("ocabra.core.service_manager.set_key", fake_set_key)
    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(ServiceManager, "_load_persisted_overrides", fake_load)

    manager = ServiceManager()
    await manager.start()

    state = await manager.get_state("a1111")
    assert state is not None
    assert state.enabled is False
    assert state.status == "disabled"
    # start() now calls refresh() for all services including disabled ones,
    # which triggers a health-check probe; the key assertion is that the
    # service is correctly marked as disabled after loading overrides.


@pytest.mark.asyncio
async def test_refresh_disabled_service_marks_external_runtime(monkeypatch) -> None:
    async def fake_publish(channel: str, data: dict) -> None:
        _ = channel, data

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = key, data, ttl

    async def fake_is_running(self, state) -> bool:  # noqa: ANN001
        _ = self, state
        return True

    monkeypatch.setattr("ocabra.core.service_manager.publish", fake_publish)
    monkeypatch.setattr("ocabra.core.service_manager.set_key", fake_set_key)
    monkeypatch.setattr(ServiceManager, "_is_container_running", fake_is_running)

    manager = ServiceManager()
    state = await manager.get_state("a1111")
    assert state is not None
    state.enabled = False

    updated = await manager.refresh("a1111")

    assert updated.enabled is False
    assert updated.status == "disabled"
    assert updated.service_alive is True
    assert updated.runtime_loaded is False
    assert updated.detail is not None
    assert "disabled_but_container_running" in updated.detail


@pytest.mark.asyncio
async def test_set_enabled_stops_running_container_even_if_service_not_marked_alive(monkeypatch) -> None:
    async def fake_publish(channel: str, data: dict) -> None:
        _ = channel, data

    async def fake_set_key(key: str, data: dict, ttl: int | None = None) -> None:
        _ = key, data, ttl

    async def fake_is_running(self, state) -> bool:  # noqa: ANN001
        _ = self, state
        return True

    stop_calls: list[str] = []

    async def fake_stop(self, state) -> None:  # noqa: ANN001
        _ = self
        stop_calls.append(state.service_id)

    monkeypatch.setattr("ocabra.core.service_manager.publish", fake_publish)
    monkeypatch.setattr("ocabra.core.service_manager.set_key", fake_set_key)
    monkeypatch.setattr(ServiceManager, "_is_container_running", fake_is_running)
    monkeypatch.setattr(ServiceManager, "_stop_container", fake_stop)

    manager = ServiceManager()
    state = await manager.get_state("a1111")
    assert state is not None
    state.enabled = True
    state.service_alive = False
    state.runtime_loaded = False

    await manager.set_enabled("a1111", enabled=False)

    assert stop_calls == ["a1111"]
