"""Suscripción a ``models_activity`` por ``/ocabra/ws`` (dashboard, v0.61.0)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api.internal import ws as ws_module
from ocabra.core.auth_manager import create_access_token


class _SilentPubSub:
    async def subscribe(self, *channels: str) -> None:
        return None

    async def listen(self):
        while True:  # noqa: ASYNC110 — doble de test: pubsub sin mensajes
            await asyncio.sleep(3600)
            yield {}

    async def aclose(self) -> None:
        return None


def test_activity_is_pushed_only_after_subscribe_and_stops_on_unsubscribe(monkeypatch):
    calls = {"n": 0}

    async def fake_build(_websocket):
        calls["n"] += 1
        return {"activity": {"m": {"inFlight": 1}}, "contexts": {}, "vramByGpu": {}}

    monkeypatch.setattr("ocabra.api.internal.models.build_models_activity", fake_build)
    monkeypatch.setattr(ws_module, "_ACTIVITY_PUSH_SECONDS", 0.05)
    ws_module._activity_cache.update({"at": 0.0, "data": None})

    app = FastAPI()
    app.include_router(ws_module.router, prefix="/ocabra")
    token = create_access_token("u", "user")
    fake_redis = SimpleNamespace(pubsub=lambda: _SilentPubSub())

    with patch("ocabra.api.internal.ws.get_redis", return_value=fake_redis):
        with TestClient(app, cookies={"ocabra_session": token}) as client:
            with client.websocket_connect("/ocabra/ws") as websocket:
                # Sin suscripción no se calcula nada.
                assert calls["n"] == 0
                websocket.send_json({"subscribe": "models_activity"})
                first = websocket.receive_json()
                second = websocket.receive_json()
                websocket.send_json({"unsubscribe": "models_activity"})

    assert first["type"] == second["type"] == "models_activity"
    assert first["data"]["activity"]["m"]["inFlight"] == 1


def test_shared_cache_computes_once_per_tick_for_several_tabs(monkeypatch):
    calls = {"n": 0}

    async def fake_build(_websocket):
        calls["n"] += 1
        return {"activity": {}}

    monkeypatch.setattr("ocabra.api.internal.models.build_models_activity", fake_build)
    ws_module._activity_cache.update({"at": 0.0, "data": None})

    async def three_tabs():
        await asyncio.gather(*(ws_module._shared_models_activity(None) for _ in range(3)))

    asyncio.run(three_tabs())
    assert calls["n"] == 1
