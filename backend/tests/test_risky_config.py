"""SEG-04: options that run code need system_admin, through any endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ocabra.api import _risky_config
from ocabra.api._deps_auth import UserContext
from ocabra.api._risky_config import RiskyConfigMiddleware, risky_settings


def _app(role: str):
    app = FastAPI()
    app.add_middleware(RiskyConfigMiddleware)
    seen: dict = {}

    @app.patch("/ocabra/models/{model_id:path}")
    async def patch(model_id: str, request: Request):
        seen["body"] = await request.json()  # the body still reaches the endpoint
        return {"ok": True}

    @app.post("/ocabra/profiles/{profile_id}")
    async def profile(profile_id: str):
        return {"ok": True}

    user = AsyncMock(return_value=UserContext(user_id="u", username="u", role=role))
    return app, seen, user


@pytest.mark.parametrize(
    "body",
    [
        {"extraConfig": {"vllm": {"trust_remote_code": True}}},
        {"extra_config": {"vllm": {"extra_args": ["--worker-cls", "evil.Worker"]}}},
        {"load_overrides": {"vllm": {"tool_parser_plugin": "/data/models/x/plugin.py"}}},
        {"extraConfig": {"tensorrt_llm": {"serve_bin": "/tmp/x"}}},
        {"trustRemoteCode": True},
    ],
)
def test_a_model_manager_cannot_set_code_running_options(monkeypatch, body):
    app, seen, user = _app("model_manager")
    monkeypatch.setattr("ocabra.api._deps_auth.get_current_user", user)

    resp = TestClient(app).patch("/ocabra/models/vllm/x", json=body)

    assert resp.status_code == 403
    assert "system_admin" in resp.json()["detail"]
    assert "body" not in seen  # never reached the endpoint


def test_an_admin_can_and_the_body_reaches_the_endpoint(monkeypatch):
    app, seen, user = _app("system_admin")
    monkeypatch.setattr("ocabra.api._deps_auth.get_current_user", user)
    body = {"extraConfig": {"vllm": {"trust_remote_code": True}}}

    resp = TestClient(app).patch("/ocabra/models/vllm/x", json=body)

    assert resp.status_code == 200 and seen["body"] == body


@pytest.mark.parametrize(
    "body",
    [
        {"extraConfig": {"vllm": {"trust_remote_code": False}}},  # turning it off
        {"extraConfig": {"vllm": {"extra_args": []}}},
        {"extraConfig": {"llama_cpp": {"ctx_size": 32768}}},  # anything else
    ],
)
def test_turning_them_off_or_other_options_need_no_extra_role(monkeypatch, body):
    app, _seen, user = _app("model_manager")
    monkeypatch.setattr("ocabra.api._deps_auth.get_current_user", user)

    assert TestClient(app).patch("/ocabra/models/vllm/x", json=body).status_code == 200
    user.assert_not_awaited()  # no risky option, no auth lookup in the middleware


def test_paths_report_where_each_option_is():
    assert risky_settings({"a": [{"vllm": {"extraArgs": ["-x"]}}]}) == ["a[0].vllm.extraArgs"]


def test_the_middleware_is_installed_in_the_app():
    from ocabra.main import app

    assert any(m.cls is _risky_config.RiskyConfigMiddleware for m in app.user_middleware)
