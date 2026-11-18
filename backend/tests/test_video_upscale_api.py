"""BUG-20: ``/ocabra/video/upscale`` recorded no stats and ignored routers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user


def _app(backend_upscale):
    from ocabra.api.internal.video import router

    app = FastAPI()
    app.include_router(router, prefix="/ocabra")
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="u1", username="vhs", role="user"
    )
    state = SimpleNamespace(backend_type="flashvsr", backend_model_id="flashvsr-v1")
    backend = MagicMock()
    backend.health_check = AsyncMock(return_value=True)
    backend.upscale_video = backend_upscale
    pool = MagicMock()
    pool.get_worker.return_value = object()
    pool.get_backend = AsyncMock(return_value=backend)
    mm = MagicMock()
    mm.begin_request.return_value = "r1"
    app.state.worker_pool = pool
    app.state.model_manager = mm
    app.state.profile_registry = MagicMock()
    app.state.router_resolver = object()
    profile = SimpleNamespace(base_model_id="flashvsr/flashvsr-v1", load_overrides=None)
    return app, state, profile


def _post(app, state, profile):
    with (
        patch(
            "ocabra.api.internal.video.resolve_profile",
            AsyncMock(return_value=(profile, state)),
        ) as resolve,
        patch("ocabra.api.internal.video.check_capability"),
        patch("ocabra.api.internal.video.record_direct_request", AsyncMock()) as record,
    ):
        resp = TestClient(app, raise_server_exceptions=False).post(
            "/ocabra/video/upscale",
            files={"file": ("seg.mp4", b"\x00" * 1024, "video/mp4")},
            data={"model": "recomendado-upscale"},
        )
    return resp, resolve, record


def test_an_upscale_is_recorded_and_resolves_routers():
    app, state, profile = _app(
        AsyncMock(return_value=(b"mp4", {"x-ocabra-fps": "24", "x-ocabra-frames": "120"}))
    )
    resp, resolve, record = _post(app, state, profile)

    assert resp.status_code == 200
    assert resolve.await_args.kwargs["router_resolver"] is app.state.router_resolver
    kw = record.await_args.kwargs
    assert kw["request_kind"] == "video_upscale"
    assert kw["status_code"] == 200
    assert kw["endpoint_path"] == "/ocabra/video/upscale"
    assert kw["work_size_meta"] == {"input_bytes": 1024, "input_frames": 120}


def test_a_failed_upscale_is_recorded_with_its_status():
    req = httpx.Request("POST", "http://worker/upscale")
    err = httpx.HTTPStatusError(
        "x", request=req, response=httpx.Response(507, text="no VRAM", request=req)
    )
    app, state, profile = _app(AsyncMock(side_effect=err))
    resp, _resolve, record = _post(app, state, profile)

    assert resp.status_code == 507
    assert record.await_args.kwargs["status_code"] == 507
    app.state.model_manager.end_request.assert_called_once_with("flashvsr/flashvsr-v1", "r1")
