"""BUG-20: ``/ocabra/video/upscale`` recorded no stats and ignored routers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user


def _app(backend_upscale):
    from ocabra.api.internal.video import router

    app = FastAPI()
    app.include_router(router, prefix="/ocabra")
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="u1", username="vhs", role="user"
    )
    # resolve_profile devuelve el estado del worker cargado: model_id es su clave.
    state = SimpleNamespace(
        model_id="flashvsr/flashvsr-v1", backend_type="flashvsr", backend_model_id="flashvsr-v1"
    )
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


def test_only_whitelisted_worker_headers_are_forwarded():
    app, state, profile = _app(
        AsyncMock(
            return_value=(
                b"mp4",
                {
                    "x-ocabra-fps": "24",
                    "X-Ocabra-Frames": "120",
                    "x-ocabra-vram-peak-mb": "9000",
                    "content-length": "999999",
                    "server": "uvicorn",
                    "set-cookie": "a=b",
                    "x-ocabra-debug-path": "/data/internal",
                    "x-ocabra-seconds": "1" * 500,
                },
            )
        )
    )
    resp, _resolve, _record = _post(app, state, profile)

    assert resp.status_code == 200
    assert resp.headers["x-ocabra-fps"] == "24"
    assert resp.headers["x-ocabra-frames"] == "120"
    assert resp.headers["x-ocabra-vram-peak-mb"] == "9000"
    assert resp.headers["content-length"] == "3"
    for dropped in ("set-cookie", "x-ocabra-debug-path", "x-ocabra-seconds"):
        assert dropped not in resp.headers
    assert resp.headers.get("server") != "uvicorn"


def test_a_segment_over_the_cap_gets_413_without_reaching_the_worker(monkeypatch):
    from ocabra.config import settings

    monkeypatch.setattr(settings, "video_upscale_max_upload_mb", 1)
    upscale = AsyncMock(return_value=(b"mp4", {}))
    app, state, profile = _app(upscale)
    with (
        patch(
            "ocabra.api.internal.video.resolve_profile",
            AsyncMock(return_value=(profile, state)),
        ) as resolve,
        patch("ocabra.api.internal.video.check_capability"),
        patch("ocabra.api.internal.video.record_direct_request", AsyncMock()),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        too_big = client.post(
            "/ocabra/video/upscale",
            files={"file": ("seg.mp4", b"\x00" * (1024 * 1024 + 1), "video/mp4")},
            data={"model": "recomendado-upscale"},
        )
        exact = client.post(
            "/ocabra/video/upscale",
            files={"file": ("seg.mp4", b"\x00" * (1024 * 1024), "video/mp4")},
            data={"model": "recomendado-upscale"},
        )

    assert too_big.status_code == 413
    assert exact.status_code == 200
    assert upscale.await_count == 1
    assert resolve.await_count == 1


async def test_read_capped_stops_at_the_cap_without_size():
    """Chunked read path (``size`` unknown): stops as soon as it passes the cap."""
    from ocabra.api.internal.video import _read_capped

    reads = 0

    class _Upload:
        size = None

        async def read(self, n):
            nonlocal reads
            reads += 1
            return b"\x00" * n

    with pytest.raises(HTTPException) as exc:
        await _read_capped(_Upload(), 3 * 1024 * 1024)
    assert exc.value.status_code == 413
    assert reads == 4
