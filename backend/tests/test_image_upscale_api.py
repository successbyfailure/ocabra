"""``POST /v1/images/upscale``: one photo through the video upscaler models."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user

MODULE = "ocabra.api.openai.image_upscale"


class _Status:
    value = "loaded"


def _state(backend_type="upscaler", image_upscaling=True):
    return SimpleNamespace(
        model_id=f"{backend_type}/m",
        backend_type=backend_type,
        backend_model_id="m",
        status=_Status(),
        capabilities=SimpleNamespace(image_upscaling=image_upscaling),
    )


def _app(upscale=None, *, role="user", registry_profile=None, manager_state=None):
    from ocabra.api.openai import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="u1", username="vhs", role=role
    )
    backend = MagicMock()
    backend.health_check = AsyncMock(return_value=True)
    backend.upscale_image = upscale or AsyncMock(
        return_value=(
            b"PNGDATA",
            "image/png",
            {"x-ocabra-delivered-resolution": "40x24", "x-ocabra-tiles": "1"},
        )
    )
    pool = MagicMock()
    pool.get_worker.return_value = object()
    pool.get_backend = AsyncMock(return_value=backend)
    mm = MagicMock()
    mm.begin_request.return_value = "r1"
    mm.get_state = AsyncMock(return_value=manager_state)
    registry = MagicMock()
    registry.get = AsyncMock(return_value=registry_profile)
    app.state.worker_pool = pool
    app.state.model_manager = mm
    app.state.profile_registry = registry
    app.state.router_resolver = MagicMock(is_router=MagicMock(return_value=False))
    return app, backend


_PROFILE = SimpleNamespace(
    profile_id="realesrgan-x4plus",
    base_model_id="upscaler/realesrgan-x4plus",
    load_overrides=None,
)


def _post(app, *, state=None, data=None, files=None):
    data = {"model": "realesrgan-x4plus", **(data or {})}
    files = files if files is not None else {"image": ("p.jpg", b"\xff\xd8jpeg", "image/jpeg")}
    with (
        patch(
            f"{MODULE}.resolve_profile",
            AsyncMock(return_value=(_PROFILE, state or _state())),
        ) as resolve,
        patch(f"{MODULE}.build_model_status_headers", AsyncMock(return_value={})),
    ):
        resp = TestClient(app, raise_server_exceptions=False).post(
            "/v1/images/upscale", files=files, data=data
        )
    return resp, resolve


def _code(resp):
    return resp.json()["detail"]["error"]["code"]


def test_devuelve_la_imagen_con_su_content_type_y_las_estadisticas():
    app, backend = _app()
    resp, resolve = _post(app, data={"scale": "2", "output_format": "jpg"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == b"PNGDATA"
    assert resp.headers["x-ocabra-delivered-resolution"] == "40x24"
    # Routers resolve with the capability, like the other image routes.
    assert resolve.await_args.kwargs["required_capability"] == "image_upscaling"
    assert resolve.await_args.kwargs["router_resolver"] is app.state.router_resolver
    kw = backend.upscale_image.await_args.kwargs
    assert kw["output_format"] == "jpeg"  # jpg is an alias
    assert kw["scale"] == 2.0
    assert kw["content_type"] == "image/jpeg"
    assert backend.upscale_image.await_args.args == ("m", b"\xff\xd8jpeg")


def test_b64_json_devuelve_el_formato_de_openai():
    app, _backend = _app()
    resp, _ = _post(app, data={"response_format": "b64_json"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    item = resp.json()["data"][0]
    assert base64.b64decode(item["b64_json"]) == b"PNGDATA"
    assert item["output_format"] == "png"
    assert item["size"] == "40x24"
    assert isinstance(resp.json()["created"], int)


@pytest.mark.parametrize(
    "data, files, param, code",
    [
        ({}, {"other": ("x", b"x", "text/plain")}, "image", "missing_required_parameter"),
        ({"model": ""}, None, "model", "missing_required_parameter"),
        ({"output_format": "gif"}, None, "output_format", "invalid_output_format"),
        ({"response_format": "url"}, None, "response_format", "invalid_response_format"),
        ({"scale": "0"}, None, "scale", "invalid_value"),
        ({"target_height": "-3"}, None, "target_height", "invalid_value"),
        ({"output_quality": "101"}, None, "output_quality", "invalid_value"),
        ({}, {"image": ("p.png", b"", "image/png")}, "image", "empty_image"),
    ],
)
def test_peticiones_invalidas_dan_400_sin_tocar_el_modelo(data, files, param, code):
    app, backend = _app()
    resp, resolve = _post(app, data=data, files=files)

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error"]["param"] == param
    # Con code: VHS solo podía reconocer los errores por el status (20/11).
    assert _code(resp) == code
    resolve.assert_not_awaited()
    backend.upscale_image.assert_not_awaited()


def test_un_modelo_de_otro_backend_da_400_antes_de_cargarlo():
    chat = SimpleNamespace(
        profile_id="qwen3.8:27b", base_model_id="llama_cpp/qwen3.8", load_overrides=None
    )
    app, backend = _app(
        role="system_admin", registry_profile=chat, manager_state=_state("llama_cpp", False)
    )
    resp, resolve = _post(app, data={"model": "qwen3.8:27b"})

    assert resp.status_code == 400
    assert _code(resp) == "model_not_capable"
    resolve.assert_not_awaited()  # nothing was loaded


def test_sin_acceso_al_modelo_no_se_revela_y_decide_resolve_profile():
    chat = SimpleNamespace(
        profile_id="qwen3.8:27b", base_model_id="llama_cpp/qwen3.8", load_overrides=None
    )
    app, _ = _app(registry_profile=chat, manager_state=_state("llama_cpp", False))
    resp, resolve = _post(app, state=_state("llama_cpp", False), data={"model": "qwen3.8:27b"})

    resolve.assert_awaited_once()
    assert resp.status_code == 400 and _code(resp) == "model_not_capable"


def test_flashvsr_solo_hace_video_y_da_model_not_capable():
    app, backend = _app()
    resp, _ = _post(app, state=_state("flashvsr", image_upscaling=False))

    assert resp.status_code == 400
    assert _code(resp) == "model_not_capable"
    backend.upscale_image.assert_not_awaited()


@pytest.mark.parametrize(
    "status, code", [(400, "invalid_image"), (413, "image_too_large"), (422, "invalid_value")]
)
def test_los_errores_de_cliente_del_worker_se_propagan(status, code):
    req = httpx.Request("POST", "http://worker/upscale/image")
    err = httpx.HTTPStatusError(
        "x",
        request=req,
        response=httpx.Response(status, json={"detail": "La imagen es enorme"}, request=req),
    )
    app, _ = _app(AsyncMock(side_effect=err))
    resp, _ = _post(app)

    assert resp.status_code == status
    assert resp.json()["detail"]["error"]["message"] == "La imagen es enorme"
    assert _code(resp) == code
    app.state.model_manager.begin_request.assert_called_once()
    app.state.model_manager.end_request.assert_called_once_with("upscaler/realesrgan-x4plus", "r1")


def test_el_worker_caido_se_recarga_una_vez_y_luego_503():
    app, backend = _app()
    backend.health_check = AsyncMock(return_value=False)
    mm = app.state.model_manager
    mm.unload = AsyncMock()
    mm.load = AsyncMock()
    mm.touch_last_request_at = AsyncMock()
    resp, _ = _post(app)

    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "10"
    mm.load.assert_awaited_once()


def test_se_clasifica_como_image_upscale():
    from ocabra.core.admission import classify_request_kind
    from ocabra.stats.collector import _extract_work_size_meta

    assert classify_request_kind("/v1/images/upscale") == "image_upscale"
    assert _extract_work_size_meta("image_upscale", None, None) == {"n_images": 1}


async def test_capacidades_upscaler_si_flashvsr_no():
    from ocabra.backends.flashvsr_backend import FlashVSRBackend
    from ocabra.backends.upscaler_backend import UpscalerBackend
    from ocabra.core.model_manager import declared_capabilities

    assert (await UpscalerBackend().get_capabilities("m")).image_upscaling is True
    assert (await FlashVSRBackend().get_capabilities("m")).image_upscaling is False
    cold = SimpleNamespace(
        backend_type="upscaler",
        backend_model_id="realesrgan-x4plus",
        model_id="upscaler/realesrgan-x4plus",
        extra_config={},
    )
    with patch(
        "ocabra.core.model_manager.ModelManager._resolve_context_length_fallback",
        return_value=0,
    ):
        caps = declared_capabilities(cold)
    assert caps.image_upscaling and caps.video_upscaling


async def test_el_adaptador_manda_la_foto_al_worker():
    from ocabra.backends.upscaler_backend import UpscalerBackend

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx.Response(
            200,
            content=b"IMG",
            headers={"content-type": "image/webp", "x-ocabra-tiles": "4", "other": "x"},
        )

    backend = UpscalerBackend()
    backend._workers["m"] = SimpleNamespace(port=1234)
    real_client = httpx.AsyncClient
    with patch(
        "ocabra.backends.upscaler_backend.httpx.AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    ):
        body, media_type, stats = await backend.upscale_image(
            "m",
            b"raw",
            filename="a.png",
            content_type="image/png",
            output_format="webp",
            scale=None,
            target_height=720,
        )

    assert (body, media_type, stats) == (b"IMG", "image/webp", {"x-ocabra-tiles": "4"})
    assert seen["url"].startswith("http://127.0.0.1:1234/upscale/image?")
    assert "output_format=webp" in seen["url"] and "target_height=720" in seen["url"]
    assert "scale=" not in seen["url"]  # None is not sent
    assert b"raw" in seen["body"]
