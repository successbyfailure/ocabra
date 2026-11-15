"""Visión (y demás capacidades de chat) en los routers — 2026-11-15.

``recomendado-sota`` elegía destino sin mirar si la petición llevaba
imágenes: con un destino sin visión cargado, la respuesta era 400 aunque
otro destino de la lista sí podía. Y anunciaba ``vision: false`` porque
copiaba las capacidades de su primer destino.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.api.openai._deps import chat_required_capabilities, ollama_required_capabilities
from ocabra.api.openai.chat import _detect_message_content_kinds
from ocabra.backends.base import BackendCapabilities
from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus
from ocabra.core.router_resolver import RouterResolver

IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


@dataclass
class _Perfil:
    profile_id: str
    base_model_id: str | None
    enabled: bool = True
    routing_targets: list[str] | None = None
    load_overrides: dict | None = None
    profile_kind: str = "execution"


def _estado(mid, *, vision, tools=True, status=ModelStatus.CONFIGURED):
    return ModelState(
        model_id=mid,
        display_name=mid,
        backend_type="llama_cpp",
        status=status,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(chat=True, tools=tools, vision=vision),
    )


class _Registro:
    def __init__(self, *p):
        self._p = {x.profile_id: x for x in p}

    async def get(self, pid):
        return self._p.get(pid)


class _Gestor:
    def __init__(self, e):
        self._e = e

    async def get_state(self, mid):
        return self._e.get(mid)

    def is_busy(self, mid):
        return False

    def any_busy_worker_except(self, target_id, *, also_exclude=()):
        return False


def _sota():
    router = _Perfil("sota", None, routing_targets=["ornith", "gemma"], profile_kind="router")
    estados = {
        # El destino sin visión está cargado: antes ganaba siempre.
        "m/ornith": _estado("m/ornith", vision=False, status=ModelStatus.LOADED),
        "m/gemma": _estado("m/gemma", vision=True),
    }
    reg = _Registro(router, _Perfil("ornith", "m/ornith"), _Perfil("gemma", "m/gemma"))
    return router, RouterResolver(profile_registry=reg, model_manager=_Gestor(estados))


@pytest.mark.asyncio
async def test_con_imagen_salta_el_destino_caliente_sin_vision():
    router, resolver = _sota()
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "¿qué es?"}, IMG]}]}
    wanted = chat_required_capabilities(body, _detect_message_content_kinds(body["messages"]))
    assert wanted == ("vision",)
    elegido = await resolver.pick(router, request_body=body, required_capability=wanted)
    assert elegido.target_profile_id == "gemma"


@pytest.mark.asyncio
async def test_sin_imagen_gana_el_caliente_como_siempre():
    router, resolver = _sota()
    body = {"messages": [{"role": "user", "content": "hola"}]}
    wanted = chat_required_capabilities(body, _detect_message_content_kinds(body["messages"]))
    assert wanted == ()
    elegido = await resolver.pick(router, request_body=body, required_capability=wanted or None)
    assert elegido.target_profile_id == "ornith"


@pytest.mark.asyncio
async def test_varias_capacidades_a_la_vez():
    router = _Perfil("r", None, routing_targets=["a", "b", "c"], profile_kind="router")
    estados = {
        "m/a": _estado("m/a", vision=True, tools=False, status=ModelStatus.LOADED),
        "m/b": _estado("m/b", vision=False, tools=True, status=ModelStatus.LOADED),
        "m/c": _estado("m/c", vision=True, tools=True),
    }
    reg = _Registro(router, _Perfil("a", "m/a"), _Perfil("b", "m/b"), _Perfil("c", "m/c"))
    resolver = RouterResolver(profile_registry=reg, model_manager=_Gestor(estados))
    elegido = await resolver.pick(router, required_capability=("tools", "vision"))
    assert elegido.target_profile_id == "c"


def test_capacidades_de_un_cuerpo_openai():
    msgs = [
        {"role": "user", "content": [IMG, {"type": "input_audio", "input_audio": {}}]},
    ]
    body = {"messages": msgs, "tools": [{"type": "function"}]}
    assert chat_required_capabilities(body, _detect_message_content_kinds(msgs)) == (
        "tools",
        "vision",
        "audio_input",
    )


@pytest.mark.parametrize(
    ("body", "esperado"),
    [
        ({"messages": [{"role": "user", "content": "x", "images": ["AAAA"]}]}, ("vision",)),
        ({"prompt": "x", "images": ["AAAA"]}, ("vision",)),
        ({"messages": [{"role": "user", "content": "x"}], "tools": [{}]}, ("tools",)),
        ({"messages": [{"role": "user", "content": "x"}]}, ()),
    ],
)
def test_capacidades_de_un_cuerpo_ollama(body, esperado):
    assert ollama_required_capabilities(body) == esperado


def test_v1_models_el_router_anuncia_vision_si_algun_destino_la_tiene():
    from fastapi.testclient import TestClient

    from tests.test_openai_api import FakeProfileRegistry, _make_app, _make_test_profile

    estados = {"ornith": _estado("ornith", vision=False), "gemma": _estado("gemma", vision=True)}
    app = _make_app()
    app.state.model_manager.get_state = AsyncMock(side_effect=lambda mid: estados.get(mid))
    router = SimpleNamespace(
        profile_id="recomendado-sota",
        profile_kind="router",
        base_model_id=None,
        display_name="SOTA",
        description=None,
        category="llm",
        enabled=True,
        is_default=False,
        routing_targets=["ornith", "gemma"],
        router_context=None,
    )
    app.state.profile_registry = FakeProfileRegistry(
        [_make_test_profile("ornith", "ornith"), _make_test_profile("gemma", "gemma"), router]
    )
    data = TestClient(app).get("/v1/models").json()["data"]
    entrada = next(m for m in data if m["id"] == "recomendado-sota")
    assert entrada["ocabra"]["capabilities"]["vision"] is True
    assert "vision" in entrada["capabilities"]  # lista estilo Ollama (OpenWebUI)


@pytest.mark.asyncio
async def test_ningun_destino_con_vision_es_400_no_503():
    from fastapi import HTTPException

    from ocabra.api.openai._deps import router_pick_error
    from ocabra.core.router_resolver import NoCapableTargetError, NoRoutableTargetError

    router = _Perfil("r", None, routing_targets=["a"], profile_kind="router")
    reg = _Registro(router, _Perfil("a", "m/a"))
    resolver = RouterResolver(
        profile_registry=reg, model_manager=_Gestor({"m/a": _estado("m/a", vision=False)})
    )
    with pytest.raises(NoCapableTargetError) as exc:
        await resolver.pick(router, required_capability=("vision",))
    http = router_pick_error(exc.value)
    assert isinstance(http, HTTPException) and http.status_code == 400
    assert router_pick_error(NoRoutableTargetError("x")).status_code == 503


def test_chat_streaming_con_imagen_a_router_sin_vision_es_400(monkeypatch):
    """En streaming el error del router se escapaba como 500."""
    from fastapi.testclient import TestClient

    from tests.test_openai_api import FakeProfileRegistry, _make_app, _make_test_profile

    estados = {"a": _estado("a", vision=False, status=ModelStatus.LOADED)}
    app = _make_app()
    app.state.model_manager.get_state = AsyncMock(side_effect=lambda mid: estados.get(mid))
    router = SimpleNamespace(
        profile_id="r",
        profile_kind="router",
        base_model_id=None,
        display_name="r",
        description=None,
        category="llm",
        enabled=True,
        is_default=False,
        routing_targets=["a"],
        router_context=None,
        request_defaults={},
        load_overrides=None,
    )
    registro = FakeProfileRegistry([_make_test_profile("a", "a"), router])
    app.state.profile_registry = registro
    app.state.router_resolver = RouterResolver(
        profile_registry=registro, model_manager=app.state.model_manager
    )
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "r",
            "stream": True,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}, IMG]}],
        },
    )
    assert resp.status_code == 400, resp.text
    assert "model_not_capable" in resp.text


@pytest.mark.asyncio
async def test_el_worker_derivado_comparte_sus_capacidades_con_el_base(monkeypatch):
    """ornith-1.5:35b solo lo sirve su worker de preset (``::hash``): el base
    nunca carga y se quedaba con ``vision: false`` deducido por el nombre."""
    from ocabra.core import model_manager as mm_mod
    from ocabra.core.model_manager import ModelManager, resolve_display_capabilities

    guardados = []
    monkeypatch.setattr(mm_mod, "set_key", AsyncMock(side_effect=lambda k, v: guardados.append(k)))
    mm = ModelManager(worker_pool=None)
    base = _estado("llamacpp_mks/ornith", vision=False, tools=False)
    base.capabilities = BackendCapabilities()  # nunca cargado: sin señal
    derivado = _estado("llamacpp_mks/ornith::abc", vision=True, status=ModelStatus.LOADED)
    derivado.capabilities = BackendCapabilities(
        chat=True, tools=True, vision=True, reasoning=True, context_length=65536
    )
    mm._states = {base.model_id: base, derivado.model_id: derivado}

    await mm._share_capabilities_with_base(derivado)

    caps, fuente = resolve_display_capabilities(base)
    assert fuente == "cached"
    assert caps["vision"] is True and caps["tools"] is True and caps["reasoning"] is True
    assert base.capabilities.context_length == 0  # lo numérico es del preset
    assert guardados == ["model:state:llamacpp_mks/ornith"]


@pytest.mark.asyncio
async def test_un_base_cargado_conserva_sus_capacidades(monkeypatch):
    from ocabra.core import model_manager as mm_mod
    from ocabra.core.model_manager import ModelManager

    monkeypatch.setattr(mm_mod, "set_key", AsyncMock())
    mm = ModelManager(worker_pool=None)
    base = _estado("m/x", vision=True, status=ModelStatus.LOADED)
    derivado = _estado("m/x::abc", vision=False, status=ModelStatus.LOADED)
    mm._states = {base.model_id: base, derivado.model_id: derivado}
    await mm._share_capabilities_with_base(derivado)
    assert base.capabilities.vision is True


@pytest.mark.asyncio
async def test_exclude_targets_salta_el_destino_caliente_que_ya_fallo():
    router = _Perfil("r", None, routing_targets=["a", "b"], profile_kind="router")
    estados = {
        "m/a": _estado("m/a", vision=True, status=ModelStatus.LOADED),
        "m/b": _estado("m/b", vision=True),
    }
    reg = _Registro(router, _Perfil("a", "m/a"), _Perfil("b", "m/b"))
    resolver = RouterResolver(profile_registry=reg, model_manager=_Gestor(estados))
    assert (await resolver.pick(router, required_capability="vision")).target_profile_id == "a"
    otro = await resolver.pick(router, required_capability="vision", exclude_targets={"a"})
    assert otro.target_profile_id == "b"


def test_vision_no_room_sin_router_es_503_claro():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import ocabra.main as main_mod
    from ocabra.core.worker_pool import VisionNoRoomError

    app = FastAPI()
    app.add_exception_handler(VisionNoRoomError, main_mod._vision_no_room_handler)

    @app.get("/x")
    async def _x():
        raise VisionNoRoomError("The vision projector needs ~1232 MB on GPU 1")

    resp = TestClient(app).get("/x")
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "vision_no_room"


def test_stream_error_details_vision_no_room():
    from ocabra.api.openai._deps import stream_error_details
    from ocabra.core.worker_pool import VisionNoRoomError

    assert stream_error_details(VisionNoRoomError("x"))[1] == "vision_no_room"


def test_chat_router_reintenta_en_otro_destino_tras_vision_no_room():
    from fastapi.testclient import TestClient

    import ocabra.main as main_mod
    from ocabra.core.worker_pool import VisionNoRoomError
    from tests.test_openai_api import FakeProfileRegistry, _make_app, _make_test_profile

    estados = {
        "a": _estado("a", vision=True, status=ModelStatus.LOADED),
        "b": _estado("b", vision=True, status=ModelStatus.LOADED),
    }
    app = _make_app()
    app.add_exception_handler(VisionNoRoomError, main_mod._vision_no_room_handler)
    mm = app.state.model_manager
    mm.get_state = AsyncMock(side_effect=lambda mid: estados.get(mid))
    mm.list_states = AsyncMock(return_value=list(estados.values()))
    router = SimpleNamespace(
        profile_id="r",
        profile_kind="router",
        base_model_id=None,
        display_name="r",
        description=None,
        category="llm",
        enabled=True,
        is_default=False,
        routing_targets=["a", "b"],
        router_context=None,
        request_defaults={},
        load_overrides=None,
    )
    registro = FakeProfileRegistry([_make_test_profile("a", "a"), _make_test_profile("b", "b"), router])
    app.state.profile_registry = registro
    app.state.router_resolver = RouterResolver(profile_registry=registro, model_manager=mm)

    llamadas = []

    async def _forward(worker_key, path, body, **kw):
        llamadas.append(worker_key)
        if worker_key == "a":
            raise VisionNoRoomError("no room on GPU 1")
        return {"choices": [{"message": {"role": "assistant", "content": "Rojo"}}]}

    app.state.worker_pool.forward_request = AsyncMock(side_effect=_forward)
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "r",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}, IMG]}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert llamadas == ["a", "b"]
