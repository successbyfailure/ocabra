"""Imagen al estilo OpenAI (2026-11-15).

- Un router con destinos que hacen una sola cosa: el endpoint decide
  (``required_capability``) y el router salta los que no la tienen.
- Los nombres de OpenAI (``gpt-image-1``, ``dall-e-*``) se resuelven por el
  modo compatibilidad, con permisos sobre el nombre pedido.
- ``image[]`` en ediciones, ``url`` por defecto con dall-e y ``quality`` →
  pasos de inferencia.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.api._deps_auth import UserContext
from ocabra.api.openai import images as images_mod
from ocabra.backends.base import BackendCapabilities
from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus
from ocabra.core.router_resolver import NoRoutableTargetError, RouterResolver


@dataclass
class _Perfil:
    profile_id: str
    base_model_id: str | None
    enabled: bool = True
    routing_targets: list[str] | None = None
    load_overrides: dict | None = None
    profile_kind: str = "execution"


def _estado(model_id: str, *, gen: bool, edit: bool, status=ModelStatus.CONFIGURED):
    return ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="diffusers",
        status=status,
        load_policy=LoadPolicy.ON_DEMAND,
        capabilities=BackendCapabilities(image_generation=gen, image_editing=edit),
    )


class _Registro:
    def __init__(self, *perfiles):
        self._p = {p.profile_id: p for p in perfiles}

    async def get(self, pid):
        return self._p.get(pid)


class _Gestor:
    def __init__(self, estados):
        self._e = estados

    async def get_state(self, mid):
        return self._e.get(mid)

    def is_busy(self, mid):
        return False

    def any_busy_worker_except(self, target_id, *, also_exclude=()):
        return False


def _resolver(*, edit_cargado: bool = False):
    router = _Perfil(
        "recomendado-imagen",
        None,
        routing_targets=["gen", "edit"],
        profile_kind="router",
    )
    estados = {
        "diffusers/gen": _estado("diffusers/gen", gen=True, edit=False),
        "diffusers/edit": _estado(
            "diffusers/edit",
            gen=False,
            edit=True,
            status=ModelStatus.LOADED if edit_cargado else ModelStatus.CONFIGURED,
        ),
    }
    registro = _Registro(
        router, _Perfil("gen", "diffusers/gen"), _Perfil("edit", "diffusers/edit")
    )
    return router, RouterResolver(profile_registry=registro, model_manager=_Gestor(estados))


@pytest.mark.asyncio
async def test_el_endpoint_decide_el_destino():
    router, resolver = _resolver()
    gen = await resolver.pick(router, required_capability="image_generation")
    edit = await resolver.pick(router, required_capability="image_editing")
    assert gen.target_profile_id == "gen"
    assert edit.target_profile_id == "edit"


@pytest.mark.asyncio
async def test_un_destino_caliente_que_no_sirve_no_gana():
    """Antes, "caliente gana": con el editor cargado, una generación iba a él."""
    router, resolver = _resolver(edit_cargado=True)
    gen = await resolver.pick(router, required_capability="image_generation")
    assert gen.target_profile_id == "gen"
    assert any(c.outcome == "not_capable" for c in gen.considered)


@pytest.mark.asyncio
async def test_sin_capacidad_pedida_se_comporta_como_antes():
    router, resolver = _resolver(edit_cargado=True)
    assert (await resolver.pick(router)).target_profile_id == "edit"


@pytest.mark.asyncio
async def test_ningun_destino_capaz_da_error():
    router, resolver = _resolver()
    with pytest.raises(NoRoutableTargetError):
        await resolver.pick(router, required_capability="music_generation")


# ── modo compatibilidad ────────────────────────────────────────────────


class _Compat:
    def __init__(self, pins):
        self._pins = pins

    def resolve(self, name):
        return self._pins.get(name or "dall-e-2")


def _peticion(pins):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(openai_image_compat=_Compat(pins))))


def _usuario(*modelos):
    return UserContext(
        user_id=None,
        username="u",
        role="user",
        group_ids=[],
        accessible_model_ids=set(modelos),
        is_anonymous=False,
    )


def test_nombre_de_openai_resuelve_a_su_perfil_con_permiso_del_nombre():
    pins = {"gpt-image-1": "recomendado-imagen"}
    model_id, user = images_mod._resolve_openai_image_compat(
        _peticion(pins), _usuario("gpt-image-1"), "gpt-image-1"
    )
    assert model_id == "recomendado-imagen"
    assert user.can_access_model("recomendado-imagen")
    assert not user.can_access_model("qwen-image-2.1")


def test_sin_permiso_sobre_el_nombre_de_openai_es_404():
    with pytest.raises(Exception) as exc:
        images_mod._resolve_openai_image_compat(
            _peticion({"gpt-image-1": "recomendado-imagen"}), _usuario(), "gpt-image-1"
        )
    assert getattr(exc.value, "status_code", None) == 404


def test_sin_modelo_se_busca_como_dall_e_2():
    model_id, _ = images_mod._resolve_openai_image_compat(
        _peticion({"dall-e-2": "recomendado-imagen"}), _usuario("dall-e-2"), ""
    )
    assert model_id == "recomendado-imagen"


def test_un_nombre_propio_pasa_sin_tocar():
    user = _usuario("qwen-image-2.1")
    model_id, mismo = images_mod._resolve_openai_image_compat(
        _peticion({"gpt-image-1": "recomendado-imagen"}), user, "qwen-image-2.1"
    )
    assert model_id == "qwen-image-2.1" and mismo is user


@pytest.mark.parametrize(
    ("modelo", "explicito", "esperado"),
    [
        ("dall-e-3", None, "url"),
        ("dall-e-2", None, "url"),
        ("", None, "url"),
        ("gpt-image-1", None, "b64_json"),
        ("qwen-image-2.1", None, "b64_json"),
        ("dall-e-3", "b64_json", "b64_json"),
    ],
)
def test_formato_por_defecto_como_openai(modelo, explicito, esperado):
    assert images_mod._default_response_format(modelo, explicito) == esperado


@pytest.mark.parametrize(
    ("cliente", "esperado"),
    [
        ({"quality": "low"}, 10),
        ({"quality": "high"}, 30),
        ({"quality": "hd"}, 30),
        ({"quality": "medium"}, 20),
        ({"quality": "auto"}, 20),
        ({"quality": "high", "num_inference_steps": 7}, 20),  # gana el del cliente
        ({}, 20),
    ],
)
def test_quality_escala_los_pasos_del_perfil(cliente, esperado):
    merged = {"num_inference_steps": 20}
    images_mod._apply_quality(merged, cliente)
    assert merged["num_inference_steps"] == esperado


# ── image[] en ediciones ───────────────────────────────────────────────


def test_edicion_acepta_image_corchetes(monkeypatch):
    from fastapi.testclient import TestClient

    from tests.test_openai_api import FakeProfileRegistry, _make_app, _make_test_profile

    estado = _estado("diffusers/edit", gen=False, edit=True, status=ModelStatus.LOADED)
    app = _make_app(model_state=estado, worker_result={"images": [{"b64_json": "aGk="}]})
    app.state.profile_registry = FakeProfileRegistry(
        [_make_test_profile("diffusers/edit", "diffusers/edit", category="image")]
    )
    app.state.worker_pool.forward_request = AsyncMock(
        return_value={"images": [{"b64_json": "aGk="}]}
    )
    monkeypatch.setattr(images_mod, "try_proxy_multipart", AsyncMock(return_value=None))

    resp = TestClient(app).post(
        "/v1/images/edits",
        data={"model": "diffusers/edit", "prompt": "pon un sombrero"},
        files=[
            ("image[]", ("a.png", b"\x89PNG-a", "image/png")),
            ("image[]", ("b.png", b"\x89PNG-b", "image/png")),
        ],
    )
    assert resp.status_code == 200, resp.text
    cuerpo = app.state.worker_pool.forward_request.await_args.args[2]
    assert isinstance(cuerpo["image_b64"], list) and len(cuerpo["image_b64"]) == 2


def test_edicion_sin_imagen_es_400(monkeypatch):
    from fastapi.testclient import TestClient

    from tests.test_openai_api import _make_app

    app = _make_app()
    monkeypatch.setattr(images_mod, "try_proxy_multipart", AsyncMock(return_value=None))
    resp = TestClient(app).post(
        "/v1/images/edits", data={"model": "test-model", "prompt": "x"}
    )
    assert resp.status_code == 400


def test_v1_models_el_router_de_imagen_anuncia_generar_y_editar(monkeypatch):
    """Antes copiaba las capacidades de su primer destino (solo genera)."""
    from fastapi.testclient import TestClient

    from tests.test_openai_api import FakeProfileRegistry, _make_app, _make_test_profile

    estados = {
        "gen": _estado("gen", gen=True, edit=False),
        "edit": _estado("edit", gen=False, edit=True),
    }
    app = _make_app()
    app.state.model_manager.get_state = AsyncMock(side_effect=lambda mid: estados.get(mid))
    router = SimpleNamespace(
        profile_id="recomendado-imagen",
        profile_kind="router",
        base_model_id=None,
        display_name="Imagen",
        description=None,
        category="image",
        enabled=True,
        is_default=False,
        routing_targets=["gen", "edit"],
        router_context=None,
    )
    app.state.profile_registry = FakeProfileRegistry(
        [_make_test_profile("gen", "gen", "image"), _make_test_profile("edit", "edit", "image"), router]
    )
    data = TestClient(app).get("/v1/models").json()["data"]
    caps = next(m for m in data if m["id"] == "recomendado-imagen")["ocabra"]["capabilities"]
    assert caps["image_generation"] is True
    assert caps["image_editing"] is True
    # Lo que el router no filtra por endpoint sigue siendo el del primer destino.
    assert caps["chat"] is False
