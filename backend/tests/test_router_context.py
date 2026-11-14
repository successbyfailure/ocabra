"""Contexto que anuncia un router: max (histórico), common o fixed.

2026-11-14: ``recomendado-sota`` anunciaba 1M porque uno de sus nueve
destinos (nemotron-3-nano) tiene un preset de 1M; los otros ocho llegan a
262k como mucho.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from ocabra.core.router_resolver import router_advertised_context
from ocabra.schemas.profiles import ProfileUpdate, RouterContextPolicy, RouterCreate
from tests.test_profiles import _make_profile, _make_session

SOTA = [262144, 262144, 262144, 229376, 262144, 245760, 262144, 131072, 1048576]


def test_por_defecto_es_el_maximo():
    assert router_advertised_context(None, SOTA) == 1048576
    assert router_advertised_context({"mode": "max"}, SOTA) == 1048576


def test_common_ignora_el_destino_atipico():
    assert router_advertised_context({"mode": "common"}, SOTA) == 262144


@pytest.mark.parametrize(
    ("valores", "esperado"),
    [
        ([1048576], 1048576),
        ([1048576, 262144], 262144),  # la mayoría de 2 son los 2
        ([1048576, 262144, 131072], 262144),
        ([1048576, 1048576, 131072], 1048576),
        ([0, 262144, 0], 262144),  # destinos sin contexto conocido no cuentan
        ([], 0),
    ],
)
def test_common_es_lo_que_alcanza_mas_de_la_mitad(valores, esperado):
    assert router_advertised_context({"mode": "common"}, valores) == esperado


def test_fixed_nunca_por_encima_de_lo_que_algun_destino_sirve():
    assert router_advertised_context({"mode": "fixed", "tokens": 200000}, SOTA) == 200000
    assert router_advertised_context({"mode": "fixed", "tokens": 4_000_000}, SOTA) == 1048576


def test_fixed_exige_tokens():
    with pytest.raises(ValidationError):
        RouterContextPolicy(mode="fixed")
    with pytest.raises(ValidationError):
        RouterContextPolicy(mode="fixed", tokens=0)
    with pytest.raises(ValidationError):
        RouterContextPolicy(mode="median")
    assert RouterContextPolicy(mode="fixed", tokens=262144).tokens == 262144


def test_los_esquemas_de_router_lo_aceptan():
    body = RouterCreate(
        profile_id="r", routing_targets=["a"], router_context={"mode": "common"}
    )
    assert body.router_context.mode == "common"
    patch = ProfileUpdate(router_context=None).model_dump(exclude_unset=True)
    assert patch == {"router_context": None}


@pytest.mark.asyncio
async def test_un_perfil_de_ejecucion_no_puede_declararlo():
    from ocabra.core.profile_registry import ProfileRegistry

    registry = ProfileRegistry()
    registry._profiles["p"] = _make_profile(profile_id="p")
    with pytest.raises(ValueError, match="router_context"):
        await registry.create(
            _make_session(),
            profile_id="otro",
            base_model_id="vllm/my-model",
            router_context={"mode": "common"},
        )


@pytest.mark.asyncio
async def test_el_router_lo_guarda():
    from ocabra.core.profile_registry import ProfileRegistry

    registry = ProfileRegistry()
    registry._profiles["target"] = _make_profile(profile_id="target")
    profile = await registry.create(
        _make_session(),
        profile_id="recomendado",
        profile_kind="router",
        base_model_id=None,
        routing_targets=["target"],
        router_context={"mode": "fixed", "tokens": 262144},
    )
    assert profile.router_context == {"mode": "fixed", "tokens": 262144}


@pytest.mark.parametrize(
    ("politica", "esperado"),
    [(None, 1048576), ({"mode": "common"}, 262144), ({"mode": "fixed", "tokens": 200000}, 200000)],
)
def test_v1_models_anuncia_el_contexto_segun_la_politica(monkeypatch, politica, esperado):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    import ocabra.api.openai.models as models_mod
    from ocabra.backends.base import BackendCapabilities
    from ocabra.core.model_manager import LoadPolicy, ModelState, ModelStatus
    from tests.test_openai_api import FakeProfileRegistry, _make_app, _make_test_profile

    ventanas = {"grande": 1048576, "media-a": 262144, "media-b": 262144}
    estados = {
        mid: ModelState(
            model_id=mid,
            display_name=mid,
            backend_type="llama_cpp",
            status=ModelStatus.CONFIGURED,
            load_policy=LoadPolicy.ON_DEMAND,
            capabilities=BackendCapabilities(chat=True, streaming=True, context_length=ctx),
        )
        for mid, ctx in ventanas.items()
    }

    async def _contexto(profile, model_manager, base_state):
        ctx = ventanas[profile.profile_id]
        return {"model_max": ctx, "effective_max": ctx, "default": ctx, "active": None}

    monkeypatch.setattr(models_mod, "_context_metadata", _contexto)
    app = _make_app()
    app.state.model_manager.get_state = AsyncMock(side_effect=lambda mid: estados.get(mid))
    perfiles = [_make_test_profile(mid, mid) for mid in ventanas]
    router = SimpleNamespace(
        profile_id="recomendado-sota",
        profile_kind="router",
        base_model_id=None,
        display_name="SOTA",
        description=None,
        category="llm",
        enabled=True,
        is_default=False,
        routing_targets=["media-a", "grande", "media-b"],
        router_context=politica,
    )
    app.state.profile_registry = FakeProfileRegistry([*perfiles, router])

    data = TestClient(app).get("/v1/models").json()["data"]
    entrada = next(m for m in data if m["id"] == "recomendado-sota")
    assert entrada["context_length"] == esperado
    assert entrada["ocabra"]["capabilities"]["context_length"] == esperado
    assert entrada["ocabra"]["context"]["default"] <= esperado
    assert entrada["ocabra"]["capabilities"]["recommended_input_tokens"] < esperado
