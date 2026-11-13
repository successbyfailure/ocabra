"""Descargar un modelo con presets descarga su worker derivado (2026-11-13).

Con presets, el modelo corre como ``modelo::hash``. Descargar el id base
mataba ese proceso a través del estado base y dejaba el derivado en LOADED con
el pid muerto: el vigilante lo tomaba por una caída y lo reiniciaba a los 5 s
con su configuración vieja. Así se perdió un cambio de plantilla en gemma4:26b.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.api.internal import models as models_api
from ocabra.core.model_manager import ModelStatus


def _client(states):
    app = FastAPI()
    app.include_router(models_api.router, prefix="/ocabra")
    unloaded: list[str] = []

    async def get_state(mid):
        return states.get(mid)

    async def list_states(include_derived=False):
        return [s for k, s in states.items() if include_derived or "::" not in k]

    async def unload(mid, reason=None):
        unloaded.append(mid)
        states[mid].status = ModelStatus.UNLOADED

    app.state.model_manager = SimpleNamespace(
        get_state=get_state, list_states=list_states, unload=AsyncMock(side_effect=unload)
    )
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id=None,
        username="t",
        role="system_admin",
        group_ids=[],
        accessible_model_ids=set(),
        is_anonymous=False,
    )
    return TestClient(app), unloaded


def _state(mid, status):
    return SimpleNamespace(model_id=mid, status=status, to_dict=lambda: {"model_id": mid})


def test_unloading_the_base_unloads_its_loaded_derived_worker():
    states = {
        "llamacpp_mks/gemma4:26b": _state("llamacpp_mks/gemma4:26b", ModelStatus.LOADED),
        "llamacpp_mks/gemma4:26b::cb3f": _state(
            "llamacpp_mks/gemma4:26b::cb3f", ModelStatus.LOADED
        ),
        "llamacpp_mks/gemma4:26b::dead": _state(
            "llamacpp_mks/gemma4:26b::dead", ModelStatus.UNLOADED
        ),
        "llamacpp_mks/gemma4:12b::x": _state("llamacpp_mks/gemma4:12b::x", ModelStatus.LOADED),
    }
    client, unloaded = _client(states)

    assert client.post("/ocabra/models/llamacpp_mks/gemma4:26b/unload").status_code == 200

    # El derivado cargado primero, por su id; ni el ya descargado ni el de otro modelo.
    assert unloaded == ["llamacpp_mks/gemma4:26b::cb3f", "llamacpp_mks/gemma4:26b"]


def test_unloading_a_derived_worker_only_unloads_that_one():
    states = {
        "llamacpp_mks/gemma4:26b": _state("llamacpp_mks/gemma4:26b", ModelStatus.LOADED),
        "llamacpp_mks/gemma4:26b::cb3f": _state(
            "llamacpp_mks/gemma4:26b::cb3f", ModelStatus.LOADED
        ),
    }
    client, unloaded = _client(states)
    client.post("/ocabra/models/llamacpp_mks/gemma4:26b::cb3f/unload")
    assert unloaded == ["llamacpp_mks/gemma4:26b::cb3f"]
