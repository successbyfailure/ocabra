"""``list_states()`` oculta los workers derivados y no el resto.

El filtro se escribio como ``model_id in self._persisted_model_ids``, que
significa "tiene fila en model_configs", no "no es derivado". Los modelos que
``sync_ollama_models`` descubre instalados en Ollama pero sin registrar en
oCabra no tienen fila, asi que desaparecian del catalogo: ``GET /ocabra/models``
sincronizaba el inventario y en la linea siguiente los descartaba.
"""

import asyncio

import pytest

from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState


@pytest.fixture
def mm() -> ModelManager:
    return ModelManager(worker_pool=object())


def _registrar_persistido(mm: ModelManager, model_id: str) -> ModelState:
    """Simula un modelo con fila en ``model_configs``."""
    state = ModelState(
        model_id=model_id,
        display_name=model_id,
        backend_type="llama_cpp",
        load_policy=LoadPolicy.ON_DEMAND,
    )
    mm._states[model_id] = state
    mm._load_locks[model_id] = asyncio.Lock()
    mm._persisted_model_ids.add(model_id)
    return state


async def test_los_modelos_de_ollama_sin_registrar_salen_en_el_listado(mm):
    anadidos = await mm.sync_ollama_models(["gemma3:4b", "llama3.2:3b"])
    assert anadidos == 2

    visibles = sorted(s.model_id for s in await mm.list_states())
    assert visibles == ["ollama/gemma3:4b", "ollama/llama3.2:3b"]


async def test_los_workers_derivados_siguen_ocultos(mm):
    _registrar_persistido(mm, "llama_cpp/gemma4:26b")
    # Un worker derivado por un preset de runtime: mismo canonico + ::<hash>.
    derivado = ModelState(
        model_id="llama_cpp/gemma4:26b::a1b2c3",
        display_name="gemma4:26b (override)",
        backend_type="llama_cpp",
        load_policy=LoadPolicy.ON_DEMAND,
    )
    mm._states[derivado.model_id] = derivado

    visibles = [s.model_id for s in await mm.list_states()]
    assert visibles == ["llama_cpp/gemma4:26b"]

    # Y siguen estando cuando se piden explicitamente.
    con_derivados = sorted(s.model_id for s in await mm.list_states(include_derived=True))
    assert con_derivados == ["llama_cpp/gemma4:26b", "llama_cpp/gemma4:26b::a1b2c3"]


async def test_un_id_gguf_con_doble_colon_no_se_confunde_con_un_derivado(mm):
    """Los ids GGUF llevan ``repo::fichero``, que no es un sufijo de derivado.

    Es la razon por la que el filtro no puede ser un simple "contiene ::".
    """
    gguf = "llama_cpp/unsloth/Qwen3-32B-GGUF::Qwen3-32B-Q4_K_M.gguf"
    _registrar_persistido(mm, gguf)

    visibles = [s.model_id for s in await mm.list_states()]
    assert visibles == [gguf]
