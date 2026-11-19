"""BUG-30: ``::`` es a la vez el ``repo::fichero`` de un GGUF y el sufijo de
preset ``::<hash>`` de un worker. Los sitios del ModelManager que buscaban el
base de un worker partían por el primer/último ``::`` y confundían un
despliegue GGUF con un worker derivado (y el ``repo::fichero::hash`` con el
``repo``). Ahora usan ``strip_worker_suffix``/``is_derived_worker_key``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.backends.base import BackendCapabilities
from ocabra.core import model_manager as mm_module
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelState, ModelStatus

GGUF = "llamacpp_mks/mradermacher/Llama-eus-8B-GGUF::Llama-eus-8B.Q4_K_M"
REPO = "llamacpp_mks/mradermacher/Llama-eus-8B-GGUF"
WORKER = f"{GGUF}::781a0367a8ba"


def _state(model_id: str, **kw) -> ModelState:
    kw.setdefault("status", ModelStatus.CONFIGURED)
    return ModelState(model_id=model_id, display_name=model_id, backend_type="llamacpp_mks", **kw)


def _mm(*states: ModelState) -> ModelManager:
    mm = ModelManager(worker_pool=None)
    for st in states:
        mm._states[st.model_id] = st
    return mm


def test_a_pinned_gguf_deployment_is_autoloaded():
    """``repo::file`` is a deployment, not a derived worker: it was skipped."""
    mm = _mm(_state(GGUF, load_policy=LoadPolicy.PIN))
    assert mm._pinned_needing_load() == [GGUF]


def test_another_gguf_deployment_does_not_satisfy_the_repo_pin():
    """``repo::file`` loaded is not a preset worker of a PIN on ``repo``."""
    mm = _mm(
        _state(REPO, load_policy=LoadPolicy.PIN),
        _state(GGUF, load_policy=LoadPolicy.ON_DEMAND, status=ModelStatus.LOADED),
    )
    assert mm._pinned_needing_load() == [REPO]
    mm._states[WORKER] = _state(WORKER, status=ModelStatus.LOADED)
    assert mm._pin_held_by_derived(GGUF)


@pytest.mark.asyncio
async def test_a_gguf_pin_resolves_its_preset_worker():
    mm = _mm(_state(GGUF, load_policy=LoadPolicy.PIN))
    mm.pin_worker_resolver = AsyncMock(return_value=WORKER)
    assert await mm._pin_load_target(GGUF) == WORKER
    mm.pin_worker_resolver.assert_awaited_once_with(GGUF)


@pytest.mark.asyncio
async def test_a_gguf_preset_worker_shares_capabilities_with_its_own_base(monkeypatch):
    monkeypatch.setattr(mm_module, "set_key", AsyncMock())
    repo = _state(REPO)
    base = _state(GGUF)
    worker = _state(WORKER, status=ModelStatus.LOADED)
    worker.capabilities = BackendCapabilities(chat=True, vision=True)
    mm = _mm(repo, base, worker)

    await mm._share_capabilities_with_base(worker)

    assert base.capabilities.vision is True
    assert repo.capabilities.vision is False  # used to land here


@pytest.mark.asyncio
async def test_a_gguf_deployment_does_not_share_capabilities_with_the_repo(monkeypatch):
    monkeypatch.setattr(mm_module, "set_key", AsyncMock())
    repo = _state(REPO)
    base = _state(GGUF, status=ModelStatus.LOADED)
    base.capabilities = BackendCapabilities(chat=True, vision=True)
    mm = _mm(repo, base)
    await mm._share_capabilities_with_base(base)
    assert repo.capabilities.vision is False


def test_canonical_id_of_an_unknown_key_keeps_the_gguf_file():
    mm = _mm()
    assert mm._canonical_runtime_model_id(GGUF) == GGUF
    assert mm._canonical_runtime_model_id(WORKER) == GGUF
    assert (
        mm._canonical_runtime_model_id("whisper/openai/whisper-large-v3-turbo::diarize")
        == "whisper/openai/whisper-large-v3-turbo"
    )


@pytest.mark.asyncio
async def test_a_gguf_preset_worker_inherits_from_its_own_base():
    base = _state(GGUF)
    base.model_key = "mradermacher/Llama-eus-8B-GGUF::Llama-eus-8B.Q4_K_M"
    base.artifact = {"kind": "gguf", "file": "Llama-eus-8B.Q4_K_M.gguf"}
    mm = _mm(base)
    mm._extract_llama_cpp_fingerprint = AsyncMock(return_value=(None, None, None))
    worker = await mm.add_model(WORKER, "llamacpp_mks", persist=False)
    assert worker.model_key == base.model_key
    assert worker.artifact == base.artifact


@pytest.mark.asyncio
async def test_orphan_purge_keeps_the_state_of_a_gguf_preset_worker(monkeypatch):
    keys = [f"model:state:{WORKER}", "model:state:gone/model::0b2a99a4d66f"]
    deleted: list[str] = []

    async def scan_iter(match=None, count=None):
        for key in keys:
            yield key

    async def delete(key):
        deleted.append(key)
        return 1

    client = SimpleNamespace(scan_iter=scan_iter, delete=delete)
    monkeypatch.setattr("ocabra.redis_client.get_redis", lambda: client)
    mm = _mm(_state(GGUF))
    await mm._purge_orphan_redis_states()
    assert deleted == ["model:state:gone/model::0b2a99a4d66f"]


@pytest.mark.asyncio
async def test_backends_api_counts_two_gguf_files_of_one_repo_as_two_models():
    from ocabra.api.internal.backends import _loaded_by_backend

    states = [
        SimpleNamespace(model_id=GGUF, backend_type="llamacpp_mks", status="loaded"),
        SimpleNamespace(
            model_id=f"{REPO}::Other.Q8_0", backend_type="llamacpp_mks", status="loaded"
        ),
        SimpleNamespace(model_id=WORKER, backend_type="llamacpp_mks", status="loaded"),
    ]

    async def list_states(include_derived=False):
        return states

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(model_manager=SimpleNamespace(list_states=list_states))
        )
    )
    assert await _loaded_by_backend(request) == {"llamacpp_mks": 2}
