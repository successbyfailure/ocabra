"""Un modelo de embeddings no se anuncia como modelo de razonamiento.

``chat`` y ``tools`` ya se calculaban con ``not embeddings``, pero ``reasoning``
no. Los GGUF de ``qwen3-embedding`` estan construidos sobre la misma base que
qwen3, asi que traen su plantilla de chat (con ``<think>``) y se anunciaban como
razonadores en ``/v1/models``. Razonar es una propiedad de la generacion.
"""

from unittest.mock import AsyncMock

import pytest

from ocabra.backends.llama_cpp_backend import LlamaCppBackend


@pytest.fixture
def backend(monkeypatch):
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._model_configs = {}
    # La plantilla dice que hay razonamiento y herramientas (es la de qwen3).
    monkeypatch.setattr(
        LlamaCppBackend,
        "_template_caps",
        AsyncMock(return_value={"tools": True, "reasoning": True}),
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_resolve_model_file", lambda self, mid, opts: (_ for _ in ()).throw(FileNotFoundError)
    )
    return b


async def test_un_modelo_de_embeddings_no_razona(backend):
    backend._model_configs["llamacpp_mks/qwen3-embedding-0.6b"] = {"embedding": True}
    caps = await backend.get_capabilities("llamacpp_mks/qwen3-embedding-0.6b")
    assert caps.embeddings is True
    assert caps.reasoning is False
    assert caps.chat is False
    assert caps.tools is False


async def test_el_nombre_tambien_basta_para_detectar_embeddings(backend):
    # Sin la opcion explicita: se detecta por el nombre ("embedding").
    backend._model_configs["llamacpp_mks/qwen3-embedding-8b"] = {}
    caps = await backend.get_capabilities("llamacpp_mks/qwen3-embedding-8b")
    assert caps.embeddings is True
    assert caps.reasoning is False


async def test_un_modelo_generativo_si_razona(backend):
    backend._model_configs["llamacpp_mks/qwen3-32b"] = {}
    caps = await backend.get_capabilities("llamacpp_mks/qwen3-32b")
    assert caps.embeddings is False
    assert caps.reasoning is True
    assert caps.chat is True


async def test_un_reasoning_on_explicito_no_resucita_el_razonamiento(backend):
    """``--reasoning on`` no tiene sentido en un modelo de embeddings."""
    backend._model_configs["llamacpp_mks/qwen3-embedding-4b"] = {
        "embedding": True,
        "reasoning": "on",
    }
    caps = await backend.get_capabilities("llamacpp_mks/qwen3-embedding-4b")
    assert caps.reasoning is False
