"""Characterization tests for ocabra.core.capabilities (REF-05 step 1)."""

from __future__ import annotations

from types import SimpleNamespace

from ocabra.core import capabilities as caps
from ocabra.core import model_manager as mm


def _state(backend_type: str, model_id: str = "", extra: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        backend_type=backend_type,
        model_id=model_id,
        backend_model_id=model_id,
        extra_config=extra or {},
    )


def test_declared_capabilities_llm() -> None:
    c = caps.declared_capabilities(_state("vllm", "Qwen/Qwen3-8B"))
    assert c.chat and c.completion and c.streaming and c.tools and c.reasoning


def test_declared_capabilities_embeddings() -> None:
    c = caps.declared_capabilities(_state("llama_cpp", "bge-embed"))
    assert c.embeddings and c.pooling
    assert not c.chat


def test_declared_capabilities_whisper() -> None:
    assert caps.declared_capabilities(_state("whisper", "large-v3")).audio_transcription


def test_resolve_context_length_fallback() -> None:
    assert (
        caps.resolve_context_length_fallback(_state("vllm", extra={"context_length": 8192})) == 8192
    )
    assert (
        caps.resolve_context_length_fallback(_state("vllm", extra={"vllm": {"maxModelLen": 4096}}))
        == 4096
    )
    assert caps.resolve_context_length_fallback(_state("vllm", extra={})) == 0


def test_capabilities_populated() -> None:
    from ocabra.backends.base import BackendCapabilities

    assert not caps.capabilities_populated(BackendCapabilities())
    populated = BackendCapabilities()
    populated.chat = True
    assert caps.capabilities_populated(populated)


def test_model_manager_reexports() -> None:
    assert mm.declared_capabilities is caps.declared_capabilities
    assert mm.capabilities_populated is caps.capabilities_populated
    assert mm.configured_reasoning_default is caps.configured_reasoning_default
    # The static wrapper still delegates to the free function.
    s = _state("vllm", extra={"context_length": 1234})
    assert mm.ModelManager._resolve_context_length_fallback(s) == 1234
