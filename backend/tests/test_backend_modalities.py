"""Tests for the multi-modal BackendInterface and WorkerPool modality helpers."""

from __future__ import annotations

import pytest

from ocabra.backends.base import BackendInterface, ModalityType

# ---------------------------------------------------------------------------
# Collect all concrete backend classes
# ---------------------------------------------------------------------------

_BACKEND_CLASSES: list[type[BackendInterface]] = []


def _collect_backends() -> list[type[BackendInterface]]:
    """Import every *Backend class so we can iterate over them."""
    if _BACKEND_CLASSES:
        return _BACKEND_CLASSES

    from ocabra.backends._mock import MockBackend
    from ocabra.backends.bitnet_backend import BitnetBackend
    from ocabra.backends.diffusers_backend import DiffusersBackend
    from ocabra.backends.llama_cpp_backend import LlamaCppBackend
    from ocabra.backends.ollama_backend import OllamaBackend
    from ocabra.backends.sglang_backend import SGLangBackend
    from ocabra.backends.tensorrt_llm_backend import TensorRTLLMBackend
    from ocabra.backends.tts_backend import TTSBackend
    from ocabra.backends.vllm_backend import VLLMBackend
    from ocabra.backends.whisper_backend import WhisperBackend

    _BACKEND_CLASSES.extend([
        VLLMBackend,
        LlamaCppBackend,
        SGLangBackend,
        OllamaBackend,
        BitnetBackend,
        DiffusersBackend,
        WhisperBackend,
        TTSBackend,
        TensorRTLLMBackend,
        MockBackend,
    ])
    return _BACKEND_CLASSES


# ---------------------------------------------------------------------------
# Tests: modality declarations
# ---------------------------------------------------------------------------


class TestModalityDeclarations:
    def test_all_backends_declare_modalities(self):
        """Every registered backend must declare at least one modality."""
        for cls in _collect_backends():
            modalities = cls.supported_modalities()
            assert len(modalities) > 0, (
                f"{cls.__name__}.supported_modalities() returned an empty set"
            )
            for m in modalities:
                assert isinstance(m, ModalityType), (
                    f"{cls.__name__} returned non-ModalityType value: {m!r}"
                )

    def test_vllm_supports_text_and_embeddings(self):
        from ocabra.backends.vllm_backend import VLLMBackend

        modalities = VLLMBackend.supported_modalities()
        assert ModalityType.TEXT_GENERATION in modalities
        assert ModalityType.EMBEDDINGS in modalities

    def test_whisper_supports_only_transcription(self):
        from ocabra.backends.whisper_backend import WhisperBackend

        modalities = WhisperBackend.supported_modalities()
        assert modalities == {ModalityType.AUDIO_TRANSCRIPTION}

    def test_diffusers_supports_only_image_generation(self):
        from ocabra.backends.diffusers_backend import DiffusersBackend

        modalities = DiffusersBackend.supported_modalities()
        assert modalities == {ModalityType.IMAGE_GENERATION}

    def test_tts_supports_only_audio_speech(self):
        from ocabra.backends.tts_backend import TTSBackend

        modalities = TTSBackend.supported_modalities()
        assert modalities == {ModalityType.AUDIO_SPEECH}

    def test_bitnet_supports_only_text_generation(self):
        from ocabra.backends.bitnet_backend import BitnetBackend

        modalities = BitnetBackend.supported_modalities()
        assert modalities == {ModalityType.TEXT_GENERATION}

    def test_tensorrt_supports_only_text_generation(self):
        from ocabra.backends.tensorrt_llm_backend import TensorRTLLMBackend

        modalities = TensorRTLLMBackend.supported_modalities()
        assert modalities == {ModalityType.TEXT_GENERATION}


# ---------------------------------------------------------------------------
# Tests: WorkerPool modality helpers
# ---------------------------------------------------------------------------


class TestWorkerPoolModalities:
    def _make_pool(self):
        from ocabra.backends._mock import MockBackend
        from ocabra.backends.diffusers_backend import DiffusersBackend
        from ocabra.backends.vllm_backend import VLLMBackend
        from ocabra.backends.whisper_backend import WhisperBackend
        from ocabra.core.worker_pool import WorkerPool

        pool = WorkerPool()
        pool.register_backend("vllm", VLLMBackend())
        pool.register_backend("whisper", WhisperBackend())
        pool.register_backend("diffusers", DiffusersBackend())
        pool.register_backend("mock", MockBackend())
        return pool

    def test_get_backends_for_modality_text(self):
        pool = self._make_pool()
        text_backends = pool.get_backends_for_modality(ModalityType.TEXT_GENERATION)
        assert "vllm" in text_backends
        assert "mock" in text_backends
        assert "whisper" not in text_backends
        assert "diffusers" not in text_backends

    def test_get_backends_for_modality_image(self):
        pool = self._make_pool()
        image_backends = pool.get_backends_for_modality(ModalityType.IMAGE_GENERATION)
        assert image_backends == ["diffusers"]

    def test_get_backends_for_modality_excludes_disabled(self):
        pool = self._make_pool()
        pool.register_disabled_backend("vllm", "no GPU available")
        text_backends = pool.get_backends_for_modality(ModalityType.TEXT_GENERATION)
        assert "vllm" not in text_backends
        assert "mock" in text_backends

    def test_supports_modality_true(self):
        pool = self._make_pool()
        assert pool.supports_modality("vllm", ModalityType.TEXT_GENERATION) is True
        assert pool.supports_modality("vllm", ModalityType.EMBEDDINGS) is True

    def test_supports_modality_false(self):
        pool = self._make_pool()
        assert pool.supports_modality("whisper", ModalityType.TEXT_GENERATION) is False
        assert pool.supports_modality("nonexistent", ModalityType.TEXT_GENERATION) is False


# ---------------------------------------------------------------------------
# Tests: unsupported modality methods raise NotImplementedError
# ---------------------------------------------------------------------------


class TestUnsupportedModalityRaises:
    @pytest.mark.asyncio
    async def test_generate_image_on_text_backend_raises(self):
        from ocabra.backends._mock import MockBackend

        backend = MockBackend()
        with pytest.raises(NotImplementedError, match="does not support image generation"):
            await backend.generate_image("some-model", "a cat")

    @pytest.mark.asyncio
    async def test_transcribe_on_text_backend_raises(self):
        from ocabra.backends._mock import MockBackend

        backend = MockBackend()
        with pytest.raises(NotImplementedError, match="does not support audio transcription"):
            await backend.transcribe("some-model", b"audio-bytes")

    @pytest.mark.asyncio
    async def test_synthesize_speech_on_text_backend_raises(self):
        from ocabra.backends._mock import MockBackend

        backend = MockBackend()
        with pytest.raises(NotImplementedError, match="does not support speech synthesis"):
            await backend.synthesize_speech("some-model", "hello world")

    @pytest.mark.asyncio
    async def test_rerank_on_text_backend_raises(self):
        from ocabra.backends._mock import MockBackend

        backend = MockBackend()
        with pytest.raises(NotImplementedError, match="does not support reranking"):
            await backend.rerank("some-model", "query", ["doc1", "doc2"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_on_base_raises(self):
        from ocabra.backends._mock import MockBackend

        backend = MockBackend()
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            await backend.generate_embeddings("some-model", ["text"])
