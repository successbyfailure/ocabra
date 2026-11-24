from unittest.mock import AsyncMock, patch

import pytest

from ocabra.backends.whisper_backend import (
    DIARIZATION_OVERHEAD_MB,
    WhisperBackend,
    _build_transcription_multipart,
)


class _DummyResponse:
    def __init__(self, *, payload=None, text="", status_code=200):
        self._payload = payload or {}
        self.text = text
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("request failed")


class _DummyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, files=None, data=None):
        self.calls.append({"url": url, "files": files, "data": data})
        return self.response


class _FakeProcess:
    def __init__(self):
        self.pid = 4321
        self.returncode = None


@pytest.mark.asyncio
async def test_whisper_capabilities_and_vram_lookup():
    backend = WhisperBackend()

    caps = await backend.get_capabilities("openai/whisper-large-v3")
    assert caps.audio_transcription is True

    vram = await backend.get_vram_estimate_mb("openai/whisper-large-v3")
    assert vram == 6000


@pytest.mark.asyncio
async def test_whisper_vram_lookup_diarization_variant_adds_overhead():
    backend = WhisperBackend()

    vram = await backend.get_vram_estimate_mb("openai/whisper-medium::diarize")
    assert vram == 3000 + DIARIZATION_OVERHEAD_MB


@pytest.mark.asyncio
async def test_whisper_load_uses_base_model_id_and_diarization_flags():
    backend = WhisperBackend()

    with (
        patch(
            "ocabra.backends.whisper_backend.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_FakeProcess()),
        ) as create_proc,
        patch.object(backend, "_wait_until_healthy", new=AsyncMock(return_value=True)),
    ):
        info = await backend.load(
            "openai/whisper-medium::diarize",
            [1],
            port=18012,
            extra_config={"diarization_enabled": True},
        )

    assert info.model_id == "openai/whisper-medium::diarize"
    args = create_proc.await_args.args
    assert "--model-id" in args
    assert "medium" in args
    assert "--diarize" in args
    assert "--diarization-model-id" in args


@pytest.mark.asyncio
async def test_el_modelo_de_diarizacion_llega_aunque_la_diarizacion_no_este_activa_al_cargar():
    """La diarización se decide por petición y ``diarization_enabled`` ya no
    forma parte de la identidad del worker: el modelo configurado tiene que
    llegar igual. Antes solo se pasaba con --diarize y un perfil con
    pyannote community-1 diarizaba con la 3.1 (2026-11-20)."""
    backend = WhisperBackend()

    with (
        patch(
            "ocabra.backends.whisper_backend.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_FakeProcess()),
        ) as create_proc,
        patch.object(backend, "_wait_until_healthy", new=AsyncMock(return_value=True)),
    ):
        await backend.load(
            "openai/whisper-large-v3",
            [0],
            port=18013,
            extra_config={
                "whisper": {"diarization_model_id": "pyannote/speaker-diarization-community-1"}
            },
        )

    args = list(create_proc.await_args.args)
    assert "--diarize" not in args
    i = args.index("--diarization-model-id")
    assert args[i + 1] == "pyannote/speaker-diarization-community-1"


@pytest.mark.asyncio
async def test_whisper_load_accepts_nested_whisper_config():
    backend = WhisperBackend()

    with (
        patch(
            "ocabra.backends.whisper_backend.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_FakeProcess()),
        ) as create_proc,
        patch.object(backend, "_wait_until_healthy", new=AsyncMock(return_value=True)),
    ):
        await backend.load(
            "openai/whisper-medium::diarize",
            [0],
            port=18013,
            extra_config={
                "whisper": {
                    "baseModelId": "openai/whisper-small",
                    "diarizationEnabled": True,
                    "diarizationModelId": "pyannote/custom-diarization",
                }
            },
        )

    args = create_proc.await_args.args
    assert "small" in args
    assert "--diarize" in args
    assert "pyannote/custom-diarization" in args


@pytest.mark.asyncio
async def test_build_multipart_includes_diarize_when_present():
    files, data = _build_transcription_multipart(
        {
            "file": ("sample.wav", b"123", "audio/wav"),
            "diarize": True,
            "response_format": "verbose_json",
        }
    )

    assert files["file"][0] == "sample.wav"
    assert data["diarize"] == "true"
    assert data["response_format"] == "verbose_json"


@pytest.mark.asyncio
async def test_whisper_vram_lookup_nvidia_nemo_models() -> None:
    backend = WhisperBackend()

    parakeet_vram = await backend.get_vram_estimate_mb("nvidia/parakeet-tdt-0.6b-v3")
    canary_vram = await backend.get_vram_estimate_mb("nvidia/canary-1b-v2")

    assert parakeet_vram == 4500
    assert canary_vram == 9000

@pytest.mark.asyncio
async def test_whisper_load_uses_configured_startup_timeout():
    backend = WhisperBackend()

    with (
        patch(
            "ocabra.backends.whisper_backend.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_FakeProcess()),
        ),
        patch("ocabra.backends.whisper_backend.settings.whisper_startup_timeout_s", 321),
        patch.object(backend, "_wait_until_healthy", new=AsyncMock(return_value=True)) as wait_healthy,
    ):
        await backend.load(
            "openai/whisper-medium::diarize",
            [1],
            port=18012,
            extra_config={"diarization_enabled": True},
        )

    wait_healthy.assert_awaited_once_with("openai/whisper-medium::diarize", timeout_s=321)
