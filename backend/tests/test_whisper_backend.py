from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ocabra.backends.base import WorkerInfo
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
async def test_whisper_forward_request_reformats_openai_transcription_payload():
    backend = WhisperBackend()
    backend._workers["openai/whisper-small"] = SimpleNamespace(
        info=WorkerInfo(
            backend_type="whisper",
            model_id="openai/whisper-small",
            gpu_indices=[0],
            port=18001,
            pid=1000,
            vram_used_mb=1200,
        ),
        process=SimpleNamespace(returncode=None),
    )

    response = _DummyResponse(payload={"text": "hola mundo"})
    dummy_client = _DummyClient(response)

    with patch("ocabra.backends.whisper_backend.httpx.AsyncClient", return_value=dummy_client):
        payload = await backend.forward_request(
            "openai/whisper-small",
            "/transcribe",
            {
                "file": ("sample.wav", b"binary-audio", "audio/wav"),
                "language": "es",
                "response_format": "json",
                "timestamp_granularities": ["segment"],
            },
        )

    assert payload == {"text": "hola mundo"}
    assert len(dummy_client.calls) == 1
    sent = dummy_client.calls[0]
    assert sent["files"]["file"][0] == "sample.wav"
    assert sent["files"]["file"][2] == "audio/wav"
    assert sent["data"]["language"] == "es"
    assert sent["data"]["timestamp_granularities"] == ["segment"]


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
