from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ocabra.backends.base import WorkerInfo
from ocabra.backends.tts_backend import OPENAI_VOICES, VOICE_MAPPINGS, TTSBackend


class _DummyResponse:
    def __init__(self, *, content=b"", status_code=200, headers=None, payload=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("request failed")

    def json(self):
        return self._payload


class _DummyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, json=None):
        self.calls.append({"method": "post", "url": url, "json": json})
        return self.response

    async def get(self, url):
        self.calls.append({"method": "get", "url": url})
        return self.response


@pytest.mark.asyncio
async def test_tts_capabilities_and_vram_lookup():
    backend = TTSBackend()

    caps = await backend.get_capabilities("Qwen/Qwen3-TTS")
    assert caps.tts is True

    vram = await backend.get_vram_estimate_mb("Qwen/Qwen3-TTS")
    assert vram == 8192


@pytest.mark.asyncio
async def test_tts_forward_request_returns_audio_content_and_content_type():
    backend = TTSBackend()
    backend._workers["Qwen/Qwen3-TTS"] = SimpleNamespace(
        info=WorkerInfo(
            backend_type="tts",
            model_id="Qwen/Qwen3-TTS",
            gpu_indices=[1],
            port=18011,
            pid=1010,
            vram_used_mb=8192,
        ),
        process=SimpleNamespace(returncode=None),
    )

    response = _DummyResponse(
        content=b"FAKE_MP3",
        headers={"content-type": "audio/mpeg"},
    )
    dummy_client = _DummyClient(response)

    with patch("ocabra.backends.tts_backend.httpx.AsyncClient", return_value=dummy_client):
        payload = await backend.forward_request(
            "Qwen/Qwen3-TTS",
            "/synthesize",
            {
                "input": "Hola mundo",
                "voice": "nova",
                "response_format": "mp3",
                "speed": 1.0,
            },
        )

    assert payload["content"] == b"FAKE_MP3"
    assert payload["content_type"] == "audio/mpeg"
    sent = dummy_client.calls[0]["json"]
    assert sent["voice"] == VOICE_MAPPINGS["qwen3-tts"]["nova"]


def test_tts_voice_mapping_has_all_openai_voices_for_each_family():
    for family, mapping in VOICE_MAPPINGS.items():
        for voice in OPENAI_VOICES:
            assert voice in mapping, f"Missing voice '{voice}' for family '{family}'"
            assert mapping[voice]
