
import pytest

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


def test_tts_voice_mapping_has_all_openai_voices_for_each_family():
    for family, mapping in VOICE_MAPPINGS.items():
        for voice in OPENAI_VOICES:
            assert voice in mapping, f"Missing voice '{voice}' for family '{family}'"
            assert mapping[voice]
