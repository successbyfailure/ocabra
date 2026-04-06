"""Tests for ChatterboxBackend — capabilities, VRAM, forward_request, streaming, voice cloning."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ocabra.backends.base import WorkerInfo
from ocabra.backends.chatterbox_backend import (
    CHATTERBOX_LANGUAGES,
    CHATTERBOX_VOICE_MAPPINGS,
    ChatterboxBackend,
)


class _DummyResponse:
    def __init__(self, *, content=b"", status_code=200, headers=None, payload=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or {}

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


def _make_worker(model_id: str = "ResembleAI/chatterbox-turbo", port: int = 18050):
    return SimpleNamespace(
        info=WorkerInfo(
            backend_type="chatterbox",
            model_id=model_id,
            gpu_indices=[0],
            port=port,
            pid=9999,
            vram_used_mb=4096,
        ),
        process=SimpleNamespace(returncode=None),
        log_file=None,
    )


# ── Capabilities & VRAM ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_chatterbox_capabilities():
    backend = ChatterboxBackend()
    caps = await backend.get_capabilities("ResembleAI/chatterbox-turbo")
    assert caps.tts is True
    assert caps.streaming is True
    # Other capabilities must be False
    assert caps.chat is False
    assert caps.vision is False
    assert caps.embeddings is False


@pytest.mark.asyncio
async def test_chatterbox_vram_turbo():
    backend = ChatterboxBackend()
    vram = await backend.get_vram_estimate_mb("ResembleAI/chatterbox-turbo")
    assert vram == 4096


@pytest.mark.asyncio
async def test_chatterbox_vram_full():
    backend = ChatterboxBackend()
    # A generic "chatterbox" model without "turbo" should return 8192
    vram = await backend.get_vram_estimate_mb("ResembleAI/chatterbox-multilingual")
    assert vram == 8192


# ── Forward request: synthesize ──────────────────────────────────


@pytest.mark.asyncio
async def test_forward_request_synthesize():
    backend = ChatterboxBackend()
    backend._workers["ResembleAI/chatterbox-turbo"] = _make_worker()

    response = _DummyResponse(
        content=b"FAKE_MP3",
        headers={"content-type": "audio/mpeg"},
    )
    dummy_client = _DummyClient(response)

    with patch("ocabra.backends.chatterbox_backend.httpx.AsyncClient", return_value=dummy_client):
        result = await backend.forward_request(
            "ResembleAI/chatterbox-turbo",
            "/synthesize",
            {
                "input": "Hello world",
                "voice": "nova",
                "response_format": "mp3",
                "speed": 1.0,
            },
        )

    assert result["content"] == b"FAKE_MP3"
    assert result["content_type"] == "audio/mpeg"
    sent = dummy_client.calls[0]["json"]
    assert sent["voice"] == "default"
    assert sent["text"] == "Hello world"


# ── Forward request: voice cloning with voice_ref ────────────────


@pytest.mark.asyncio
async def test_forward_request_with_voice_ref():
    backend = ChatterboxBackend()
    backend._workers["ResembleAI/chatterbox-turbo"] = _make_worker()

    response = _DummyResponse(
        content=b"CLONED_AUDIO",
        headers={"content-type": "audio/mpeg"},
    )
    dummy_client = _DummyClient(response)

    with patch("ocabra.backends.chatterbox_backend.httpx.AsyncClient", return_value=dummy_client):
        result = await backend.forward_request(
            "ResembleAI/chatterbox-turbo",
            "/synthesize",
            {
                "input": "Cloned voice test",
                "voice": "alloy",
                "response_format": "mp3",
                "speed": 1.0,
                "voice_ref": "/data/profiles/tts-glados/reference.wav",
            },
        )

    assert result["content"] == b"CLONED_AUDIO"
    sent = dummy_client.calls[0]["json"]
    assert sent["voice_ref"] == "/data/profiles/tts-glados/reference.wav"


# ── Forward request: voices endpoint ─────────────────────────────


@pytest.mark.asyncio
async def test_forward_request_voices():
    backend = ChatterboxBackend()
    backend._workers["ResembleAI/chatterbox-turbo"] = _make_worker()

    voices_payload = {
        "voices": ["default"],
        "model_type": "chatterbox",
        "languages": CHATTERBOX_LANGUAGES,
        "supports_voice_clone": True,
    }
    response = _DummyResponse(payload=voices_payload)
    dummy_client = _DummyClient(response)

    with patch("ocabra.backends.chatterbox_backend.httpx.AsyncClient", return_value=dummy_client):
        result = await backend.forward_request(
            "ResembleAI/chatterbox-turbo",
            "/voices",
            {},
        )

    assert result["voices"] == ["default"]
    assert result["supports_voice_clone"] is True


# ── Forward stream ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forward_stream():
    backend = ChatterboxBackend()
    backend._workers["ResembleAI/chatterbox-turbo"] = _make_worker()

    chunks = [b"chunk1", b"chunk2", b"chunk3"]

    class _StreamResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_bytes(self):
            for c in chunks:
                yield c

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class _StreamClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, method, url, json=None):
            return _StreamResponse()

    with patch(
        "ocabra.backends.chatterbox_backend.httpx.AsyncClient", return_value=_StreamClient()
    ):
        collected = []
        async for chunk in backend.forward_stream(
            "ResembleAI/chatterbox-turbo",
            "/synthesize/stream",
            {
                "input": "Stream test sentence one. Sentence two.",
                "voice": "alloy",
                "response_format": "mp3",
                "speed": 1.0,
            },
        ):
            collected.append(chunk)

    assert collected == chunks


# ── Voice mapping ────────────────────────────────────────────────


def test_all_openai_voices_mapped():
    openai_voices = ("alloy", "echo", "fable", "nova", "onyx", "shimmer")
    for voice in openai_voices:
        assert voice in CHATTERBOX_VOICE_MAPPINGS
        assert CHATTERBOX_VOICE_MAPPINGS[voice] == "default"


# ── Health check ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_no_worker():
    backend = ChatterboxBackend()
    assert await backend.health_check("nonexistent") is False


@pytest.mark.asyncio
async def test_health_check_dead_process():
    backend = ChatterboxBackend()
    worker = _make_worker()
    worker.process.returncode = 1  # process died
    backend._workers["ResembleAI/chatterbox-turbo"] = worker
    assert await backend.health_check("ResembleAI/chatterbox-turbo") is False


# ── Unload ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unload_noop_when_missing():
    backend = ChatterboxBackend()
    # Should not raise
    await backend.unload("nonexistent")


# ── Error: forward with no loaded worker ─────────────────────────


@pytest.mark.asyncio
async def test_forward_request_raises_for_missing_worker():
    backend = ChatterboxBackend()
    with pytest.raises(KeyError, match="not loaded"):
        await backend.forward_request("nonexistent", "/synthesize", {"input": "test"})
