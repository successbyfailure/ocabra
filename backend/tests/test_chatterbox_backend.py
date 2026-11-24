"""Tests for ChatterboxBackend — capabilities, VRAM, forward_request, streaming, voice cloning."""

from types import SimpleNamespace

import pytest

from ocabra.backends.base import WorkerInfo
from ocabra.backends.chatterbox_backend import (
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
