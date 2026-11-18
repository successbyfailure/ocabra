"""Work done per request kind (stats/collector.py)."""

from __future__ import annotations

import io
import wave

import pytest

from ocabra.stats.collector import (
    _audio_duration_s,
    _classify_request_kind,
    _extract_work_size_meta,
)


def test_each_kind_records_its_work():
    assert _extract_work_size_meta("tts", {"input": "hola mundo"}, None) == {"text_chars": 10}
    assert _extract_work_size_meta("tts", {"text": "hola"}, None) == {"text_chars": 4}
    assert _extract_work_size_meta("music", {"duration_seconds": 30}, None) == {
        "duration_seconds": 30.0
    }
    assert _extract_work_size_meta("embedding", {"input": ["a", "b", "c"]}, None) == {"n_inputs": 3}
    assert _extract_work_size_meta("ollama_embedding", {"input": "a"}, None) == {"n_inputs": 1}
    assert _extract_work_size_meta("rerank", {"query": "q", "documents": ["a", "b"]}, None) == {
        "n_documents": 2
    }
    meta = _extract_work_size_meta(
        "image_generation", {"prompt": "x", "size": "512x512", "n": 2}, {"data": [{}, {}]}
    )
    assert meta == {"resolution_w": 512, "resolution_h": 512, "n_images": 2}
    assert _extract_work_size_meta("audio_transcription", None, {"duration": 12.5}) == {
        "audio_seconds": 12.5
    }


def test_responses_and_music_are_classified():
    assert _classify_request_kind("/v1/responses") == "responses"
    assert _classify_request_kind("/v1/audio/generate") == "music"


def _wav(seconds: float, rate: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(rate * seconds))
    return buf.getvalue()


@pytest.mark.asyncio
async def test_generated_audio_length():
    assert await _audio_duration_s(_wav(2.0), "audio/wav") == 2.0
    # A streamed WAV declares a placeholder size: the bytes decide.
    streamed = bytearray(_wav(1.5))
    data_at = streamed.find(b"data")
    streamed[data_at + 4 : data_at + 8] = b"\xff\xff\xff\xff"
    assert await _audio_duration_s(bytes(streamed), "audio/wav") == 1.5
    assert await _audio_duration_s(b"\0" * 48000, "audio/pcm") == 1.0
    assert await _audio_duration_s(b"", "audio/mpeg") is None


@pytest.mark.asyncio
async def test_energy_comes_from_the_preset_worker_gpus(monkeypatch):
    """The base model is not loaded; its preset worker runs on GPUs 0 and 1.
    Before, the row got no GPU and no energy (86 % of chat rows)."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    import ocabra.database as database
    from ocabra.stats import collector

    base = SimpleNamespace(
        backend_type="llamacpp_mks", model_key="qwen3.8:27b", current_gpu=[], preferred_gpu=None
    )
    worker = SimpleNamespace(
        backend_type="llamacpp_mks", model_key="qwen3.8:27b", current_gpu=[1, 0]
    )

    async def get_state(key):
        return {"llamacpp_mks/qwen3.8:27b": base, "llamacpp_mks/qwen3.8:27b::abc": worker}.get(key)

    async def resolve(request, model_id):
        return "llamacpp_mks/qwen3.8:27b"

    monkeypatch.setattr(collector, "_resolve_inflight_model_id", resolve)
    rows = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def add(self, row):
            rows.append(row)

        async def commit(self):
            pass

    monkeypatch.setattr(database, "AsyncSessionLocal", lambda: _Session())
    gpus = {0: SimpleNamespace(power_draw_w=150.0), 1: SimpleNamespace(power_draw_w=300.0)}
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                model_manager=SimpleNamespace(get_state=get_state, list_states=None),
                gpu_manager=SimpleNamespace(_states=gpus),
            )
        ),
        state=SimpleNamespace(
            auth_user=None,
            inflight_request_id="r",
            runtime_telemetry=None,
            runtime_worker_key="llamacpp_mks/qwen3.8:27b::abc",
        ),
        headers={},
        client=None,
    )
    await collector._record_stat(
        request=request,
        model_id="qwen3.8:27b",
        started_at=datetime.now(UTC),
        duration_ms=36_000.0,
        error_message=None,
        status_code=200,
        endpoint_path="/v1/chat/completions",
        request_kind="chat",
        input_tokens=10,
        output_tokens=100,
    )
    [row] = [r for r in rows if getattr(r, "request_kind", None) == "chat"]
    assert row.gpu_index == 1
    assert row.energy_wh == pytest.approx((150 + 300) * 36 / 3600)
