"""Speed of the requests not measured in generated tokens (stats/aggregator.py)."""

from __future__ import annotations

from types import SimpleNamespace

from ocabra.stats.aggregator import work_speeds


def _row(kind, ms, meta=None, inp=None, out=None, error=None, status=200):
    return SimpleNamespace(
        request_kind=kind, duration_ms=ms, work_size_meta=meta, input_tokens=inp,
        output_tokens=out, error=error, status_code=status,
    )


def _by_unit(rows):
    return {(tuple(s["kinds"]), s["unit"]): s for s in work_speeds(rows)}


def test_transcription_is_seconds_of_audio_per_second():
    rows = [
        _row("audio_transcription", 2000, {"audio_seconds": 60}),
        _row("audio_transcription", 1000, {"audio_seconds": 30}),
        _row("audio_transcription", 500, None),  # no audio length: left out
        _row("audio_transcription", 900, {"audio_seconds": 99}, error="boom"),
    ]
    s = _by_unit(rows)[(("audio_transcription",), "x_realtime")]
    assert s["value"] == 30.0 and s["samples"] == 2


def test_speech_embeddings_rerank_images():
    rows = [
        _row("tts", 2000, {"text_chars": 300, "audio_seconds": 20}),
        _row("embedding", 100, {"n_inputs": 10}, inp=500),
        _row("rerank", 200, {"n_documents": 40}),
        _row("image_generation", 12000, {"n_images": 2}),
    ]
    s = _by_unit(rows)
    assert s[(("tts",), "x_realtime")]["value"] == 10.0
    assert s[(("tts",), "chars_per_s")]["value"] == 150.0
    assert s[(("embedding",), "inputs_per_s")]["value"] == 100.0
    assert s[(("embedding",), "tokens_per_s")]["value"] == 5000.0
    assert s[(("rerank",), "documents_per_s")]["value"] == 200.0
    assert s[(("image_generation",), "s_per_image")]["value"] == 6.0


def test_chat_has_no_work_speed_here():
    assert work_speeds([_row("chat", 1000, None, inp=10, out=100)]) == []
