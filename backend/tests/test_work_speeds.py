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


def test_video_upscale_is_frames_per_second():
    rows = [
        _row("video_upscale", 10000, {"input_frames": 240, "input_bytes": 1}),
        _row("video_upscale", 5000, {"input_bytes": 1}),  # worker gave no frames
    ]
    s = _by_unit(rows)[(("video_upscale",), "frames_per_s")]
    assert s["value"] == 24.0 and s["samples"] == 1


def test_photo_upscale_is_megapixels_per_second_without_the_load():
    """VHS 20/11: 10 s warm vs 45 s for a 4K photo, and a cold load of 8 s
    counted as work. The speed is the network's megapixels per second of
    service, not per second waited."""
    warm = _row("image_upscale", 10_000, {"n_images": 1, "model_mp": 33.178})
    cold = _row("image_upscale", 18_000, {"n_images": 1, "model_mp": 33.178})
    cold.runtime_telemetry = {"load_ms": 8_000}
    s = _by_unit([warm, cold])
    assert s[(("image_upscale",), "mp_per_s")]["value"] == 3.32
    assert s[(("image_upscale",), "s_per_image")]["value"] == 10.0
