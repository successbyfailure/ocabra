from itertools import count
from types import SimpleNamespace
import sys

import numpy as np

from workers import whisper_worker


class _TempFileCtx:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_fake_tempfiles(monkeypatch):
    idx = count()

    def _fake_named_tempfile(*, suffix: str, delete: bool):
        _ = delete
        return _TempFileCtx(f"/tmp/nemo-chunk-{next(idx)}{suffix}")

    monkeypatch.setattr(whisper_worker.tempfile, "NamedTemporaryFile", _fake_named_tempfile)


def test_chunk_audio_for_nemo_merges_tiny_tail(monkeypatch):
    writes: list[int] = []

    def _read(path, always_2d=False):
        _ = path, always_2d
        # 20s chunk (320000) + 1 sample tail -> should be merged into one chunk.
        return np.zeros(320001, dtype=np.float32), 16000

    def _write(path, audio, sample_rate, subtype):
        _ = path, sample_rate, subtype
        writes.append(int(np.asarray(audio).shape[0]))

    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(read=_read, write=_write))
    _patch_fake_tempfiles(monkeypatch)

    chunk_paths, cleanup = whisper_worker._chunk_audio_for_nemo("/tmp/input.wav", chunk_seconds=20)

    assert len(chunk_paths) == 1
    assert len(cleanup) == 1
    assert writes == [320001]


def test_chunk_audio_for_nemo_keeps_regular_tail(monkeypatch):
    writes: list[int] = []

    def _read(path, always_2d=False):
        _ = path, always_2d
        # 20s chunk + 10000 tail samples (>0.5s threshold=8000) -> keep 2 chunks.
        return np.zeros(330000, dtype=np.float32), 16000

    def _write(path, audio, sample_rate, subtype):
        _ = path, sample_rate, subtype
        writes.append(int(np.asarray(audio).shape[0]))

    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(read=_read, write=_write))
    _patch_fake_tempfiles(monkeypatch)

    chunk_paths, cleanup = whisper_worker._chunk_audio_for_nemo("/tmp/input.wav", chunk_seconds=20)

    assert len(chunk_paths) == 2
    assert len(cleanup) == 2
    assert writes == [320000, 10000]


class _FakeWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class _FakeSegment:
    def __init__(self, start, end, text, words=None):
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class _FakeWhisperModel:
    """Records the kwargs faster-whisper would have been called with."""

    def __init__(self, segments):
        self._segments = segments
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append(kwargs)
        info = SimpleNamespace(language="es")
        return list(self._segments), info


def test_run_transcription_word_granularity_emits_words():
    segments = [
        _FakeSegment(
            0.0,
            2.0,
            " hola mundo",
            words=[
                _FakeWord(" hola", 0.0, 1.0),
                _FakeWord(" mundo", 1.0, 2.0),
            ],
        )
    ]
    model = _FakeWhisperModel(segments)

    result = whisper_worker._run_transcription(
        "/tmp/a.wav",
        model,
        language="es",
        prompt=None,
        temperature=0.0,
        diarize=False,
        diarization_pipeline=None,
        timestamp_granularities=["word", "segment"],
    )

    # faster-whisper must be asked for word timestamps and VAD segmentation.
    assert model.calls[0]["word_timestamps"] is True
    assert model.calls[0]["vad_filter"] is True
    # Per-segment words and a top-level words array (OpenAI-compatible).
    assert result["segments"][0]["words"][0]["word"] == " hola"
    assert [w["word"] for w in result["words"]] == [" hola", " mundo"]


def test_run_transcription_segment_granularity_omits_words():
    segments = [
        _FakeSegment(
            0.0,
            2.0,
            " hola",
            words=[_FakeWord(" hola", 0.0, 2.0)],
        )
    ]
    model = _FakeWhisperModel(segments)

    result = whisper_worker._run_transcription(
        "/tmp/a.wav",
        model,
        language="es",
        prompt=None,
        temperature=0.0,
        diarize=False,
        diarization_pipeline=None,
        timestamp_granularities=["segment"],
    )

    # Without word granularity, word data must not leak into the response.
    assert "words" not in result
    assert "words" not in result["segments"][0]
    assert model.calls[0]["word_timestamps"] is False


# Speaker-splitting logic moved to workers/diarization_lib.py (shared with the
# whisper.cpp diarization sidecar) — see tests/test_diarization_lib.py.
