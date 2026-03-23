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
