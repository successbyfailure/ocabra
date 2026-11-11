import sys
from itertools import count
from types import SimpleNamespace

import pytest

# numpy no es dependencia de oCabra: la usa el worker de audio, que vive en su
# propio entorno. Sin este skip el ImportError rompia la recolectada de TODA la
# suite (no solo de este fichero), asi que habia que lanzar pytest con
# --ignore=tests/test_whisper_worker.py para que corriese algo.
np = pytest.importorskip("numpy", reason="numpy solo esta en el entorno del worker de audio")

from workers import whisper_worker  # noqa: E402


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


# -- the on-demand diarizer is dropped when idle (2026-11-11) --------------------


@pytest.mark.asyncio
async def test_an_idle_diarizer_is_unloaded_but_never_mid_diarization(monkeypatch):
    import asyncio
    import time

    rt = whisper_worker.runtime
    monkeypatch.setattr(rt, "diarize_default", False)
    monkeypatch.setattr(rt, "diarizer_idle_unload_s", 300.0)
    monkeypatch.setattr(rt, "diarization_pipeline", object())
    monkeypatch.setattr(rt, "embedding_inference", object())
    dropped = []
    monkeypatch.setattr(
        whisper_worker,
        "_drop_diarizer",
        lambda: dropped.append(True) or setattr(rt, "diarization_pipeline", None),
    )

    async def tick():
        task = asyncio.create_task(whisper_worker._reap_idle_diarizer(interval_s=0.01))
        await asyncio.sleep(0.05)
        task.cancel()

    # Used 10 s ago: kept.
    monkeypatch.setattr(rt, "last_diarization_at", time.monotonic() - 10)
    await tick()
    assert not dropped
    # Idle long enough but a diarization is running: kept.
    monkeypatch.setattr(rt, "last_diarization_at", time.monotonic() - 400)
    monkeypatch.setattr(rt, "diarizations_in_flight", 1)
    await tick()
    assert not dropped
    # Idle and nothing running: dropped.
    monkeypatch.setattr(rt, "diarizations_in_flight", 0)
    await tick()
    assert dropped


@pytest.mark.asyncio
async def test_a_profile_that_always_diarizes_keeps_its_diarizer(monkeypatch):
    import asyncio
    import time

    rt = whisper_worker.runtime
    monkeypatch.setattr(rt, "diarize_default", True)
    monkeypatch.setattr(rt, "diarizer_idle_unload_s", 1.0)
    monkeypatch.setattr(rt, "diarization_pipeline", object())
    monkeypatch.setattr(rt, "diarizations_in_flight", 0)
    monkeypatch.setattr(rt, "last_diarization_at", time.monotonic() - 999)
    task = asyncio.create_task(whisper_worker._reap_idle_diarizer(interval_s=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    assert rt.diarization_pipeline is not None


def test_run_transcription_reports_progress_as_segments_decode():
    """faster-whisper entrega los segmentos al decodificarlos: el fin de cada
    uno es el avance real. Al terminar de transcribir, todo el audio."""
    seen: list[float] = []
    progress: dict = {"phase": "transcribing", "startedMonotonic": 0.0, "updatedMonotonic": 0.0}

    def segments():
        for i, end in enumerate((30.0, 60.0, 95.5)):
            # Lo que había apuntado al pedir este segmento.
            seen.append(progress.get("processedSeconds", 0.0))
            yield SimpleNamespace(start=end - 30, end=end, text=f" t{i}", no_speech_prob=0.0, avg_logprob=-0.1, words=None)

    class FakeModel:
        def transcribe(self, *args, **kwargs):
            return segments(), SimpleNamespace(duration=120.0, language="es", language_probability=0.99)

    result = whisper_worker._run_transcription(
        "/tmp/x.wav", FakeModel(), "es", None, 0.0, False, None, ["segment"], progress=progress
    )
    assert seen == [0.0, 30.0, 60.0]
    assert progress["audioSeconds"] == 120.0
    assert progress["segments"] == 3
    # Tras el último segmento (VAD se salta el silencio final): todo el audio.
    assert progress["processedSeconds"] == 120.0
    assert len(result["segments"]) == 3
    # La duración viaja en el resultado (y de ahí en la cabecera X-Audio-Duration).
    assert result["duration"] == 120.0


def test_progress_endpoint_lists_jobs():
    # Construir la app no carga el modelo: eso lo hace el lifespan al arrancar.
    app = whisper_worker.create_app("tiny", [])
    whisper_worker.runtime.progress.clear()
    whisper_worker.runtime.progress["req-1"] = {
        "phase": "transcribing",
        "audioSeconds": 120.0,
        "processedSeconds": 30.0,
        "segments": 1,
        "startedMonotonic": whisper_worker.time.monotonic() - 2,
        "updatedMonotonic": whisper_worker.time.monotonic() - 1,
    }
    import asyncio
    import json

    routes = {r.path: r for r in app.routes}
    assert "/progress" in routes
    jobs = json.loads(asyncio.run(routes["/progress"].endpoint()).body)["jobs"]
    assert jobs["req-1"]["processedSeconds"] == 30.0
    assert jobs["req-1"]["audioSeconds"] == 120.0
    assert 1500 <= jobs["req-1"]["elapsedMs"] <= 3500
    whisper_worker.runtime.progress.clear()
