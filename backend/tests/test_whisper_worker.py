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
async def test_a_worker_started_with_diarize_also_frees_its_idle_diarizer(monkeypatch):
    """2026-11-12: con --diarize el diarizador no se liberaba nunca (~2 GB de
    la 3060 retenidos por un worker que diarizaba de vez en cuando). Se
    recarga bajo demanda en la siguiente diarizacion."""
    import asyncio
    import time

    rt = whisper_worker.runtime
    monkeypatch.setattr(rt, "diarize_default", True)
    monkeypatch.setattr(rt, "diarizer_idle_unload_s", 1.0)
    monkeypatch.setattr(rt, "diarization_pipeline", object())
    monkeypatch.setattr(rt, "embedding_inference", object())
    monkeypatch.setattr(rt, "diarizations_in_flight", 0)
    monkeypatch.setattr(rt, "last_diarization_at", time.monotonic() - 999)
    monkeypatch.setattr(
        whisper_worker, "_drop_diarizer", lambda: setattr(rt, "diarization_pipeline", None)
    )
    task = asyncio.create_task(whisper_worker._reap_idle_diarizer(interval_s=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    assert rt.diarization_pipeline is None


def test_run_transcription_reports_progress_as_segments_decode():
    """faster-whisper entrega los segmentos al decodificarlos: el fin de cada
    uno es el avance real. Al terminar de transcribir, todo el audio."""
    seen: list[float] = []
    progress: dict = {"phase": "preparing", "startedMonotonic": 0.0, "updatedMonotonic": 0.0}

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
    # Decodificar y el VAD terminan al devolver el generador: de ahí en
    # adelante transcribe, y su velocidad se mide desde ese momento.
    assert progress["phase"] == "transcribing"
    assert progress["transcribeStartedMonotonic"] > 0
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
    assert jobs["req-1"]["transcribingMs"] is None
    whisper_worker.runtime.progress.clear()


# --- Preparación rápida de audios largos (19/11) ------------------------------
#
# Copias literales de lo que hace faster-whisper 1.2.1 (collect_chunks y
# FeatureExtractor.__call__/stft), para comprobar sin faster-whisper instalado
# que las versiones del worker dan exactamente lo mismo. Si faster-whisper está
# (entorno del worker), se compara además contra el original de verdad.


def _reference_collect_chunks(audio, chunks, sampling_rate=16000, max_duration=float("inf")):
    if not chunks:
        return [np.array([], dtype=np.float32)], [{"offset": 0, "duration": 0, "segments": []}]
    audio_chunks, chunks_metadata, current_segments = [], [], []
    current_duration = total_duration = 0
    current_audio = np.array([], dtype=np.float32)
    for chunk in chunks:
        if current_duration + chunk["end"] - chunk["start"] > max_duration * sampling_rate:
            audio_chunks.append(current_audio)
            chunks_metadata.append({
                "offset": total_duration / sampling_rate,
                "duration": current_duration / sampling_rate,
                "segments": current_segments,
            })
            total_duration += current_duration
            current_segments = []
            current_audio = audio[chunk["start"] : chunk["end"]]
            current_duration = chunk["end"] - chunk["start"]
        else:
            current_segments.append(chunk)
            current_audio = np.concatenate((current_audio, audio[chunk["start"] : chunk["end"]]))
            current_duration += chunk["end"] - chunk["start"]
    audio_chunks.append(current_audio)
    chunks_metadata.append({
        "offset": total_duration / sampling_rate,
        "duration": current_duration / sampling_rate,
        "segments": current_segments,
    })
    return audio_chunks, chunks_metadata


class _ReferenceFeatureExtractor:
    """El log-mel de faster-whisper 1.2.1, sin las comprobaciones de entrada."""

    def __init__(self, feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.time_per_frame = hop_length / sampling_rate
        self.sampling_rate = sampling_rate
        rng = np.random.default_rng(7)
        self.mel_filters = rng.random((feature_size, n_fft // 2 + 1)).astype("float32") / 50

    @staticmethod
    def stft(
        input_array,
        n_fft,
        hop_length=None,
        win_length=None,
        window=None,
        center=True,
        mode="reflect",
        normalized=False,
        onesided=None,
        return_complex=None,
    ):
        _ = win_length, normalized, onesided, return_complex
        squeeze = input_array.ndim == 1
        if squeeze:
            input_array = np.expand_dims(input_array, axis=0)
        if center:
            pad = n_fft // 2
            input_array = np.pad(input_array, ((0, 0), (pad, pad)), mode=mode)
        batch, length = input_array.shape
        n_frames = 1 + (length - n_fft) // hop_length
        frames = np.lib.stride_tricks.as_strided(
            input_array,
            (batch, n_frames, n_fft),
            (input_array.strides[0], hop_length * input_array.strides[1], input_array.strides[1]),
        )
        output = np.fft.rfft(frames * window, n=n_fft, axis=-1).transpose((0, 2, 1))
        return output.squeeze(0) if squeeze else output

    def __call__(self, waveform, padding=160, chunk_length=None):
        if chunk_length is not None:
            self.n_samples = chunk_length * self.sampling_rate
            self.nb_max_frames = self.n_samples // self.hop_length
        if waveform.dtype is not np.float32:
            waveform = waveform.astype(np.float32)
        if padding:
            waveform = np.pad(waveform, (0, padding))
        window = np.hanning(self.n_fft + 1)[:-1].astype("float32")
        stft = self.stft(
            waveform, self.n_fft, self.hop_length, window=window, return_complex=True
        ).astype("complex64")
        magnitudes = np.abs(stft[..., :-1]) ** 2
        mel_spec = self.mel_filters @ magnitudes
        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0


def _speech_chunks(rng, total, n):
    cuts = np.sort(rng.choice(np.arange(1, total), size=2 * n, replace=False))
    return [{"start": int(cuts[2 * i]), "end": int(cuts[2 * i + 1])} for i in range(n)]


@pytest.mark.parametrize("max_duration", [float("inf"), 1.0, 0.25, 0.0001])
def test_collect_chunks_linear_is_identical_to_faster_whisper(max_duration):
    """faster-whisper 1.2.1 junta los trozos con voz concatenando uno a uno
    (cuadrático): con 1 h 35 min eran 150 s de los ~190 de preparación. La
    versión lineal tiene que dar los mismos arrays, dtype y metadatos."""
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(16000 * 20).astype(np.float32)
    chunks = _speech_chunks(rng, audio.shape[0], 40)
    references = [_reference_collect_chunks]
    try:
        from faster_whisper.vad import collect_chunks as real
    except ImportError:
        pass
    else:
        references.append(real)
    got_audio, got_meta = whisper_worker._collect_chunks_linear(
        audio, chunks, max_duration=max_duration
    )
    for reference in references:
        ref_audio, ref_meta = reference(audio, chunks, max_duration=max_duration)
        assert got_meta == ref_meta
        assert len(got_audio) == len(ref_audio)
        for got, ref in zip(got_audio, ref_audio, strict=True):
            assert got.dtype == ref.dtype
            assert np.array_equal(got, ref)


def test_collect_chunks_linear_without_speech():
    audio, meta = whisper_worker._collect_chunks_linear(np.zeros(10, dtype=np.float32), [])
    ref_audio, ref_meta = _reference_collect_chunks(np.zeros(10, dtype=np.float32), [])
    assert meta == ref_meta
    assert audio[0].dtype == ref_audio[0].dtype
    assert audio[0].size == 0


@pytest.mark.parametrize("samples", [0, 1, 159, 400, 16000 * 3 + 7, 16000 * 9])
def test_blocked_log_mel_is_bit_identical(monkeypatch, samples):
    """El log-mel por tramos y en hilos da exactamente lo mismo que el de
    faster-whisper, incluido el suelo relativo al máximo GLOBAL (no por
    tramo). Tramos pequeños para que haya muchos y un resto."""
    monkeypatch.setattr(whisper_worker, "_LOG_MEL_FRAMES_PER_BLOCK", 37)
    monkeypatch.setattr(whisper_worker, "_log_mel_threads", lambda: 4)
    rng = np.random.default_rng(samples)
    wave = (rng.standard_normal(samples) * 0.1).astype(np.float32)
    if samples > 16000 * 5:
        wave[: 16000 * 2] *= 1e-6  # un tramo casi en silencio: lo toca el suelo global
    base = _ReferenceFeatureExtractor()
    fast = whisper_worker._make_fast_feature_extractor(base)
    expected = base(wave)
    got = fast(wave)
    assert got.dtype == expected.dtype == np.float32
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)
    # Conserva lo que faster-whisper lee del extractor.
    assert fast.nb_max_frames == base.nb_max_frames
    assert fast.sampling_rate == base.sampling_rate


def test_blocked_log_mel_matches_real_faster_whisper(monkeypatch):
    feature_extractor = pytest.importorskip("faster_whisper.feature_extractor")
    monkeypatch.setattr(whisper_worker, "_LOG_MEL_FRAMES_PER_BLOCK", 101)
    base = feature_extractor.FeatureExtractor(feature_size=128)
    fast = whisper_worker._make_fast_feature_extractor(base)
    wave = (np.random.default_rng(3).standard_normal(16000 * 45) * 0.1).astype(np.float32)
    assert np.array_equal(fast(wave), base(wave))
    assert np.array_equal(fast(wave, chunk_length=20), base(wave, chunk_length=20))
    assert fast.nb_max_frames == base.nb_max_frames


def test_install_fast_preprocessing_patches_faster_whisper_once(monkeypatch):
    fake_transcribe = SimpleNamespace(collect_chunks=_reference_collect_chunks)
    fake_vad = SimpleNamespace(collect_chunks=_reference_collect_chunks)
    fake_package = SimpleNamespace(transcribe=fake_transcribe, vad=fake_vad)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_package)
    monkeypatch.setitem(sys.modules, "faster_whisper.transcribe", fake_transcribe)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", fake_vad)
    model = SimpleNamespace(feature_extractor=_ReferenceFeatureExtractor())

    whisper_worker._install_fast_preprocessing(model, "cuda")
    first = model.feature_extractor
    assert fake_transcribe.collect_chunks is whisper_worker._collect_chunks_linear
    assert fake_vad.collect_chunks is whisper_worker._collect_chunks_linear
    assert isinstance(first, _ReferenceFeatureExtractor)
    assert first._ocabra_fast
    whisper_worker._install_fast_preprocessing(model, "cpu")
    assert model.feature_extractor is first  # idempotente


def test_run_transcription_reports_audio_seconds_while_preparing(monkeypatch):
    """La duración sale de la cabecera antes de decodificar: el panel la
    enseña durante la preparación. Luego manda la del audio decodificado."""
    progress: dict = {"phase": "preparing", "startedMonotonic": 0.0, "updatedMonotonic": 0.0}
    seen: dict = {}
    monkeypatch.setattr(whisper_worker, "_probe_audio_seconds", lambda path: 5700.0)

    class FakeModel:
        def transcribe(self, *args, **kwargs):
            seen.update(phase=progress["phase"], audio=progress.get("audioSeconds"))
            return iter(()), SimpleNamespace(duration=5699.999, language="es")

    whisper_worker._run_transcription(
        "/tmp/x.mp3", FakeModel(), None, None, 0.0, False, None, ["segment"], progress=progress
    )
    assert seen == {"phase": "preparing", "audio": 5700.0}
    assert progress["audioSeconds"] == 5699.999


def test_run_transcription_keeps_probed_seconds_without_info_duration(monkeypatch):
    progress: dict = {"phase": "preparing", "startedMonotonic": 0.0, "updatedMonotonic": 0.0}
    monkeypatch.setattr(whisper_worker, "_probe_audio_seconds", lambda path: 12.5)

    class FakeModel:
        def transcribe(self, *args, **kwargs):
            return iter(()), SimpleNamespace(language="es")

    whisper_worker._run_transcription(
        "/tmp/x.mp3", FakeModel(), None, None, 0.0, False, None, ["segment"], progress=progress
    )
    assert progress["audioSeconds"] == 12.5


def test_probe_audio_seconds_unknown_file_is_none(tmp_path):
    assert whisper_worker._probe_audio_seconds(str(tmp_path / "missing.mp3")) is None
