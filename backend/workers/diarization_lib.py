"""Shared speaker-diarization helpers (pyannote) — extracted from whisper_worker.py
so whisper.cpp's diarization sidecar (diarization_worker.py) and the faster-whisper
worker share one implementation instead of two copies that can drift.

Import discipline: every heavy dependency (torch, pyannote.audio, numpy, av,
torchaudio) is imported LOCALLY inside the function that needs it, never at
module level. That makes this module safe to import from the main oCabra API
process too (ocabra/backends/whispercpp_backend.py loads it by file path to
reuse the pure merge/formatting functions below) — as long as the caller never
invokes load_diarization_pipeline/embed_waveform/diarize_audio_file there,
nothing ever touches torch/pyannote outside a backend's own isolated venv.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"

logger = logging.getLogger("ocabra.diarization_lib")


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "":
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Invalid boolean value. Use true/false.")


def patch_torchaudio_compat() -> None:
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):

        class _CompatAudioMetaData:  # pragma: no cover
            def __init__(
                self,
                sample_rate: int,
                num_frames: int,
                num_channels: int,
                bits_per_sample: int = 0,
                encoding: str = "PCM_S",
            ) -> None:
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

        torchaudio.AudioMetaData = _CompatAudioMetaData

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "ffmpeg"
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None
    if not hasattr(torchaudio, "info"):
        import soundfile as sf

        def _compat_info(path: str, backend: str | None = None):  # pragma: no cover
            del backend
            metadata = sf.info(path)
            return torchaudio.AudioMetaData(
                sample_rate=int(metadata.samplerate),
                num_frames=int(metadata.frames),
                num_channels=int(metadata.channels),
                bits_per_sample=0,
                encoding="PCM_S",
            )

        torchaudio.info = _compat_info


def load_diarization_pipeline(model_id: str, device: str, hf_token: str | None) -> Any:
    patch_torchaudio_compat()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Diarization requested but pyannote.audio is not installed. "
            "Install backend extras '[audio]' including pyannote.audio."
        ) from exc

    kwargs: dict[str, Any] = {}
    if hf_token:
        kwargs["token"] = hf_token

    # Resolve local model directory before attempting a remote download.
    pipeline_source: str = model_id
    models_dir = os.environ.get("MODELS_DIR", "")
    if models_dir:
        local_hf = Path(models_dir) / "huggingface" / model_id.replace("/", "--")
        if local_hf.is_dir() and (local_hf / "config.yaml").exists():
            pipeline_source = str(local_hf)
            kwargs.pop("token", None)  # local path needs no auth

    try:
        import torch
        from torch.torch_version import TorchVersion

        torch.serialization.add_safe_globals([TorchVersion])
    except Exception:
        pass

    # pyannote ≥4.0 always calls get_plda() even for AgglomerativeClustering
    # which doesn't use PLDA. Patch it to return None on download failures so
    # the pipeline still loads when PLDA is unavailable (gated HF access).
    _patch_targets: list[Any] = []
    _orig_get_plda = None
    try:
        import pyannote.audio.pipelines.speaker_diarization as _sd_mod
        from pyannote.audio.pipelines.utils import getter as _getter

        _orig_get_plda = _getter.get_plda

        def _safe_get_plda(plda, **kw):  # type: ignore[override]
            try:
                return _orig_get_plda(plda, **kw)
            except Exception:
                return None

        # Must patch both the canonical location and the imported reference.
        _getter.get_plda = _safe_get_plda
        _sd_mod.get_plda = _safe_get_plda
        _patch_targets = [(_getter, "get_plda"), (_sd_mod, "get_plda")]
    except Exception:
        pass

    try:
        pipeline = Pipeline.from_pretrained(pipeline_source, **kwargs)
    finally:
        for _mod, _attr in _patch_targets:
            setattr(_mod, _attr, _orig_get_plda)

    try:
        import torch

        target_device = torch.device("cuda" if device == "cuda" else "cpu")
        pipeline.to(target_device)
    except Exception:
        pass

    return pipeline


def embed_waveform(pipeline: Any, waveform: Any, sample_rate: int) -> list[float] | None:
    """Compute a single speaker embedding for a mono 16 kHz waveform crop, reusing
    the diarization pipeline's OWN (already-loaded) ``PretrainedSpeakerEmbedding``
    — avoids a second gated download and matches how the pipeline clusters speakers.
    Its call signature is ``embedder(batch=(B, channels, samples)) -> (B, dim)``.
    Returns a plain float list, or None if unavailable."""
    import numpy as _np

    emb = getattr(pipeline, "_embedding", None)
    if emb is None:
        return None
    try:
        wf = waveform
        if hasattr(wf, "dim"):
            if wf.dim() == 1:
                wf = wf.unsqueeze(0)  # (samples,) -> (1, samples)
            batch = wf.unsqueeze(0)  # (channels, samples) -> (1, channels, samples)
        else:
            import torch as _torch

            batch = _torch.as_tensor(wf, dtype=_torch.float32).reshape(1, 1, -1)
        out = emb(batch)  # ndarray (1, dim)
        arr = _np.asarray(getattr(out, "data", out), dtype=float)
        if arr.ndim > 1:
            arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        arr = arr.ravel()
        if arr.size == 0 or bool(_np.any(_np.isnan(arr))):
            return None
        return [float(x) for x in arr.tolist()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedding_extract_failed: %s", exc)
        return None


def collect_diarization_turns(annotation: Any) -> list[dict[str, Any]]:
    # pyannote ≥4.0 returns DiarizeOutput; pyannote 3.x returns Annotation directly.
    if hasattr(annotation, "speaker_diarization"):
        annotation = annotation.speaker_diarization
    turns: list[dict[str, Any]] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": str(speaker),
            }
        )
    turns.sort(key=lambda item: item["start"])
    return turns


def speaker_for_interval(
    start: float,
    end: float,
    diarization_turns: list[dict[str, Any]],
) -> str | None:
    """Return the speaker whose diarization turn overlaps [start, end] most."""
    best_speaker: str | None = None
    best_overlap = 0.0
    for turn in diarization_turns:
        overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]
    return best_speaker


def segment_from_words(
    segment_id: int,
    words: list[dict[str, Any]],
    speaker: str | None,
) -> dict[str, Any]:
    segment: dict[str, Any] = {
        "id": segment_id,
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": "".join(word["word"] for word in words).strip(),
        "words": words,
    }
    if speaker is not None:
        segment["speaker"] = speaker
    return segment


def split_segments_by_speaker(
    segments: list[dict[str, Any]],
    diarization_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Re-segment the transcript so each segment maps to a single speaker.

    When a segment has word-level timestamps, consecutive words are grouped by
    their overlapping diarization speaker and emitted as separate segments on
    speaker changes. Segments without words fall back to whole-segment overlap.
    """
    result: list[dict[str, Any]] = []
    next_id = 0

    for segment in segments:
        words = segment.get("words")
        if not words:
            speaker = speaker_for_interval(
                float(segment["start"]), float(segment["end"]), diarization_turns
            )
            new_segment = {**segment, "id": next_id}
            if speaker is not None:
                new_segment["speaker"] = speaker
            result.append(new_segment)
            next_id += 1
            continue

        group_words: list[dict[str, Any]] = []
        group_speaker: str | None = None
        for word in words:
            speaker = speaker_for_interval(
                float(word["start"]), float(word["end"]), diarization_turns
            )
            if group_words and speaker != group_speaker:
                result.append(segment_from_words(next_id, group_words, group_speaker))
                next_id += 1
                group_words = []
            group_speaker = speaker
            group_words.append(word)

        if group_words:
            result.append(segment_from_words(next_id, group_words, group_speaker))
            next_id += 1

    return result


class DiarizationOutOfMemory(MemoryError):
    """The diarizer ran out of GPU memory even at batch size 1."""


_BATCH_ATTRS = ("segmentation_batch_size", "embedding_batch_size")
# Below this the diarizer gets slow for little memory back; past it oCabra
# frees memory on the card instead.
MIN_DIARIZATION_BATCH = 4


def _is_out_of_memory(exc: BaseException) -> bool:
    # pyannote turns CUDA OOM into MemoryError("batch_size (32) is probably too
    # large"); other paths raise torch.cuda.OutOfMemoryError (a RuntimeError).
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


def _lower_batch_sizes(pipeline: Any) -> dict[str, int] | None:
    """Halve the pipeline's batch sizes (32 -> 16 -> 8 -> 4). Returns the new
    values, or None when they are all at the floor already."""
    lowered: dict[str, int] = {}
    for attr in _BATCH_ATTRS:
        try:
            current = int(getattr(pipeline, attr))
        except Exception:  # noqa: BLE001 - attribute missing on this pipeline
            continue
        if current > MIN_DIARIZATION_BATCH:
            new = max(MIN_DIARIZATION_BATCH, current // 2)
            setattr(pipeline, attr, new)
            lowered[attr] = new
    return lowered or None


def _release_cuda_cache() -> None:
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def run_pipeline_with_backoff(pipeline: Any, audio_input: Any) -> Any:
    """Run the diarizer, lowering its batch sizes each time it runs out of GPU
    memory. Slower, but it fits next to a model that filled the card: qwen3.8
    on both GPUs with its projector left pyannote no room at batch 32
    (2026-11-11). The lowered sizes stay until the diarizer is unloaded,
    because the card stays as full as it was. When even batch
    ``MIN_DIARIZATION_BATCH`` runs out of memory it raises
    :class:`DiarizationOutOfMemory`, so the caller can free memory and retry.
    """
    while True:
        try:
            return pipeline(audio_input)
        except Exception as exc:
            if not _is_out_of_memory(exc):
                raise
            lowered = _lower_batch_sizes(pipeline)
            _release_cuda_cache()
            if lowered is None:
                raise DiarizationOutOfMemory(
                    "diarization_out_of_memory: no GPU memory for the diarizer even at "
                    f"batch size {MIN_DIARIZATION_BATCH}"
                ) from exc
            logger.warning("diarization_out_of_memory_retrying: %s", lowered)


def diarize_audio_file(
    pipeline: Any,
    audio_path: str,
    *,
    sr_target: int = 16000,
    with_embeddings: bool = True,
) -> dict[str, Any]:
    """Run ``pipeline`` over ``audio_path`` and return turns/speakers/embeddings.

    pyannote ≥4.0 uses torchcodec (requires FFmpeg shared libs) for audio I/O.
    Since torchcodec may not be available, load the audio via PyAV (bundled
    with faster-whisper, also present in whisper.cpp's diarization sidecar
    venv) and pass a waveform dict instead.
    """
    import av as _av
    import numpy as _np
    import torch as _torch

    frames: list[_np.ndarray] = []
    with _av.open(audio_path) as container:
        resampler = _av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sr_target)
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                frames.append(resampled.to_ndarray())
    waveform_np = _np.concatenate(frames, axis=1) if frames else _np.zeros((1, 1), dtype=_np.float32)
    waveform = _torch.from_numpy(waveform_np)
    audio_input: Any = {"waveform": waveform, "sample_rate": sr_target}

    annotation = run_pipeline_with_backoff(pipeline, audio_input)
    turns = collect_diarization_turns(annotation)
    speakers = sorted({turn["speaker"] for turn in turns})

    embeddings: dict[str, list[float]] = {}
    if with_embeddings and turns:
        by_speaker: dict[str, list[dict[str, Any]]] = {}
        for turn in turns:
            by_speaker.setdefault(turn["speaker"], []).append(turn)
        for speaker, speaker_turns in by_speaker.items():
            longest = max(speaker_turns, key=lambda t: t["end"] - t["start"])
            start_sample = int(float(longest["start"]) * sr_target)
            end_sample = int(float(longest["end"]) * sr_target)
            if end_sample - start_sample < int(sr_target * 0.3):  # need ≥0.3 s for a stable embedding
                continue
            vector = embed_waveform(pipeline, waveform[:, start_sample:end_sample], sr_target)
            if vector:
                embeddings[str(speaker)] = vector

    return {"turns": turns, "speakers": speakers, "embeddings": embeddings}


def to_text(segments: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    current_speaker: str | None = None
    for segment in segments:
        speaker = segment.get("speaker")
        text = segment["text"].strip()
        if speaker:
            if speaker != current_speaker:
                current_speaker = speaker
                lines.append(f"[{speaker}] {text}")
            else:
                lines.append(text)
        else:
            lines.append(text)
    return "\n".join(lines)


def to_srt(segments: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, segment in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_ts(segment['start'])} --> {format_srt_ts(segment['end'])}")
        speaker = segment.get("speaker")
        text = segment["text"].strip()
        lines.append(f"[{speaker}] {text}" if speaker else text)
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: Sequence[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.append(f"{format_vtt_ts(segment['start'])} --> {format_vtt_ts(segment['end'])}")
        speaker = segment.get("speaker")
        text = segment["text"].strip()
        lines.append(f"<v {speaker}>{text}" if speaker else text)
        lines.append("")
    return "\n".join(lines)


def format_srt_ts(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    millis = int((value - int(value)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_vtt_ts(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    millis = int((value - int(value)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
