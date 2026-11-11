import argparse
import asyncio
import gc
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

if TYPE_CHECKING:  # numpy es una dependencia opcional del backend de audio
    import numpy as np

# diarization_lib.py is a sibling script, not an installed package. Running
# this file directly (python .../whisper_worker.py, the production path) puts
# its own directory on sys.path[0] automatically; importing it as
# workers.whisper_worker (tests, via the implicit namespace package) does not
# — only the repo's backend/ root is on sys.path then. Make the sibling
# directory resolvable in both cases instead of relying on which one happens
# to already be the case.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diarization_lib import (  # noqa: E402
    DEFAULT_DIARIZATION_MODEL_ID,
    diarize_audio_file,
    load_diarization_pipeline,
    parse_optional_bool,
    split_segments_by_speaker,
    to_srt,
    to_text,
    to_vtt,
)

# CTranslate2 shares model weights across workers on the same device — extra
# workers cost ~tens-to-low-hundreds of MB each (measured empirically), not a
# full model copy. Compute is the real limit, not VRAM: on an RTX 3060 with
# large-v3-turbo (int8), ~14 workers was the point where per-request latency
# crossed real-time (RTF~1.0); 5 keeps comfortable headroom (RTF~0.3-0.4)
# without needing per-deployment tuning before anyone measures their own case.
DEFAULT_NUM_WORKERS = 5


class WhisperRuntime:
    def __init__(self) -> None:
        self.model_id: str = ""
        self.gpu_indices: list[int] = []
        self.device: str = "cpu"
        self.compute_type: str = "int8"
        self.model: Any = None
        self.error: str | None = None
        self.diarize_default: bool = False
        self.diarization_model_id: str = DEFAULT_DIARIZATION_MODEL_ID
        self.diarization_pipeline: Any = None
        self.diarization_error: str | None = None
        self.embedding_inference: Any = None  # pyannote speaker-embedding model (lazy)
        self.num_workers: int = DEFAULT_NUM_WORKERS
        # The diarizer loads on demand and holds ~2.6 GB more of VRAM (4
        # concurrent diarized transcriptions peaked at 6.1 GB against 3.4 GB
        # without, and PyTorch kept 5.3 GB reserved afterwards, 2026-11-11).
        # Drop it once nobody has diarized for this long; 0 keeps it.
        self.diarizer_idle_unload_s: float = 300.0
        self.diarizations_in_flight: int = 0
        self.last_diarization_at: float = 0.0


runtime = WhisperRuntime()


def create_app(
    model_id: str,
    gpu_indices: list[int],
    diarize_default: bool = False,
    diarization_model_id: str = DEFAULT_DIARIZATION_MODEL_ID,
    num_workers: int = DEFAULT_NUM_WORKERS,
    diarizer_idle_unload_s: float = 300.0,
) -> FastAPI:
    runtime.diarizer_idle_unload_s = float(diarizer_idle_unload_s)
    runtime.model_id = model_id
    runtime.gpu_indices = gpu_indices
    runtime.diarize_default = diarize_default
    runtime.diarization_model_id = diarization_model_id
    runtime.num_workers = num_workers

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"} else "cpu"
        runtime.compute_type = "float16" if runtime.device == "cuda" else "int8"
        if runtime.device == "cuda" and not _cuda_whisper_available():
            runtime.device = "cpu"
            runtime.compute_type = "int8"

        try:
            runtime.model = _create_whisper_model(
                runtime.model_id,
                runtime.device,
                runtime.compute_type,
                runtime.num_workers,
            )
            runtime.error = None

            runtime.diarization_pipeline = None
            runtime.diarization_error = None
            if runtime.diarize_default:
                runtime.diarization_pipeline = await asyncio.to_thread(
                    load_diarization_pipeline,
                    runtime.diarization_model_id,
                    runtime.device,
                    os.getenv("HF_TOKEN", "").strip() or None,
                )
        except Exception as exc:
            runtime.model = None
            runtime.error = str(exc)

        reaper = asyncio.create_task(_reap_idle_diarizer())
        yield
        reaper.cancel()

        runtime.model = None
        runtime.diarization_pipeline = None

    app = FastAPI(title="oCabra Whisper Worker", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        if runtime.model is None:
            raise HTTPException(status_code=503, detail=runtime.error or "Model not ready")
        if runtime.diarize_default and runtime.diarization_pipeline is None:
            detail = runtime.diarization_error or "Diarization pipeline not ready"
            raise HTTPException(status_code=503, detail=detail)
        return JSONResponse({"status": "ok"})

    @app.get("/info")
    async def info() -> JSONResponse:
        return JSONResponse(
            {
                "backend": "whisper",
                "model_id": runtime.model_id,
                "gpu_indices": runtime.gpu_indices,
                "device": runtime.device,
                "compute_type": runtime.compute_type,
                "num_workers": runtime.num_workers,
                "loaded": runtime.model is not None,
                "error": runtime.error,
                "diarize_default": runtime.diarize_default,
                "diarization_model_id": runtime.diarization_model_id,
                "diarization_ready": runtime.diarization_pipeline is not None,
                "diarization_error": runtime.diarization_error,
            }
        )

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),  # noqa: B008
        language: str | None = Form(default=None),  # noqa: B008
        prompt: str | None = Form(default=None),  # noqa: B008
        response_format: str = Form(default="json"),  # noqa: B008
        temperature: float = Form(default=0.0),  # noqa: B008
        timestamp_granularities: list[str] = Form(default=["segment"]),  # noqa: B008
        diarize: str | None = Form(default=None),  # noqa: B008
        vad_filter: str | None = Form(default=None),  # noqa: B008
        condition_on_previous_text: str | None = Form(default=None),  # noqa: B008
    ):
        if runtime.model is None:
            raise HTTPException(status_code=503, detail=runtime.error or "Whisper model not loaded")

        try:
            diarize_request = parse_optional_bool(diarize)
            vad_request = parse_optional_bool(vad_filter)
            condition_request = parse_optional_bool(condition_on_previous_text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Defaults chosen to prevent Whisper hallucination loops swallowing real
        # speech: VAD drops non-speech, and NOT conditioning on previous text stops
        # a degenerate loop from poisoning the context to the end of the file
        # (same approach as WhisperX). Callers may override either.
        vad_effective = True if vad_request is None else vad_request
        condition_effective = False if condition_request is None else condition_request

        use_diarization = runtime.diarize_default if diarize_request is None else diarize_request
        if use_diarization and runtime.diarization_pipeline is None:
            try:
                runtime.diarization_pipeline = await asyncio.to_thread(
                    load_diarization_pipeline,
                    runtime.diarization_model_id,
                    runtime.device,
                    os.getenv("HF_TOKEN", "").strip() or None,
                )
                runtime.diarization_error = None
            except Exception as exc:
                runtime.diarization_error = str(exc)
                raise HTTPException(status_code=503, detail=runtime.diarization_error) from exc

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        if use_diarization:
            runtime.diarizations_in_flight += 1

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            try:
                result = await asyncio.to_thread(
                    _run_transcription,
                    tmp_path,
                    runtime.model,
                    language,
                    prompt,
                    temperature,
                    use_diarization,
                    runtime.diarization_pipeline,
                    timestamp_granularities,
                    vad_filter=vad_effective,
                    condition_on_previous_text=condition_effective,
                )
            except Exception as exc:
                if runtime.device == "cuda" and _is_missing_cudnn_error(exc):
                    runtime.model = await asyncio.to_thread(
                        _create_whisper_model,
                        runtime.model_id,
                        "cpu",
                        "int8",
                        runtime.num_workers,
                    )
                    runtime.device = "cpu"
                    runtime.compute_type = "int8"
                    runtime.error = None
                    result = await asyncio.to_thread(
                        _run_transcription,
                        tmp_path,
                        runtime.model,
                        language,
                        prompt,
                        temperature,
                        use_diarization,
                        runtime.diarization_pipeline,
                        timestamp_granularities,
                    )
                else:
                    raise
        finally:
            if use_diarization:
                runtime.diarizations_in_flight -= 1
                runtime.last_diarization_at = time.monotonic()
            with suppress(FileNotFoundError):
                os.unlink(tmp_path)

        normalized_format = response_format.lower()
        text = result["text"]
        segments = result["segments"]
        detected_language = result["language"]

        if normalized_format == "text":
            return PlainTextResponse(to_text(segments))
        if normalized_format == "srt":
            return PlainTextResponse(to_srt(segments))
        if normalized_format == "vtt":
            return PlainTextResponse(to_vtt(segments))
        if normalized_format == "verbose_json":
            payload = {
                "text": text,
                "language": detected_language,
                "segments": segments,
                "timestamp_granularities": timestamp_granularities,
            }
            if result.get("words") is not None:
                payload["words"] = result["words"]
            if result.get("diarization") is not None:
                payload["diarization"] = result["diarization"]
                payload["speakers"] = result.get("speakers", [])
                if result.get("speaker_embeddings"):
                    payload["speaker_embeddings"] = result["speaker_embeddings"]
            return JSONResponse(payload)

        return JSONResponse({"text": text})

    return app




def _cuda_whisper_available() -> bool:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:
        pass

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False

class _NemoModelAdapter:
    def __init__(self, model: Any) -> None:
        self._model = model

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
        condition_on_previous_text: bool = False,
        **_kwargs: Any,
    ) -> tuple[list[Any], Any]:
        # NeMo checkpoints here produce a single transcript without word-level
        # timestamps; accept (and ignore) the faster-whisper-only options — the
        # ``**_kwargs`` catch-all keeps the shared call site from breaking NeMo
        # when new Whisper-only knobs are added.
        _ = initial_prompt, temperature, word_timestamps, vad_filter, condition_on_previous_text
        text = _nemo_transcribe_text(self._model, audio_path=audio_path, language=language)
        segment = SimpleNamespace(start=0.0, end=0.0, text=text)
        info = SimpleNamespace(language=language or "unknown")
        return [segment], info


def _is_nemo_model_id(model_id: str) -> bool:
    candidate = Path(model_id)
    if candidate.suffix.lower() == ".nemo":
        return True
    return candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".nemo"


def _create_whisper_model(
    model_id: str, device: str, compute_type: str, num_workers: int = DEFAULT_NUM_WORKERS
) -> Any:
    if _is_nemo_model_id(model_id):
        # NeMo models don't go through CTranslate2 — no multi-worker concept here.
        return _create_nemo_model(model_id=model_id, device=device)

    from faster_whisper import WhisperModel

    return WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
        num_workers=max(1, num_workers),
    )


def _create_nemo_model(model_id: str, device: str) -> _NemoModelAdapter:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "NeMo STT model requested but torch is not installed. Install backend extras '[audio]'."
        ) from exc

    try:
        from nemo.collections.asr.models import ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "NeMo STT model requested but nemo_toolkit is not installed. "
            "Install backend extras that include nemo_toolkit[asr]."
        ) from exc

    map_location = "cuda" if device == "cuda" else "cpu"
    model = ASRModel.restore_from(restore_path=model_id, map_location=map_location)
    model = model.eval()
    if device == "cuda":
        model = model.to(torch.device("cuda"))
    return _NemoModelAdapter(model)


def _decode_audio_with_av(audio_path: str) -> tuple["np.ndarray", int]:
    import av
    import numpy as np

    container = av.open(audio_path)
    try:
        stream = container.streams.audio[0]
        chunks: list[np.ndarray] = []
        sample_rate = int(stream.rate or 16000)
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            # PyAV may return planar/interleaved data. Normalize to [samples, channels].
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim == 2:
                if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                    arr = arr.T
            else:
                arr = arr.reshape(arr.shape[0], -1)

            if arr.dtype.kind in {"i", "u"}:
                max_abs = float(np.iinfo(arr.dtype).max)
                arr = arr.astype(np.float32) / max_abs
            else:
                arr = arr.astype(np.float32)
            chunks.append(arr)

        if not chunks:
            raise RuntimeError("Unable to decode audio stream with PyAV")

        samples = np.concatenate(chunks, axis=0)
        return samples, sample_rate
    finally:
        container.close()


def _prepare_audio_for_nemo(audio_path: str) -> tuple[str, str | None]:
    import numpy as np
    import soundfile as sf

    try:
        samples, sample_rate = sf.read(audio_path, always_2d=True)
    except Exception:
        # Some containers (e.g. m4a/aac) are not readable via libsndfile.
        samples, sample_rate = _decode_audio_with_av(audio_path)

    # NeMo transcribe() expects mono waveform at 16kHz for these ASR checkpoints.
    mono = np.mean(samples, axis=1, dtype=np.float32)

    target_sr = 16000
    if int(sample_rate) != target_sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=int(sample_rate), target_sr=target_sr)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, mono, target_sr, subtype="PCM_16")
        return tmp.name, tmp.name


def _extract_nemo_text(result: Any) -> str:
    if isinstance(result, list):
        if not result:
            return ""
        first = result[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for key in ("text", "pred_text", "transcript"):
                value = first.get(key)
                if isinstance(value, str):
                    return value.strip()
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text.strip()
        return str(first).strip()

    if isinstance(result, dict):
        for key in ("text", "pred_text", "transcript"):
            value = result.get(key)
            if isinstance(value, str):
                return value.strip()

    if isinstance(result, str):
        return result.strip()

    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text.strip()

    return str(result).strip()


def _run_nemo_transcribe_once(model: Any, path: str, language: str | None) -> str:
    transcribe_attempts: list[dict[str, Any]] = []
    if language:
        transcribe_attempts.append({"paths2audio_files": [path], "language_id": language})
        transcribe_attempts.append({"audio": [path], "language": language})
        transcribe_attempts.append({
            "audio": [path],
            "taskname": "asr",
            "source_lang": language,
            "target_lang": "en",
            "pnc": "yes",
        })
    transcribe_attempts.append({"paths2audio_files": [path]})
    transcribe_attempts.append({"audio": [path]})
    transcribe_attempts.append({
        "audio": [path],
        "taskname": "asr",
        "source_lang": "en",
        "target_lang": "en",
        "pnc": "yes",
    })

    result: Any | None = None
    for kwargs in transcribe_attempts:
        try:
            result = model.transcribe(**kwargs)
            break
        except TypeError:
            continue

    if result is None:
        result = model.transcribe([path])
    return _extract_nemo_text(result)


def _chunk_audio_for_nemo(
    prepared_path: str,
    chunk_seconds: int = 20,
    min_tail_seconds: float = 0.5,
) -> tuple[list[str], list[str]]:
    import soundfile as sf

    audio, sample_rate = sf.read(prepared_path, always_2d=False)
    if audio is None:
        return [prepared_path], []

    total_samples = int(audio.shape[0]) if hasattr(audio, "shape") else 0
    chunk_size = int(sample_rate) * chunk_seconds
    if total_samples <= chunk_size or chunk_size <= 0:
        return [prepared_path], []

    chunk_paths: list[str] = []
    cleanup: list[str] = []
    min_tail_samples = max(1, int(float(sample_rate) * float(min_tail_seconds)))
    start = 0
    while start < total_samples:
        remaining = total_samples - start
        if remaining <= chunk_size:
            end = total_samples
        else:
            end = start + chunk_size
            tail = total_samples - end
            # Avoid creating tiny tail chunks that can fail NeMo feature normalization.
            if 0 < tail < min_tail_samples:
                end = total_samples
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio[start:end], int(sample_rate), subtype="PCM_16")
            chunk_paths.append(tmp.name)
            cleanup.append(tmp.name)
        start = end
    return chunk_paths, cleanup


def _nemo_transcribe_text(model: Any, audio_path: str, language: str | None) -> str:
    prepared_path, cleanup_path = _prepare_audio_for_nemo(audio_path)
    chunk_cleanup: list[str] = []
    try:
        chunk_paths, chunk_cleanup = _chunk_audio_for_nemo(prepared_path, chunk_seconds=20)
        texts: list[str] = []
        for chunk in chunk_paths:
            try:
                text = _run_nemo_transcribe_once(model, chunk, language)
            except ValueError as exc:
                message = str(exc).lower()
                if "normalize_batch" in message and "length 1" in message:
                    continue
                raise
            if text:
                texts.append(text.strip())
        return " ".join(t for t in texts if t).strip()
    finally:
        for chunk in chunk_cleanup:
            with suppress(FileNotFoundError):
                os.unlink(chunk)
        if cleanup_path:
            with suppress(FileNotFoundError):
                os.unlink(cleanup_path)


def _is_missing_cudnn_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "libcudnn_ops_infer.so.8" in message:
        return True
    if "libcudnn_ops.so.9" in message:
        return True
    return "cudnn" in message and "cannot open shared object file" in message

def _run_transcription(
    audio_path: str,
    model: Any,
    language: str | None,
    prompt: str | None,
    temperature: float,
    diarize: bool,
    diarization_pipeline: Any,
    timestamp_granularities: Sequence[str] | None = None,
    *,
    vad_filter: bool = True,
    condition_on_previous_text: bool = False,
) -> dict[str, Any]:
    granularities = list(timestamp_granularities or ["segment"])
    expose_words = "word" in granularities
    diarizing = bool(diarize and diarization_pipeline is not None)
    # Word timestamps are needed when the caller requests word granularity, and
    # also internally for diarization so segments can be split on speaker turns.
    need_words = expose_words or diarizing

    segments, info = model.transcribe(
        audio_path,
        language=language,
        initial_prompt=prompt,
        temperature=temperature,
        word_timestamps=need_words,
        # VAD splits the audio on silences so faster-whisper emits per-utterance
        # segments instead of collapsing everything into one block.
        vad_filter=vad_filter,
        # False stops a hallucination loop from propagating via the decoder
        # context to the end of the file (see the /transcribe defaults).
        condition_on_previous_text=condition_on_previous_text,
    )

    segment_list: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for idx, segment in enumerate(segments):
        text_parts.append(segment.text)
        entry: dict[str, Any] = {
            "id": idx,
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            # Confidence signals used to drop hallucinations over silence/noise
            # (Whisper invents "Gracias/Adiós/…" fillers on non-speech).
            "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
            "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
        }
        seg_words = getattr(segment, "words", None)
        if need_words and seg_words:
            entry["words"] = [
                {
                    "word": word.word,
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": float(getattr(word, "probability", 0.0) or 0.0),
                }
                for word in seg_words
            ]
        segment_list.append(entry)

    diarization_turns: list[dict[str, Any]] | None = None
    speakers: list[str] = []
    speaker_embeddings: dict[str, list[float]] = {}
    if diarizing:
        diar_result = diarize_audio_file(diarization_pipeline, audio_path)
        diarization_turns = diar_result["turns"]
        # Split segments on speaker turns (using word timestamps when present)
        # so each segment carries the speaker that actually spoke it.
        segment_list = split_segments_by_speaker(segment_list, diarization_turns)
        # The speaker roster reflects every speaker in the diarization track,
        # not just the ones that happened to win a segment.
        speakers = diar_result["speakers"]
        speaker_embeddings = diar_result["embeddings"]

    # Drop the internal word data unless the caller asked for word granularity.
    if not expose_words:
        for segment in segment_list:
            segment.pop("words", None)

    if diarization_turns is not None:
        final_text = to_text(segment_list)
    else:
        final_text = " ".join(part.strip() for part in text_parts if part).strip()

    payload: dict[str, Any] = {
        "text": final_text,
        "segments": segment_list,
        "language": getattr(info, "language", language or "unknown"),
    }
    if expose_words:
        payload["words"] = [
            word for segment in segment_list for word in segment.get("words", [])
        ]
    if diarization_turns is not None:
        payload["diarization"] = diarization_turns
        payload["speakers"] = speakers
        if speaker_embeddings:
            payload["speaker_embeddings"] = speaker_embeddings
    return payload



def _drop_diarizer() -> None:
    """Release the diarization models and give their VRAM back to the card."""
    runtime.diarization_pipeline = None
    runtime.embedding_inference = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - CPU-only or no torch: nothing to give back
        pass


async def _reap_idle_diarizer(interval_s: float = 30.0) -> None:
    """Unload the on-demand diarizer once it has been idle long enough.

    Never while a diarization is running, and never for a profile that
    diarizes every request (``diarize_default``): that one needs it loaded.
    """
    while True:
        await asyncio.sleep(interval_s)
        idle_limit = runtime.diarizer_idle_unload_s
        if (
            idle_limit > 0
            and runtime.diarization_pipeline is not None
            and not runtime.diarize_default
            and runtime.diarizations_in_flight == 0
            and time.monotonic() - runtime.last_diarization_at >= idle_limit
        ):
            await asyncio.to_thread(_drop_diarizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="oCabra Whisper worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu-indices", default="")
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--diarization-model-id", default=DEFAULT_DIARIZATION_MODEL_ID)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--diarizer-idle-unload-s", type=float, default=300.0)
    args = parser.parse_args()

    gpu_indices = [int(item) for item in args.gpu_indices.split(",") if item.strip()]
    app = create_app(
        args.model_id,
        gpu_indices,
        diarize_default=bool(args.diarize),
        diarization_model_id=args.diarization_model_id,
        num_workers=args.num_workers,
        diarizer_idle_unload_s=args.diarizer_idle_unload_s,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
