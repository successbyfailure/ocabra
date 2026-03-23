import argparse
import asyncio
import os
import tempfile
from collections.abc import Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"


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


runtime = WhisperRuntime()


def create_app(
    model_id: str,
    gpu_indices: list[int],
    diarize_default: bool = False,
    diarization_model_id: str = DEFAULT_DIARIZATION_MODEL_ID,
) -> FastAPI:
    runtime.model_id = model_id
    runtime.gpu_indices = gpu_indices
    runtime.diarize_default = diarize_default
    runtime.diarization_model_id = diarization_model_id

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
            )
            runtime.error = None

            runtime.diarization_pipeline = None
            runtime.diarization_error = None
            if runtime.diarize_default:
                runtime.diarization_pipeline = await asyncio.to_thread(
                    _load_diarization_pipeline,
                    runtime.diarization_model_id,
                    runtime.device,
                    os.getenv("HF_TOKEN", "").strip() or None,
                )
        except Exception as exc:
            runtime.model = None
            runtime.error = str(exc)

        yield

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
    ):
        if runtime.model is None:
            raise HTTPException(status_code=503, detail=runtime.error or "Whisper model not loaded")

        try:
            diarize_request = _parse_optional_bool(diarize)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        use_diarization = runtime.diarize_default if diarize_request is None else diarize_request
        if use_diarization and runtime.diarization_pipeline is None:
            try:
                runtime.diarization_pipeline = await asyncio.to_thread(
                    _load_diarization_pipeline,
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
                )
            except Exception as exc:
                if runtime.device == "cuda" and _is_missing_cudnn_error(exc):
                    runtime.model = await asyncio.to_thread(
                        _create_whisper_model,
                        runtime.model_id,
                        "cpu",
                        "int8",
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
                    )
                else:
                    raise
        finally:
            with suppress(FileNotFoundError):
                os.unlink(tmp_path)

        normalized_format = response_format.lower()
        text = result["text"]
        segments = result["segments"]
        detected_language = result["language"]

        if normalized_format == "text":
            return PlainTextResponse(text)
        if normalized_format == "srt":
            return PlainTextResponse(_to_srt(segments))
        if normalized_format == "vtt":
            return PlainTextResponse(_to_vtt(segments))
        if normalized_format == "verbose_json":
            payload = {
                "text": text,
                "language": detected_language,
                "segments": segments,
                "timestamp_granularities": timestamp_granularities,
            }
            if result.get("diarization") is not None:
                payload["diarization"] = result["diarization"]
                payload["speakers"] = result.get("speakers", [])
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
    ) -> tuple[list[Any], Any]:
        _ = initial_prompt, temperature
        text = _nemo_transcribe_text(self._model, audio_path=audio_path, language=language)
        segment = SimpleNamespace(start=0.0, end=0.0, text=text)
        info = SimpleNamespace(language=language or "unknown")
        return [segment], info


def _is_nemo_model_id(model_id: str) -> bool:
    candidate = Path(model_id)
    if candidate.suffix.lower() == ".nemo":
        return True
    return candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".nemo"


def _create_whisper_model(model_id: str, device: str, compute_type: str) -> Any:
    if _is_nemo_model_id(model_id):
        return _create_nemo_model(model_id=model_id, device=device)

    from faster_whisper import WhisperModel

    return WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
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


def _prepare_audio_for_nemo(audio_path: str) -> tuple[str, str | None]:
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(audio_path, always_2d=True)
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


def _chunk_audio_for_nemo(prepared_path: str, chunk_seconds: int = 20) -> tuple[list[str], list[str]]:
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
    start = 0
    while start < total_samples:
        end = min(total_samples, start + chunk_size)
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
            text = _run_nemo_transcribe_once(model, chunk, language)
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
) -> dict[str, Any]:
    segments, info = model.transcribe(
        audio_path,
        language=language,
        initial_prompt=prompt,
        temperature=temperature,
    )

    segment_list: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for idx, segment in enumerate(segments):
        text_parts.append(segment.text)
        segment_list.append(
            {
                "id": idx,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
        )

    diarization_turns: list[dict[str, Any]] | None = None
    speakers: list[str] = []
    if diarize and diarization_pipeline is not None:
        annotation = diarization_pipeline(audio_path)
        diarization_turns = _collect_diarization_turns(annotation)
        speakers = _attach_speakers(segment_list, diarization_turns)

    payload = {
        "text": " ".join(part.strip() for part in text_parts if part).strip(),
        "segments": segment_list,
        "language": getattr(info, "language", language or "unknown"),
    }
    if diarization_turns is not None:
        payload["diarization"] = diarization_turns
        payload["speakers"] = speakers
    return payload


def _patch_torchaudio_compat() -> None:
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


def _load_diarization_pipeline(model_id: str, device: str, hf_token: str | None) -> Any:
    _patch_torchaudio_compat()
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
        kwargs["use_auth_token"] = hf_token

    try:
        import torch
        from torch.torch_version import TorchVersion

        torch.serialization.add_safe_globals([TorchVersion])
    except Exception:
        pass

    pipeline = Pipeline.from_pretrained(model_id, **kwargs)

    try:
        import torch

        target_device = torch.device("cuda" if device == "cuda" else "cpu")
        pipeline.to(target_device)
    except Exception:
        pass

    return pipeline


def _collect_diarization_turns(annotation: Any) -> list[dict[str, Any]]:
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


def _attach_speakers(
    segments: list[dict[str, Any]],
    diarization_turns: list[dict[str, Any]],
) -> list[str]:
    speakers: set[str] = set()

    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        best_speaker: str | None = None
        best_overlap = 0.0

        for turn in diarization_turns:
            overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        if best_speaker is not None:
            segment["speaker"] = best_speaker
            speakers.add(best_speaker)

    return sorted(speakers)


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized == "":
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError("Invalid 'diarize' value. Use true/false.")


def _to_srt(segments: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, segment in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_format_srt_ts(segment['start'])} --> {_format_srt_ts(segment['end'])}")
        lines.append(segment["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _to_vtt(segments: Sequence[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.append(f"{_format_vtt_ts(segment['start'])} --> {_format_vtt_ts(segment['end'])}")
        lines.append(segment["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _format_srt_ts(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    millis = int((value - int(value)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _format_vtt_ts(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    millis = int((value - int(value)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="oCabra Whisper worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu-indices", default="")
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--diarization-model-id", default=DEFAULT_DIARIZATION_MODEL_ID)
    args = parser.parse_args()

    gpu_indices = [int(item) for item in args.gpu_indices.split(",") if item.strip()]
    app = create_app(
        args.model_id,
        gpu_indices,
        diarize_default=bool(args.diarize),
        diarization_model_id=args.diarization_model_id,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
