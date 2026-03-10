import argparse
import asyncio
import os
import tempfile
from collections.abc import Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse


class WhisperRuntime:
    def __init__(self) -> None:
        self.model_id: str = ""
        self.gpu_indices: list[int] = []
        self.device: str = "cpu"
        self.compute_type: str = "int8"
        self.model: Any = None
        self.error: str | None = None


runtime = WhisperRuntime()


def create_app(model_id: str, gpu_indices: list[int]) -> FastAPI:
    runtime.model_id = model_id
    runtime.gpu_indices = gpu_indices

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"} else "cpu"
        runtime.compute_type = "float16" if runtime.device == "cuda" else "int8"

        try:
            from faster_whisper import WhisperModel

            runtime.model = WhisperModel(
                runtime.model_id,
                device=runtime.device,
                compute_type=runtime.compute_type,
            )
            runtime.error = None
        except Exception as exc:
            runtime.model = None
            runtime.error = str(exc)

        yield

        runtime.model = None

    app = FastAPI(title="oCabra Whisper Worker", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        if runtime.model is None:
            raise HTTPException(status_code=503, detail=runtime.error or "Model not ready")
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
    ):
        if runtime.model is None:
            raise HTTPException(status_code=503, detail=runtime.error or "Whisper model not loaded")

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = await asyncio.to_thread(
                _run_transcription,
                tmp_path,
                runtime.model,
                language,
                prompt,
                temperature,
            )
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
            return JSONResponse(payload)

        return JSONResponse({"text": text})

    return app


def _run_transcription(
    audio_path: str,
    model: Any,
    language: str | None,
    prompt: str | None,
    temperature: float,
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

    return {
        "text": " ".join(part.strip() for part in text_parts if part).strip(),
        "segments": segment_list,
        "language": getattr(info, "language", language or "unknown"),
    }


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
    args = parser.parse_args()

    gpu_indices = [int(item) for item in args.gpu_indices.split(",") if item.strip()]
    app = create_app(args.model_id, gpu_indices)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
