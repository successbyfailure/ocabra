"""Standalone pyannote diarization sidecar for backends with no Python process
of their own to lazy-load pyannote into (whisper.cpp's whisper-server is a
native binary — there's nowhere inside it to bolt this onto, unlike
workers/whisper_worker.py where faster-whisper and pyannote share one process).

Launched lazily by WhisperCppBackend on the first diarize=true request against
a given model_id, and reused for the life of that model's worker. Takes the
raw audio plus the already-produced (non-diarized) transcription segments,
runs pyannote, merges speaker labels in, and returns the fully-formatted
response body — the caller (whispercpp_backend.py, running in the main oCabra
API process with no torch/pyannote installed) never needs to import anything
from diarization_lib.py itself, just forward bytes over HTTP exactly like it
already does for whisper-server.

Pipeline is loaded eagerly at startup (not lazily per-request): this process
is itself the lazy part — WhisperCppBackend only spawns it once diarization is
actually requested, so there is no benefit to a second lazy gate inside it,
and eager loading avoids the duplicate-concurrent-load race that exists in
whisper_worker.py's per-request lazy path (two requests arriving before the
first load finishes would otherwise build two pipelines concurrently).
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diarization_lib import (  # noqa: E402
    DEFAULT_DIARIZATION_MODEL_ID,
    diarize_audio_file,
    load_diarization_pipeline,
    split_segments_by_speaker,
    to_srt,
    to_text,
    to_vtt,
)


class _Runtime:
    def __init__(self) -> None:
        self.model_id: str = DEFAULT_DIARIZATION_MODEL_ID
        self.device: str = "cpu"
        self.pipeline: Any = None
        self.error: str | None = None


runtime = _Runtime()


def create_app(model_id: str, device: str) -> FastAPI:
    runtime.model_id = model_id
    runtime.device = device

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            runtime.pipeline = await asyncio.to_thread(
                load_diarization_pipeline,
                runtime.model_id,
                runtime.device,
                os.getenv("HF_TOKEN", "").strip() or None,
            )
            runtime.error = None
        except Exception as exc:
            runtime.pipeline = None
            runtime.error = str(exc)
        yield
        runtime.pipeline = None

    app = FastAPI(title="oCabra Diarization Sidecar", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        if runtime.pipeline is None:
            raise HTTPException(status_code=503, detail=runtime.error or "Pipeline not ready")
        return JSONResponse({"status": "ok"})

    @app.post("/diarize")
    async def diarize(
        file: UploadFile = File(...),  # noqa: B008
        segments_json: str = Form(...),  # noqa: B008
        response_format: str = Form(default="verbose_json"),  # noqa: B008
    ):
        if runtime.pipeline is None:
            raise HTTPException(status_code=503, detail=runtime.error or "Pipeline not ready")

        try:
            segments = json.loads(segments_json)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid segments_json: {exc}") from exc

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            diar_result = await asyncio.to_thread(diarize_audio_file, runtime.pipeline, tmp_path)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(tmp_path)

        turns = diar_result["turns"]
        merged_segments = split_segments_by_speaker(segments, turns)
        final_text = to_text(merged_segments)

        normalized_format = response_format.lower()
        if normalized_format == "text":
            return PlainTextResponse(final_text)
        if normalized_format == "srt":
            return PlainTextResponse(to_srt(merged_segments))
        if normalized_format == "vtt":
            return PlainTextResponse(to_vtt(merged_segments))
        if normalized_format == "verbose_json":
            payload: dict[str, Any] = {
                "text": final_text,
                "segments": merged_segments,
                "diarization": turns,
                "speakers": diar_result["speakers"],
            }
            if diar_result["embeddings"]:
                payload["speaker_embeddings"] = diar_result["embeddings"]
            return JSONResponse(payload)
        return JSONResponse({"text": final_text})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="oCabra diarization sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--diarization-model-id", default=DEFAULT_DIARIZATION_MODEL_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    app = create_app(args.diarization_model_id, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
