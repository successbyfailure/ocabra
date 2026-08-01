"""oCabra VibeASR worker — HTTP shim around the persistent ``asr_stream_server``.

VibeASR.cpp (microsoft/VibeVoice-ASR-BitNet) ships a native ``asr_stream_server``
that loads the VAE + LM GGUFs **once** and then streams transcriptions over a
simple stdin/stdout protocol:

    stdin :  one audio-file path per line ("EXIT" to terminate)
    stdout:  "---READY---" once models are loaded, then per request the decoded
             tokens (one per line) terminated by "---END---".

This worker keeps that process alive and exposes the ``/health`` + ``/transcribe``
HTTP contract the oCabra OpenAI ``/v1/audio/transcriptions`` route expects (it
POSTs a multipart ``file`` to ``http://127.0.0.1:<port>/transcribe``). Uploaded
audio is transcoded to 24 kHz mono WAV with ffmpeg before being handed to the
server (which expects a decoded PCM file path).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

_READY_MARKER = "---READY---"
_END_MARKER = "---END---"


class VibeAsrRuntime:
    def __init__(self) -> None:
        self.model_id: str = ""
        self.stream_bin: str = "asr_stream_server"
        self.vae_model: str = ""
        self.lm_model: str = ""
        self.threads: int = 8
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.ready: bool = False
        self.error: str | None = None


runtime = VibeAsrRuntime()


async def _read_until(marker: str, timeout_s: float) -> list[str]:
    """Read stdout lines until ``marker``; return the lines before it.

    Raises on EOF (process died) or timeout.
    """
    proc = runtime.process
    assert proc is not None and proc.stdout is not None
    lines: list[str] = []

    async def _loop() -> list[str]:
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                raise RuntimeError("asr_stream_server exited unexpectedly")
            # Keep leading spaces (tokens carry their own spacing); drop only \n.
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line.strip() == marker:
                return lines
            lines.append(line)

    return await asyncio.wait_for(_loop(), timeout=timeout_s)


def _transcode_to_wav(src: str) -> str:
    """Transcode any input to 24 kHz mono WAV via ffmpeg; return the new path.

    Falls back to the original path when ffmpeg is unavailable.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    dst = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        [ffmpeg, "-y", "-i", src, "-ar", "24000", "-ac", "1", dst],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return dst


def create_app(
    model_id: str, stream_bin: str, vae_model: str, lm_model: str, threads: int
) -> FastAPI:
    runtime.model_id = model_id
    runtime.stream_bin = stream_bin
    runtime.vae_model = vae_model
    runtime.lm_model = lm_model
    runtime.threads = threads

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for label, path in (
            ("stream-server", stream_bin),
            ("vae-model", vae_model),
            ("lm-model", lm_model),
        ):
            resolved = shutil.which(path) if label == "stream-server" else None
            if resolved is None and not Path(path).is_file():  # noqa: ASYNC240
                runtime.error = f"{label} not found: {path}"
                yield
                return
        try:
            runtime.process = await asyncio.create_subprocess_exec(
                stream_bin,
                "--vae-model",
                vae_model,
                "--lm-model",
                lm_model,
                "-t",
                str(threads),
                "--greedy",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,
            )
            # Block until the server signals it has loaded both models.
            await _read_until(_READY_MARKER, timeout_s=180.0)
            runtime.ready = True
            runtime.error = None
        except Exception as exc:  # noqa: BLE001
            runtime.ready = False
            runtime.error = str(exc)
        yield
        proc = runtime.process
        runtime.ready = False
        if proc is not None and proc.returncode is None:
            with suppress(Exception):
                if proc.stdin is not None:
                    proc.stdin.write(b"EXIT\n")
                    await proc.stdin.drain()
                await asyncio.wait_for(proc.wait(), timeout=8)
            if proc.returncode is None:
                with suppress(ProcessLookupError):
                    proc.kill()

    app = FastAPI(title="oCabra VibeASR Worker", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        if not runtime.ready:
            raise HTTPException(status_code=503, detail=runtime.error or "VibeASR not ready")
        return JSONResponse({"status": "ok"})

    @app.get("/info")
    async def info() -> JSONResponse:
        return JSONResponse(
            {
                "backend": "vibeasr",
                "model_id": runtime.model_id,
                "vae_model": runtime.vae_model,
                "lm_model": runtime.lm_model,
                "threads": runtime.threads,
                "ready": runtime.ready,
                "error": runtime.error,
            }
        )

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),  # noqa: B008
        language: str | None = Form(default=None),  # noqa: B008
        response_format: str = Form(default="json"),  # noqa: B008
        temperature: float = Form(default=0.0),  # noqa: B008
    ):
        if not runtime.ready or runtime.process is None:
            raise HTTPException(status_code=503, detail=runtime.error or "VibeASR not ready")

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        src_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
        await asyncio.to_thread(Path(src_path).write_bytes, audio_bytes)
        wav_path = src_path
        try:
            wav_path = await asyncio.to_thread(_transcode_to_wav, src_path)
            text = await _transcribe_path(wav_path)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr.decode("utf-8", "replace") if exc.stderr else str(exc))[-1500:]
            raise HTTPException(status_code=500, detail=f"ffmpeg failed: {detail}") from exc
        except (RuntimeError, TimeoutError) as exc:
            runtime.ready = False  # the stream server died — force a reload on next request
            raise HTTPException(status_code=500, detail=f"asr_stream_server failed: {exc}") from exc
        finally:
            for p in {src_path, wav_path}:
                with suppress(FileNotFoundError):
                    os.unlink(p)

        normalized = response_format.lower()
        if normalized == "text":
            return PlainTextResponse(text)
        if normalized == "verbose_json":
            return JSONResponse(
                {
                    "text": text,
                    "language": language or "unknown",
                    "segments": [{"id": 0, "start": 0.0, "end": 0.0, "text": text}],
                }
            )
        return JSONResponse({"text": text})

    return app


async def _transcribe_path(wav_path: str) -> str:
    """Feed one audio path to the persistent server and collect its transcription."""
    proc = runtime.process
    assert proc is not None and proc.stdin is not None
    async with runtime.lock:
        proc.stdin.write(f"{wav_path}\n".encode())
        await proc.stdin.drain()
        tokens = await _read_until(_END_MARKER, timeout_s=600.0)
    # Tokens carry their own leading spaces; join verbatim then trim the ends.
    return "".join(tokens).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="oCabra VibeASR worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--stream-server-bin", required=True)
    parser.add_argument("--vae-model", required=True)
    parser.add_argument("--lm-model", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu-indices", default="")
    args = parser.parse_args()

    app = create_app(
        args.model_id,
        args.stream_server_bin,
        args.vae_model,
        args.lm_model,
        args.threads,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
