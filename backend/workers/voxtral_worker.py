"""
oCabra Voxtral TTS Worker — Mistral Voxtral via vllm-omni

Launches ``vllm serve MODEL --omni`` as a subprocess and exposes the
standard oCabra TTS worker interface on top:

  GET  /health           — liveness (proxied to vllm /health)
  GET  /info             — runtime metadata
  GET  /voices           — preset Voxtral voices
  POST /synthesize       — full-text synthesis
  POST /synthesize/stream — sentence-by-sentence streaming synthesis
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import re
import struct
import subprocess
import sys
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Voxtral preset voices (from Mistral docs)
# ---------------------------------------------------------------------------
VOXTRAL_VOICES = [
    "jessica", "emma", "allison", "lily", "sarah",
    "vivienne", "aurora", "liam", "michael", "james",
    "daniel", "ethan", "theodore", "lucas",
    "casual_male", "professional_male",
    "casual_female", "professional_female",
    "soft_female", "warm_male",
]

# Map OpenAI standard voices to Voxtral voices
OPENAI_TO_VOXTRAL: dict[str, str] = {
    "alloy":   "casual_male",
    "echo":    "liam",
    "fable":   "vivienne",
    "onyx":    "professional_male",
    "nova":    "jessica",
    "shimmer": "emma",
}

FORMAT_CONTENT_TYPES: dict[str, str] = {
    "mp3":  "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
    "aac":  "audio/aac",
    "flac": "audio/flac",
    "wav":  "audio/wav",
    "pcm":  "audio/pcm",
}

_MIN_SENTENCE_CHARS = 8

SUPPORTED_LANGUAGES = [
    "English", "French", "Spanish", "German", "Italian",
    "Portuguese", "Dutch", "Arabic", "Hindi",
]

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

class VoxtralRuntime:
    def __init__(self) -> None:
        self.model_id: str = ""
        self.gpu_indices: list[int] = []
        self.vllm_port: int = 0
        self.vllm_process: asyncio.subprocess.Process | None = None
        self.error: str | None = None
        self.ready: bool = False


runtime = VoxtralRuntime()


# ---------------------------------------------------------------------------
# vllm-omni process management
# ---------------------------------------------------------------------------

def _resolve_model_path(model_id: str) -> str:
    """Resolve model_id to a local path or HF repo ID."""
    models_dir = os.environ.get("MODELS_DIR", "")
    hf_id = model_id
    for prefix in ("tts/", "voxtral/"):
        if hf_id.startswith(prefix):
            hf_id = hf_id[len(prefix):]
            break
    if models_dir:
        local = os.path.join(models_dir, "huggingface", hf_id.replace("/", "--"))
        if os.path.isdir(local):
            return local
    return hf_id


def _find_vllm_omni_bin(python_bin: str) -> str | None:
    """Locate the ``vllm-omni`` console script.

    Checks the same bin directory as *python_bin*, then PATH.
    """
    import shutil
    bin_dir = os.path.dirname(python_bin)
    candidate = os.path.join(bin_dir, "vllm-omni")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return shutil.which("vllm-omni")


async def _start_vllm_omni(model_id: str, port: int, gpu_indices: list[int]) -> asyncio.subprocess.Process:
    """Launch ``vllm-omni serve MODEL`` on *port*."""
    model_path = _resolve_model_path(model_id)

    # Determine which python/binary to use for vllm-omni.
    # Priority: VOXTRAL_PYTHON_BIN env → vllm-omni CLI → python -m vllm_omni
    python_bin = sys.executable
    venv_python = os.environ.get("VOXTRAL_PYTHON_BIN", "")
    if venv_python and os.path.isfile(venv_python):
        python_bin = venv_python

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_indices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)

    # vllm-omni installs a `vllm-omni` console_script that wraps vllm with
    # omni multimodal support (TTS, etc.).  Use it if available; otherwise
    # fall back to `python -m vllm_omni.entrypoints.cli.main`.
    vllm_omni_bin = _find_vllm_omni_bin(python_bin)

    if vllm_omni_bin:
        cmd = [
            vllm_omni_bin, "serve", model_path,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--served-model-name", model_id,
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "4",
            "--trust-remote-code",
        ]
    else:
        cmd = [
            python_bin, "-m", "vllm_omni.entrypoints.cli.main",
            "serve", model_path,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--served-model-name", model_id,
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "4",
            "--trust-remote-code",
        ]

    log_path = f"/tmp/voxtral-worker-{model_id.replace('/', '__')}.log"
    log_file = open(log_path, "ab")  # noqa: SIM115

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=log_file,
        stderr=log_file,
    )
    return proc


async def _wait_vllm_healthy(port: int, timeout_s: int = 300) -> bool:
    """Poll vllm /health until it responds 200."""
    for _ in range(timeout_s * 2):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"http://127.0.0.1:{port}/health")
                if r.status_code == 200:
                    return True
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Audio helpers (same as tts_worker.py)
# ---------------------------------------------------------------------------

def _wav_params(wav_bytes: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth() * 8


def _wav_to_pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _make_streaming_wav_header(sample_rate: int, channels: int, bits: int) -> bytes:
    byte_rate   = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFF_FFFF, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits,
        b"data", 0xFFFF_FFFF,
    )


def _soundfile_encode(wav_bytes: bytes, fmt: str) -> bytes | None:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None
    sf_map: dict[str, tuple[str, str | None]] = {
        "mp3": ("MP3", None), "opus": ("OGG", "VORBIS"), "flac": ("FLAC", None),
    }
    entry = sf_map.get(fmt)
    if not entry:
        return None
    sf_format, sf_subtype = entry
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        import numpy as np
        pcm16 = np.frombuffer(frames, dtype=np.int16)
        samples = pcm16.astype(np.float32) / 32768.0
        out = io.BytesIO()
        kw: dict = {"format": sf_format}
        if sf_subtype:
            kw["subtype"] = sf_subtype
        sf.write(out, samples, sr, **kw)
        return out.getvalue()
    except Exception:
        return None


def _encode_audio(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    fmt = fmt.lower().strip()
    content_type = FORMAT_CONTENT_TYPES.get(fmt, "audio/wav")
    if fmt == "wav":
        return wav_bytes, content_type
    if fmt == "pcm":
        return _wav_to_pcm(wav_bytes), content_type
    encoded = _soundfile_encode(wav_bytes, fmt)
    if encoded:
        return encoded, content_type
    return wav_bytes, "audio/wav"  # fallback


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?…;])\s+|\n+", text.strip())
    result: list[str] = []
    buf = ""
    for part in raw:
        part = part.strip()
        if not part:
            continue
        buf = (buf + " " + part).strip() if buf else part
        if len(buf) >= _MIN_SENTENCE_CHARS and re.search(r"[.!?…;]$", buf):
            result.append(buf)
            buf = ""
    if buf:
        if result and len(buf) < _MIN_SENTENCE_CHARS:
            result[-1] = result[-1] + " " + buf
        else:
            result.append(buf)
    return result or [text.strip()]


def _map_voice(voice: str) -> str:
    """Map OpenAI voice names to Voxtral voices, or pass through."""
    v = voice.lower().strip()
    return OPENAI_TO_VOXTRAL.get(v, v if v in VOXTRAL_VOICES else "casual_male")


# ---------------------------------------------------------------------------
# Synthesis via vllm-omni
# ---------------------------------------------------------------------------

async def _synthesize_via_vllm(
    text: str,
    voice: str,
    response_format: str = "wav",
) -> bytes:
    """Call the vllm-omni /v1/audio/speech endpoint and return audio bytes."""
    mapped_voice = _map_voice(voice)

    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(
            f"http://127.0.0.1:{runtime.vllm_port}/v1/audio/speech",
            json={
                "model": runtime.model_id,
                "input": text,
                "voice": mapped_voice,
                "response_format": "wav",  # always get WAV, encode ourselves
            },
        )
        r.raise_for_status()
        return r.content


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0
    language: str = "Auto"
    reference_audio: str | None = None
    reference_text: str | None = None
    speaker: str | None = None
    instruct: str | None = None


def create_app(model_id: str, port: int, gpu_indices: list[int], vllm_port: int) -> FastAPI:
    runtime.model_id = model_id
    runtime.gpu_indices = gpu_indices
    runtime.vllm_port = vllm_port

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import logging
        log = logging.getLogger("voxtral_worker")

        try:
            log.info("Starting vllm-omni for %s on port %d...", model_id, vllm_port)
            runtime.vllm_process = await _start_vllm_omni(model_id, vllm_port, gpu_indices)

            startup_timeout = int(os.environ.get("VOXTRAL_STARTUP_TIMEOUT_S", "300"))
            if await _wait_vllm_healthy(vllm_port, timeout_s=startup_timeout):
                runtime.ready = True
                runtime.error = None
                log.info("vllm-omni ready for %s on port %d (pid=%d)", model_id, vllm_port, runtime.vllm_process.pid)
            else:
                runtime.error = f"vllm-omni failed to start within {startup_timeout}s"
                log.error(runtime.error)
        except Exception as exc:
            runtime.error = str(exc)
            log.error("Failed to start vllm-omni: %s", exc)

        yield

        # Shutdown
        if runtime.vllm_process and runtime.vllm_process.returncode is None:
            runtime.vllm_process.terminate()
            try:
                await asyncio.wait_for(runtime.vllm_process.wait(), timeout=15)
            except TimeoutError:
                runtime.vllm_process.kill()
                await runtime.vllm_process.wait()

    app = FastAPI(title="oCabra Voxtral TTS Worker", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        if not runtime.ready:
            raise HTTPException(status_code=503, detail=runtime.error or "Not ready")
        # Also check vllm is still alive
        if runtime.vllm_process and runtime.vllm_process.returncode is not None:
            raise HTTPException(status_code=503, detail="vllm-omni process died")
        return JSONResponse({"status": "ok", "vllm_port": runtime.vllm_port})

    @app.get("/info")
    async def info() -> JSONResponse:
        return JSONResponse({
            "backend": "voxtral",
            "model_id": runtime.model_id,
            "gpu_indices": runtime.gpu_indices,
            "vllm_port": runtime.vllm_port,
            "ready": runtime.ready,
            "error": runtime.error,
            "vllm_alive": runtime.vllm_process is not None and runtime.vllm_process.returncode is None,
        })

    @app.get("/voices")
    async def voices() -> JSONResponse:
        return JSONResponse({
            "voices": list(VOXTRAL_VOICES),
            "model_type": "voxtral",
            "speakers": list(VOXTRAL_VOICES),
            "languages": SUPPORTED_LANGUAGES,
            "supports_voice_clone": False,
        })

    @app.post("/synthesize")
    async def synthesize(body: SynthesizeRequest) -> StreamingResponse:
        if not body.input.strip():
            raise HTTPException(status_code=400, detail="'input' text is required")
        if not runtime.ready:
            raise HTTPException(status_code=503, detail=runtime.error or "Not ready")

        fmt = body.response_format.lower()
        wav_bytes = await _synthesize_via_vllm(body.input.strip(), body.voice)
        encoded, content_type = _encode_audio(wav_bytes, fmt)

        return StreamingResponse(
            io.BytesIO(encoded),
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="speech.{fmt}"'},
        )

    @app.post("/synthesize/stream")
    async def synthesize_stream(body: SynthesizeRequest) -> StreamingResponse:
        if not body.input.strip():
            raise HTTPException(status_code=400, detail="'input' text is required")
        if not runtime.ready:
            raise HTTPException(status_code=503, detail=runtime.error or "Not ready")

        fmt       = body.response_format.lower()
        sentences = _split_sentences(body.input.strip())
        content_type = FORMAT_CONTENT_TYPES.get(fmt, "audio/mpeg")

        async def _generate():
            wav_header_sent = False
            for sentence in sentences:
                wav_bytes = await _synthesize_via_vllm(sentence, body.voice)
                if not wav_bytes:
                    continue
                if fmt == "wav":
                    if not wav_header_sent:
                        sr, ch, bits = _wav_params(wav_bytes)
                        yield _make_streaming_wav_header(sr, ch, bits)
                        wav_header_sent = True
                    yield _wav_to_pcm(wav_bytes)
                elif fmt == "pcm":
                    yield _wav_to_pcm(wav_bytes)
                else:
                    encoded, _ = _encode_audio(wav_bytes, fmt)
                    yield encoded

        return StreamingResponse(
            _generate(),
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="speech.{fmt}"'},
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="oCabra Voxtral TTS worker (vllm-omni wrapper)")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True, help="Worker API port")
    parser.add_argument("--vllm-port", type=int, default=0, help="Internal vllm-omni port (auto if 0)")
    parser.add_argument("--gpu-indices", default="")
    args = parser.parse_args()

    gpu_indices = [int(i) for i in args.gpu_indices.split(",") if i.strip()]

    # Auto-assign vllm internal port: worker port + 1000
    vllm_port = args.vllm_port if args.vllm_port else args.port + 1000

    app = create_app(args.model_id, args.port, gpu_indices, vllm_port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
