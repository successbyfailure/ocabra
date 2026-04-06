"""
oCabra Chatterbox TTS Worker — ResembleAI Chatterbox Multilingual

Standalone FastAPI worker that loads a Chatterbox model and exposes
the standard oCabra TTS worker interface:

  GET  /health           — liveness probe
  GET  /info             — runtime metadata
  GET  /voices           — available voices (minimal: "default" + voice cloning)
  POST /synthesize       — full-text synthesis, returns audio
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
import wave
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHATTERBOX_LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "nl",
    "pl",
    "ro",
    "hu",
    "cs",
    "el",
    "sv",
    "da",
    "fi",
    "bg",
    "hr",
    "sk",
    "sl",
    "lt",
    "lv",
    "et",
    "mt",
]

FORMAT_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

_MIN_SENTENCE_CHARS = 8

# Default sample rate for Chatterbox output
_DEFAULT_SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


class ChatterboxRuntime:
    def __init__(self) -> None:
        self.model_id: str = ""
        self.gpu_indices: list[int] = []
        self.device: str = "cpu"
        self.model: Any = None
        self.sample_rate: int = _DEFAULT_SAMPLE_RATE
        self.is_turbo: bool = False
        self.error: str | None = None


runtime = ChatterboxRuntime()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _resolve_model_path(model_id: str) -> str:
    """Resolve model_id to a local path or HF repo ID."""
    models_dir = os.environ.get("MODELS_DIR", "")
    hf_id = model_id
    for prefix in ("chatterbox/", "tts/"):
        if hf_id.startswith(prefix):
            hf_id = hf_id[len(prefix) :]
            break
    if models_dir:
        local = os.path.join(models_dir, "huggingface", hf_id.replace("/", "--"))
        if os.path.isdir(local):
            return local
    return hf_id


def _is_turbo(model_id: str) -> bool:
    return "turbo" in model_id.lower()


def _patch_perth_watermarker() -> None:
    """Replace missing PerthImplicitWatermarker with DummyWatermarker.

    The perth C++ extension often fails to compile, leaving
    PerthImplicitWatermarker as None.  Chatterbox's __init__ then crashes
    with ``TypeError: 'NoneType' object is not callable``.  Patching the
    module before import lets the model load without watermarking.
    """
    import perth

    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
        logger.warning("perth_implicit_watermarker_unavailable, using DummyWatermarker")


def _load_chatterbox(model_id: str, device: str) -> tuple[Any, int, bool]:
    """Load Chatterbox model. Returns (model, sample_rate, is_turbo)."""
    import torch  # noqa: F401 — ensure torch is available

    _patch_perth_watermarker()

    is_turbo = _is_turbo(model_id)

    if is_turbo:
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # type: ignore[import]

        model = ChatterboxTurboTTS.from_pretrained(device=device)
    else:
        from chatterbox.tts import ChatterboxTTS  # type: ignore[import]

        model = ChatterboxTTS.from_pretrained(device=device)

    # Chatterbox outputs at 24000 Hz; use model attribute if available
    sample_rate = getattr(model, "sr", _DEFAULT_SAMPLE_RATE)
    return model, sample_rate, is_turbo


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _numpy_to_wav(audio: Any, sample_rate: int) -> bytes:
    import numpy as np

    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    max_abs = float(np.max(np.abs(audio_np)))
    if max_abs > 0:
        audio_np = audio_np / max_abs
    pcm = (audio_np * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _tensor_to_wav(audio_tensor: Any, sample_rate: int) -> bytes:
    """Convert a torch.Tensor to WAV bytes."""
    import torch

    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.cpu().numpy()
    else:
        audio_np = audio_tensor
    return _numpy_to_wav(audio_np, sample_rate)


def _wav_params(wav_bytes: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth() * 8


def _wav_to_pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _make_streaming_wav_header(sample_rate: int, channels: int, bits: int) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFF_FFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        0xFFFF_FFFF,
    )


def _soundfile_encode(wav_bytes: bytes, fmt: str) -> bytes | None:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None
    sf_map: dict[str, tuple[str, str | None]] = {
        "mp3": ("MP3", None),
        "opus": ("OGG", "VORBIS"),
        "flac": ("FLAC", None),
    }
    entry = sf_map.get(fmt)
    if not entry:
        return None
    sf_format, sf_subtype = entry
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
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


def _ffmpeg_encode(wav_bytes: bytes, fmt: str) -> bytes:
    """Encode WAV bytes to *fmt* via soundfile or ffmpeg fallback."""
    import logging

    encoded = _soundfile_encode(wav_bytes, fmt)
    if encoded:
        return encoded

    _codec: dict[str, list[str]] = {
        "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "4"],
        "opus": ["-f", "ogg", "-codec:a", "opus", "-strict", "-2", "-b:a", "64k"],
        "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "128k"],
        "flac": ["-f", "flac", "-codec:a", "flac"],
    }
    codec_args = _codec.get(fmt)
    if not codec_args:
        return wav_bytes

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
        *codec_args,
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, input=wav_bytes, capture_output=True, timeout=60)
        if result.returncode == 0 and result.stdout:
            return result.stdout
        logging.getLogger(__name__).warning(
            "ffmpeg encode failed for %s (rc=%d): %s",
            fmt,
            result.returncode,
            result.stderr[:200].decode(errors="replace"),
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("ffmpeg encode error for %s: %s", fmt, exc)
    return wav_bytes


def _encode_audio(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    fmt = fmt.lower().strip()
    content_type = FORMAT_CONTENT_TYPES.get(fmt, "audio/wav")
    if fmt == "wav":
        return wav_bytes, content_type
    if fmt == "pcm":
        return _wav_to_pcm(wav_bytes), content_type
    return _ffmpeg_encode(wav_bytes, fmt), content_type


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def _load_voice_ref(voice_ref: str) -> Any:
    """Load a voice reference audio file for voice cloning.

    Accepts a file path. Returns the loaded audio tensor suitable
    for passing to Chatterbox's generate method.
    """
    import torchaudio  # type: ignore[import]

    wav, sr = torchaudio.load(voice_ref)
    return wav, sr


def _synthesize(
    rt: ChatterboxRuntime,
    text: str,
    voice_ref: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> bytes:
    """Synthesize text using the loaded Chatterbox model."""
    import torch

    kwargs: dict[str, Any] = {
        "text": text,
    }

    # Voice cloning via reference audio
    if voice_ref and os.path.isfile(voice_ref):
        audio_prompt, sr = _load_voice_ref(voice_ref)
        kwargs["audio_prompt"] = audio_prompt

    with torch.inference_mode():
        wav = rt.model.generate(**kwargs)

    return _tensor_to_wav(wav, rt.sample_rate)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "default"
    language: str = "en"
    speed: float = 1.0
    response_format: str = "mp3"
    voice_ref: str | None = None  # path to reference audio for voice cloning


def create_app(model_id: str, gpu_indices: list[int]) -> FastAPI:
    runtime.model_id = model_id
    runtime.gpu_indices = gpu_indices

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import logging

        log = logging.getLogger("chatterbox_worker")

        use_cuda = os.getenv("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}
        runtime.device = "cuda" if use_cuda else "cpu"

        try:
            log.info("Loading Chatterbox model '%s' on %s...", model_id, runtime.device)
            model, sample_rate, is_turbo = await asyncio.to_thread(
                _load_chatterbox, model_id, runtime.device
            )
            runtime.model = model
            runtime.sample_rate = sample_rate
            runtime.is_turbo = is_turbo
            runtime.error = None
            log.info(
                "Chatterbox model loaded: %s (turbo=%s, sr=%d)",
                model_id,
                is_turbo,
                sample_rate,
            )
        except ImportError as exc:
            runtime.error = (
                f"Chatterbox dependencies not installed: {exc}. "
                "Install with: pip install chatterbox-tts"
            )
            log.error(runtime.error)
        except Exception as exc:
            runtime.error = str(exc)
            log.error("Failed to load Chatterbox model '%s': %s", model_id, exc)

        yield

        runtime.model = None

    app = FastAPI(title="oCabra Chatterbox TTS Worker", lifespan=lifespan)

    def _is_loaded() -> bool:
        return runtime.model is not None

    @app.get("/health")
    async def health() -> JSONResponse:
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Model not loaded")
        return JSONResponse({"status": "ok", "model_loaded": True, "is_turbo": runtime.is_turbo})

    @app.get("/info")
    async def info() -> JSONResponse:
        return JSONResponse(
            {
                "backend": "chatterbox",
                "model_id": runtime.model_id,
                "gpu_indices": runtime.gpu_indices,
                "device": runtime.device,
                "loaded": _is_loaded(),
                "is_turbo": runtime.is_turbo,
                "sample_rate": runtime.sample_rate,
                "error": runtime.error,
            }
        )

    @app.get("/voices")
    async def voices() -> JSONResponse:
        return JSONResponse(
            {
                "voices": ["default"],
                "model_type": "chatterbox",
                "speakers": [],
                "languages": list(CHATTERBOX_LANGUAGES),
                "supports_voice_clone": True,
            }
        )

    @app.post("/synthesize")
    async def synthesize(body: SynthesizeRequest) -> StreamingResponse:
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="'text' is required")
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Model not loaded")

        fmt = body.response_format.lower()

        wav_bytes = await asyncio.to_thread(
            _synthesize,
            runtime,
            body.text.strip(),
            body.voice_ref,
        )
        encoded, content_type = _encode_audio(wav_bytes, fmt)

        return StreamingResponse(
            io.BytesIO(encoded),
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="speech.{fmt}"'},
        )

    @app.post("/synthesize/stream")
    async def synthesize_stream(body: SynthesizeRequest) -> StreamingResponse:
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="'text' is required")
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Model not loaded")

        fmt = body.response_format.lower()
        sentences = _split_sentences(body.text.strip())
        content_type = FORMAT_CONTENT_TYPES.get(fmt, "audio/mpeg")

        async def _generate():
            wav_header_sent = False
            for sentence in sentences:
                wav_bytes = await asyncio.to_thread(
                    _synthesize,
                    runtime,
                    sentence,
                    body.voice_ref,
                )
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
    parser = argparse.ArgumentParser(description="oCabra Chatterbox TTS worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu-indices", default="")
    args = parser.parse_args()

    gpu_indices = [int(i) for i in args.gpu_indices.split(",") if i.strip()]
    app = create_app(args.model_id, gpu_indices)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
