"""
oCabra TTS Worker — Qwen3-TTS via qwen-tts package

Supports:
  - Qwen3-TTS-12Hz-1.7B-Base    → voice cloning via reference audio
  - Qwen3-TTS-12Hz-1.7B-CustomVoice → preset speakers (Vivian, Ryan, Serena, …)

Endpoints:
  GET  /health     — liveness probe
  GET  /info       — runtime metadata
  GET  /voices     — list voices / speakers + model_type
  POST /synthesize — generate speech
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import math
import os
import struct
import tempfile
import wave
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUSTOMVOICE_SPEAKERS = [
    "vivian",    # Bright, slightly edgy young female (Chinese)
    "serena",    # Warm, gentle young female (Chinese)
    "uncle_fu",  # Seasoned male, low mellow timbre (Chinese)
    "dylan",     # Youthful Beijing male (Chinese Beijing Dialect)
    "eric",      # Lively Chengdu male (Chinese Sichuan Dialect)
    "ryan",      # Dynamic male, rhythmic drive (English)
    "aiden",     # Sunny American male (English)
    "ono_anna",  # Playful Japanese female (Japanese)
    "sohee",     # Warm Korean female (Korean)
]

# Map OpenAI voice names to CustomVoice speakers (lowercase)
OPENAI_TO_CUSTOMVOICE: dict[str, str] = {
    "alloy":   "ryan",
    "echo":    "aiden",
    "fable":   "serena",
    "onyx":    "uncle_fu",
    "nova":    "vivian",
    "shimmer": "sohee",
}

OPENAI_VOICES = ("alloy", "echo", "fable", "nova", "onyx", "shimmer")

SUPPORTED_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

class TTSRuntime:
    def __init__(self) -> None:
        self.model_id: str = ""
        self.gpu_indices: list[int] = []
        self.device: str = "cpu"
        self.qwen_model: Any = None   # Qwen3TTSModel wrapper
        self.model_type: str = "placeholder"  # "base" | "custom_voice" | "placeholder"
        self.speakers: list[str] = []
        self.error: str | None = None


runtime = TTSRuntime()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_model_path(model_id: str) -> str:
    models_dir = os.environ.get("MODELS_DIR", "")
    hf_id = model_id
    for prefix in ("tts/", "whisper/", "vllm/", "diffusers/"):
        if hf_id.startswith(prefix):
            hf_id = hf_id[len(prefix):]
            break
    if models_dir:
        local = os.path.join(models_dir, "huggingface", hf_id.replace("/", "--"))
        if os.path.isdir(local):
            return local
    return hf_id


def _load_qwen_tts(model_id: str, device: str) -> tuple[Any, str]:
    """Load Qwen3-TTS model; returns (Qwen3TTSModel, model_type)."""
    import torch
    from qwen_tts import Qwen3TTSModel  # type: ignore[import]

    path = _resolve_model_path(model_id)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    qwen_model = Qwen3TTSModel.from_pretrained(
        path,
        dtype=dtype,
        device_map=device,
    )
    model_type: str = getattr(qwen_model.model, "tts_model_type", "base")
    return qwen_model, model_type


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _numpy_to_wav(audio: Any, sample_rate: int) -> bytes:
    import numpy as np
    audio_np = np.asarray(audio, dtype=np.float32)
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


def _generate_tone(text: str, speed: float = 1.0) -> bytes:
    """Placeholder tone when no model is loaded."""
    sample_rate = 22050
    duration = max(0.5, min(len(text) * 0.08 / max(0.1, speed), 10.0))
    num_samples = int(sample_rate * duration)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            t = i / sample_rate
            envelope = 1.0 if t < duration * 0.9 else (duration - t) / (duration * 0.1)
            sample = int(32767 * 0.4 * envelope * math.sin(2 * math.pi * 440.0 * t))
            wf.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


def _decode_audio(reference_audio: str) -> str:
    """Decode base64 audio to a temp file, return path."""
    data = base64.b64decode(reference_audio)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _synthesize(
    rt: TTSRuntime,
    text: str,
    voice: str,
    speed: float,
    language: str,
    reference_audio: str | None,
    reference_text: str | None,
    speaker: str | None,
    instruct: str | None,
) -> bytes:
    import logging
    log = logging.getLogger(__name__)

    if rt.qwen_model is None:
        return _generate_tone(text=text, speed=speed)

    try:
        if rt.model_type == "custom_voice":
            # Use preset speakers: speaker param takes priority; fall back to OpenAI→speaker mapping
            resolved_speaker = speaker or OPENAI_TO_CUSTOMVOICE.get(voice, "Ryan")
            wavs, sr = rt.qwen_model.generate_custom_voice(
                text=text,
                speaker=resolved_speaker,
                language=language or "Auto",
                instruct=instruct or None,
            )
        elif rt.model_type == "base":
            # Voice cloning — requires reference audio
            if not reference_audio:
                # Without reference audio fall back to a default speaker embedding via x-vector mode
                wavs, sr = rt.qwen_model.generate_voice_clone(
                    text=text,
                    language=language or "Auto",
                    ref_audio=None,
                    x_vector_only_mode=True,
                )
            else:
                ref_path = _decode_audio(reference_audio)
                try:
                    wavs, sr = rt.qwen_model.generate_voice_clone(
                        text=text,
                        language=language or "Auto",
                        ref_audio=ref_path,
                        ref_text=reference_text or None,
                    )
                finally:
                    try:
                        os.unlink(ref_path)
                    except OSError:
                        pass
        else:
            return _generate_tone(text=text, speed=speed)

        if wavs and len(wavs) > 0:
            return _numpy_to_wav(wavs[0], sr)
        return _generate_tone(text=text, speed=speed)

    except Exception as exc:
        log.warning("TTS synthesis error (%s): %s", rt.model_type, exc, exc_info=True)
        return _generate_tone(text=text, speed=speed)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    input: str
    voice: str = "alloy"
    response_format: str = "wav"
    speed: float = 1.0
    language: str = "Auto"
    # Voice cloning (Base model)
    reference_audio: Optional[str] = None   # base64-encoded audio
    reference_text: Optional[str] = None    # transcript of reference audio
    # CustomVoice model
    speaker: Optional[str] = None           # explicit speaker name (overrides voice mapping)
    instruct: Optional[str] = None          # style instruction


def create_app(model_id: str, gpu_indices: list[int]) -> FastAPI:
    runtime.model_id = model_id
    runtime.gpu_indices = gpu_indices

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        use_cuda = os.getenv("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}
        runtime.device = "cuda" if use_cuda else "cpu"

        try:
            qwen_model, model_type = await asyncio.to_thread(
                _load_qwen_tts, model_id, runtime.device
            )
            runtime.qwen_model = qwen_model
            runtime.model_type = model_type
            runtime.error = None
            # Populate speakers from model
            if model_type == "custom_voice" and hasattr(qwen_model.model, "get_supported_speakers"):
                try:
                    runtime.speakers = list(qwen_model.model.get_supported_speakers())
                except Exception:
                    runtime.speakers = list(CUSTOMVOICE_SPEAKERS)
        except Exception as exc:
            runtime.qwen_model = None
            runtime.model_type = "placeholder"
            runtime.error = str(exc)
            import logging
            logging.getLogger(__name__).error("Failed to load TTS model: %s", exc)

        yield

        runtime.qwen_model = None

    app = FastAPI(title="oCabra TTS Worker", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "model_loaded": runtime.qwen_model is not None})

    @app.get("/info")
    async def info() -> JSONResponse:
        return JSONResponse({
            "backend": "tts",
            "model_id": runtime.model_id,
            "model_type": runtime.model_type,
            "gpu_indices": runtime.gpu_indices,
            "device": runtime.device,
            "loaded": runtime.qwen_model is not None,
            "error": runtime.error,
        })

    @app.get("/voices")
    async def voices() -> JSONResponse:
        if runtime.model_type == "custom_voice":
            speakers = runtime.speakers or CUSTOMVOICE_SPEAKERS
            voices_list = speakers
        else:
            speakers = []
            voices_list = list(OPENAI_VOICES)
        return JSONResponse({
            "voices": voices_list,
            "model_type": runtime.model_type,
            "speakers": speakers,
            "languages": SUPPORTED_LANGUAGES,
            "supports_voice_clone": runtime.model_type == "base",
        })

    @app.post("/synthesize")
    async def synthesize(body: SynthesizeRequest) -> StreamingResponse:
        if not body.input.strip():
            raise HTTPException(status_code=400, detail="'input' text is required")

        speed = max(0.25, min(body.speed, 4.0))

        audio_bytes = await asyncio.to_thread(
            _synthesize,
            runtime,
            body.input.strip(),
            body.voice,
            speed,
            body.language,
            body.reference_audio,
            body.reference_text,
            body.speaker,
            body.instruct,
        )

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="oCabra TTS worker")
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
