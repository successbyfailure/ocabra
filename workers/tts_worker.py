import argparse
import asyncio
import io
import json
import math
import os
import wave
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

OPENAI_VOICES = ("alloy", "echo", "fable", "onyx", "nova", "shimmer")

VOICE_MAPPINGS: dict[str, dict[str, str]] = {
    "qwen3-tts": {
        "alloy": "zh-CN-XiaoxiaoNeural",
        "echo": "zh-CN-YunxiNeural",
        "fable": "en-GB-SoniaNeural",
        "onyx": "en-US-GuyNeural",
        "nova": "en-US-AriaNeural",
        "shimmer": "en-US-JennyNeural",
    },
    "kokoro": {
        "alloy": "af",
        "echo": "am",
        "fable": "bf",
        "onyx": "bm",
        "nova": "cf",
        "shimmer": "cm",
    },
    "bark": {
        "alloy": "v2/en_speaker_0",
        "echo": "v2/en_speaker_1",
        "fable": "v2/en_speaker_2",
        "onyx": "v2/en_speaker_3",
        "nova": "v2/en_speaker_4",
        "shimmer": "v2/en_speaker_5",
    },
}

FORMAT_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


class SynthesisRequest(BaseModel):
    input: str = Field(min_length=1)
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0


class TTSRuntime:
    def __init__(self) -> None:
        self.model_id: str = ""
        self.gpu_indices: list[int] = []
        self.family: str = "kokoro"
        self.available_voices: list[str] = []
        self.error: str | None = None


runtime = TTSRuntime()


def create_app(model_id: str, gpu_indices: list[int]) -> FastAPI:
    runtime.model_id = model_id
    runtime.gpu_indices = gpu_indices

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.family = _infer_tts_family(runtime.model_id)
        runtime.available_voices = _resolve_available_voices(runtime.model_id, runtime.family)
        runtime.error = None
        yield

    app = FastAPI(title="oCabra TTS Worker", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        if runtime.error:
            raise HTTPException(status_code=503, detail=runtime.error)
        return JSONResponse({"status": "ok"})

    @app.get("/info")
    async def info() -> JSONResponse:
        return JSONResponse(
            {
                "backend": "tts",
                "model_id": runtime.model_id,
                "gpu_indices": runtime.gpu_indices,
                "family": runtime.family,
                "voices": runtime.available_voices,
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                "error": runtime.error,
            }
        )

    @app.get("/voices")
    async def voices() -> JSONResponse:
        return JSONResponse(
            [
                {"openai_voice": voice, "model_voice": VOICE_MAPPINGS[runtime.family][voice]}
                for voice in OPENAI_VOICES
            ]
        )

    @app.post("/synthesize")
    async def synthesize(request: SynthesisRequest) -> StreamingResponse:
        model_voice = VOICE_MAPPINGS[runtime.family].get(
            request.voice.lower(), VOICE_MAPPINGS[runtime.family]["alloy"]
        )

        audio_bytes, actual_format = await asyncio.to_thread(
            _synthesize_audio,
            request.input,
            model_voice,
            request.response_format,
            request.speed,
        )

        media_type = FORMAT_CONTENT_TYPES.get(actual_format, "audio/wav")
        headers = {"X-Model-Voice": model_voice, "X-Model-Family": runtime.family}
        return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type, headers=headers)

    return app


def _resolve_available_voices(model_id: str, family: str) -> list[str]:
    model_index_path = Path(os.getenv("MODELS_DIR", "/data/models")) / model_id / "model_index.json"
    if model_index_path.exists():
        try:
            data = json.loads(model_index_path.read_text(encoding="utf-8"))
            voices = data.get("voices")
            if isinstance(voices, Sequence):
                return [str(item) for item in voices]
        except Exception:
            pass

    return [VOICE_MAPPINGS[family][voice] for voice in OPENAI_VOICES]


def _infer_tts_family(model_id: str) -> str:
    normalized = model_id.lower()
    if "qwen3-tts" in normalized or "qwen" in normalized:
        return "qwen3-tts"
    if "kokoro" in normalized:
        return "kokoro"
    if "bark" in normalized:
        return "bark"
    return "kokoro"


def _synthesize_audio(text: str, voice: str, response_format: str, speed: float) -> tuple[bytes, str]:
    """Return (audio_bytes, actual_format).

    This is a placeholder that generates a synthetic tone.  The actual format
    produced is always WAV regardless of *response_format* because real encoder
    libraries (e.g. pydub) are not available in this stub.  The caller must use
    the returned *actual_format* to set the correct Content-Type.
    """
    del voice  # Voice controls are backend-specific; placeholder signal is deterministic.
    audio_bytes = _generate_tone(text=text, response_format="wav", speed=speed)
    return audio_bytes, "wav"


def _generate_tone(text: str, response_format: str, speed: float) -> bytes:
    sample_rate = 22050
    channels = 1
    sample_width = 2
    safe_speed = min(max(speed, 0.25), 4.0)

    base_duration = max(0.4, min(8.0, len(text) / 18.0))
    duration = base_duration / safe_speed
    total_samples = int(sample_rate * duration)

    frequency = 220.0 + (len(text) % 80)
    amplitude = 0.25
    pcm = bytearray()

    for i in range(total_samples):
        value = math.sin((2 * math.pi * frequency * i) / sample_rate)
        sample = int(value * (32767 * amplitude))
        pcm.extend(sample.to_bytes(2, byteorder="little", signed=True))

    fmt = response_format.lower()
    if fmt == "pcm":
        return bytes(pcm)

    # For container formats, generate WAV as a baseline payload.
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(pcm))

    return wav_buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="oCabra TTS worker")
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
