import argparse
import asyncio
import io
import json
import math
import os
import re
import struct
import subprocess
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
    "mp3":  "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
    "aac":  "audio/aac",
    "flac": "audio/flac",
    "wav":  "audio/wav",
    "pcm":  "audio/pcm",
}

_MIN_SENTENCE_CHARS = 8


class SynthesisRequest(BaseModel):
    input: str = Field(min_length=1)
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0
    language: str = "Auto"
    reference_audio: str | None = None
    reference_text: str | None = None
    speaker: str | None = None
    instruct: str | None = None


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
        fmt = request.response_format.lower()

        wav_bytes, _ = await asyncio.to_thread(
            _synthesize_audio,
            request.input,
            model_voice,
            "wav",   # always generate WAV internally
            request.speed,
        )
        encoded, content_type = _encode_audio(wav_bytes, fmt)

        headers = {"X-Model-Voice": model_voice, "X-Model-Family": runtime.family}
        return StreamingResponse(
            io.BytesIO(encoded),
            media_type=content_type,
            headers={**headers, "Content-Disposition": f'attachment; filename="speech.{fmt}"'},
        )

    @app.post("/synthesize/stream")
    async def synthesize_stream(request: SynthesisRequest) -> StreamingResponse:
        model_voice = VOICE_MAPPINGS[runtime.family].get(
            request.voice.lower(), VOICE_MAPPINGS[runtime.family]["alloy"]
        )
        fmt          = request.response_format.lower()
        content_type = FORMAT_CONTENT_TYPES.get(fmt, "audio/mpeg")
        sentences    = _split_sentences(request.input)

        async def _generate():
            wav_header_sent = False
            for sentence in sentences:
                wav_bytes, _ = await asyncio.to_thread(
                    _synthesize_audio, sentence, model_voice, "wav", request.speed,
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


# ---------------------------------------------------------------------------
# Audio format conversion (stub — same helpers as backend/workers/tts_worker.py)
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


def _ffmpeg_encode(wav_bytes: bytes, fmt: str) -> bytes:
    _codec: dict[str, list[str]] = {
        "mp3":  ["-f", "mp3",  "-codec:a", "libmp3lame", "-q:a", "4"],
        "opus": ["-f", "ogg",  "-codec:a", "libopus",    "-b:a", "64k"],
        "aac":  ["-f", "adts", "-codec:a", "aac",        "-b:a", "128k"],
        "flac": ["-f", "flac", "-codec:a", "flac"],
    }
    codec_args = _codec.get(fmt)
    if not codec_args:
        return wav_bytes
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "wav", "-i", "pipe:0", *codec_args, "pipe:1"],
            input=wav_bytes, capture_output=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except Exception:
        pass
    return wav_bytes


def _encode_audio(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    fmt = fmt.lower().strip()
    content_type = FORMAT_CONTENT_TYPES.get(fmt, "audio/wav")
    if fmt == "wav":
        return wav_bytes, content_type
    if fmt == "pcm":
        return _wav_to_pcm(wav_bytes), content_type
    return _ffmpeg_encode(wav_bytes, fmt), content_type


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
        result.append(buf)
    return result or [text.strip()]


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
