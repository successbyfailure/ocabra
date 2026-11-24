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
import sys
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _audio_common import (  # noqa: E402
    FORMAT_CONTENT_TYPES,
)
from _audio_common import (
    encode_audio as _encode_audio,
)
from _audio_common import (
    make_streaming_wav_header as _make_streaming_wav_header,
)
from _audio_common import (
    split_sentences as _split_sentences,
)
from _audio_common import (
    wav_params as _wav_params,
)
from _audio_common import (
    wav_to_pcm as _wav_to_pcm,
)

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
        self.family: str = "placeholder"  # "qwen" | "kokoro" | "bark" | "placeholder"
        self.qwen_model: Any = None       # Qwen3TTSModel wrapper
        self.qwen_type: str = "base"      # "base" | "custom_voice"
        self.kokoro_pipeline: Any = None   # kokoro.KPipeline
        self.bark_model: Any = None       # BarkModel
        self.bark_processor: Any = None   # AutoProcessor
        self.speakers: list[str] = []
        self.error: str | None = None


runtime = TTSRuntime()


# Kokoro preset voices
KOKORO_VOICES: list[str] = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_liam", "am_michael", "am_onyx",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

OPENAI_TO_KOKORO: dict[str, str] = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
}

BARK_SPEAKERS: list[str] = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
    "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8",
    "v2/en_speaker_9",
]

OPENAI_TO_BARK: dict[str, str] = {
    "alloy": "v2/en_speaker_0",
    "echo": "v2/en_speaker_1",
    "fable": "v2/en_speaker_2",
    "onyx": "v2/en_speaker_3",
    "nova": "v2/en_speaker_4",
    "shimmer": "v2/en_speaker_5",
}


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def _detect_family(model_id: str) -> str:
    low = model_id.lower()
    if "kokoro" in low:
        return "kokoro"
    if "bark" in low:
        return "bark"
    if "qwen" in low:
        return "qwen"
    return "qwen"  # default


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


def _load_kokoro(model_id: str, device: str) -> Any:
    """Load Kokoro TTS pipeline."""
    from kokoro import KPipeline  # type: ignore[import]

    # Kokoro uses language codes for pipeline init: a=American, b=British, j=Japanese, etc.
    # Default to American English
    pipeline = KPipeline(lang_code="a", device=device)
    return pipeline


def _load_bark(model_id: str, device: str) -> tuple[Any, Any]:
    """Load Bark TTS model; returns (model, processor)."""
    import torch
    from transformers import AutoProcessor, BarkModel  # type: ignore[import]

    path = _resolve_model_path(model_id)
    processor = AutoProcessor.from_pretrained(path)
    model = BarkModel.from_pretrained(
        path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    return model, processor


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


def _synthesize_qwen(
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
    if rt.qwen_type == "custom_voice":
        resolved_speaker = speaker or OPENAI_TO_CUSTOMVOICE.get(voice, "Ryan")
        wavs, sr = rt.qwen_model.generate_custom_voice(
            text=text,
            speaker=resolved_speaker,
            language=language or "Auto",
            instruct=instruct or None,
        )
    elif rt.qwen_type == "base":
        if not reference_audio:
            raise ValueError(
                "Qwen3-TTS Base model requires 'reference_audio' (base64-encoded audio) "
                "for voice cloning. Use a CustomVoice model variant for speaker-based "
                "synthesis without a reference audio."
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
        raise RuntimeError(f"Unknown Qwen TTS model type: {rt.qwen_type}")

    if wavs and len(wavs) > 0:
        return _numpy_to_wav(wavs[0], sr)
    raise RuntimeError("Qwen TTS returned empty audio")


def _synthesize_kokoro(
    rt: TTSRuntime,
    text: str,
    voice: str,
    speed: float,
) -> bytes:
    import numpy as np
    import torch

    # Map OpenAI voice to Kokoro voice
    kokoro_voice = OPENAI_TO_KOKORO.get(voice.lower(), voice)
    if kokoro_voice not in KOKORO_VOICES and not kokoro_voice.startswith(("af_", "am_", "bf_", "bm_")):
        kokoro_voice = "af_heart"

    segments = list(rt.kokoro_pipeline(text, voice=kokoro_voice, speed=speed))
    if not segments:
        raise RuntimeError("Kokoro returned no audio segments")

    # Kokoro returns torch.Tensor audio at 24000 Hz (fixed)
    parts = []
    for seg in segments:
        audio = seg.audio
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        parts.append(np.asarray(audio, dtype=np.float32))

    all_audio = np.concatenate(parts)
    return _numpy_to_wav(all_audio, 24000)


def _synthesize_bark(
    rt: TTSRuntime,
    text: str,
    voice: str,
) -> bytes:
    import numpy as np

    bark_voice = OPENAI_TO_BARK.get(voice.lower(), voice)
    if bark_voice not in BARK_SPEAKERS:
        bark_voice = "v2/en_speaker_0"

    inputs = rt.bark_processor(text, voice_preset=bark_voice, return_tensors="pt")
    inputs = {k: v.to(rt.device) for k, v in inputs.items()}

    speech = rt.bark_model.generate(**inputs)
    audio = speech.cpu().numpy().squeeze().astype(np.float32)
    # Bark outputs at 24000 Hz
    sample_rate = getattr(rt.bark_model.generation_config, "sample_rate", 24000)
    return _numpy_to_wav(audio, sample_rate)


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

    try:
        if rt.family == "qwen":
            return _synthesize_qwen(
                rt, text, voice, speed, language,
                reference_audio, reference_text, speaker, instruct,
            )
        elif rt.family == "kokoro":
            return _synthesize_kokoro(rt, text, voice, speed)
        elif rt.family == "bark":
            return _synthesize_bark(rt, text, voice)
        else:
            raise RuntimeError(f"No TTS engine loaded (family={rt.family})")
    except Exception as exc:
        log.error("TTS synthesis error (%s): %s", rt.family, exc, exc_info=True)
        raise


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
    reference_audio: str | None = None   # base64-encoded audio
    reference_text: str | None = None    # transcript of reference audio
    # CustomVoice model
    speaker: str | None = None           # explicit speaker name (overrides voice mapping)
    instruct: str | None = None          # style instruction


def create_app(model_id: str, gpu_indices: list[int]) -> FastAPI:
    runtime.model_id = model_id
    runtime.gpu_indices = gpu_indices

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import logging
        log = logging.getLogger(__name__)

        use_cuda = os.getenv("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}
        runtime.device = "cuda" if use_cuda else "cpu"
        runtime.family = _detect_family(model_id)

        try:
            if runtime.family == "qwen":
                qwen_model, qwen_type = await asyncio.to_thread(
                    _load_qwen_tts, model_id, runtime.device
                )
                runtime.qwen_model = qwen_model
                runtime.qwen_type = qwen_type
                if qwen_type == "custom_voice" and hasattr(qwen_model.model, "get_supported_speakers"):
                    try:
                        runtime.speakers = list(qwen_model.model.get_supported_speakers())
                    except Exception:
                        runtime.speakers = list(CUSTOMVOICE_SPEAKERS)
                log.info("Loaded Qwen3-TTS (%s)", qwen_type)

            elif runtime.family == "kokoro":
                pipeline = await asyncio.to_thread(_load_kokoro, model_id, runtime.device)
                runtime.kokoro_pipeline = pipeline
                runtime.speakers = list(KOKORO_VOICES)
                log.info("Loaded Kokoro TTS")

            elif runtime.family == "bark":
                bark_model, bark_processor = await asyncio.to_thread(
                    _load_bark, model_id, runtime.device
                )
                runtime.bark_model = bark_model
                runtime.bark_processor = bark_processor
                runtime.speakers = list(BARK_SPEAKERS)
                log.info("Loaded Bark TTS")

            else:
                raise RuntimeError(f"Unknown TTS family: {runtime.family}")

            runtime.error = None

        except Exception as exc:
            runtime.family = "placeholder"
            runtime.error = str(exc)
            log.error("Failed to load TTS model '%s': %s", model_id, exc)

        yield

        runtime.qwen_model = None
        runtime.kokoro_pipeline = None
        runtime.bark_model = None
        runtime.bark_processor = None

    app = FastAPI(title="oCabra TTS Worker", lifespan=lifespan)

    def _is_loaded() -> bool:
        if runtime.family == "qwen":
            return runtime.qwen_model is not None
        if runtime.family == "kokoro":
            return runtime.kokoro_pipeline is not None
        if runtime.family == "bark":
            return runtime.bark_model is not None
        return False

    @app.get("/health")
    async def health() -> JSONResponse:
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Model not loaded")
        return JSONResponse({"status": "ok", "model_loaded": True, "family": runtime.family})

    @app.get("/info")
    async def info() -> JSONResponse:
        return JSONResponse({
            "backend": "tts",
            "model_id": runtime.model_id,
            "family": runtime.family,
            "gpu_indices": runtime.gpu_indices,
            "device": runtime.device,
            "loaded": _is_loaded(),
            "error": runtime.error,
        })

    @app.get("/voices")
    async def voices() -> JSONResponse:
        if runtime.family == "kokoro":
            voices_list = list(KOKORO_VOICES)
            languages = ["English", "British English", "Japanese", "Korean", "Chinese"]
            openai_voice_map: dict[str, str] | None = dict(OPENAI_TO_KOKORO)
        elif runtime.family == "bark":
            voices_list = list(BARK_SPEAKERS)
            languages = ["English"]
            openai_voice_map = dict(OPENAI_TO_BARK)
        elif runtime.family == "qwen" and runtime.qwen_type == "custom_voice":
            voices_list = runtime.speakers or list(CUSTOMVOICE_SPEAKERS)
            languages = list(SUPPORTED_LANGUAGES)
            openai_voice_map = dict(OPENAI_TO_CUSTOMVOICE)
        else:
            # qwen "base" is voice-cloning only (``voice`` is ignored unless
            # no ``reference_audio`` is given, in which case it 400s) — no
            # OpenAI name maps to anything real here.
            voices_list = list(OPENAI_VOICES)
            languages = list(SUPPORTED_LANGUAGES)
            openai_voice_map = None

        return JSONResponse({
            "voices": voices_list,
            "model_type": runtime.family,
            "speakers": runtime.speakers,
            "languages": languages,
            "supports_voice_clone": runtime.family == "qwen" and runtime.qwen_type == "base",
            "openai_voice_map": openai_voice_map,
        })

    @app.post("/synthesize")
    async def synthesize(body: SynthesizeRequest) -> StreamingResponse:
        """Synthesise *input* and return a single audio file in *response_format*."""
        if not body.input.strip():
            raise HTTPException(status_code=400, detail="'input' text is required")
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Model not loaded")

        speed = max(0.25, min(body.speed, 4.0))
        fmt   = body.response_format.lower()

        try:
            wav_bytes = await asyncio.to_thread(
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        encoded, content_type = _encode_audio(wav_bytes, fmt)

        return StreamingResponse(
            io.BytesIO(encoded),
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="speech.{fmt}"'},
        )

    @app.post("/synthesize/stream")
    async def synthesize_stream(body: SynthesizeRequest) -> StreamingResponse:
        """Synthesise *input* sentence-by-sentence and stream audio chunks progressively."""
        if not body.input.strip():
            raise HTTPException(status_code=400, detail="'input' text is required")
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Model not loaded")

        speed     = max(0.25, min(body.speed, 4.0))
        fmt       = body.response_format.lower()
        sentences = _split_sentences(body.input.strip())
        content_type = FORMAT_CONTENT_TYPES.get(fmt, "audio/mpeg")

        async def _generate():
            # ``_wav_to_pcm`` (numpy copy) and ``_encode_audio`` (ffmpeg
            # subprocess for mp3/opus/aac/flac fallbacks) are blocking.
            # Running them on the asyncio loop thread stalls Starlette's
            # chunked response writer between sentences, which collapses the
            # whole stream into a single late burst at the end (observed
            # TTFB equal to total duration). Offload to a worker thread so
            # the loop can push the previous chunk to the client while the
            # next encode runs.
            wav_header_sent = False
            for sentence in sentences:
                wav_bytes = await asyncio.to_thread(
                    _synthesize,
                    runtime,
                    sentence,
                    body.voice,
                    speed,
                    body.language,
                    body.reference_audio,
                    body.reference_text,
                    body.speaker,
                    body.instruct,
                )
                if not wav_bytes:
                    continue

                if fmt == "wav":
                    if not wav_header_sent:
                        sr, ch, bits = _wav_params(wav_bytes)
                        yield _make_streaming_wav_header(sr, ch, bits)
                        wav_header_sent = True
                    yield await asyncio.to_thread(_wav_to_pcm, wav_bytes)
                elif fmt == "pcm":
                    yield await asyncio.to_thread(_wav_to_pcm, wav_bytes)
                else:
                    encoded, _ = await asyncio.to_thread(_encode_audio, wav_bytes, fmt)
                    yield encoded

        return StreamingResponse(
            _generate(),
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="speech.{fmt}"'},
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
