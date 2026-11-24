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
import logging
import os
import sys
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
        logging.getLogger(__name__).warning(
            "perth_implicit_watermarker_unavailable, using DummyWatermarker"
        )


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


def _decode_reference_audio(base64_audio: str) -> str:
    """Decode base64-encoded audio to a temporary WAV file. Returns file path."""
    import base64
    import tempfile

    data = base64.b64decode(base64_audio)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


_ALLOWED_VOICE_REF_DIRS = (
    "/data/profiles/",
    "/data/models/",
    "/tmp/",
)


def _is_voice_ref_path_safe(path: str) -> bool:
    """Check that a voice_ref path is within an allowed oCabra-controlled directory.

    Prevents path traversal attacks by resolving the real path and checking
    it starts with one of the allowed prefixes.
    """
    if not path:
        return False
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    return any(resolved.startswith(prefix) for prefix in _ALLOWED_VOICE_REF_DIRS)


def _resolve_voice_ref(
    voice_ref: str | None, reference_audio: str | None
) -> str | None:
    """Resolve voice reference from either a file path or base64 audio.

    Only paths within oCabra-controlled directories are accepted.
    Returns a file path (possibly temporary) or None.
    """
    if voice_ref:
        if not _is_voice_ref_path_safe(voice_ref):
            logging.getLogger(__name__).warning(
                "voice_ref path rejected (outside allowed dirs): %s", voice_ref
            )
            return None
        if os.path.isfile(voice_ref):
            return voice_ref
    if reference_audio:
        return _decode_reference_audio(reference_audio)
    return None


def _synthesize(
    rt: ChatterboxRuntime,
    text: str,
    voice_ref: str | None = None,
    reference_audio: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> bytes:
    """Synthesize text using the loaded Chatterbox model."""
    import torch

    kwargs: dict[str, Any] = {
        "text": text,
    }

    # Voice cloning via reference audio (file path or base64)
    resolved_ref = _resolve_voice_ref(voice_ref, reference_audio)
    tmp_ref = None
    if resolved_ref:
        # Track if we created a temp file that needs cleanup
        if resolved_ref != voice_ref:
            tmp_ref = resolved_ref
        kwargs["audio_prompt_path"] = resolved_ref

    try:
        with torch.inference_mode():
            wav = rt.model.generate(**kwargs)
        return _tensor_to_wav(wav, rt.sample_rate)
    finally:
        if tmp_ref:
            try:
                os.unlink(tmp_ref)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    text: str | None = None
    input: str | None = None  # OpenAI-compat alias for text
    voice: str = "default"
    language: str = "en"
    speed: float = 1.0
    response_format: str = "mp3"
    voice_ref: str | None = None  # path to reference audio for voice cloning
    reference_audio: str | None = None  # base64-encoded reference audio (OpenAI-compat)
    reference_text: str | None = None  # transcript of reference audio (unused but accepted)

    @property
    def resolved_text(self) -> str:
        """Return text content, preferring 'input' (OpenAI) over 'text'."""
        return (self.input or self.text or "").strip()


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
                # ``SynthesizeRequest.voice`` is never read by this worker
                # (see ``synthesize``/``_synthesize`` below) — OpenAI voice
                # names (alloy, echo...) have no effect, every request gets
                # the single "default" voice or the configured voice clone.
                "openai_voice_map": None,
            }
        )

    @app.post("/synthesize")
    async def synthesize(body: SynthesizeRequest) -> StreamingResponse:
        if not body.resolved_text:
            raise HTTPException(status_code=400, detail="'text' is required")
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Model not loaded")

        fmt = body.response_format.lower()

        try:
            wav_bytes = await asyncio.to_thread(
                _synthesize,
                runtime,
                body.resolved_text,
                body.voice_ref,
                body.reference_audio,
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
        if not body.resolved_text:
            raise HTTPException(status_code=400, detail="'text' is required")
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Model not loaded")

        fmt = body.response_format.lower()
        sentences = _split_sentences(body.resolved_text)
        content_type = FORMAT_CONTENT_TYPES.get(fmt, "audio/mpeg")

        async def _generate():
            # Both ``_wav_to_pcm`` (numpy copy) and ``_encode_audio`` (ffmpeg
            # subprocess for mp3/opus/aac/flac fallbacks) are blocking. Running
            # them on the asyncio loop thread stalls Starlette's chunked
            # response writer between sentences, which collapses the whole
            # stream into a single late burst at the end — observed TTFB equal
            # to total duration before this fix. Offload to a worker thread so
            # the loop can push the previous chunk to the client while the
            # next encode runs.
            wav_header_sent = False
            for sentence in sentences:
                wav_bytes = await asyncio.to_thread(
                    _synthesize,
                    runtime,
                    sentence,
                    body.voice_ref,
                    body.reference_audio,
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
