"""
oCabra Piper TTS Worker — rhasspy/piper1-gpl (VITS, CPU-first)

Standalone FastAPI worker that loads a single Piper voice and exposes the
standard oCabra TTS worker interface:

  GET  /health           — liveness probe
  GET  /info              — runtime metadata
  GET  /voices            — the one voice this worker serves
  POST /synthesize        — full-text synthesis, returns audio
  POST /synthesize/stream — streaming synthesis (Piper yields per internal
                            sentence chunk natively — no manual sentence
                            splitting needed here, unlike the other workers)

One worker process serves exactly one voice (one ``.onnx``/``.onnx.json``
pair), mirroring the chatterbox/tts worker convention of one subprocess per
model_id. Runs entirely on CPU — no CUDA, no GPU indices used.
"""

from __future__ import annotations

import argparse
import io
import logging
import struct
import subprocess
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

FORMAT_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

_PIPER_VOICES_REPO = "rhasspy/piper-voices"


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


class PiperRuntime:
    def __init__(self) -> None:
        self.model_id: str = ""
        self.voice_name: str = ""
        self.voice: Any = None
        self.sample_rate: int = 22050
        self.languages: list[str] = []
        self.speaker_id_map: dict[str, int] = {}
        self.error: str | None = None


runtime = PiperRuntime()


# ---------------------------------------------------------------------------
# Voice resolution / loading
# ---------------------------------------------------------------------------


def _split_voice_name(voice_name: str) -> tuple[str, str, str, str]:
    """Split a Piper voice name into (lang, lang_region, speaker, quality).

    Follows the rhasspy/piper-voices repo layout, e.g.
    ``es_ES-davefx-medium`` -> ("es", "es_ES", "davefx", "medium").
    """
    try:
        lang_region, speaker, quality = voice_name.split("-", 2)
    except ValueError as exc:
        raise ValueError(
            f"Invalid Piper voice name '{voice_name}' — expected "
            "'<lang>_<REGION>-<speaker>-<quality>' (e.g. 'es_ES-davefx-medium')"
        ) from exc
    lang = lang_region.split("_")[0]
    return lang, lang_region, speaker, quality


def _download_voice_file(voice_name: str, suffix: str, voices_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    lang, lang_region, speaker, quality = _split_voice_name(voice_name)
    repo_filename = f"{lang}/{lang_region}/{speaker}/{quality}/{voice_name}{suffix}"
    local_path = hf_hub_download(
        repo_id=_PIPER_VOICES_REPO,
        filename=repo_filename,
        local_dir=str(voices_dir),
    )
    return Path(local_path)


def _resolve_voice_files(
    voice_name: str,
    voices_dir: Path,
    onnx_override: str | None,
    config_override: str | None,
) -> tuple[Path, Path]:
    """Resolve (onnx_path, config_path) for *voice_name*.

    Explicit overrides win (used for custom/fine-tuned voices that don't
    live in the rhasspy/piper-voices repo — e.g. a GLaDOS voice trained
    locally). Otherwise look in ``voices_dir``, downloading from
    rhasspy/piper-voices on first use.
    """
    if onnx_override and config_override:
        onnx_path, config_path = Path(onnx_override), Path(config_override)
        if onnx_path.is_file() and config_path.is_file():
            return onnx_path, config_path

    voices_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = voices_dir / f"{voice_name}.onnx"
    config_path = voices_dir / f"{voice_name}.onnx.json"
    if not onnx_path.is_file():
        onnx_path = _download_voice_file(voice_name, ".onnx", voices_dir)
    if not config_path.is_file():
        config_path = _download_voice_file(voice_name, ".onnx.json", voices_dir)
    return onnx_path, config_path


def _load_piper_voice(
    voice_name: str,
    voices_dir: Path,
    onnx_override: str | None,
    config_override: str | None,
) -> tuple[Any, int, list[str], dict[str, int]]:
    """Load a Piper voice. Returns (voice, sample_rate, languages, speaker_id_map)."""
    import json

    from piper import PiperVoice

    onnx_path, config_path = _resolve_voice_files(
        voice_name, voices_dir, onnx_override, config_override
    )
    voice = PiperVoice.load(onnx_path, config_path=config_path)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    sample_rate = int(cfg.get("audio", {}).get("sample_rate") or 22050)
    language = cfg.get("language", {})
    lang_code = language.get("code") if isinstance(language, dict) else None
    languages = [lang_code] if lang_code else []
    speaker_id_map = cfg.get("speaker_id_map") or {}
    return voice, sample_rate, languages, speaker_id_map


# ---------------------------------------------------------------------------
# Audio helpers (same shape as chatterbox_worker.py / tts_worker.py)
# ---------------------------------------------------------------------------


def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


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
    encoded = _soundfile_encode(wav_bytes, fmt)
    if encoded:
        return encoded

    codec: dict[str, list[str]] = {
        "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "4"],
        "opus": ["-f", "ogg", "-codec:a", "opus", "-strict", "-2", "-b:a", "64k"],
        "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "128k"],
        "flac": ["-f", "flac", "-codec:a", "flac"],
    }
    codec_args = codec.get(fmt)
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
# Sentence splitting (same shape as chatterbox_worker.py / tts_worker.py, so
# TTFB behaviour is consistent across backends regardless of whether the
# underlying engine does its own internal chunking)
# ---------------------------------------------------------------------------

_MIN_SENTENCE_CHARS = 8


def _split_sentences(text: str) -> list[str]:
    import re

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


def _synthesis_config(rt: PiperRuntime, speaker: str | None, speed: float) -> Any:
    from piper.config import SynthesisConfig

    speaker_id = None
    if speaker and rt.speaker_id_map:
        speaker_id = rt.speaker_id_map.get(speaker)
    length_scale = 1.0 / speed if speed and speed > 0 else None
    return SynthesisConfig(speaker_id=speaker_id, length_scale=length_scale)


def _synthesize_full(rt: PiperRuntime, text: str, speaker: str | None, speed: float) -> bytes:
    """Synthesize *text* (any length — Piper splits internally) in one call. Returns WAV bytes."""
    syn_config = _synthesis_config(rt, speaker, speed)
    pcm_chunks = [chunk.audio_int16_bytes for chunk in rt.voice.synthesize(text, syn_config)]
    return _pcm16_to_wav(b"".join(pcm_chunks), rt.sample_rate)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    text: str | None = None
    input: str | None = None  # OpenAI-compat alias for text
    voice: str = "default"  # speaker name within a multi-speaker voice, if any
    speed: float = 1.0
    response_format: str = "wav"

    @property
    def resolved_text(self) -> str:
        return (self.input or self.text or "").strip()


def create_app(
    model_id: str,
    voice_name: str,
    voices_dir: Path,
    onnx_override: str | None,
    config_override: str | None,
) -> FastAPI:
    runtime.model_id = model_id
    runtime.voice_name = voice_name

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import asyncio

        log = logging.getLogger("piper_worker")
        try:
            log.info("Loading Piper voice '%s'...", voice_name)
            voice, sample_rate, languages, speaker_id_map = await asyncio.to_thread(
                _load_piper_voice, voice_name, voices_dir, onnx_override, config_override
            )
            runtime.voice = voice
            runtime.sample_rate = sample_rate
            runtime.languages = languages
            runtime.speaker_id_map = speaker_id_map
            runtime.error = None
            log.info(
                "Piper voice loaded: %s (sr=%d, languages=%s, speakers=%d)",
                voice_name,
                sample_rate,
                languages,
                len(speaker_id_map),
            )
        except ImportError as exc:
            runtime.error = f"Piper dependencies not installed: {exc}. Install with: pip install piper-tts"
            log.error(runtime.error)
        except Exception as exc:
            runtime.error = str(exc)
            log.error("Failed to load Piper voice '%s': %s", voice_name, exc)

        yield

        runtime.voice = None

    app = FastAPI(title="oCabra Piper TTS Worker", lifespan=lifespan)

    def _is_loaded() -> bool:
        return runtime.voice is not None

    @app.get("/health")
    async def health() -> JSONResponse:
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Voice not loaded")
        return JSONResponse({"status": "ok", "model_loaded": True})

    @app.get("/info")
    async def info() -> JSONResponse:
        return JSONResponse(
            {
                "backend": "piper",
                "model_id": runtime.model_id,
                "voice": runtime.voice_name,
                "device": "cpu",
                "loaded": _is_loaded(),
                "sample_rate": runtime.sample_rate,
                "error": runtime.error,
            }
        )

    @app.get("/voices")
    async def voices() -> JSONResponse:
        return JSONResponse(
            {
                "voices": [runtime.voice_name],
                "model_type": "piper",
                "speakers": list(runtime.speaker_id_map.keys()),
                "languages": runtime.languages,
                "supports_voice_clone": False,
            }
        )

    @app.post("/synthesize")
    async def synthesize(body: SynthesizeRequest) -> StreamingResponse:
        import asyncio

        if not body.resolved_text:
            raise HTTPException(status_code=400, detail="'text' is required")
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Voice not loaded")

        fmt = body.response_format.lower()
        wav_bytes = await asyncio.to_thread(
            _synthesize_full, runtime, body.resolved_text, body.voice, body.speed
        )
        encoded, content_type = _encode_audio(wav_bytes, fmt)

        return StreamingResponse(
            io.BytesIO(encoded),
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="speech.{fmt}"'},
        )

    @app.post("/synthesize/stream")
    async def synthesize_stream(body: SynthesizeRequest) -> StreamingResponse:
        import asyncio

        if not body.resolved_text:
            raise HTTPException(status_code=400, detail="'text' is required")
        if not _is_loaded():
            raise HTTPException(status_code=503, detail=runtime.error or "Voice not loaded")

        fmt = body.response_format.lower()
        sentences = _split_sentences(body.resolved_text)
        content_type = FORMAT_CONTENT_TYPES.get(fmt, "audio/wav")

        async def _generate():
            # One to_thread() call per sentence (not one for the whole text)
            # so the event loop can flush each chunk to the client while the
            # next sentence is still synthesizing — see the TTFB-collapse
            # note in chatterbox_worker.py for why this matters.
            wav_header_sent = False
            for sentence in sentences:
                wav_bytes = await asyncio.to_thread(
                    _synthesize_full, runtime, sentence, body.voice, body.speed
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
    parser = argparse.ArgumentParser(description="oCabra Piper TTS worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--voice", required=True, help="Piper voice name, e.g. es_ES-davefx-medium")
    parser.add_argument("--voices-dir", required=True)
    parser.add_argument("--onnx-path", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    app = create_app(
        args.model_id,
        args.voice,
        Path(args.voices_dir),
        args.onnx_path,
        args.config_path,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
