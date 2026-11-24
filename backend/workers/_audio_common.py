"""Utilidades de audio compartidas por los workers de TTS.

``tts_worker``, ``chatterbox_worker``, ``piper_worker`` y ``voxtral_worker`` son
scripts independientes que se lanzan como subproceso (``python worker.py …``) y
repetían el mismo manejo de WAV/PCM, la codificación a mp3/opus/flac y el
troceado en frases. Este módulo reúne esas piezas comunes. Se importa como
módulo de primer nivel (los workers insertan su carpeta en ``sys.path``), por lo
que no depende del paquete ``ocabra`` ni carga nada pesado al importarse: numpy
y soundfile se importan de forma perezosa dentro de las funciones que los usan.

El comportamiento es idéntico al que tenía cada worker; solo cambia de sitio.
"""

from __future__ import annotations

import io
import logging
import struct
import subprocess
import wave

# Content-Type por formato de salida.
FORMAT_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# Longitud mínima (caracteres) de una frase antes de emitir un trozo de TTS en
# modo streaming; los fragmentos más cortos se funden con el siguiente.
MIN_SENTENCE_CHARS = 8


def wav_params(wav_bytes: bytes) -> tuple[int, int, int]:
    """Return (sample_rate, channels, sample_width_bits) from WAV header."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth() * 8


def wav_to_pcm(wav_bytes: bytes) -> bytes:
    """Strip WAV header and return raw PCM16LE frames."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def make_streaming_wav_header(sample_rate: int, channels: int, bits: int) -> bytes:
    """44-byte RIFF/WAV header with 0xFFFFFFFF sizes for unknown-length streaming.

    Most decoders (including Android MediaPlayer) accept this for progressive playback.
    """
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


def soundfile_encode(wav_bytes: bytes, fmt: str) -> bytes | None:
    """Encode WAV bytes to *fmt* using soundfile (in-process, no subprocess).

    Returns encoded bytes on success, ``None`` on failure.
    Supports: mp3, ogg/opus, flac.
    """
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
        kwargs: dict = {"format": sf_format}
        if sf_subtype:
            kwargs["subtype"] = sf_subtype
        sf.write(out, samples, sr, **kwargs)
        return out.getvalue()
    except Exception:
        return None


def ffmpeg_encode(wav_bytes: bytes, fmt: str) -> bytes:
    """Encode WAV bytes to *fmt*.

    Tries soundfile first (fast, in-process).  Falls back to ffmpeg subprocess
    if available.  Returns original WAV bytes if all methods fail.
    """
    # ---- soundfile (preferred) ----
    encoded = soundfile_encode(wav_bytes, fmt)
    if encoded:
        return encoded

    # ---- ffmpeg subprocess (fallback) ----
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
    return wav_bytes  # fallback to WAV


def encode_audio(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    """Convert internal WAV bytes to the requested *fmt*.

    Returns (encoded_bytes, content_type).
    """
    fmt = fmt.lower().strip()
    content_type = FORMAT_CONTENT_TYPES.get(fmt, "audio/wav")
    if fmt == "wav":
        return wav_bytes, content_type
    if fmt == "pcm":
        return wav_to_pcm(wav_bytes), content_type
    return ffmpeg_encode(wav_bytes, fmt), content_type


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences suitable for progressive TTS synthesis.

    Fragments shorter than MIN_SENTENCE_CHARS are merged with the next sentence
    to avoid synthesising tiny clips (e.g. "Ok.") that sound unnatural in isolation.
    """
    import re

    # Split on .!?…; followed by whitespace, or on newlines
    raw = re.split(r"(?<=[.!?…;])\s+|\n+", text.strip())

    result: list[str] = []
    buf = ""
    for part in raw:
        part = part.strip()
        if not part:
            continue
        buf = (buf + " " + part).strip() if buf else part
        # Yield when buffer ends with a sentence-ending punctuation and is long enough
        if len(buf) >= MIN_SENTENCE_CHARS and re.search(r"[.!?…;]$", buf):
            result.append(buf)
            buf = ""
    if buf:
        # Short remainder — merge with last sentence to avoid tiny fragments
        if result and len(buf) < MIN_SENTENCE_CHARS:
            result[-1] = result[-1] + " " + buf
        else:
            result.append(buf)
    return result or [text.strip()]
