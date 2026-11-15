"""Audio formats of the Realtime API, converted at the edges of a session.

Inside a :class:`~ocabra.core.realtime_session.RealtimeSession` the input
buffer is always PCM16 mono at 16 kHz (what VAD and Whisper expect) and TTS
audio is produced at whatever rate the TTS worker uses. Clients speak other
formats:

- Beta dialect: ``input_audio_format``/``output_audio_format`` = ``"pcm16"``,
  ``"g711_ulaw"`` or ``"g711_alaw"``.
- GA dialect: ``session.audio.input.format`` = ``{"type": "audio/pcm",
  "rate": 24000}``, ``{"type": "audio/pcmu"}`` or ``{"type": "audio/pcma"}``.

This module turns client input into 16 kHz PCM16 and 16/22/24 kHz TTS PCM16
into what the client asked for. Pure Python on purpose: the API image ships
neither numpy nor scipy, and ``audioop`` is gone in Python 3.13. Voice at
these rates is a few tens of thousands of samples per second, well within
reach.
"""

from __future__ import annotations

import io
import math
import sys
import wave
from array import array
from operator import mul
from typing import Any

#: Internal rate of the session's input buffer (VAD + STT).
INTERNAL_INPUT_RATE = 16000

PCM16 = "pcm16"
ULAW = "g711_ulaw"
ALAW = "g711_alaw"

_GA_TYPES = {
    "audio/pcm": PCM16,
    "audio/pcmu": ULAW,
    "audio/pcma": ALAW,
}
_BETA_TYPES = {PCM16: PCM16, ULAW: ULAW, ALAW: ALAW, "pcm": PCM16}

#: Default rate of each format. Beta ``pcm16`` stays at 16 kHz because that is
#: what oCabra documented and what its existing clients send; the GA dialect
#: (``audio/pcm``) defaults to 24 kHz, as the GA spec says.
_G711_RATE = 8000
_BETA_PCM_RATE = 16000
_GA_PCM_RATE = 24000


def parse_audio_format(value: Any, *, ga: bool, rate: Any = None) -> tuple[str, int] | None:
    """``(kind, sample_rate)`` for a format in either dialect, or ``None``.

    ``value`` is a beta string (``"pcm16"``…) or a GA object
    (``{"type": "audio/pcm", "rate": 24000}``). ``rate`` is an explicit rate
    from elsewhere (oCabra's beta extension ``input_audio_sample_rate``).
    """
    explicit = rate
    if isinstance(value, dict):
        kind = _GA_TYPES.get(str(value.get("type") or "").lower())
        explicit = value.get("rate", explicit)
        ga = True
    elif isinstance(value, str):
        key = value.strip().lower()
        kind = _GA_TYPES.get(key) or _BETA_TYPES.get(key)
        if key in _GA_TYPES:
            ga = True
    else:
        return None
    if kind is None:
        return None
    if kind in (ULAW, ALAW):
        return kind, _G711_RATE
    try:
        sample_rate = int(explicit) if explicit else 0
    except (TypeError, ValueError):
        sample_rate = 0
    if sample_rate <= 0:
        sample_rate = _GA_PCM_RATE if ga else _BETA_PCM_RATE
    return kind, sample_rate


def ga_format(kind: str, sample_rate: int) -> dict[str, Any]:
    """The GA ``format`` object for a ``(kind, rate)`` pair."""
    if kind == ULAW:
        return {"type": "audio/pcmu"}
    if kind == ALAW:
        return {"type": "audio/pcma"}
    return {"type": "audio/pcm", "rate": int(sample_rate)}


# -- G.711 ------------------------------------------------------------------


def _ulaw_to_linear(byte: int) -> int:
    u = ~byte & 0xFF
    exponent = (u >> 4) & 0x07
    sample = (((u & 0x0F) << 3) + 0x84) << exponent
    sample -= 0x84
    return -sample if u & 0x80 else sample


def _linear_to_ulaw(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    if sample < 0:
        sample = -sample
    sample = min(sample, 32635) + 0x84
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not sample & mask:
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _alaw_to_linear(byte: int) -> int:
    a = byte ^ 0x55
    exponent = (a >> 4) & 0x07
    mantissa = a & 0x0F
    if exponent == 0:
        sample = (mantissa << 4) + 8
    else:
        sample = ((mantissa << 4) + 0x108) << (exponent - 1)
    return sample if a & 0x80 else -sample


def _linear_to_alaw(sample: int) -> int:
    sign = 0x80 if sample >= 0 else 0
    magnitude = min(abs(sample), 32767)
    if magnitude < 256:
        exponent, mantissa = 0, magnitude >> 4
    else:
        exponent = min(7, magnitude.bit_length() - 8)
        mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (sign | (exponent << 4) | mantissa) ^ 0x55


_ULAW_DECODE = [_ulaw_to_linear(b) for b in range(256)]
_ALAW_DECODE = [_alaw_to_linear(b) for b in range(256)]


def _pcm16_samples(data: bytes) -> array:
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def _pcm16_bytes(samples) -> bytes:
    out = array("h", samples)
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


# -- resampling --------------------------------------------------------------


def _lowpass(cutoff: float, taps: int) -> list[float]:
    """Blackman-windowed sinc; ``cutoff`` in cycles per input sample (< 0.5)."""
    m = taps - 1
    h = []
    for n in range(taps):
        x = n - m / 2
        s = 2 * cutoff if x == 0 else math.sin(2 * math.pi * cutoff * x) / (math.pi * x)
        w = 0.42 - 0.5 * math.cos(2 * math.pi * n / m) + 0.08 * math.cos(4 * math.pi * n / m)
        h.append(s * w)
    total = sum(h)
    return [v / total for v in h]


class StreamResampler:
    """Resample a PCM16 stream fed in arbitrary chunks, keeping state between
    them so chunk boundaries add no clicks and durations stay exact.

    Downsampling first low-passes at the new Nyquist (anti-aliasing), then
    interpolates linearly; upsampling interpolates linearly.
    """

    def __init__(self, src_rate: int, dst_rate: int, taps: int = 24) -> None:
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self._step = self.src_rate / self.dst_rate
        self._pos = 0.0
        self._buf: list[float] = []
        self._fir: list[float] | None = None
        self._hist: list[float] = []
        if self.dst_rate < self.src_rate:
            self._fir = _lowpass(0.45 * self.dst_rate / self.src_rate, taps)[::-1]
            self._hist = [0.0] * (taps - 1)

    def process(self, samples) -> list[int]:
        if self.src_rate == self.dst_rate:
            return list(samples)
        if self._fir is not None:
            ext = self._hist + [float(s) for s in samples]
            taps = len(self._fir)
            fir = self._fir
            filtered = [sum(map(mul, fir, ext[n : n + taps])) for n in range(len(samples))]
            self._hist = ext[len(ext) - (taps - 1) :]
        else:
            filtered = [float(s) for s in samples]
        self._buf.extend(filtered)
        out: list[int] = []
        buf = self._buf
        pos = self._pos
        step = self._step
        last = len(buf) - 1
        while pos < last:
            i = int(pos)
            frac = pos - i
            value = buf[i] + (buf[i + 1] - buf[i]) * frac
            out.append(max(-32768, min(32767, int(round(value)))))
            pos += step
        drop = int(pos)
        if drop:
            del buf[:drop]
            pos -= drop
        self._pos = pos
        return out


# -- session edges -----------------------------------------------------------


class InputAudioDecoder:
    """Client audio (any supported format/rate) → PCM16 mono 16 kHz."""

    def __init__(self, kind: str = PCM16, sample_rate: int = INTERNAL_INPUT_RATE) -> None:
        self.kind = kind
        self.sample_rate = int(sample_rate)
        self._carry = b""
        self._resampler = (
            StreamResampler(self.sample_rate, INTERNAL_INPUT_RATE)
            if self.sample_rate != INTERNAL_INPUT_RATE
            else None
        )

    @property
    def passthrough(self) -> bool:
        return self.kind == PCM16 and self._resampler is None

    def feed(self, data: bytes) -> bytes:
        if self.kind == ULAW:
            samples = [_ULAW_DECODE[b] for b in data]
        elif self.kind == ALAW:
            samples = [_ALAW_DECODE[b] for b in data]
        else:
            data = self._carry + data
            if len(data) % 2:
                self._carry, data = data[-1:], data[:-1]
            else:
                self._carry = b""
            if self._resampler is None:
                return data
            samples = _pcm16_samples(data)
        if self._resampler is not None:
            samples = self._resampler.process(samples)
        return _pcm16_bytes(samples)


class OutputAudioEncoder:
    """PCM16 mono at ``src_rate`` (TTS) → the client's output format."""

    def __init__(self, kind: str, sample_rate: int, src_rate: int) -> None:
        self.kind = kind
        self.sample_rate = int(sample_rate)
        self._resampler = (
            StreamResampler(int(src_rate), self.sample_rate)
            if int(src_rate) != self.sample_rate
            else None
        )

    def feed(self, pcm16: bytes) -> bytes:
        if self.kind == PCM16 and self._resampler is None:
            return pcm16
        samples = _pcm16_samples(pcm16[: len(pcm16) - len(pcm16) % 2])
        if self._resampler is not None:
            samples = self._resampler.process(samples)
        if self.kind == ULAW:
            return bytes(_linear_to_ulaw(s) for s in samples)
        if self.kind == ALAW:
            return bytes(_linear_to_alaw(s) for s in samples)
        return _pcm16_bytes(samples)


def bytes_per_second(kind: str, sample_rate: int) -> int:
    return int(sample_rate) * (1 if kind in (ULAW, ALAW) else 2)


def split_wav(data: bytes) -> tuple[bytes, int] | None:
    """``(pcm16_frames, sample_rate)`` of a mono 16-bit WAV, or ``None``."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getsampwidth() != 2:
                return None
            frames = wav.readframes(wav.getnframes())
            rate = wav.getframerate()
            channels = wav.getnchannels()
    except (wave.Error, EOFError):
        return None
    if channels == 2:
        stereo = _pcm16_samples(frames)
        frames = _pcm16_bytes((stereo[i] + stereo[i + 1]) // 2 for i in range(0, len(stereo), 2))
    elif channels != 1:
        return None
    return frames, rate


def openai_vad_threshold_to_rms(threshold: Any, default_rms: float) -> float | None:
    """OpenAI's ``turn_detection.threshold`` is a speech probability (0–1,
    default 0.5); oCabra's VAD compares RMS energy (default 0.02). Values
    above 0.1 are read as OpenAI's scale and mapped linearly so that 0.5 is
    the default energy; smaller ones are taken as a raw RMS (oCabra's own
    clients). Passing 0.5 straight through demanded 25× the normal level and
    the VAD never fired — the Playground has been sending exactly that.
    """
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value > 0.1:
        return default_rms * (value / 0.5)
    return value
