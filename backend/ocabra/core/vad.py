"""
Simple Voice Activity Detection (VAD) using RMS energy.

No ML dependencies — uses only struct for PCM16 processing.
Designed as a drop-in component for the Realtime API session;
can be replaced with Silero VAD in a future iteration.
"""

from __future__ import annotations

import struct
from enum import StrEnum


class VadEvent(StrEnum):
    """Events emitted by the VAD processor."""

    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPED = "speech_stopped"


class SimpleVAD:
    """Voice Activity Detection by RMS energy level.

    Processes PCM16 audio chunks and tracks speech/silence state transitions.
    Returns :class:`VadEvent` on state changes, or ``None`` when no transition occurs.

    Attributes:
        SPEECH_THRESHOLD: RMS level above which audio is considered speech.
        SILENCE_DURATION_MS: Milliseconds of silence required to end a speech segment.
        MIN_SPEECH_MS: Minimum milliseconds of speech required to emit SPEECH_STARTED.
    """

    SPEECH_THRESHOLD: float = 0.02
    SILENCE_DURATION_MS: int = 800
    MIN_SPEECH_MS: int = 200

    def __init__(
        self,
        threshold: float | None = None,
        silence_duration_ms: int | None = None,
        prefix_padding_ms: int | None = None,
    ) -> None:
        if threshold is not None:
            self.SPEECH_THRESHOLD = threshold
        if silence_duration_ms is not None:
            self.SILENCE_DURATION_MS = silence_duration_ms
        # prefix_padding_ms accepted for API compat but not used in the simple impl
        self._prefix_padding_ms = prefix_padding_ms or 300

        self._is_speaking = False
        self._speech_ms: float = 0.0
        self._silence_ms: float = 0.0
        self._speech_started_emitted = False

    def reset(self) -> None:
        """Reset internal state."""
        self._is_speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._speech_started_emitted = False

    def process_chunk(self, pcm16_bytes: bytes, sample_rate: int = 16000) -> VadEvent | None:
        """Process a chunk of PCM16LE audio and return a VAD event if a state transition occurs.

        Args:
            pcm16_bytes: Raw PCM16 little-endian audio bytes.
            sample_rate: Sample rate in Hz (default 16000).

        Returns:
            VadEvent.SPEECH_STARTED when speech begins (after MIN_SPEECH_MS),
            VadEvent.SPEECH_STOPPED when silence exceeds SILENCE_DURATION_MS,
            or None if no transition.
        """
        if len(pcm16_bytes) < 2:
            return None

        # Calculate RMS from PCM16 samples
        num_samples = len(pcm16_bytes) // 2
        samples = struct.unpack(f"<{num_samples}h", pcm16_bytes[: num_samples * 2])

        sum_sq = 0.0
        for s in samples:
            normalized = s / 32768.0
            sum_sq += normalized * normalized
        rms = (sum_sq / num_samples) ** 0.5 if num_samples > 0 else 0.0

        # Duration of this chunk in milliseconds
        chunk_ms = (num_samples / sample_rate) * 1000.0

        is_speech = rms > self.SPEECH_THRESHOLD

        if is_speech:
            self._silence_ms = 0.0
            self._speech_ms += chunk_ms

            if not self._is_speaking and self._speech_ms >= self.MIN_SPEECH_MS:
                self._is_speaking = True
                self._speech_started_emitted = True
                return VadEvent.SPEECH_STARTED
        else:
            if self._is_speaking:
                self._silence_ms += chunk_ms
                if self._silence_ms >= self.SILENCE_DURATION_MS:
                    self._is_speaking = False
                    self._speech_ms = 0.0
                    self._silence_ms = 0.0
                    self._speech_started_emitted = False
                    return VadEvent.SPEECH_STOPPED
            else:
                # Not speaking and no speech detected — reset speech accumulator
                self._speech_ms = 0.0

        return None
