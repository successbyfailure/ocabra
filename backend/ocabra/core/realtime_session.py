"""
RealtimeSession — Core session manager for the OpenAI Realtime API WebSocket.

Coordinates the pipeline: audio buffer -> Whisper STT -> LLM streaming -> TTS -> audio out.
Emits protocol-compliant server events over the WebSocket connection.

Tool calls are not yet implemented (documented as future work).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
import uuid
import wave
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from ocabra.core.vad import SimpleVAD, VadEvent

if TYPE_CHECKING:
    from fastapi import WebSocket

    from ocabra.api._deps_auth import UserContext
    from ocabra.core.model_manager import ModelManager
    from ocabra.core.worker_pool import WorkerPool

logger = structlog.get_logger(__name__)

# Sentence-splitting regex: split on period, exclamation, question mark, or newline
# followed by whitespace or end-of-string.
_SENTENCE_RE = re.compile(r"(?<=[.!?\n])\s+")

# Default audio parameters
_SAMPLE_RATE = 24000  # TTS output sample rate
_INPUT_SAMPLE_RATE = 16000  # Input audio sample rate (PCM16 from client)
# Transcription-only streaming: how often to re-transcribe the growing buffer for
# partial hypotheses, and the minimum audio before we bother.
_PARTIAL_INTERVAL_S = 0.6
_PARTIAL_MIN_BYTES = int(_INPUT_SAMPLE_RATE * 2 * 0.5)  # 0.5 s of PCM16


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    """Longest common leading run of two word lists (LocalAgreement-2 stable prefix)."""
    out: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


# Cosine similarity above which two speaker embeddings are treated as the same
# person across segments (pyannote 3.1 embeddings; tune per deployment).
_SPEAKER_SIM_THRESHOLD = 0.60


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
_CHANNELS = 1
_SAMPLE_WIDTH = 2  # 16-bit


def _new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:24]}"


def _new_item_id() -> str:
    return f"item_{uuid.uuid4().hex[:24]}"


def _new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = _INPUT_SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16LE bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for incremental TTS."""
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


class RealtimeSession:
    """Manages a single Realtime API WebSocket session.

    Attributes:
        ws: The WebSocket connection.
        llm_model_id: Canonical model ID for the LLM.
        stt_model_id: Canonical model ID for STT (Whisper).
        tts_model_id: Canonical model ID for TTS.
        voice: TTS voice name.
        instructions: System instructions for the LLM.
        audio_buffer: Accumulated input audio (PCM16).
        conversation: Conversation history for LLM context.
        vad_enabled: Whether server-side VAD is active.
    """

    def __init__(
        self,
        ws: WebSocket,
        model_id: str,
        worker_pool: WorkerPool,
        model_manager: ModelManager,
        user: UserContext | None = None,
        transcription_only: bool = False,
    ) -> None:
        from ocabra.config import settings

        self.ws = ws
        self.llm_model_id = model_id
        # Transcription-only mode (OpenAI Realtime transcription session,
        # ?intent=transcription): audio -> VAD -> Whisper -> transcript events,
        # no LLM/TTS and no response generation. ``model`` is the STT model.
        self.transcription_only = transcription_only
        # Streaming partial-hypothesis state (transcription-only mode).
        self._partial_task: asyncio.Task[None] | None = None
        self._partial_item_id: str | None = None
        self._partial_prev_words: list[str] = []
        self._partial_emitted: list[str] = []
        # Cumulative committed audio seconds, to make segment timestamps absolute.
        self._audio_offset_s: float = 0.0
        # Optional STT language/prompt hints (input_audio_transcription.language).
        self._stt_language: str | None = None
        self._stt_prompt: str | None = None
        # Diarization on the final transcript (input_audio_transcription.diarize).
        self._stt_diarize: bool = False
        # Session-wide speaker registry for consistent diarization across segments:
        # each entry {id, centroid (embedding), count}. Local pyannote labels
        # (SPEAKER_00…) are matched to a global "speaker_N" by cosine similarity.
        self._speaker_registry: list[dict[str, Any]] = []
        # Apply server-configured defaults for STT/TTS models
        self.stt_model_id: str | None = (
            model_id if transcription_only else (settings.realtime_default_stt_model or None)
        )
        self.tts_model_id: str | None = (
            None if transcription_only else (settings.realtime_default_tts_model or None)
        )
        self.voice = "alloy"
        self.instructions = ""
        self.audio_buffer = bytearray()
        # Conversation entries may contain either plain text ``content`` or a
        # list of OpenAI-style content parts (e.g. ``input_audio``) when the
        # LLM supports native audio input.
        self.conversation: list[dict[str, Any]] = []
        self.vad_enabled = True
        self.modalities: list[str] = ["text", "audio"]
        self.input_audio_format = "pcm16"
        self.output_audio_format = "pcm16"

        self._worker_pool = worker_pool
        self._model_manager = model_manager
        self._user = user
        self._vad = SimpleVAD()
        self._session_id = f"sess_{uuid.uuid4().hex[:24]}"
        self._cancel_event = asyncio.Event()
        self._response_task: asyncio.Task[None] | None = None
        self._background_load_task: asyncio.Task[None] | None = None
        self._llm_ready = asyncio.Event()
        self._tts_ready = asyncio.Event()
        # Cached capability flag — populated lazily on first turn from the
        # selected LLM's ``ModelState.capabilities.audio_input``.
        # ``None`` = not resolved yet; ``True``/``False`` = decision known.
        self._native_audio_input: bool | None = None
        # Explicit client override via ``session.update.input_audio_routing``:
        # one of ``"auto"`` (default), ``"native"`` (force native), ``"stt"``
        # (force Whisper path even if the LLM advertises audio_input).
        self._input_audio_routing: str = "auto"
        # When native-audio mode is active, also run Whisper in parallel just
        # to surface a user transcript via ``conversation.item.input_audio_
        # transcription.completed`` (OpenAI Realtime standard). Off-by-default
        # would force the UI to render "<audio>" placeholders — on by default
        # is the friendlier UX. Toggle via ``session.update.transcribe_user_audio``.
        self._transcribe_user_audio: bool = True
        # Cached capability flag for symmetric output bypass — populated lazily
        # from ``audio_output``. Currently no model in the registry declares
        # this true, but the hook is in place for Qwen2.5-Omni / GPT-4o-realtime
        # style models that emit audio directly without a separate TTS step.
        self._native_audio_output: bool | None = None

    # ── Public entry point ──────────────────────────────────────────────

    async def run(self) -> None:
        """Main session loop: receive client events and dispatch them.

        Cold-start strategy depends on the routing decision for the
        selected LLM:

        Classic (Whisper -> text -> LLM): STT is on the critical path, so
        Phase 1 blocks until Whisper is loaded.

        Native-audio LLM: Whisper is no longer critical — the LLM ingests
        audio directly. STT moves to Phase 3 (background) when
        ``transcribe_user_audio`` is true (so the parallel transcript is
        still available eventually), or is skipped entirely when it's
        false. The session is usable from the first byte.
        """
        # Transcription-only: only STT matters — load it, then serve. No LLM/TTS.
        if self.transcription_only:
            self._llm_ready.set()
            self._tts_ready.set()
            if self.stt_model_id:
                await self._load_model_with_progress("stt", self.stt_model_id)
            await self._send_session_created()
            try:
                while True:
                    raw = await self.ws.receive_text()
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        await self._send_error("Invalid JSON", "invalid_json")
                        continue
                    await self._dispatch(message.get("type", ""), message)
            except Exception:
                raise
            return

        # Mark models that are already loaded as ready
        if self._worker_pool.get_worker(self.llm_model_id):
            self._llm_ready.set()
        if not self.tts_model_id or self._worker_pool.get_worker(self.tts_model_id):
            self._tts_ready.set()

        # Phase 0: resolve routing so the cold-start strategy knows whether
        # STT belongs on the critical path or in the background.
        will_use_native = await self._should_use_native_audio()
        stt_needed = bool(self.stt_model_id) and (
            (not will_use_native) or self._transcribe_user_audio
        )

        # Phase 1: Load STT synchronously only when it's on the critical
        # path (classic Whisper -> text -> LLM flow).
        if stt_needed and not will_use_native:
            await self._load_model_with_progress("stt", self.stt_model_id)

        # Phase 2: Session is usable — send created, client can start talking
        await self._send_session_created()

        # Phase 3: Load LLM + TTS (+ optional STT for native flow) in background
        stt_bg = stt_needed and will_use_native
        need_bg = (
            not self._llm_ready.is_set()
            or not self._tts_ready.is_set()
            or stt_bg
        )
        if need_bg:
            self._background_load_task = asyncio.create_task(
                self._load_remaining_models(include_stt=stt_bg),
                name="realtime-bg-load",
            )

        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error("Invalid JSON", "invalid_json")
                    continue

                event_type = message.get("type", "")
                await self._dispatch(event_type, message)
        except Exception:
            # WebSocketDisconnect or other connection errors — handled by caller
            raise

    async def _load_model_with_progress(self, role: str, model_id: str | None) -> bool:
        """Load a single model, sending progress events. Returns True if ready.

        Safe to call concurrently for the same model — model_manager.load()
        holds a per-model lock so the second caller waits for the first.
        """
        if not model_id:
            return False
        if self._worker_pool.get_worker(model_id):
            return True
        try:
            state = await self._model_manager.get_state(model_id)
            if not state:
                await self._send_error(
                    f"Model '{model_id}' ({role}) not configured", "model_not_found"
                )
                return False
            if state.status.value == "loaded":
                return True

            logger.info("realtime_cold_start_loading", model_id=model_id, role=role)
            await self._send_event(
                "session.loading",
                model=model_id,
                role=role,
                message=f"Loading {role}: {state.display_name or model_id}",
            )
            await self._model_manager.load(model_id)
            logger.info("realtime_cold_start_loaded", model_id=model_id, role=role)
            return True
        except Exception as exc:
            logger.warning(
                "realtime_auto_load_failed", model_id=model_id, role=role, error=str(exc)
            )
            await self._send_error(
                f"Failed to load {role} model '{model_id}': {exc}", "model_load_error"
            )
            return False

    async def _load_remaining_models(self, include_stt: bool = False) -> None:
        """Background task: load LLM, TTS, optionally STT after session starts."""

        async def _load_llm():
            ok = await self._load_model_with_progress("llm", self.llm_model_id)
            if ok:
                self._llm_ready.set()

        async def _load_tts():
            if self.tts_model_id:
                ok = await self._load_model_with_progress("tts", self.tts_model_id)
                if ok:
                    self._tts_ready.set()
            else:
                self._tts_ready.set()

        async def _load_stt():
            await self._load_model_with_progress("stt", self.stt_model_id)

        coros = [_load_llm(), _load_tts()]
        if include_stt and self.stt_model_id:
            coros.append(_load_stt())
        await asyncio.gather(*coros, return_exceptions=True)

    # ── Event dispatch ──────────────────────────────────────────────────

    async def _dispatch(self, event_type: str, message: dict[str, Any]) -> None:
        """Route a client event to its handler."""
        handlers: dict[str, Any] = {
            "session.update": self._handle_session_update,
            "input_audio_buffer.append": self._handle_audio_append,
            "input_audio_buffer.commit": self._handle_audio_commit,
            "input_audio_buffer.clear": self._handle_audio_clear,
            "response.create": self._handle_response_create,
            "response.cancel": self._handle_response_cancel,
        }

        handler = handlers.get(event_type)
        if handler is None:
            await self._send_error(
                f"Unknown event type: {event_type}",
                "unknown_event",
            )
            return

        try:
            await handler(message)
        except Exception as exc:
            logger.exception("realtime_event_handler_error", event_type=event_type)
            await self._send_error(str(exc), "internal_error")

    # ── Client event handlers ───────────────────────────────────────────

    async def _handle_session_update(self, message: dict[str, Any]) -> None:
        """Handle session.update — configure VAD, voice, instructions, models."""
        session_cfg = message.get("session", {})

        if "instructions" in session_cfg:
            self.instructions = session_cfg["instructions"]
        if "voice" in session_cfg:
            self.voice = session_cfg["voice"]
        if "modalities" in session_cfg:
            self.modalities = session_cfg["modalities"]
        if "input_audio_format" in session_cfg:
            self.input_audio_format = session_cfg["input_audio_format"]
        if "output_audio_format" in session_cfg:
            self.output_audio_format = session_cfg["output_audio_format"]

        # STT model / language / prompt from input_audio_transcription (OpenAI standard)
        iat = session_cfg.get("input_audio_transcription")
        if isinstance(iat, dict):
            if "model" in iat:
                self.stt_model_id = iat["model"]
            if iat.get("language"):
                self._stt_language = str(iat["language"])
            if iat.get("prompt"):
                self._stt_prompt = str(iat["prompt"])
            if "diarize" in iat:
                self._stt_diarize = bool(iat["diarize"])

        # TTS model (extension: not in official API, but useful for oCabra)
        if "tts_model" in session_cfg:
            self.tts_model_id = session_cfg["tts_model"]

        # oCabra extension: explicit input-audio routing override. Auto by
        # default — picks native when the LLM advertises ``audio_input``.
        if "input_audio_routing" in session_cfg:
            routing = str(session_cfg["input_audio_routing"]).lower()
            if routing in ("auto", "native", "stt"):
                self._input_audio_routing = routing
                # Invalidate cached decisions so the next turn re-evaluates.
                self._native_audio_input = None
                self._native_audio_output = None

        # oCabra extension: when native-audio routing is active, optionally
        # still run Whisper in parallel to populate the user's transcript on
        # the UI (default true). Set to false to save the STT compute.
        if "transcribe_user_audio" in session_cfg:
            self._transcribe_user_audio = bool(session_cfg["transcribe_user_audio"])

        # Auto-load changed models. Mirror the cold-start strategy from
        # ``run()``: STT is only on the critical path for the classic
        # Whisper -> text -> LLM flow. With a native-audio LLM, STT (when
        # needed for the parallel transcript) is loaded in the background
        # so the dispatch loop never blocks here.
        if iat:
            will_use_native = await self._should_use_native_audio()
            stt_needed = bool(self.stt_model_id) and (
                (not will_use_native) or self._transcribe_user_audio
            )
            if stt_needed and not will_use_native:
                await self._load_model_with_progress("stt", self.stt_model_id)
            elif stt_needed and will_use_native:
                asyncio.create_task(
                    self._load_model_with_progress("stt", self.stt_model_id),
                    name="realtime-session-update-stt",
                )
        if "tts_model" in session_cfg and self.tts_model_id:
            self._tts_ready.clear()
            asyncio.create_task(self._load_remaining_models())

        # VAD configuration
        turn_detection = session_cfg.get("turn_detection")
        if turn_detection is not None:
            if turn_detection is False or (
                isinstance(turn_detection, dict) and turn_detection.get("type") == "none"
            ):
                self.vad_enabled = False
                self._vad.reset()
            elif isinstance(turn_detection, dict):
                self.vad_enabled = True
                self._vad = SimpleVAD(
                    threshold=turn_detection.get("threshold"),
                    silence_duration_ms=turn_detection.get("silence_duration_ms"),
                    prefix_padding_ms=turn_detection.get("prefix_padding_ms"),
                )

        await self._send_event(
            "session.updated",
            session=self._session_snapshot(),
        )

    async def _handle_audio_append(self, message: dict[str, Any]) -> None:
        """Handle input_audio_buffer.append — accumulate PCM16 audio."""
        audio_b64 = message.get("audio", "")
        if not audio_b64:
            return

        try:
            pcm_bytes = base64.b64decode(audio_b64)
        except Exception:
            await self._send_error("Invalid base64 audio data", "invalid_audio")
            return

        self.audio_buffer.extend(pcm_bytes)

        # Run VAD if enabled
        if self.vad_enabled:
            vad_event = self._vad.process_chunk(pcm_bytes, _INPUT_SAMPLE_RATE)
            if vad_event == VadEvent.SPEECH_STARTED:
                await self._send_event(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=self._buffer_duration_ms(),
                )
                # Stream partial hypotheses for this segment as it's spoken.
                self._start_partial()
            elif vad_event == VadEvent.SPEECH_STOPPED:
                await self._send_event(
                    "input_audio_buffer.speech_stopped",
                    audio_end_ms=self._buffer_duration_ms(),
                )
                # Auto-commit on speech stop
                await self._commit_and_respond()

    async def _handle_audio_commit(self, message: dict[str, Any]) -> None:
        """Handle input_audio_buffer.commit — process the buffered audio."""
        await self._commit_and_respond()

    async def _handle_audio_clear(self, message: dict[str, Any]) -> None:
        """Handle input_audio_buffer.clear — discard buffered audio."""
        self.audio_buffer.clear()
        self._vad.reset()
        await self._send_event("input_audio_buffer.cleared")

    async def _handle_response_create(self, message: dict[str, Any]) -> None:
        """Handle response.create — generate a response (optionally from text)."""
        if self.transcription_only:
            # No LLM in a transcription session; flush any buffered audio so its
            # transcript is emitted, but don't generate a response.
            if self.audio_buffer:
                await self._commit_audio_only()
            await self._send_error(
                "response.create is not available in a transcription-only session.",
                "unsupported_in_transcription_mode",
            )
            return
        # If there is buffered audio, commit it first
        if self.audio_buffer:
            await self._commit_audio_only()

        # Allow override instructions/conversation in the response.create payload
        create_cfg = message.get("response", {})
        instructions_override = create_cfg.get("instructions")

        await self._generate_response(instructions_override=instructions_override)

    async def _handle_response_cancel(self, message: dict[str, Any]) -> None:
        """Handle response.cancel — cancel an in-progress response."""
        self._cancel_event.set()
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except (asyncio.CancelledError, Exception):
                pass
        self._response_task = None

    # ── Pipeline helpers ────────────────────────────────────────────────

    async def _should_use_native_audio(self) -> bool:
        """Decide whether the selected LLM can ingest raw audio directly.

        Caches the answer on the session — the LLM does not change mid-session
        unless explicitly re-routed via ``session.update.input_audio_routing``.
        """
        if self._input_audio_routing == "native":
            self._native_audio_input = True
            return True
        if self._input_audio_routing == "stt":
            self._native_audio_input = False
            return False
        if self._native_audio_input is not None:
            return self._native_audio_input
        try:
            state = await self._model_manager.get_state(self.llm_model_id)
        except Exception:
            state = None
        caps = getattr(state, "capabilities", None) if state else None
        self._native_audio_input = bool(getattr(caps, "audio_input", False))
        return self._native_audio_input

    async def _should_use_native_audio_output(self) -> bool:
        """Symmetric hook to ``_should_use_native_audio`` for output bypass.

        When the LLM advertises ``audio_output`` it emits PCM/Opus chunks in
        its response stream directly (Qwen2.5-Omni, GPT-4o-realtime). In that
        case the TTS step is redundant. Today no model in the registry sets
        this flag — the helper exists so the pipeline already routes around
        TTS when one shows up. Returns False if the LLM cannot be resolved.
        """
        if self._native_audio_output is not None:
            return self._native_audio_output
        try:
            state = await self._model_manager.get_state(self.llm_model_id)
        except Exception:
            state = None
        caps = getattr(state, "capabilities", None) if state else None
        self._native_audio_output = bool(getattr(caps, "audio_output", False))
        return self._native_audio_output

    async def _commit_and_respond(self) -> None:
        """Commit the audio buffer and trigger the full pipeline.

        In transcription-only mode we stop after the transcript: no LLM response.
        """
        await self._commit_audio_only()
        if not self.transcription_only:
            await self._generate_response()

    async def _commit_transcription(self) -> None:
        """Transcription-only commit: finalize the current segment with timestamps
        (and speakers if diarized), reusing the partial segment's item_id so the
        client can reconcile ``.delta`` partials with this ``.completed``."""
        await self._stop_partial()
        item_id = self._partial_item_id or _new_item_id()
        self._partial_item_id = None
        self._partial_prev_words = []
        self._partial_emitted = []

        if not self.audio_buffer:
            await self._send_event("input_audio_buffer.committed", item_id=item_id)
            return

        pcm_data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        self._vad.reset()
        await self._send_event("input_audio_buffer.committed", item_id=item_id)

        offset = self._audio_offset_s
        result = await self._transcribe_verbose(pcm_data)
        self._audio_offset_s += len(pcm_data) / 2 / _INPUT_SAMPLE_RATE

        # Consistent diarization across segments: remap this segment's local
        # pyannote speaker labels to session-global ids via the embedding registry.
        mapping = self._reconcile_speakers(result.get("speaker_embeddings") or {})
        if mapping:
            for seg in result["segments"]:
                if seg.get("speaker") in mapping:
                    seg["speaker"] = mapping[seg["speaker"]]
            for wd in result["words"]:
                if wd.get("speaker") in mapping:
                    wd["speaker"] = mapping[wd["speaker"]]
            result["speakers"] = [mapping.get(s, s) for s in result["speakers"]]

        def _abs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out = []
            for e in entries:
                e2 = dict(e)
                for k in ("start", "end"):
                    v = e2.get(k)
                    if isinstance(v, (int, float)):
                        e2[k] = round(float(v) + offset, 3)
                out.append(e2)
            return out

        transcript = result["text"]
        if transcript:
            self.conversation.append({"role": "user", "content": transcript})
        await self._send_event(
            "conversation.item.input_audio_transcription.completed",
            item_id=item_id,
            content_index=0,
            transcript=transcript,
            # oCabra extensions (standard clients ignore unknown fields):
            segments=_abs(result["segments"]),
            words=_abs(result["words"]),
            speakers=result["speakers"],
        )

    async def _commit_audio_only(self) -> None:
        """Commit the audio buffer.

        Routing depends on the selected LLM's ``audio_input`` capability:
        - native audio LLM → embed the WAV as an ``input_audio`` content part
          and let the LLM transcribe + answer in one shot (no Whisper).
        - everything else → send to Whisper as before, store the transcript
          as a plain text user message.
        """
        if self.transcription_only:
            await self._commit_transcription()
            return

        if not self.audio_buffer:
            await self._send_event("input_audio_buffer.committed", item_id=_new_item_id())
            return

        pcm_data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        self._vad.reset()

        item_id = _new_item_id()
        await self._send_event("input_audio_buffer.committed", item_id=item_id)

        if not self.transcription_only and await self._should_use_native_audio():
            wav_bytes = _pcm16_to_wav(pcm_data, _INPUT_SAMPLE_RATE)
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
            self.conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        }
                    ],
                }
            )
            await self._send_event(
                "conversation.item.created",
                item={
                    "id": item_id,
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            # Transcript is left empty here — populated by the
                            # parallel STT task below (if enabled) via the
                            # standard ``input_audio_transcription.completed``
                            # event the UI already knows how to render.
                            "transcript": "",
                        }
                    ],
                },
            )
            if self._transcribe_user_audio and self.stt_model_id:
                asyncio.create_task(
                    self._emit_parallel_user_transcript(pcm_data, item_id),
                    name="realtime-parallel-stt",
                )
            return

        # STT path: transcribe audio
        transcript = await self._transcribe(pcm_data)

        if transcript:
            self.conversation.append({"role": "user", "content": transcript})
            await self._send_event(
                "conversation.item.created",
                item={
                    "id": item_id,
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "transcript": transcript,
                        }
                    ],
                },
            )

    async def _generate_response(
        self,
        instructions_override: str | None = None,
    ) -> None:
        """Run the LLM -> TTS pipeline and stream results back."""
        # Cancel any previous response
        self._cancel_event.set()
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except (asyncio.CancelledError, Exception):
                pass

        self._cancel_event = asyncio.Event()
        self._response_task = asyncio.create_task(
            self._run_pipeline(instructions_override),
            name="realtime-pipeline",
        )

    async def _run_pipeline(self, instructions_override: str | None = None) -> None:
        """Execute the full STT -> LLM -> TTS pipeline."""
        response_id = _new_response_id()
        output_item_id = _new_item_id()
        start_time = time.monotonic()

        # Surface routing decision per-turn for ops/debugging. The decision
        # itself was made (and cached) at commit time; this just reports it.
        mode = "native_audio" if self._native_audio_input else "stt"
        logger.debug(
            "realtime_turn_route",
            mode=mode,
            llm=self.llm_model_id,
            session_id=self._session_id,
        )

        # Wait for LLM to be ready (may still be loading from cold start)
        if not self._llm_ready.is_set():
            await self._send_event(
                "session.loading",
                model=self.llm_model_id,
                role="llm",
                message="Waiting for LLM to finish loading...",
            )
            try:
                await asyncio.wait_for(self._llm_ready.wait(), timeout=300)
            except TimeoutError:
                await self._send_error(
                    "LLM model failed to load within timeout", "model_load_timeout"
                )
                return

        await self._send_event(
            "response.created",
            response={
                "id": response_id,
                "status": "in_progress",
                "output": [],
            },
        )

        try:
            # Build LLM messages
            messages = self._build_llm_messages(instructions_override)

            # Resolve symmetric output routing once per turn. When the LLM
            # emits audio natively (audio_output capability) we skip the
            # discrete TTS step — the response audio chunks come through the
            # same LLM stream. Today no model declares this; keeps the door
            # open for Qwen2.5-Omni / GPT-4o-realtime style backends.
            tts_bypassed_by_native = await self._should_use_native_audio_output()
            tts_enabled = (
                "audio" in self.modalities
                and self.tts_model_id
                and not tts_bypassed_by_native
            )

            # Stream LLM response
            full_text = ""
            sentence_buffer = ""
            audio_sent = False

            async for text_chunk in self._stream_llm(messages):
                if self._cancel_event.is_set():
                    break

                full_text += text_chunk

                # Send text transcript delta
                await self._send_event(
                    "response.audio_transcript.delta",
                    response_id=response_id,
                    item_id=output_item_id,
                    content_index=0,
                    delta=text_chunk,
                )

                # Buffer text and send TTS for complete sentences
                sentence_buffer += text_chunk
                sentences = _split_sentences(sentence_buffer)

                if len(sentences) > 1:
                    # All but the last are complete sentences
                    for sentence in sentences[:-1]:
                        if self._cancel_event.is_set():
                            break
                        if tts_enabled:
                            await self._synthesize_and_send(sentence, response_id, output_item_id)
                            audio_sent = True
                    sentence_buffer = sentences[-1]

            # Flush remaining text
            if sentence_buffer.strip() and not self._cancel_event.is_set():
                if "audio" in self.modalities and self.tts_model_id:
                    await self._synthesize_and_send(sentence_buffer, response_id, output_item_id)
                    audio_sent = True

            if not self._cancel_event.is_set():
                # Add assistant message to conversation
                if full_text:
                    self.conversation.append({"role": "assistant", "content": full_text})

                # Send completion events
                await self._send_event(
                    "response.audio_transcript.done",
                    response_id=response_id,
                    item_id=output_item_id,
                    content_index=0,
                    transcript=full_text,
                )

                if audio_sent:
                    await self._send_event(
                        "response.audio.done",
                        response_id=response_id,
                        item_id=output_item_id,
                        content_index=0,
                    )

                elapsed = time.monotonic() - start_time
                await self._send_event(
                    "response.done",
                    response={
                        "id": response_id,
                        "status": "completed",
                        "output": [
                            {
                                "id": output_item_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "audio",
                                        "transcript": full_text,
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "total_tokens": 0,  # TODO: get real usage from LLM
                            "input_tokens": 0,
                            "output_tokens": 0,
                        },
                    },
                    metadata={
                        "elapsed_s": round(elapsed, 3),
                    },
                )

        except asyncio.CancelledError:
            await self._send_event(
                "response.done",
                response={
                    "id": response_id,
                    "status": "cancelled",
                    "output": [],
                },
            )
        except Exception as exc:
            logger.exception("realtime_pipeline_error")
            await self._send_error(
                f"Pipeline error: {exc}",
                "pipeline_error",
            )
            await self._send_event(
                "response.done",
                response={
                    "id": response_id,
                    "status": "failed",
                    "output": [],
                },
            )

    async def _emit_parallel_user_transcript(
        self, pcm_data: bytes, item_id: str
    ) -> None:
        """Run Whisper in parallel and emit a transcription event.

        Only fires when native-audio mode is active and
        ``session.transcribe_user_audio`` is true. Failures are silent — the
        LLM response is the primary signal; the transcript is purely cosmetic.
        """
        try:
            transcript = await self._transcribe(pcm_data)
        except Exception as exc:
            logger.warning(
                "realtime_parallel_stt_failed",
                session_id=self._session_id,
                error=str(exc),
            )
            return
        if not transcript:
            return
        await self._send_event(
            "conversation.item.input_audio_transcription.completed",
            item_id=item_id,
            content_index=0,
            transcript=transcript,
        )

    # ── STT (Whisper) ───────────────────────────────────────────────────

    def _stt_data(self, base: dict[str, Any]) -> dict[str, Any]:
        """Add the session's language/prompt hints to a Whisper request body."""
        if self._stt_language:
            base["language"] = self._stt_language
        if self._stt_prompt:
            base["prompt"] = self._stt_prompt
        return base

    async def _transcribe(self, pcm_data: bytes) -> str:
        """Send PCM16 audio to Whisper worker for transcription."""
        if not self.stt_model_id:
            logger.warning("realtime_no_stt_model", session_id=self._session_id)
            return ""

        wav_bytes = _pcm16_to_wav(pcm_data, _INPUT_SAMPLE_RATE)

        worker = self._worker_pool.get_worker(self.stt_model_id)
        if not worker:
            try:
                await self._model_manager.load(self.stt_model_id)
                worker = self._worker_pool.get_worker(self.stt_model_id)
            except Exception:
                pass
            if not worker:
                logger.warning("realtime_stt_worker_missing", model_id=self.stt_model_id)
                return ""

        url = f"http://127.0.0.1:{worker.port}/transcribe"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    url,
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data=self._stt_data({"response_format": "json", "temperature": "0.0"}),
                )
                resp.raise_for_status()
                result = resp.json()
                return result.get("text", "").strip()
        except Exception as exc:
            logger.warning("realtime_stt_error", error=str(exc))
            return ""

    async def _transcribe_verbose(self, pcm_data: bytes) -> dict[str, Any]:
        """Transcribe with segment/word timestamps (and speakers if the profile
        diarizes). Returns ``{text, segments, words, speakers}``; empty on error."""
        empty = {"text": "", "segments": [], "words": [], "speakers": [], "speaker_embeddings": {}}
        if not self.stt_model_id:
            return empty
        worker = self._worker_pool.get_worker(self.stt_model_id)
        if not worker:
            return empty
        wav_bytes = _pcm16_to_wav(pcm_data, _INPUT_SAMPLE_RATE)
        url = f"http://127.0.0.1:{worker.port}/transcribe"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    url,
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data=self._stt_data({
                        "response_format": "verbose_json",
                        "temperature": "0.0",
                        "timestamp_granularities": ["segment", "word"],
                        **({"diarize": "true"} if self._stt_diarize else {}),
                    }),
                )
                resp.raise_for_status()
                r = resp.json()
                return {
                    "text": (r.get("text") or "").strip(),
                    "segments": r.get("segments") or [],
                    "words": r.get("words") or [],
                    "speakers": r.get("speakers") or [],
                    "speaker_embeddings": r.get("speaker_embeddings") or {},
                }
        except Exception as exc:
            logger.warning("realtime_stt_verbose_error", error=str(exc))
            return empty

    def _reconcile_speakers(self, local_embeddings: dict[str, list[float]]) -> dict[str, str]:
        """Map a segment's local pyannote labels (SPEAKER_00…) to session-global
        ``speaker_N`` ids by matching embeddings against the running registry (or
        registering a new global speaker). Keeps diarization consistent across the
        independently-diarized streaming segments."""
        mapping: dict[str, str] = {}
        for local_label, emb in local_embeddings.items():
            if not emb:
                continue
            best_id: str | None = None
            best_sim = -1.0
            for entry in self._speaker_registry:
                sim = _cosine(emb, entry["centroid"])
                if sim > best_sim:
                    best_sim, best_id = sim, entry["id"]
            if best_id is not None and best_sim >= _SPEAKER_SIM_THRESHOLD:
                entry = next(e for e in self._speaker_registry if e["id"] == best_id)
                n = entry["count"]
                entry["centroid"] = [
                    (c * n + e2) / (n + 1) for c, e2 in zip(entry["centroid"], emb)
                ]
                entry["count"] = n + 1
                mapping[local_label] = best_id
            else:
                new_id = f"speaker_{len(self._speaker_registry) + 1}"
                self._speaker_registry.append({"id": new_id, "centroid": list(emb), "count": 1})
                mapping[local_label] = new_id
        return mapping

    def _start_partial(self) -> None:
        """Begin emitting partial hypotheses for the current (uncommitted) segment."""
        if not self.transcription_only or self._partial_task is not None:
            return
        self._partial_item_id = _new_item_id()
        self._partial_prev_words = []
        self._partial_emitted = []
        self._partial_task = asyncio.create_task(
            self._partial_loop(self._partial_item_id), name="realtime-partial-stt"
        )

    async def _stop_partial(self) -> None:
        task = self._partial_task
        self._partial_task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task

    async def _partial_loop(self, item_id: str) -> None:
        """Re-transcribe the growing buffer on an interval and emit the newly
        stable word-prefix as ``input_audio_transcription.delta`` (LocalAgreement-2:
        a word is committed once two consecutive hypotheses agree on it)."""
        try:
            while True:
                await asyncio.sleep(_PARTIAL_INTERVAL_S)
                buf = bytes(self.audio_buffer)
                if len(buf) < _PARTIAL_MIN_BYTES:
                    continue
                text = await self._transcribe(buf)
                words = text.split()
                if not words:
                    continue
                stable = _common_prefix(self._partial_prev_words, words)
                self._partial_prev_words = words
                if len(stable) > len(self._partial_emitted):
                    delta_words = stable[len(self._partial_emitted):]
                    self._partial_emitted = stable
                    await self._send_event(
                        "conversation.item.input_audio_transcription.delta",
                        item_id=item_id,
                        content_index=0,
                        delta=" ".join(delta_words) + " ",
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — partials must never kill the session
            logger.warning("realtime_partial_error", error=str(exc))

    # ── LLM streaming ───────────────────────────────────────────────────

    async def _stream_llm(self, messages: list[dict[str, Any]]) -> AsyncIteratorWrapper:
        """Stream text from the LLM via the shared chat-completions pipeline.

        Routes through :meth:`WorkerPool.forward_stream` so the same code path
        as ``/v1/chat/completions`` is reused (Ollama URL resolution, reasoning
        field normalisation, Langfuse tracing). Native-audio LLMs receive the
        message verbatim — including ``input_audio`` content parts.

        Yields text content deltas.
        """
        worker = self._worker_pool.get_worker(self.llm_model_id)
        if not worker:
            # Try on-demand load as fallback
            try:
                await self._model_manager.load(self.llm_model_id)
                worker = self._worker_pool.get_worker(self.llm_model_id)
            except Exception as exc:
                logger.warning(
                    "realtime_llm_load_failed", model_id=self.llm_model_id, error=str(exc)
                )
            if not worker:
                logger.warning("realtime_llm_worker_missing", model_id=self.llm_model_id)
                return

        state = await self._model_manager.get_state(self.llm_model_id)
        if not state:
            logger.warning("realtime_llm_state_missing", model_id=self.llm_model_id)
            return

        backend_model_id = state.backend_model_id or self.llm_model_id

        body = {
            "model": backend_model_id,
            "messages": messages,
            "stream": True,
        }

        line_buf = b""
        try:
            async for chunk in self._worker_pool.forward_stream(
                self.llm_model_id, "/v1/chat/completions", body
            ):
                if self._cancel_event.is_set():
                    break
                line_buf += chunk
                while b"\n" in line_buf:
                    raw_line, line_buf = line_buf.split(b"\n", 1)
                    line = raw_line.rstrip(b"\r").decode("utf-8", errors="ignore")
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
        except httpx.HTTPStatusError as exc:
            logger.warning("realtime_llm_http_error", status=exc.response.status_code)
        except Exception as exc:
            logger.warning("realtime_llm_stream_error", error=str(exc))

    # ── TTS ─────────────────────────────────────────────────────────────

    async def _synthesize_and_send(
        self,
        text: str,
        response_id: str,
        item_id: str,
    ) -> None:
        """Synthesize text to audio and send as response.audio.delta events."""
        if not self.tts_model_id:
            return

        # Wait for TTS to be ready (may still be loading)
        if not self._tts_ready.is_set():
            try:
                await asyncio.wait_for(self._tts_ready.wait(), timeout=60)
            except TimeoutError:
                logger.warning("realtime_tts_load_timeout", model_id=self.tts_model_id)
                return  # Skip audio, text transcript was already sent

        worker = self._worker_pool.get_worker(self.tts_model_id)
        if not worker:
            try:
                await self._model_manager.load(self.tts_model_id)
                worker = self._worker_pool.get_worker(self.tts_model_id)
            except Exception:
                pass
            if not worker:
                logger.warning("realtime_tts_worker_missing", model_id=self.tts_model_id)
                return

        url = f"http://127.0.0.1:{worker.port}/synthesize"

        payload = {
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            "speed": 1.0,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                audio_bytes = resp.content

            if not audio_bytes:
                return

            # Send audio in chunks to avoid oversized WebSocket frames
            chunk_size = 16384  # ~0.5s at 16kHz 16-bit mono
            for i in range(0, len(audio_bytes), chunk_size):
                if self._cancel_event.is_set():
                    break
                chunk = audio_bytes[i : i + chunk_size]
                b64_chunk = base64.b64encode(chunk).decode("ascii")
                await self._send_event(
                    "response.audio.delta",
                    response_id=response_id,
                    item_id=item_id,
                    content_index=0,
                    delta=b64_chunk,
                )

        except Exception as exc:
            logger.warning("realtime_tts_error", error=str(exc), text=text[:80])

    # ── Message helpers ─────────────────────────────────────────────────

    def _build_llm_messages(self, instructions_override: str | None = None) -> list[dict[str, Any]]:
        """Build the messages array for the LLM, including system instructions.

        Conversation entries may carry either plain ``content`` strings (STT
        path) or a list of OpenAI-style content parts (native-audio path —
        ``input_audio``). Both forms are forwarded verbatim.
        """
        messages: list[dict[str, Any]] = []

        instructions = instructions_override or self.instructions
        if instructions:
            messages.append({"role": "system", "content": instructions})

        messages.extend(self.conversation)
        return messages

    def _buffer_duration_ms(self) -> int:
        """Calculate the duration of the audio buffer in milliseconds."""
        num_samples = len(self.audio_buffer) // _SAMPLE_WIDTH
        return int((num_samples / _INPUT_SAMPLE_RATE) * 1000)

    def _session_snapshot(self) -> dict[str, Any]:
        """Return the current session configuration as a dict."""
        return {
            "id": self._session_id,
            "model": self.llm_model_id,
            "modalities": self.modalities,
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "input_audio_transcription": (
                {"model": self.stt_model_id} if self.stt_model_id else None
            ),
            "turn_detection": (
                {
                    "type": "server_vad",
                    "threshold": self._vad.SPEECH_THRESHOLD,
                    "silence_duration_ms": self._vad.SILENCE_DURATION_MS,
                    "prefix_padding_ms": self._vad._prefix_padding_ms,
                }
                if self.vad_enabled
                else None
            ),
            "tools": [],  # Tool calls not yet implemented
            "tool_choice": "none",
            "input_audio_routing": self._input_audio_routing,
            "transcribe_user_audio": self._transcribe_user_audio,
        }

    # ── WebSocket event helpers ─────────────────────────────────────────

    async def _send_event(self, event_type: str, **kwargs: Any) -> None:
        """Send a server event to the client."""
        event: dict[str, Any] = {
            "type": event_type,
            "event_id": _new_event_id(),
        }
        event.update(kwargs)
        try:
            await self.ws.send_text(json.dumps(event))
        except Exception:
            # Connection may be closed — suppress to allow cleanup
            pass

    async def _send_error(self, message: str, code: str) -> None:
        """Send an error event to the client."""
        await self._send_event(
            "error",
            error={
                "type": "invalid_request_error",
                "code": code,
                "message": message,
            },
        )

    async def _send_session_created(self) -> None:
        """Send the initial session.created event."""
        await self._send_event(
            "session.created",
            session=self._session_snapshot(),
        )


# Type alias for the async generator used by _stream_llm.
# Python async generators don't have a named type in typing, so we use a comment.
AsyncIteratorWrapper = Any  # noqa: N816 — used only as return type hint
