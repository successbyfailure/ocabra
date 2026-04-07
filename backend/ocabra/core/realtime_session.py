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
    ) -> None:
        from ocabra.config import settings

        self.ws = ws
        self.llm_model_id = model_id
        # Apply server-configured defaults for STT/TTS models
        self.stt_model_id: str | None = settings.realtime_default_stt_model or None
        self.tts_model_id: str | None = settings.realtime_default_tts_model or None
        self.voice = "alloy"
        self.instructions = ""
        self.audio_buffer = bytearray()
        self.conversation: list[dict[str, str]] = []
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

    # ── Public entry point ──────────────────────────────────────────────

    async def run(self) -> None:
        """Main session loop: receive client events and dispatch them.

        Cold-start strategy:
        1. Load STT first (blocking) — needed immediately for audio input
        2. Send session.created — client can start talking
        3. Load LLM + TTS in background — needed when first speech turn ends
        By the time the user finishes their first sentence (~3-10s),
        LLM and TTS are likely loaded.
        """
        # Mark models that are already loaded as ready
        if self._worker_pool.get_worker(self.llm_model_id):
            self._llm_ready.set()
        if not self.tts_model_id or self._worker_pool.get_worker(self.tts_model_id):
            self._tts_ready.set()

        # Phase 1: Load STT synchronously — smallest model, needed first
        await self._load_model_with_progress("stt", self.stt_model_id)

        # Phase 2: Session is usable — send created, client can start talking
        await self._send_session_created()

        # Phase 3: Load LLM + TTS in background while user speaks
        if not self._llm_ready.is_set() or not self._tts_ready.is_set():
            self._background_load_task = asyncio.create_task(
                self._load_remaining_models(), name="realtime-bg-load"
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
                await self._send_error(f"Model '{model_id}' ({role}) not configured", "model_not_found")
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
            logger.warning("realtime_auto_load_failed", model_id=model_id, role=role, error=str(exc))
            await self._send_error(f"Failed to load {role} model '{model_id}': {exc}", "model_load_error")
            return False

    async def _load_remaining_models(self) -> None:
        """Background task: load LLM and TTS in parallel after session starts."""
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

        # Load LLM and TTS concurrently
        await asyncio.gather(_load_llm(), _load_tts(), return_exceptions=True)

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

        # STT model from input_audio_transcription (OpenAI standard)
        iat = session_cfg.get("input_audio_transcription")
        if isinstance(iat, dict) and "model" in iat:
            self.stt_model_id = iat["model"]

        # TTS model (extension: not in official API, but useful for oCabra)
        if "tts_model" in session_cfg:
            self.tts_model_id = session_cfg["tts_model"]

        # Auto-load changed models (STT sync, LLM+TTS in background)
        if iat or "tts_model" in session_cfg:
            await self._load_model_with_progress("stt", self.stt_model_id)
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

    async def _commit_and_respond(self) -> None:
        """Commit the audio buffer and trigger the full pipeline."""
        await self._commit_audio_only()
        await self._generate_response()

    async def _commit_audio_only(self) -> None:
        """Commit the audio buffer: transcribe and add to conversation."""
        if not self.audio_buffer:
            await self._send_event("input_audio_buffer.committed", item_id=_new_item_id())
            return

        pcm_data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        self._vad.reset()

        item_id = _new_item_id()
        await self._send_event("input_audio_buffer.committed", item_id=item_id)

        # STT: transcribe audio
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
                await self._send_error("LLM model failed to load within timeout", "model_load_timeout")
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
                        if "audio" in self.modalities and self.tts_model_id:
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

    # ── STT (Whisper) ───────────────────────────────────────────────────

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
                    data={"response_format": "json", "temperature": "0.0"},
                )
                resp.raise_for_status()
                result = resp.json()
                return result.get("text", "").strip()
        except Exception as exc:
            logger.warning("realtime_stt_error", error=str(exc))
            return ""

    # ── LLM streaming ───────────────────────────────────────────────────

    async def _stream_llm(self, messages: list[dict[str, str]]) -> AsyncIteratorWrapper:
        """Stream text from the LLM worker via SSE.

        Yields text content deltas.
        """
        worker = self._worker_pool.get_worker(self.llm_model_id)
        if not worker:
            # Try on-demand load as fallback
            try:
                await self._model_manager.load(self.llm_model_id)
                worker = self._worker_pool.get_worker(self.llm_model_id)
            except Exception as exc:
                logger.warning("realtime_llm_load_failed", model_id=self.llm_model_id, error=str(exc))
            if not worker:
                logger.warning("realtime_llm_worker_missing", model_id=self.llm_model_id)
                return

        state = await self._model_manager.get_state(self.llm_model_id)
        if not state:
            logger.warning("realtime_llm_state_missing", model_id=self.llm_model_id)
            return

        backend_model_id = state.backend_model_id or self.llm_model_id

        if worker.backend_type == "ollama":
            from ocabra.config import settings

            base = settings.ollama_base_url.rstrip("/")
            url = f"{base}/v1/chat/completions"
        else:
            url = f"http://127.0.0.1:{worker.port}/v1/chat/completions"

        body = {
            "model": backend_model_id,
            "messages": messages,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", url, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if self._cancel_event.is_set():
                            break
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
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

    def _build_llm_messages(self, instructions_override: str | None = None) -> list[dict[str, str]]:
        """Build the messages array for the LLM, including system instructions."""
        messages: list[dict[str, str]] = []

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
