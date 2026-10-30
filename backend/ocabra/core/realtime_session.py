"""
RealtimeSession — Core session manager for the OpenAI Realtime API WebSocket.

Coordinates the pipeline: audio buffer -> Whisper STT -> LLM streaming -> TTS -> audio out.
Emits protocol-compliant server events over the WebSocket connection.

Tool/function calling is supported (2026-10-30): ``session.update``/
``response.create`` accept ``tools``/``tool_choice``, streamed tool-call
deltas are surfaced as ``response.function_call_arguments.delta``/``.done``,
and a client closes the loop with ``conversation.item.create``
(``type: "function_call_output"``) + a follow-up ``response.create``.
``conversation.item.create``/``.truncate``/``.delete`` are also handled —
the former is also the only way to inject a text-only user message (no
audio) into a session.
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
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from starlette.websockets import WebSocketState

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
# Max consecutive partial-loop ticks where the audio buffer did not grow. When
# the WebSocket client dies without a clean close (half-closed TCP, no FIN),
# the main ``receive_text()`` blocks forever and the partial task keeps
# re-transcribing the exact same bytes once per interval, pegging the Whisper
# worker. 100 ticks × 0.6 s ≈ 60 s of silence are enough to call it dead and
# bail. Observed in production (2026-10-14): one session kept the whisper
# worker at 259 % CPU for ~2 h after the client died.
_PARTIAL_STAGNANT_MAX = 100
# Max seconds the main ``run()`` loop will block on ``receive_text()`` without
# any client message before probing/giving up. 300 s is comfortable for a
# human conversation; a healthy client sends audio chunks every ~20 ms.
_RECEIVE_IDLE_TIMEOUT_S = 300.0


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    """Longest common leading run of two word lists (LocalAgreement-2 stable prefix)."""
    out: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


# Cosine similarity above which two speaker embeddings are treated as the same
# person across segments (wespeaker/pyannote embeddings: same-speaker ~0.5-0.9,
# different ~0.0-0.3). Lower than the first cut to reduce spurious new speakers.
_SPEAKER_SIM_THRESHOLD = 0.50
# Exemplar embeddings kept per speaker; matching uses the best of the exemplars
# (more robust to intra-speaker variation than a drifting running-mean centroid).
_MAX_EXEMPLARS = 12
# Cross-session speaker registries (keyed by conversation_id) so a reconnecting
# Realtime session resumes the same speaker namespace. Pruned by TTL.
_REGISTRY_STORE: dict[str, dict[str, Any]] = {}
_REGISTRY_TTL_S = 2 * 3600.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# Whisper hallucinates "YouTube subtitle" fillers over silence/low noise; the
# streaming path (short VAD-committed clips) is especially prone. Drop a segment
# when the model itself flags it as likely non-speech, or when its whole text is a
# known filler on a short/low-confidence segment.
_NO_SPEECH_DROP = 0.6
_HALLUCINATION_PHRASES = {
    "gracias", "muchas gracias", "gracias por ver el video", "gracias por ver",
    "adios", "hasta luego", "hasta la proxima", "nos vemos", "chau",
    "subtitulos realizados por la comunidad de amara org", "subtitulos por la comunidad de amara org",
    "subtitulado por la comunidad de amara org", "amara org",
    "thanks for watching", "thank you", "thank you for watching", "bye", "goodbye",
    "you", "the", "si", "no", "a", "fue", "eh", "ah", "mmm",
}


def _normalize_phrase(text: str) -> str:
    return re.sub(r"[^\w\sáéíóúñü]", "", text.lower(), flags=re.UNICODE).strip()


def _is_hallucinated_segment(seg: dict[str, Any]) -> bool:
    text = str(seg.get("text") or "").strip()
    norm = _normalize_phrase(text)
    if not norm:
        return True
    nsp = float(seg.get("no_speech_prob") or 0.0)
    dur = float(seg.get("end") or 0.0) - float(seg.get("start") or 0.0)
    # The model itself thinks this is (mostly) non-speech.
    if nsp >= _NO_SPEECH_DROP:
        return True
    # A known filler as the *entire* segment, on a short or low-confidence clip.
    if norm in _HALLUCINATION_PHRASES and (dur < 2.0 or nsp >= 0.25):
        return True
    # A single very short token on a short clip with some non-speech signal.
    if len(norm) <= 3 and dur < 1.0 and nsp >= 0.2:
        return True
    return False


def _prune_registry_store(now: float) -> None:
    stale = [k for k, v in _REGISTRY_STORE.items() if now - v.get("ts", 0.0) > _REGISTRY_TTL_S]
    for k in stale:
        _REGISTRY_STORE.pop(k, None)
_CHANNELS = 1
_SAMPLE_WIDTH = 2  # 16-bit


def _new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:24]}"


def _new_item_id() -> str:
    return f"item_{uuid.uuid4().hex[:24]}"


def _new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def _realtime_tools_to_chat_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Realtime-API-shaped tool defs to the chat/completions shape.

    The two OpenAI API surfaces disagree on tool shape: Realtime's
    ``session.update.tools`` entries are FLAT —
    ``{"type": "function", "name", "description", "parameters"}`` — while
    ``/v1/chat/completions`` (what the LLM backend actually receives, via
    ``to_backend_body``) expects them NESTED —
    ``{"type": "function", "function": {"name", "description", "parameters"}}``.
    Forwarding the flat shape straight through silently breaks tool-calling
    (confirmed live 2026-10-30: curling the backend directly with the nested
    shape worked — ``finish_reason: "tool_calls"`` with a correct call — the
    exact same request with the flat Realtime shape instead got back empty
    content, no tool call, because the backend's tools param didn't parse).
    Tolerant of a client sending the nested shape anyway (passes through).
    """
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if isinstance(tool.get("function"), dict):
            converted.append(tool)  # already chat-completions shaped
            continue
        converted.append({
            "type": tool.get("type", "function"),
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        })
    return converted


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


def _message_word_count(messages: list[dict[str, Any]]) -> int:
    """Approximate text input tokens without inspecting embedded audio payloads."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(part["text"].split())
    return total


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
        request_recorder: Callable[..., Awaitable[None]] | None = None,
        llm_worker_key: str | None = None,
        llm_load_overrides: dict | None = None,
        llm_preferred_gpu: int | None = None,
        llm_load_policy: str | None = None,
        profile_registry: Any | None = None,
        router_resolver: Any | None = None,
    ) -> None:
        from ocabra.config import settings

        self.ws = ws
        self.llm_model_id = model_id
        # The actual worker instance to load/call for the LLM role. Profiles
        # with multiple runtime presets (most llama_cpp/vllm/sglang chat
        # models) are served under a preset-derived key like
        # "llama_cpp/gemma4-26b::<hash>" — REST endpoints resolve this via
        # plan_profile_runtime() before loading. Without it, every realtime
        # session loaded the bare base_model_id as its own separate,
        # untracked worker instance, orphaned from whatever preset instance
        # REST traffic was already using — on a single GPU that doesn't fit
        # both, so each new session evicted the other and cold-started again.
        # Reproduced live 2026-10-29: 3 consecutive realtime sessions for the
        # same model each spawned a brand new llama-server process. Defaults
        # to the bare model_id for profiles without presets (unchanged
        # behaviour) or if runtime-plan resolution fails for any reason.
        self.llm_worker_key = llm_worker_key or model_id
        self._llm_load_overrides = llm_load_overrides
        self._llm_preferred_gpu = llm_preferred_gpu
        self._llm_load_policy = llm_load_policy
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
        # Drop Whisper hallucinations over silence in streaming (default on).
        self._suppress_hallucinations: bool = True
        # Session-wide speaker registry for consistent diarization across segments:
        # each entry {id, centroid, count, exemplars}. Local pyannote labels
        # (SPEAKER_00…) are matched to a global "speaker_N" by cosine similarity.
        self._speaker_registry: list[dict[str, Any]] = []
        # Cap on distinct speakers (session.update input_audio_transcription.
        # num_speakers). At the cap we never mint a new speaker — the strongest
        # guard against id drift in long sessions.
        self._max_speakers: int = 0
        # Optional id to persist/resume the speaker namespace across reconnections.
        self._conversation_id: str | None = None
        # Last assigned global speaker, used for short/embedding-less segments.
        self._last_speaker: str | None = None
        # Apply server-configured defaults for STT/TTS models. These may name a
        # router profile (e.g. "recomendado-stt") rather than a loadable model —
        # resolved once in ``_resolve_stt_tts_defaults()`` at the top of ``run()``
        # (not here: that resolution is async and ``__init__`` can't await it).
        self._profile_registry = profile_registry
        self._router_resolver = router_resolver
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

        # Generation params (2026-10-30) — session.update/response.create can
        # set these; previously the LLM call was a bare {model, messages,
        # stream} body that silently dropped every one of them. ``tools``
        # follows the standard OpenAI Realtime shape
        # ([{"type": "function", "name", "description", "parameters"}]);
        # empty list = no tool use, matching "tools" being omitted entirely
        # in a plain chat/completions request.
        self.tools: list[dict[str, Any]] = []
        self.tool_choice: str | dict[str, Any] = "auto"
        self.temperature: float | None = None
        self.max_response_output_tokens: int | None = None
        # oCabra extensions (not in the OpenAI Realtime spec, same pattern as
        # ``tts_model``/``input_audio_routing`` already are) — reasoning_budget
        # defaults ON (not None) specifically because an uncapped thinking
        # model is the single biggest latency lever in voice mode: the
        # documented 2026-10-29 truncation incident was caused by exactly
        # this (llama-server's reasoning has no limit, -1, unless told
        # otherwise) and a 2026-10-29 test on this same stack showed capping
        # it at 200 tokens cut a qwen3.8:27b turn from "several minutes" to
        # 29s. A client can still opt out by sending reasoning_budget: null
        # explicitly in session.update.
        self.reasoning_effort: str | None = None
        self.reasoning_budget: int | None = settings.realtime_default_reasoning_budget or None

        self._worker_pool = worker_pool
        self._model_manager = model_manager
        self._user = user
        self._request_recorder = request_recorder
        self._vad = SimpleVAD()
        self._session_id = f"sess_{uuid.uuid4().hex[:24]}"
        # Bloque 20 — SessionRegistry. Injected lazily so this module doesn't
        # add a hard dependency on the registry for callers that don't need
        # it (tests, in-process embedding); ``ocabra.api.openai.realtime``
        # sets it right after construction.
        self._session_registry: Any | None = None
        self._cancel_event = asyncio.Event()
        self._response_task: asyncio.Task[None] | None = None
        self._background_load_task: asyncio.Task[None] | None = None
        self._llm_ready = asyncio.Event()
        self._tts_ready = asyncio.Event()
        # Set once the STT model is loaded (transcription-only mode loads it in the
        # background so live audio can be buffered from the first byte).
        self._stt_ready = asyncio.Event()
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

    def set_session_registry(self, registry: Any) -> None:
        """Late-bind the SessionRegistry so evict/router know this session's
        workers are held. Called by the WebSocket entrypoint before ``run``.
        """
        self._session_registry = registry

    async def _register_workers(self) -> None:
        """Notify the SessionRegistry which workers this session pins.

        Best-effort — a failure here must never break the WebSocket. The
        workers list depends on the routing decisions made during ``run``
        so we call this once STT/LLM/TTS ids are known.
        """
        if self._session_registry is None:
            return
        workers: list[str] = []
        for worker_id in (self.stt_model_id, self.llm_worker_key, self.tts_model_id):
            if worker_id and worker_id not in workers:
                workers.append(worker_id)
        if not workers:
            return
        try:
            user_id = None
            api_key_name = None
            if self._user is not None and not self._user.is_anonymous:
                user_id = self._user.user_id
                api_key_name = getattr(self._user, "api_key_name", None)
            await self._session_registry.register(
                self._session_id,
                workers_held=workers,
                user_id=user_id,
                api_key_name=api_key_name,
            )
        except Exception as exc:  # noqa: BLE001 — must not break Realtime
            logger.warning(
                "session_registry_register_failed",
                session_id=self._session_id,
                error=str(exc),
            )

    async def _heartbeat_session(self) -> None:
        """Refresh the session's last_activity in the registry. Called from
        the dispatch loop on every meaningful event."""
        if self._session_registry is None:
            return
        try:
            await self._session_registry.heartbeat(self._session_id)
        except Exception:  # noqa: BLE001
            pass

    async def _unregister_workers(self, *, reason: str = "closed") -> None:
        if self._session_registry is None:
            return
        try:
            await self._session_registry.unregister(self._session_id, reason=reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "session_registry_unregister_failed",
                session_id=self._session_id,
                error=str(exc),
            )

    async def _record_request(
        self,
        *,
        model_id: str,
        request_kind: str,
        started_at: datetime,
        started_monotonic: float,
        status_code: int,
        error_message: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Record a Realtime backend operation without disrupting the session."""
        if self._request_recorder is None:
            return
        try:
            await self._request_recorder(
                session_id=self._session_id,
                model_id=model_id,
                started_at=started_at,
                duration_ms=(time.monotonic() - started_monotonic) * 1000,
                request_kind=request_kind,
                status_code=status_code,
                error_message=error_message,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - stats must not break Realtime
            logger.warning(
                "realtime_stats_record_failed",
                session_id=self._session_id,
                model_id=model_id,
                request_kind=request_kind,
                error=str(exc),
            )

    async def _resolve_stt_tts_defaults(self) -> None:
        """Resolve the configured STT/TTS defaults to concrete, loadable model ids.

        ``settings.realtime_default_stt_model``/``realtime_default_tts_model`` may
        name a router profile (no ``base_model_id`` of its own) instead of a plain
        model id — mirrors how the LLM target is resolved once at connect via
        ``resolve_realtime_model_id()`` in ``ocabra.api.openai.realtime``: a known
        profile resolves to its base model, a router profile is resolved through
        the ``RouterResolver``, and anything else (a literal model/backend id, or
        an unknown string) passes through unchanged. Runs once here, not per-turn
        — same scope as the LLM path, which doesn't re-pick mid-session either.

        No-op when no ``profile_registry`` was injected (e.g. direct construction
        in tests) or when a default isn't configured, preserving prior behaviour.
        """
        if self._profile_registry is None:
            return
        from ocabra.api.openai.realtime import resolve_realtime_model_id

        if self.stt_model_id:
            resolved = await resolve_realtime_model_id(
                self.stt_model_id,
                profile_registry=self._profile_registry,
                model_manager=self._model_manager,
                router_resolver=self._router_resolver,
                exclude_session_id=self._session_id,
            )
            if resolved is not None:
                self.stt_model_id = resolved
        if self.tts_model_id:
            resolved = await resolve_realtime_model_id(
                self.tts_model_id,
                profile_registry=self._profile_registry,
                model_manager=self._model_manager,
                router_resolver=self._router_resolver,
                exclude_session_id=self._session_id,
            )
            if resolved is not None:
                self.tts_model_id = resolved

    async def _resolve_model_id_or_keep(self, raw: str) -> str:
        """Resolve a client-supplied model id the same way the connect-time
        defaults are resolved (``_resolve_stt_tts_defaults``) — a router
        profile id passes through ``RouterResolver``, a plain profile id
        resolves to its base model, anything else (or no profile_registry
        injected) passes through unchanged.

        Needed because ``session.update`` (``_handle_session_update``)
        assigns ``stt_model_id``/``tts_model_id`` directly from whatever the
        client sends — and the Playground resends the configured default on
        every ``ws.onopen``, which, before this fix, clobbered the already
        -resolved router target back to the raw unresolvable router profile
        id. Found live 2026-10-30: selecting a router as the Realtime
        STT/TTS default appeared to work (it resolved once in ``run()``) but
        every session actually tried to load a worker keyed by the router's
        profile id, which has no ``base_model_id``/backend and can't load.

        ``exclude_session_id`` passed down to the router pick matters MORE
        here than in ``_resolve_stt_tts_defaults`` — by the time
        ``session.update`` arrives, ``_register_workers()`` has already run,
        so this session now holds its own STT/TTS worker reservation. Without
        excluding it, the router's "prefer already loaded" scan vetoes the
        very model this session just warmed (sees it as "held by an active
        session" — itself) and falls through to picking a DIFFERENT,
        genuinely cold target instead. Reproduced live 2026-10-30: a router
        STT selection flashed a ~1s "loading" card on every session open —
        this self-veto, not a real reload — and, as a bonus, pointed
        ``self.stt_model_id`` at a different model than what Phase 1 already
        loaded, wasting that warm worker and keying a second one.
        """
        if self._profile_registry is None or not raw:
            return raw
        from ocabra.api.openai.realtime import resolve_realtime_model_id

        resolved = await resolve_realtime_model_id(
            raw,
            profile_registry=self._profile_registry,
            model_manager=self._model_manager,
            router_resolver=self._router_resolver,
            exclude_session_id=self._session_id,
        )
        return resolved if resolved is not None else raw

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
            # Serve the session immediately so a LIVE source can start streaming
            # from the first byte; load STT in the background and buffer incoming
            # audio meanwhile. Commits/partials wait for ``_stt_ready`` so nothing
            # captured during the cold start is lost.
            await self._send_session_created()
            # Bloque 20 — register the worker (STT-only in this mode) so
            # the scheduler doesn't evict it mid-transcription.
            await self._register_workers()
            if self.stt_model_id:
                self._background_load_task = asyncio.create_task(
                    self._load_stt_then_ready(), name="realtime-transcription-stt-load"
                )
            else:
                self._stt_ready.set()
            try:
                while True:
                    # Idle timeout — a half-closed TCP (client crashed, network
                    # dropped) leaves ``receive_text()`` blocked forever and
                    # background tasks pegging the whisper worker on a stale
                    # buffer. Treat ``asyncio.TimeoutError`` here as a
                    # disconnect so the outer finally clean-up runs.
                    try:
                        raw = await asyncio.wait_for(
                            self.ws.receive_text(),
                            timeout=_RECEIVE_IDLE_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError as exc:
                        logger.info(
                            "realtime_session_idle_timeout",
                            session_id=self._session_id,
                            timeout_s=_RECEIVE_IDLE_TIMEOUT_S,
                        )
                        raise RuntimeError("WebSocket idle timeout") from exc
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        await self._send_error("Invalid JSON", "invalid_json")
                        continue
                    await self._heartbeat_session()
                    await self._dispatch(message.get("type", ""), message)
            except Exception:
                raise
            finally:
                await self._unregister_workers()
            return

        # Resolve the configured STT/TTS defaults (profile id, router profile
        # id, or literal model id) before anything below reads them.
        await self._resolve_stt_tts_defaults()

        # Mark models that are already loaded as ready
        if self._worker_pool.get_worker(self.llm_worker_key):
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

        # Bloque 20 — with the routing decisions frozen (Phase 0) we know
        # which workers this session actually holds. Register them BEFORE
        # yielding to the client loop so the scheduler can't evict them
        # between now and the first heartbeat.
        await self._register_workers()

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
                # Idle timeout — see transcription-only path above.
                try:
                    raw = await asyncio.wait_for(
                        self.ws.receive_text(),
                        timeout=_RECEIVE_IDLE_TIMEOUT_S,
                    )
                except asyncio.TimeoutError as exc:
                    logger.info(
                        "realtime_session_idle_timeout",
                        session_id=self._session_id,
                        timeout_s=_RECEIVE_IDLE_TIMEOUT_S,
                    )
                    raise RuntimeError("WebSocket idle timeout") from exc
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error("Invalid JSON", "invalid_json")
                    continue

                event_type = message.get("type", "")
                await self._heartbeat_session()
                await self._dispatch(event_type, message)
        except Exception:
            # WebSocketDisconnect or other connection errors — handled by caller
            raise
        finally:
            await self._unregister_workers()

    async def _load_model_with_progress(
        self,
        role: str,
        model_id: str | None,
        *,
        worker_key: str | None = None,
        load_overrides: dict | None = None,
        preferred_gpu: int | None = None,
        load_policy: str | None = None,
    ) -> bool:
        """Load a single model, sending progress events. Returns True if ready.

        Safe to call concurrently for the same model — model_manager.load()
        holds a per-model lock so the second caller waits for the first.

        ``worker_key`` lets a role resolve to a preset-derived worker instead
        of the bare ``model_id`` (see ``ensure_worker_loaded`` and the
        ``llm_worker_key`` docstring on ``__init__``) — today only the LLM
        role uses this; STT/TTS backends have no runtime-preset system to go
        through, so they default to ``worker_key == model_id``.
        """
        if not model_id:
            return False
        key = worker_key or model_id
        if self._worker_pool.get_worker(key):
            return True
        try:
            state = await self._model_manager.get_state(key)
            if not state and key == model_id:
                await self._send_error(
                    f"Model '{model_id}' ({role}) not configured", "model_not_found"
                )
                return False
            if state and state.status.value == "loaded":
                return True

            logger.info("realtime_cold_start_loading", model_id=key, role=role)
            await self._send_event(
                "session.loading",
                model=model_id,
                role=role,
                message=f"Loading {role}: {(state.display_name if state else None) or model_id}",
            )
            if key != model_id:
                from ocabra.api.openai._deps import ensure_worker_loaded

                await ensure_worker_loaded(
                    self._model_manager,
                    model_id,
                    key,
                    load_overrides,
                    preferred_gpu=preferred_gpu,
                    load_policy=load_policy,
                )
            else:
                await self._model_manager.load(key)
            logger.info("realtime_cold_start_loaded", model_id=key, role=role)
            return True
        except Exception as exc:
            logger.warning(
                "realtime_auto_load_failed", model_id=key, role=role, error=str(exc)
            )
            await self._send_error(
                f"Failed to load {role} model '{model_id}': {exc}", "model_load_error"
            )
            return False

    async def _load_remaining_models(self, include_stt: bool = False) -> None:
        """Background task: load LLM, TTS, optionally STT after session starts."""

        async def _load_llm():
            ok = await self._load_model_with_progress(
                "llm",
                self.llm_model_id,
                worker_key=self.llm_worker_key,
                load_overrides=self._llm_load_overrides,
                preferred_gpu=self._llm_preferred_gpu,
                load_policy=self._llm_load_policy,
            )
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
            "conversation.item.create": self._handle_conversation_item_create,
            "conversation.item.truncate": self._handle_conversation_item_truncate,
            "conversation.item.delete": self._handle_conversation_item_delete,
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

        # Generation params (2026-10-30). ``tools``/``tool_choice`` are
        # standard OpenAI Realtime fields; ``reasoning_effort``/
        # ``reasoning_budget`` are oCabra extensions living in the same
        # message (unknown fields are ignored by strict OpenAI clients, same
        # pattern as ``tts_model``/``input_audio_routing`` below). Presence
        # -checked (``in session_cfg``), not truthiness-checked, so a client
        # can explicitly send e.g. ``reasoning_budget: null`` to opt out of
        # the server default set in ``__init__``.
        if "tools" in session_cfg:
            self.tools = session_cfg["tools"] if isinstance(session_cfg["tools"], list) else []
        if "tool_choice" in session_cfg:
            self.tool_choice = session_cfg["tool_choice"]
        if "temperature" in session_cfg:
            self.temperature = session_cfg["temperature"]
        if "max_response_output_tokens" in session_cfg:
            value = session_cfg["max_response_output_tokens"]
            self.max_response_output_tokens = None if value == "inf" else value
        if "reasoning_effort" in session_cfg:
            self.reasoning_effort = session_cfg["reasoning_effort"]
        if "reasoning_budget" in session_cfg:
            self.reasoning_budget = session_cfg["reasoning_budget"]

        # STT model / language / prompt from input_audio_transcription (OpenAI standard)
        iat = session_cfg.get("input_audio_transcription")
        if isinstance(iat, dict):
            if "model" in iat:
                self.stt_model_id = await self._resolve_model_id_or_keep(iat["model"])
            if iat.get("language"):
                self._stt_language = str(iat["language"])
            if iat.get("prompt"):
                self._stt_prompt = str(iat["prompt"])
            if "diarize" in iat:
                self._stt_diarize = bool(iat["diarize"])
            if "suppress_hallucinations" in iat:
                self._suppress_hallucinations = bool(iat["suppress_hallucinations"])
            _nspk = iat.get("num_speakers", iat.get("max_speakers"))
            if _nspk is not None:
                try:
                    self._max_speakers = max(0, int(_nspk))
                except (TypeError, ValueError):
                    pass
            if iat.get("conversation_id"):
                self._conversation_id = str(iat["conversation_id"])
                self._load_registry()

        # TTS model (extension: not in official API, but useful for oCabra)
        if "tts_model" in session_cfg:
            self.tts_model_id = await self._resolve_model_id_or_keep(session_cfg["tts_model"])

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
                # Barge-in (2026-10-30): real speech detected while the
                # assistant is still mid-response — interrupt it immediately
                # rather than letting it talk over the user, matching
                # OpenAI's default server_vad behavior. Previously this only
                # emitted the event; the in-progress response kept running
                # to completion regardless of the user already talking.
                if self._response_task and not self._response_task.done():
                    await self._cancel_current_response()
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

        # Allow overriding instructions and generation params for just this
        # one response (2026-10-30: previously only ``instructions`` was
        # read here — ``tools``/``tool_choice``/``temperature``/
        # ``max_response_output_tokens`` sent in the real spec's per-turn
        # ``response`` object were silently dropped).
        create_cfg = message.get("response", {})
        instructions_override = create_cfg.get("instructions")
        response_overrides = {
            key: create_cfg[key]
            for key in (
                "tools",
                "tool_choice",
                "temperature",
                "max_response_output_tokens",
                "reasoning_effort",
                "reasoning_budget",
            )
            if key in create_cfg
        }

        await self._generate_response(
            instructions_override=instructions_override,
            response_overrides=response_overrides or None,
        )

    async def _handle_response_cancel(self, message: dict[str, Any]) -> None:
        """Handle response.cancel — cancel an in-progress response."""
        await self._cancel_current_response()
        self._response_task = None

    async def _handle_conversation_item_create(self, message: dict[str, Any]) -> None:
        """Handle conversation.item.create — inject an item into the history.

        Added 2026-10-30; previously unhandled entirely (fell through to
        "Unknown event type"). Two cases matter here:

        - A text-only user message (``type: "message"``) — the ONLY way a
          client can put something in the conversation without it being
          transcribed audio; without this handler a text-capable Realtime
          client had no way to say anything at all.
        - A tool result (``type: "function_call_output"``) — how the client
          feeds a function's return value back in after executing a tool
          call from a previous ``response.done``. Required for the
          tools/function-calling round trip to actually close the loop.
        """
        item = message.get("item", {})
        item_type = item.get("type")
        item_id = item.get("id") or _new_item_id()

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output", "")
            self.conversation.append(
                {"id": item_id, "role": "tool", "tool_call_id": call_id, "content": str(output)}
            )
            await self._send_event(
                "conversation.item.created",
                item={
                    "id": item_id,
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            )
        elif item_type == "message":
            role = item.get("role", "user")
            content_parts = item.get("content", [])
            text = "".join(
                str(part.get("text", ""))
                for part in content_parts
                if isinstance(part, dict) and part.get("type") in ("input_text", "text")
            )
            self.conversation.append({"id": item_id, "role": role, "content": text})
            await self._send_event(
                "conversation.item.created",
                item={"id": item_id, "type": "message", "role": role, "content": content_parts},
            )
        else:
            await self._send_error(
                f"Unsupported conversation.item.create type: {item_type!r}",
                "unsupported_item_type",
            )

    async def _handle_conversation_item_truncate(self, message: dict[str, Any]) -> None:
        """Handle conversation.item.truncate — acknowledge only.

        This session doesn't keep a separately replayable audio buffer per
        conversation item (TTS audio is streamed straight out, not stored),
        so there's nothing to actually truncate at the audio level. Still
        ack with the matching event — a client doing standard barge-in
        bookkeeping sends this right after interrupting and would otherwise
        get "Unknown event type" for routine behavior.
        """
        await self._send_event(
            "conversation.item.truncated",
            item_id=message.get("item_id"),
            content_index=message.get("content_index", 0),
            audio_end_ms=message.get("audio_end_ms", 0),
        )

    async def _handle_conversation_item_delete(self, message: dict[str, Any]) -> None:
        """Handle conversation.item.delete — remove an item from history."""
        item_id = message.get("item_id")
        self.conversation = [entry for entry in self.conversation if entry.get("id") != item_id]
        await self._send_event("conversation.item.deleted", item_id=item_id)

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
            state = await self._model_manager.get_state(self.llm_worker_key)
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
            state = await self._model_manager.get_state(self.llm_worker_key)
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

        # If the model is still cold-loading, wait — the audio is already captured
        # in ``pcm_data`` so a commit during the cold start is transcribed once
        # ready rather than lost.
        await self._stt_ready.wait()

        offset = self._audio_offset_s
        result = await self._transcribe_verbose(pcm_data)
        self._audio_offset_s += len(pcm_data) / 2 / _INPUT_SAMPLE_RATE

        # Drop Whisper hallucinations over silence before anything else.
        if self._suppress_hallucinations:
            self._filter_hallucinations(result)

        # Consistent diarization across segments: remap this segment's local
        # pyannote labels to session-global ids via the embedding registry. Labels
        # without an embedding this segment (very short turns) inherit the last
        # global speaker for continuity instead of leaking a raw ``SPEAKER_XX``.
        embeddings = result.get("speaker_embeddings") or {}
        if embeddings or result["speakers"]:
            mapping = self._reconcile_speakers(embeddings)

            def _global(local: Any) -> str | None:
                if local is None:
                    return self._last_speaker
                if local in mapping:
                    return mapping[local]
                return self._last_speaker  # unmapped local label → continuity

            for seg in result["segments"]:
                if seg.get("speaker") is not None:
                    seg["speaker"] = _global(seg["speaker"])
            for wd in result["words"]:
                if wd.get("speaker") is not None:
                    wd["speaker"] = _global(wd["speaker"])
            result["speakers"] = sorted(
                {g for s in result["speakers"] if (g := _global(s))}
            )
            self._save_registry()

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
            self.conversation.append({"id": item_id, "role": "user", "content": transcript})
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
                    "id": item_id,
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
            self.conversation.append({"id": item_id, "role": "user", "content": transcript})
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

    async def _cancel_current_response(self) -> None:
        """Cancel the in-progress response task, if any. Shared by the
        explicit ``response.cancel`` handler and barge-in (speech detected
        while the assistant is still talking)."""
        self._cancel_event.set()
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _generate_response(
        self,
        instructions_override: str | None = None,
        response_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Run the LLM -> TTS pipeline and stream results back."""
        await self._cancel_current_response()

        self._cancel_event = asyncio.Event()
        self._response_task = asyncio.create_task(
            self._run_pipeline(instructions_override, response_overrides),
            name="realtime-pipeline",
        )

    async def _run_pipeline(
        self,
        instructions_override: str | None = None,
        response_overrides: dict[str, Any] | None = None,
    ) -> None:
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
            # index -> accumulated tool call fragments (id/name arrive once,
            # on the first fragment for that index; arguments stream in
            # pieces, same shape OpenAI's own chat/completions streaming
            # uses for tool_calls — see _stream_llm's docstring).
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            tool_call_item_ids: dict[int, str] = {}

            async for event in self._stream_llm(messages, response_overrides=response_overrides):
                if self._cancel_event.is_set():
                    break

                if event["type"] == "content":
                    text_chunk = event["text"]
                    full_text += text_chunk

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

                elif event["type"] == "tool_call_delta":
                    idx = event["index"]
                    acc = tool_calls_acc.get(idx)
                    if acc is None:
                        # First fragment for this index — id/name are only
                        # sent once, here.
                        call_id = event.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                        name = event.get("name") or ""
                        acc = {"id": call_id, "name": name, "arguments": ""}
                        tool_calls_acc[idx] = acc
                        call_item_id = _new_item_id()
                        tool_call_item_ids[idx] = call_item_id
                        await self._send_event(
                            "response.output_item.added",
                            response_id=response_id,
                            output_index=idx,
                            item={
                                "id": call_item_id,
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": "",
                            },
                        )
                        await self._send_event(
                            "conversation.item.created",
                            item={
                                "id": call_item_id,
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                            },
                        )
                    args_delta = event.get("arguments_delta") or ""
                    if args_delta:
                        acc["arguments"] += args_delta
                        await self._send_event(
                            "response.function_call_arguments.delta",
                            response_id=response_id,
                            item_id=tool_call_item_ids[idx],
                            call_id=acc["id"],
                            delta=args_delta,
                        )

                elif event["type"] == "finish":
                    pass  # finish_reason available for logging only today

            # Flush remaining text
            if sentence_buffer.strip() and not self._cancel_event.is_set():
                if "audio" in self.modalities and self.tts_model_id:
                    await self._synthesize_and_send(sentence_buffer, response_id, output_item_id)
                    audio_sent = True

            if not self._cancel_event.is_set():
                output_items: list[dict[str, Any]] = []

                # Add assistant message to conversation
                if full_text:
                    self.conversation.append(
                        {"id": output_item_id, "role": "assistant", "content": full_text}
                    )
                    output_items.append({
                        "id": output_item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "audio", "transcript": full_text}],
                    })

                # Finalize any tool calls: tell the client, and persist an
                # assistant message carrying tool_calls (standard OpenAI chat
                # shape) so the next turn's history is coherent once the
                # client answers with conversation.item.create
                # (function_call_output) + response.create.
                if tool_calls_acc:
                    assistant_tool_calls = []
                    for idx in sorted(tool_calls_acc):
                        acc = tool_calls_acc[idx]
                        await self._send_event(
                            "response.function_call_arguments.done",
                            response_id=response_id,
                            item_id=tool_call_item_ids[idx],
                            call_id=acc["id"],
                            name=acc["name"],
                            arguments=acc["arguments"],
                        )
                        assistant_tool_calls.append({
                            "id": acc["id"],
                            "type": "function",
                            "function": {"name": acc["name"], "arguments": acc["arguments"]},
                        })
                        output_items.append({
                            "id": tool_call_item_ids[idx],
                            "type": "function_call",
                            "call_id": acc["id"],
                            "name": acc["name"],
                            "arguments": acc["arguments"],
                        })
                    self.conversation.append({
                        "id": output_item_id,
                        "role": "assistant",
                        "content": full_text or None,
                        "tool_calls": assistant_tool_calls,
                    })

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
                        "output": output_items,
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

    async def _load_stt_then_ready(self) -> None:
        """Background STT load for transcription-only mode. Sets ``_stt_ready`` when
        done (or on failure, so buffered commits stop waiting and degrade cleanly)."""
        try:
            await self._load_model_with_progress("stt", self.stt_model_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("realtime_stt_bg_load_failed", error=str(exc))
        finally:
            self._stt_ready.set()

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

        started_at = datetime.now(UTC)
        started_monotonic = time.monotonic()
        status_code = 200
        error_message: str | None = None
        output_tokens = 0
        request_id = self._model_manager.begin_request(
            self.stt_model_id, source="realtime_stt"
        )
        try:
            worker = await self._ensure_stt_worker()
            if not worker:
                status_code = 503
                error_message = "STT worker unavailable"
                logger.warning("realtime_stt_worker_missing", model_id=self.stt_model_id)
                return ""
            wav_bytes = _pcm16_to_wav(pcm_data, _INPUT_SAMPLE_RATE)
            url = f"http://127.0.0.1:{worker.port}/transcribe"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    url,
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data=self._stt_data({"response_format": "json", "temperature": "0.0"}),
                )
                resp.raise_for_status()
                result = resp.json()
                text = result.get("text", "").strip()
                output_tokens = len(text.split())
                return text
        except asyncio.CancelledError:
            status_code = 499
            error_message = "Realtime request cancelled"
            raise
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_message = str(exc)
            logger.warning(
                "realtime_stt_error",
                session_id=self._session_id,
                model_id=self.stt_model_id,
                error=error_message,
            )
            return ""
        except Exception as exc:
            status_code = 502
            error_message = str(exc)
            logger.warning(
                "realtime_stt_error",
                session_id=self._session_id,
                model_id=self.stt_model_id,
                error=error_message,
            )
            return ""
        finally:
            self._model_manager.end_request(self.stt_model_id, request_id)
            await self._record_request(
                model_id=self.stt_model_id,
                request_kind="realtime_transcription",
                started_at=started_at,
                started_monotonic=started_monotonic,
                status_code=status_code,
                error_message=error_message,
                input_tokens=0,
                output_tokens=output_tokens,
            )

    async def _transcribe_verbose(self, pcm_data: bytes) -> dict[str, Any]:
        """Transcribe with segment/word timestamps (and speakers if the profile
        diarizes). Returns ``{text, segments, words, speakers}``; empty on error."""
        empty = {"text": "", "segments": [], "words": [], "speakers": [], "speaker_embeddings": {}}
        if not self.stt_model_id:
            return empty
        started_at = datetime.now(UTC)
        started_monotonic = time.monotonic()
        status_code = 200
        error_message: str | None = None
        output_tokens = 0
        request_id = self._model_manager.begin_request(
            self.stt_model_id, source="realtime_stt"
        )
        try:
            worker = await self._ensure_stt_worker()
            if not worker:
                status_code = 503
                error_message = "STT worker unavailable"
                return empty
            wav_bytes = _pcm16_to_wav(pcm_data, _INPUT_SAMPLE_RATE)
            url = f"http://127.0.0.1:{worker.port}/transcribe"
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
                text = (r.get("text") or "").strip()
                output_tokens = len(text.split())
                return {
                    "text": text,
                    "segments": r.get("segments") or [],
                    "words": r.get("words") or [],
                    "speakers": r.get("speakers") or [],
                    "speaker_embeddings": r.get("speaker_embeddings") or {},
                }
        except asyncio.CancelledError:
            status_code = 499
            error_message = "Realtime request cancelled"
            raise
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_message = str(exc)
            logger.warning(
                "realtime_stt_verbose_error",
                session_id=self._session_id,
                model_id=self.stt_model_id,
                error=error_message,
            )
            return empty
        except Exception as exc:
            status_code = 502
            error_message = str(exc)
            logger.warning(
                "realtime_stt_verbose_error",
                session_id=self._session_id,
                model_id=self.stt_model_id,
                error=error_message,
            )
            return empty
        finally:
            self._model_manager.end_request(self.stt_model_id, request_id)
            await self._record_request(
                model_id=self.stt_model_id,
                request_kind="realtime_transcription",
                started_at=started_at,
                started_monotonic=started_monotonic,
                status_code=status_code,
                error_message=error_message,
                input_tokens=0,
                output_tokens=output_tokens,
            )

    async def _ensure_stt_worker(self):
        """Return a stable STT worker, waiting through an active unload."""
        if not self.stt_model_id:
            return None

        from ocabra.config import settings
        from ocabra.core.model_manager import ModelStatus

        deadline = time.monotonic() + max(1, int(settings.model_load_wait_timeout_s))
        while time.monotonic() < deadline:
            state = await self._model_manager.get_state(self.stt_model_id)
            if state is not None and state.status == ModelStatus.UNLOADING:
                await asyncio.sleep(0.1)
                continue

            worker = self._worker_pool.get_worker(self.stt_model_id)
            if worker is not None and (
                state is None or state.status == ModelStatus.LOADED
            ):
                return worker

            try:
                await self._model_manager.load(self.stt_model_id)
            except Exception as exc:  # noqa: BLE001 - retry until bounded deadline
                logger.info(
                    "realtime_stt_load_wait",
                    session_id=self._session_id,
                    model_id=self.stt_model_id,
                    error=str(exc),
                )
                await asyncio.sleep(0.1)
                continue

            worker = self._worker_pool.get_worker(self.stt_model_id)
            if worker is not None:
                return worker

        return None

    def _filter_hallucinations(self, result: dict[str, Any]) -> None:
        """Drop hallucinated-over-silence segments from a transcription result
        (streaming only), recomputing text/words to match. Mutates ``result``."""
        segments = result.get("segments") or []
        kept = [s for s in segments if not _is_hallucinated_segment(s)]
        if len(kept) == len(segments):
            return
        keep_ranges = [(float(s.get("start") or 0.0), float(s.get("end") or 0.0)) for s in kept]
        result["segments"] = kept
        result["text"] = " ".join(str(s.get("text") or "").strip() for s in kept).strip()
        result["words"] = [
            w
            for w in (result.get("words") or [])
            if any(a <= float(w.get("start") or 0.0) <= b for a, b in keep_ranges)
        ]

    def _reconcile_speakers(self, local_embeddings: dict[str, list[float]]) -> dict[str, str]:
        """Map a segment's local pyannote labels (SPEAKER_00…) to session-global
        ``speaker_N`` ids by matching embeddings against the registry. Matching uses
        the best similarity over each speaker's exemplars (robust to intra-speaker
        variation), respects a ``num_speakers`` cap (no new speaker once reached —
        the main anti-drift guard), and only mints a new speaker above threshold."""
        mapping: dict[str, str] = {}
        for local_label, emb in local_embeddings.items():
            if not emb:
                continue
            best_id: str | None = None
            best_sim = -1.0
            for entry in self._speaker_registry:
                sim = max(
                    (_cosine(emb, ex) for ex in entry["exemplars"]),
                    default=_cosine(emb, entry["centroid"]),
                )
                if sim > best_sim:
                    best_sim, best_id = sim, entry["id"]
            at_cap = self._max_speakers > 0 and len(self._speaker_registry) >= self._max_speakers
            if best_id is not None and (best_sim >= _SPEAKER_SIM_THRESHOLD or at_cap):
                entry = next(e for e in self._speaker_registry if e["id"] == best_id)
                n = entry["count"]
                entry["centroid"] = [
                    (c * n + e2) / (n + 1) for c, e2 in zip(entry["centroid"], emb)
                ]
                entry["count"] = n + 1
                entry["exemplars"].append(list(emb))
                if len(entry["exemplars"]) > _MAX_EXEMPLARS:
                    entry["exemplars"].pop(0)
                mapping[local_label] = best_id
            else:
                new_id = f"speaker_{len(self._speaker_registry) + 1}"
                self._speaker_registry.append(
                    {"id": new_id, "centroid": list(emb), "count": 1, "exemplars": [list(emb)]}
                )
                mapping[local_label] = new_id
            self._last_speaker = mapping[local_label]
        return mapping

    def _save_registry(self) -> None:
        """Persist the speaker registry so a reconnecting session with the same
        conversation_id resumes the same speaker ids."""
        if not self._conversation_id:
            return
        _REGISTRY_STORE[self._conversation_id] = {
            "registry": self._speaker_registry,
            "last": self._last_speaker,
            "ts": time.time(),
        }

    def _load_registry(self) -> None:
        if not self._conversation_id:
            return
        now = time.time()
        _prune_registry_store(now)
        stored = _REGISTRY_STORE.get(self._conversation_id)
        if stored and now - stored.get("ts", 0.0) <= _REGISTRY_TTL_S:
            self._speaker_registry = stored["registry"]
            self._last_speaker = stored.get("last")
            logger.info(
                "realtime_speaker_registry_resumed",
                conversation_id=self._conversation_id,
                speakers=len(self._speaker_registry),
            )

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
        last_buf_len = -1
        stagnant_ticks = 0
        try:
            while True:
                await asyncio.sleep(_PARTIAL_INTERVAL_S)
                # Guard A — Starlette normally raises on the main receive_text
                # loop when the client disconnects, but a half-closed TCP (no
                # FIN delivered, e.g. network drop or client crash) leaves this
                # background task spinning forever on a dead buffer. Check the
                # WS state directly and bail.
                if self.ws.client_state != WebSocketState.CONNECTED:
                    logger.info(
                        "realtime_partial_abort_ws_closed",
                        session_id=self._session_id,
                    )
                    return
                if not self._stt_ready.is_set():
                    continue  # no partials until the model finishes cold-loading
                buf = bytes(self.audio_buffer)
                # Guard B — detect a dead peer that stopped sending audio: if
                # the buffer length hasn't grown over ``_PARTIAL_STAGNANT_MAX``
                # consecutive ticks, assume the client is gone and release the
                # whisper worker.
                if len(buf) == last_buf_len:
                    stagnant_ticks += 1
                    if stagnant_ticks >= _PARTIAL_STAGNANT_MAX:
                        logger.info(
                            "realtime_partial_abort_idle",
                            session_id=self._session_id,
                            buffer_bytes=len(buf),
                            idle_ticks=stagnant_ticks,
                        )
                        return
                    continue
                stagnant_ticks = 0
                last_buf_len = len(buf)
                if len(buf) < _PARTIAL_MIN_BYTES:
                    continue
                text = await self._transcribe(buf)
                words = text.split()
                if not words:
                    continue
                # Skip partials that are just a known filler (Whisper silence
                # hallucination) so the live preview doesn't flicker "Gracias/Adiós".
                if self._suppress_hallucinations and _normalize_phrase(text) in _HALLUCINATION_PHRASES:
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

    async def _stream_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        response_overrides: dict[str, Any] | None = None,
    ) -> AsyncIteratorWrapper:
        """Stream the LLM response via the shared chat-completions pipeline.

        Routes through :meth:`WorkerPool.forward_stream` so the same code path
        as ``/v1/chat/completions`` is reused (Ollama URL resolution, reasoning
        field normalisation, Langfuse tracing). Native-audio LLMs receive the
        message verbatim — including ``input_audio`` content parts.

        ``to_backend_body`` — the SAME translation chat.py/completions.py use
        — is reused here too (2026-10-30; previously this hand-built a bare
        ``{model, messages, stream}`` body, silently dropping every
        generation param a client could set: temperature, max_tokens, tools,
        reasoning_effort, reasoning_budget). Reusing it means Realtime
        inherits the backend-specific reasoning_effort/reasoning_budget/
        thinking-disable translation for free instead of reimplementing it.

        Yields dicts tagged by ``type``:
          - ``{"type": "content", "text": str}`` — a text delta.
          - ``{"type": "tool_call_delta", "index": int, "id": str|None,
            "name": str|None, "arguments_delta": str}`` — a fragment of a
            streamed tool call; ``id``/``name`` are only present on the first
            fragment for a given ``index`` (matches OpenAI's own streaming
            tool-call shape).
          - ``{"type": "finish", "reason": str|None}`` — end of this
            response's stream (``finish_reason`` from the backend, e.g.
            ``"stop"``/``"tool_calls"``/``"length"``).
        """
        started_at = datetime.now(UTC)
        started_monotonic = time.monotonic()
        status_code = 200
        error_message: str | None = None
        input_tokens = _message_word_count(messages)
        output_tokens = 0
        line_buf = b""
        tool_call_names: dict[int, str] = {}  # index -> name, for stats/logging only
        try:
            worker = self._worker_pool.get_worker(self.llm_worker_key)
            if not worker:
                try:
                    if self.llm_worker_key != self.llm_model_id:
                        from ocabra.api.openai._deps import ensure_worker_loaded

                        await ensure_worker_loaded(
                            self._model_manager,
                            self.llm_model_id,
                            self.llm_worker_key,
                            self._llm_load_overrides,
                            preferred_gpu=self._llm_preferred_gpu,
                            load_policy=self._llm_load_policy,
                        )
                    else:
                        await self._model_manager.load(self.llm_worker_key)
                    worker = self._worker_pool.get_worker(self.llm_worker_key)
                except Exception as exc:
                    status_code = 503
                    error_message = str(exc)
                    logger.warning(
                        "realtime_llm_load_failed",
                        model_id=self.llm_worker_key,
                        error=error_message,
                    )
                if not worker:
                    status_code = 503
                    error_message = error_message or "LLM worker unavailable"
                    logger.warning("realtime_llm_worker_missing", model_id=self.llm_worker_key)
                    return

            state = await self._model_manager.get_state(self.llm_worker_key)
            if not state:
                status_code = 503
                error_message = "LLM state unavailable"
                logger.warning("realtime_llm_state_missing", model_id=self.llm_worker_key)
                return

            from ocabra.api.openai._deps import to_backend_body

            raw_body: dict[str, Any] = {"messages": messages, "stream": True}
            gen = {**self._generation_defaults(), **(response_overrides or {})}
            if gen.get("temperature") is not None:
                raw_body["temperature"] = gen["temperature"]
            if gen.get("max_response_output_tokens") is not None:
                raw_body["max_tokens"] = gen["max_response_output_tokens"]
            if gen.get("reasoning_effort") is not None:
                raw_body["reasoning_effort"] = gen["reasoning_effort"]
            if gen.get("reasoning_budget") is not None:
                raw_body["reasoning_budget"] = gen["reasoning_budget"]
            tools = gen.get("tools")
            if tools:
                raw_body["tools"] = _realtime_tools_to_chat_format(tools)
                raw_body["tool_choice"] = gen.get("tool_choice") or "auto"
            body = to_backend_body(state, raw_body)

            async for chunk in self._worker_pool.forward_stream(
                self.llm_worker_key, "/v1/chat/completions", body
            ):
                if self._cancel_event.is_set():
                    status_code = 499
                    error_message = "Realtime response cancelled"
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
                        output_tokens += len(content.split())
                        yield {"type": "content", "text": content}
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        fn = tc.get("function") or {}
                        name = fn.get("name")
                        if name:
                            tool_call_names[idx] = name
                        yield {
                            "type": "tool_call_delta",
                            "index": idx,
                            "id": tc.get("id"),
                            "name": name,
                            "arguments_delta": fn.get("arguments") or "",
                        }
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        yield {"type": "finish", "reason": finish_reason}
        except asyncio.CancelledError:
            status_code = 499
            error_message = "Realtime request cancelled"
            raise
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_message = str(exc)
            logger.warning("realtime_llm_http_error", status=exc.response.status_code)
        except Exception as exc:
            status_code = 502
            error_message = str(exc)
            logger.warning("realtime_llm_stream_error", error=error_message)
        finally:
            await self._record_request(
                model_id=self.llm_model_id,
                request_kind="realtime_chat",
                started_at=started_at,
                started_monotonic=started_monotonic,
                status_code=status_code,
                error_message=error_message,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

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

        started_at = datetime.now(UTC)
        started_monotonic = time.monotonic()
        status_code = 200
        error_message: str | None = None
        try:
            # Wait for TTS to be ready (may still be loading)
            if not self._tts_ready.is_set():
                await asyncio.wait_for(self._tts_ready.wait(), timeout=60)

            worker = self._worker_pool.get_worker(self.tts_model_id)
            if not worker:
                await self._model_manager.load(self.tts_model_id)
                worker = self._worker_pool.get_worker(self.tts_model_id)
            if not worker:
                status_code = 503
                error_message = "TTS worker unavailable"
                logger.warning("realtime_tts_worker_missing", model_id=self.tts_model_id)
                return

            url = f"http://127.0.0.1:{worker.port}/synthesize"

            payload = {
                "input": text,
                "voice": self.voice,
                "response_format": "pcm",
                "speed": 1.0,
            }

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

        except TimeoutError:
            status_code = 503
            error_message = "TTS model load timeout"
            logger.warning("realtime_tts_load_timeout", model_id=self.tts_model_id)
        except asyncio.CancelledError:
            status_code = 499
            error_message = "Realtime request cancelled"
            raise
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_message = str(exc)
            logger.warning("realtime_tts_error", error=error_message)
        except Exception as exc:
            status_code = 502
            error_message = str(exc)
            logger.warning("realtime_tts_error", error=error_message)
        finally:
            await self._record_request(
                model_id=self.tts_model_id,
                request_kind="realtime_tts",
                started_at=started_at,
                started_monotonic=started_monotonic,
                status_code=status_code,
                error_message=error_message,
                input_tokens=len(text.split()),
                output_tokens=0,
            )

    # ── Message helpers ─────────────────────────────────────────────────

    def _build_llm_messages(self, instructions_override: str | None = None) -> list[dict[str, Any]]:
        """Build the messages array for the LLM, including system instructions.

        Conversation entries may carry either plain ``content`` strings (STT
        path) or a list of OpenAI-style content parts (native-audio path —
        ``input_audio``). Both forms are forwarded verbatim. Each entry also
        carries an ``id`` (used for ``conversation.item.delete``/matching
        SSE events to their item) that must NOT reach the backend chat
        endpoint — stripped here, not at append time, so the rest of the
        session code can keep treating ``self.conversation`` entries as
        plain chat messages with one extra key.
        """
        messages: list[dict[str, Any]] = []

        instructions = instructions_override or self.instructions
        if instructions:
            messages.append({"role": "system", "content": instructions})

        messages.extend({k: v for k, v in entry.items() if k != "id"} for entry in self.conversation)
        return messages

    def _generation_defaults(self) -> dict[str, Any]:
        """Session-level generation params, as a plain dict so
        ``_stream_llm`` can merge per-turn ``response.create`` overrides on
        top with ``{**defaults, **overrides}`` without a dozen ``if``s."""
        return {
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_budget": self.reasoning_budget,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
        }

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
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
            # oCabra extensions (standard clients ignore unknown fields):
            "reasoning_effort": self.reasoning_effort,
            "reasoning_budget": self.reasoning_budget,
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
