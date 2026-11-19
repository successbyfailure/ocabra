"""
Stats middleware — records inference request metrics.

Tracks OpenAI-compatible `/v1/*` and Ollama-compatible inference routes under
`/api/*` so the stats page reflects real usage regardless of client protocol.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ocabra.config import settings
from ocabra.core import admission, privacy

# What a request is and which worker it counts against: owned by admission
# (REF-08), shared with the stats. The old private names stay for this module
# and its tests.
from ocabra.core.admission import approx_text_tokens as _approx_text_tokens
from ocabra.core.admission import classify_request_kind as _classify_request_kind
from ocabra.core.admission import count_input_tokens as _count_input_tokens
from ocabra.core.admission import request_bound_for_peer as _request_bound_for_peer  # noqa: F401
from ocabra.core.admission import resolve_inflight_model_id as _resolve_inflight_model_id
from ocabra.core.admission import should_track as _should_track
from ocabra.core.background_tasks import spawn
from ocabra.core.request_telemetry import RequestRuntimeTelemetry, current_runtime_telemetry

logger = structlog.get_logger(__name__)


def _extract_work_size_meta(
    request_kind: str,
    request_payload: dict | None,
    response_payload: dict | None,
) -> dict | None:
    """Build the ``work_size_meta`` for the family-specific estimator.

    Called once per completed request at stat write time. Returns a dict
    with the axis a given family's regression consumes:

      * TTS              → ``text_chars``
      * Image generation → ``steps``, ``resolution_w``, ``resolution_h``
      * Transcription    → ``audio_seconds`` (from response when the
                            backend surfaces it — Whisper does).
      * ACE-Step         → ``duration_seconds`` (target output length)

    Returns None when nothing useful can be extracted — the row still
    lands with NULL and future refreshes just skip it.
    """
    if not isinstance(request_payload, dict):
        request_payload = None
    if not isinstance(response_payload, dict):
        response_payload = None

    if request_kind == "tts":
        meta: dict[str, float] = {}
        text = None
        if request_payload:
            # OpenAI's field is ``input``; some clients send ``text``.
            text = request_payload.get("input") or request_payload.get("text")
        if isinstance(text, str) and text:
            meta["text_chars"] = len(text)
        audio_s = (response_payload or {}).get("audio_seconds")
        if isinstance(audio_s, (int, float)) and audio_s > 0:
            meta["audio_seconds"] = float(audio_s)
        return meta or None

    if request_kind == "music":
        # ACE-Step: the target length of the piece. (This used to sit after
        # the TTS branch, which always returned first.)
        if request_payload:
            dur = request_payload.get("duration_seconds") or request_payload.get("duration")
            if isinstance(dur, (int, float)) and dur > 0:
                return {"duration_seconds": float(dur)}
        return None

    if request_kind in ("embedding", "ollama_embedding"):
        if request_payload is None:
            return None
        items = request_payload.get("input")
        if items is None:
            items = request_payload.get("prompt")
        if isinstance(items, list):
            return {"n_inputs": len(items)} if items else None
        return {"n_inputs": 1} if items else None

    if request_kind in ("rerank", "score", "classify", "pooling"):
        if request_payload is None:
            return None
        for key in ("documents", "texts", "text_2", "input"):
            items = request_payload.get(key)
            if isinstance(items, list) and items:
                return {"n_documents": len(items)}
            if isinstance(items, str) and items:
                return {"n_documents": 1}
        return None

    if request_kind == "image_generation":
        if request_payload is None:
            return None
        meta: dict[str, int] = {}
        steps = request_payload.get("num_inference_steps") or request_payload.get("steps")
        if isinstance(steps, (int, float)) and steps > 0:
            meta["steps"] = int(steps)
        # OpenAI: ``size`` is "1024x1024". Diffusers-native: ``width``/``height``.
        size = request_payload.get("size")
        if isinstance(size, str) and "x" in size:
            try:
                w_s, h_s = size.lower().split("x", 1)
                meta["resolution_w"] = int(w_s)
                meta["resolution_h"] = int(h_s)
            except ValueError:
                pass
        for k, dest in (("width", "resolution_w"), ("height", "resolution_h")):
            v = request_payload.get(k)
            if isinstance(v, (int, float)) and v > 0:
                meta[dest] = int(v)
        data = (response_payload or {}).get("data")
        if isinstance(data, list) and data:
            meta["n_images"] = len(data)
        else:
            n = request_payload.get("n")
            meta["n_images"] = int(n) if isinstance(n, (int, float)) and n > 0 else 1
        return meta or None

    if request_kind == "audio_transcription":
        # Whisper surfaces ``duration`` (seconds of decoded audio) on the
        # verbose_json response. Faster-whisper and Nvidia backends do the
        # same. When absent, the worker's X-Audio-Duration header fills it
        # in (see the caller).
        if response_payload is None:
            return None
        for k in ("duration", "audio_seconds"):
            v = response_payload.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return {"audio_seconds": float(v)}
        return None


def _apply_token_kind_defaults(
    in_tok: int | None,
    out_tok: int | None,
    request_kind: str,
    request_payload: dict | None,
) -> tuple[int | None, int | None]:
    """Fill in the axis that a request kind structurally can't have, so stats are
    consistent instead of a mix of null (rendered "—") and 0. Embeddings/TTS/rerank
    generate no output tokens; transcription has no text input; TTS input is counted
    from the request text (no usage payload since the response is audio)."""
    if request_kind == "embedding":
        if out_tok is None:
            out_tok = 0
    elif request_kind == "tts":
        if in_tok is None and isinstance(request_payload, dict):
            in_tok = _approx_text_tokens(request_payload.get("input"))
        if out_tok is None:
            out_tok = 0
    elif request_kind == "audio_transcription":
        if in_tok is None:
            in_tok = 0  # audio input carries no text tokens
    elif request_kind in ("rerank", "ollama_embedding", "pooling", "score", "classify"):
        if out_tok is None:
            out_tok = 0
    elif request_kind == "image_generation":
        if in_tok is None and isinstance(request_payload, dict):
            in_tok = _approx_text_tokens(request_payload.get("prompt"))
        if out_tok is None:
            out_tok = 0
    return in_tok, out_tok


# Kinds that generate text output (so a stream/content fallback makes sense when
# the backend doesn't return a usage payload).
_TEXT_GEN_KINDS = {"chat", "completion", "ollama_chat", "ollama_generate", "responses"}


def _stream_output_tokens(
    body: bytes, content_type: str, *, include_reasoning: bool = False
) -> int | None:
    """Approx output tokens (~word count) from the streamed content deltas, used when
    a streaming response carried no usage payload.

    ``include_reasoning``: also count ``delta.reasoning_content`` words, not
    just ``delta.content``. Off by default — the persisted ``output_tokens``
    stat should keep meaning "final answer length", not "final answer +
    thinking". The live-progress tee (model/GPU card) passes True: for a
    reasoning model, the entire thinking phase is otherwise invisible —
    tested live 2026-10-29, a qwen3.8:27b request sat at
    ``outputTokensEstimate: null`` through 25+ seconds of active reasoning
    because none of it had reached ``content`` yet.
    """
    if not body:
        return None
    text = body.decode("utf-8", errors="ignore")
    words = 0
    found = False
    if "event-stream" in content_type:
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") if isinstance(choice, dict) else None
                content = delta.get("content") if isinstance(delta, dict) else None
                if isinstance(content, str) and content:
                    words += len(content.split())
                    found = True
                if include_reasoning and isinstance(delta, dict):
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        words += len(reasoning.split())
                        found = True
                tool_words = _tool_call_words(
                    delta.get("tool_calls") if isinstance(delta, dict) else None
                )
                if tool_words is not None:
                    words += tool_words
                    found = True
    elif "ndjson" in content_type:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            msg = chunk.get("message") if isinstance(chunk, dict) else None
            content = (msg or {}).get("content") if isinstance(msg, dict) else None
            if not content and isinstance(chunk, dict):
                content = chunk.get("response")
            if isinstance(content, str) and content:
                words += len(content.split())
                found = True
            if include_reasoning and isinstance(msg, dict):
                thinking = msg.get("thinking")
                if isinstance(thinking, str) and thinking:
                    words += len(thinking.split())
                    found = True
            tool_words = _tool_call_words(msg.get("tool_calls") if isinstance(msg, dict) else None)
            if tool_words is not None:
                words += tool_words
                found = True
    return words if found else None


_ARG_PIECES = re.compile(r"\w+|[^\w\s]+")


def _tool_call_words(tool_calls: object) -> int | None:
    """Approx tokens generated as tool-call arguments in one stream event.

    The arguments are model output like any other: an agent writing a file
    through a tool emits the whole file there. Without this the live counter
    froze while the model kept generating — ravenx showed ~1 t/s in the
    dashboard with 156 tokens counted and 8.300 decoded by llama-server
    (2026-11-12). OpenAI streams ``function.arguments`` as string pieces;
    Ollama sends the whole call with ``arguments`` as an object.
    """
    if not isinstance(tool_calls, list):
        return None
    words = 0
    found = False
    for call in tool_calls:
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict):
            continue
        args = fn.get("arguments")
        if isinstance(args, dict | list):
            args = json.dumps(args, ensure_ascii=False)
        if isinstance(args, str) and args:
            # JSON and code have few spaces: count words and punctuation runs,
            # or a 200-character argument would weigh like one word.
            words += max(1, len(_ARG_PIECES.findall(args)))
            found = True
    return words if found else None


def _chunk_output_words(chunk: dict, *, sse: bool) -> tuple[int, bool, int, bool]:
    """Words one parsed stream event adds: ``(content, found, with_reasoning, found)``.

    The same rules as :func:`_stream_output_tokens`, applied to a single
    event so a stream can be counted as it passes instead of re-reading it.
    """
    words = 0
    found = False
    extra = 0  # reasoning/thinking words: only the live counter sees them
    extra_found = False
    if sse:
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str) and content:
                words += len(content.split())
                found = True
            if isinstance(delta, dict):
                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    extra += len(reasoning.split())
                    extra_found = True
            tool_words = _tool_call_words(
                delta.get("tool_calls") if isinstance(delta, dict) else None
            )
            if tool_words is not None:
                words += tool_words
                found = True
    else:
        msg = chunk.get("message")
        content = (msg or {}).get("content") if isinstance(msg, dict) else None
        if not content:
            content = chunk.get("response")
        if isinstance(content, str) and content:
            words += len(content.split())
            found = True
        if isinstance(msg, dict):
            thinking = msg.get("thinking")
            if isinstance(thinking, str) and thinking:
                extra += len(thinking.split())
                extra_found = True
        tool_words = _tool_call_words(msg.get("tool_calls") if isinstance(msg, dict) else None)
        if tool_words is not None:
            words += tool_words
            found = True
    return words, found, words + extra, found or extra_found


class _StreamAccumulator:
    """Incremental reader of a streamed response body (BUG-36).

    The streaming tee used to keep every chunk and re-parse the whole body:
    on each chunk until the first token, every 0.5 s for the live progress,
    and three more times at the end. CPU on the event loop grew with the
    square of the stream length and memory with the whole response. This
    parses each complete line exactly once as it passes and keeps only what
    the stats need: the word counters, the last ``usage`` event (SSE), the
    last ``done`` event (NDJSON) and the last in-band error. The results are
    the ones :func:`_stream_output_tokens`, :func:`_extract_last_payload_from_stream`
    and :func:`_extract_stream_error` give for the whole body.

    Memory: the unfinished last line, capped (``_MAX_LINE_TEXT`` for SSE and
    NDJSON, ``_MAX_LINE_OTHER`` for audio/video, where only a JSON error line
    could matter). A line longer than the cap is dropped, as if it were not
    JSON. The audio body of TTS/music is kept on request (``keep_body``),
    because its length is what ``audio_seconds`` comes from; past
    ``_MAX_AUDIO_BODY`` only the head and the total size are kept.
    """

    _MAX_LINE_TEXT = 16 * 1024 * 1024
    _MAX_LINE_OTHER = 64 * 1024
    _MAX_AUDIO_BODY = 64 * 1024 * 1024
    _TAIL_PEEK_LIMIT = 64 * 1024

    def __init__(self, content_type: str, *, keep_body: bool = False) -> None:
        self._sse = "event-stream" in content_type
        self._ndjson = not self._sse and "ndjson" in content_type
        self._text = self._sse or self._ndjson
        self._max_line = self._MAX_LINE_TEXT if self._text else self._MAX_LINE_OTHER
        self._pending = bytearray()
        self._dropping = False  # inside a line longer than the cap
        self._keep_body = keep_body
        self._body: list[bytes] = []
        self._body_size = 0
        self.total_size = 0
        self.body_truncated = False
        self.content_words = 0
        self.content_found = False
        self.all_words = 0
        self.all_found = False
        self.last_usage_payload: dict | None = None
        self.last_done_payload: dict | None = None
        self.last_error_payload: dict | None = None
        # Instrumentation for the tests: bytes handed to the line parser.
        self.bytes_parsed = 0

    # ── input ─────────────────────────────────────────────────────────
    def feed(self, chunk: bytes) -> None:
        self.total_size += len(chunk)
        if self._keep_body and chunk:
            if self._body_size + len(chunk) <= self._MAX_AUDIO_BODY:
                self._body.append(chunk)
                self._body_size += len(chunk)
            else:
                self.body_truncated = True
        newline = chunk.rfind(b"\n")
        if newline < 0:
            self._append_pending(chunk)
            return
        head = chunk[:newline]
        if self._dropping:
            # The rest of an oversized line ends here; the line is discarded.
            first = head.find(b"\n")
            self._dropping = False
            self._pending.clear()
            if first < 0:
                head = b""
            else:
                head = head[first + 1 :]
            data = head
        elif self._pending:
            data = bytes(self._pending) + head
            self._pending.clear()
        else:
            data = head
        if data:
            self._parse_block(data)
        self._append_pending(chunk[newline + 1 :])

    def finish(self) -> None:
        """Parse the last line when the stream did not end with a newline."""
        if self._pending and not self._dropping:
            self._parse_block(bytes(self._pending))
        self._pending.clear()
        self._dropping = False

    def _append_pending(self, data: bytes) -> None:
        if not data or self._dropping:
            return
        if len(self._pending) + len(data) > self._max_line:
            self._pending.clear()
            self._dropping = True
            return
        self._pending += data

    # ── parsing ───────────────────────────────────────────────────────
    def _parse_block(self, data: bytes) -> None:
        """Complete lines (``\\n``-separated). Decoded per line so a multi-byte
        character split across chunks never breaks; ``str.splitlines`` on
        each piece keeps the old whole-body segmentation exactly."""
        self.bytes_parsed += len(data)
        for raw in data.split(b"\n"):
            if not raw:
                continue
            for line in raw.decode("utf-8", errors="ignore").splitlines():
                event = self._parse_line(line)
                if event is not None:
                    self._absorb(event)

    def _parse_line(self, line: str) -> dict | None:
        if self._sse:
            if not line.startswith("data: "):
                return None
            data = line[6:].strip()
        else:
            data = line.strip()
        # Only an object matters to any reader (they all look up keys);
        # anything else — "[DONE]", audio bytes — is skipped without parsing.
        if not data.startswith("{"):
            return None
        try:
            event = json.loads(data)
        except Exception:
            return None
        return event if isinstance(event, dict) else None

    def _absorb(self, event: dict) -> None:
        if "error" in event:
            self.last_error_payload = event
        if self._sse:
            if event.get("usage"):
                self.last_usage_payload = event
        elif self._ndjson:
            if event.get("done") is True:
                self.last_done_payload = event
        if self._text:
            words, found, all_words, all_found = _chunk_output_words(event, sse=self._sse)
            self.content_words += words
            self.content_found = self.content_found or found
            self.all_words += all_words
            self.all_found = self.all_found or all_found

    # ── results ───────────────────────────────────────────────────────
    def output_tokens(self, *, include_reasoning: bool = False) -> int | None:
        if include_reasoning:
            return self.all_words if self.all_found else None
        return self.content_words if self.content_found else None

    def has_visible_output(self) -> bool:
        """Whether a content/reasoning/tool token was already streamed.

        Includes the unfinished last line when it already parses (a chunk
        that ends right after the JSON, before its newline), which is what
        the whole-body parse saw. Bounded: only until the first token, and
        only for a short line.
        """
        if self.all_words > 0:
            return True
        if not self._text or not self._pending or self._dropping:
            return False
        if len(self._pending) > self._TAIL_PEEK_LIMIT:
            return False
        for line in bytes(self._pending).decode("utf-8", errors="ignore").splitlines():
            event = self._parse_line(line)
            if event is not None and _chunk_output_words(event, sse=self._sse)[2] > 0:
                return True
        return False

    def last_payload(self) -> dict | None:
        """Same as :func:`_extract_last_payload_from_stream` on the whole body."""
        if self._sse:
            return self.last_usage_payload
        if self._ndjson:
            return self.last_done_payload
        return None

    def stream_error(self) -> tuple[str, str | None] | None:
        """Same as :func:`_extract_stream_error` on the whole body."""
        if self.last_error_payload is None:
            return None
        error = self.last_error_payload["error"]
        if isinstance(error, dict):
            return str(error.get("message") or "Streaming inference failed"), error.get("code")
        return str(error), None

    def audio(self, content_type: str) -> tuple[bytes, str] | tuple[bytes, str, int]:
        """``output_audio`` for ``_record_stat``: the body, or its head and
        the total size when it went past the cap."""
        body = b"".join(self._body)
        if self.body_truncated:
            return body, content_type, self.total_size
        return body, content_type


def _strip_subtitle_markup(text: str) -> str:
    """Drop SRT/VTT indices and timestamp lines so a word count reflects speech."""
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.isdigit() or "-->" in s or s.upper().startswith("WEBVTT"):
            continue
        kept.append(s)
    return " ".join(kept)


def _extract_usage_tokens(
    payload: dict | None,
    request_kind: str = "",
    request_payload: dict | None = None,
) -> tuple[int | None, int | None]:
    in_tok, out_tok, _cached, _reason = _extract_usage_tokens_detailed(
        payload, request_kind, request_payload
    )
    return in_tok, out_tok


def _extract_usage_tokens_detailed(
    payload: dict | None,
    request_kind: str = "",
    request_payload: dict | None = None,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Variant of :func:`_extract_usage_tokens` that also returns the
    ``cached_tokens`` and ``reasoning_tokens`` sub-counts from the OpenAI
    ``usage.*_tokens_details`` payload. When the backend doesn't emit the
    reasoning detail we fall back to counting words of
    ``choices[].message.reasoning_content`` — approximate, but much better
    than losing the signal entirely for Ollama's thinking models (which
    report the big ``completion_tokens`` without breaking out reasoning).
    """
    in_tok: int | None = None
    out_tok: int | None = None
    cached_tok: int | None = None
    reason_tok: int | None = None

    usage = payload.get("usage") if isinstance(payload, dict) else None
    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens")
        if input_tokens is None:
            input_tokens = usage.get("input_tokens")

        output_tokens = usage.get("completion_tokens")
        if output_tokens is None:
            output_tokens = usage.get("output_tokens")

        try:
            in_tok = int(input_tokens) if input_tokens is not None else None
            out_tok = int(output_tokens) if output_tokens is not None else None
        except (TypeError, ValueError):
            in_tok, out_tok = None, None

        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            raw_cached = prompt_details.get("cached_tokens")
            try:
                cached_tok = int(raw_cached) if raw_cached is not None else None
            except (TypeError, ValueError):
                cached_tok = None

        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            raw_reason = completion_details.get("reasoning_tokens")
            try:
                reason_tok = int(raw_reason) if raw_reason is not None else None
            except (TypeError, ValueError):
                reason_tok = None

    # Ollama-style normalized responses.
    if in_tok is None and out_tok is None and isinstance(payload, dict):
        prompt_eval_count = payload.get("prompt_eval_count")
        eval_count = payload.get("eval_count")
        if prompt_eval_count is not None or eval_count is not None:
            try:
                in_tok = int(prompt_eval_count) if prompt_eval_count is not None else None
                out_tok = int(eval_count) if eval_count is not None else None
            except (TypeError, ValueError):
                in_tok, out_tok = None, None
        # Ollama exposes cache hits on /api/chat as ``prompt_eval_cached_count``.
        cached_native = payload.get("prompt_eval_cached_count")
        if cached_native is not None and cached_tok is None:
            try:
                cached_tok = int(cached_native)
            except (TypeError, ValueError):
                cached_tok = None

    # Reasoning fallback — Ollama/vLLM don't always break out
    # reasoning_tokens into the details sub-object, but they do surface the
    # raw reasoning in ``message.reasoning`` / ``message.reasoning_content``.
    # Approximate from word count so dashboards see at least a lower bound.
    if reason_tok is None and isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            reasoning_text: str | None = None
            if isinstance(message, dict):
                reasoning_text = message.get("reasoning_content") or message.get("reasoning")
            if reasoning_text:
                reason_tok = _approx_text_tokens(reasoning_text)

    # Whisper-style: {"text": "..."} — use word count as output proxy.
    if request_kind == "audio_transcription" and out_tok is None and isinstance(payload, dict):
        out_tok = _approx_text_tokens(payload.get("text"))

    in_tok, out_tok = _apply_token_kind_defaults(in_tok, out_tok, request_kind, request_payload)
    return in_tok, out_tok, cached_tok, reason_tok


def _extract_generation_ms(payload: dict | None) -> int | None:
    """Pure generation time (ms), reported directly by the backend engine —
    independent of ``duration_ms``, which measures wall clock from the
    moment the HTTP request arrived (so it includes queue wait and
    cold-start time, and understates real throughput under contention).

    llama.cpp's non-streaming/final-chunk response body carries a
    ``timings.predicted_ms`` field; Ollama's carries ``eval_duration``
    (nanoseconds). Neither vLLM nor the OpenAI-shape-only backends expose
    this, so this returns ``None`` far more often than not — callers must
    treat a ``None`` as "unknown", not "zero".
    """
    if not isinstance(payload, dict):
        return None
    timings = payload.get("timings")
    if isinstance(timings, dict):
        predicted_ms = timings.get("predicted_ms")
        if predicted_ms is not None:
            try:
                return int(float(predicted_ms))
            except (TypeError, ValueError):
                pass
    eval_duration_ns = payload.get("eval_duration")
    if eval_duration_ns is not None:
        try:
            return int(float(eval_duration_ns) / 1_000_000.0)
        except (TypeError, ValueError):
            pass
    return None


def _update_engine_runtime_telemetry(request: Request, payload: dict | None) -> None:
    """Merge optional llama.cpp prompt/slot timings into request telemetry."""
    telemetry = getattr(request.state, "runtime_telemetry", None)
    if not isinstance(telemetry, RequestRuntimeTelemetry) or not isinstance(payload, dict):
        return
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        timings = {}
    raw_prompt_ms = timings.get("prompt_ms")
    try:
        if raw_prompt_ms is not None:
            telemetry.prefill_ms = max(0, int(float(raw_prompt_ms)))
    except (TypeError, ValueError):
        pass
    raw_slot = payload.get("slot_id", timings.get("slot_id"))
    try:
        if raw_slot is not None:
            telemetry.slot_id = int(raw_slot)
    except (TypeError, ValueError):
        pass


def _mark_stream_first_token(
    request: Request,
    chunks: list[bytes],
    content_type: str,
    *,
    now_monotonic: float | None = None,
) -> None:
    """Capture TTFT even when a complete stream finishes inside 500 ms.

    Live progress is deliberately throttled, but TTFT must inspect every chunk
    until the first visible content/reasoning token arrives. After that this is
    a constant-time no-op, so long streams do not acquire quadratic parsing.
    """
    telemetry = getattr(request.state, "runtime_telemetry", None)
    if not isinstance(telemetry, RequestRuntimeTelemetry) or telemetry.ttft_ms is not None:
        return
    output_tokens = _stream_output_tokens(b"".join(chunks), content_type, include_reasoning=True)
    if output_tokens is not None and output_tokens > 0:
        telemetry.mark_first_token(now_monotonic)


def _mark_stream_first_token_from(request: Request, acc: _StreamAccumulator) -> None:
    """:func:`_mark_stream_first_token` over the running parse of the stream.

    Constant work per chunk once the first token is known; before that, at
    most a peek at the unfinished last line (see ``has_visible_output``).
    """
    telemetry = getattr(request.state, "runtime_telemetry", None)
    if not isinstance(telemetry, RequestRuntimeTelemetry) or telemetry.ttft_ms is not None:
        return
    if acc.has_visible_output():
        telemetry.mark_first_token(None)


def _stream_ttft_known(request: Request) -> bool:
    """Si ``_mark_stream_first_token`` ya vio el primer token de este stream."""
    telemetry = getattr(request.state, "runtime_telemetry", None)
    return isinstance(telemetry, RequestRuntimeTelemetry) and telemetry.ttft_ms is not None


def _ollama_tps(payload: dict, clave_tokens: str, clave_ns: str) -> float | None:
    """tokens/s de un par (contador, duracion en ns) de la respuesta de Ollama."""

    tokens = payload.get(clave_tokens)
    duracion_ns = payload.get(clave_ns)
    if tokens is None or duracion_ns is None:
        return None
    try:
        tokens = float(tokens)
        segundos = float(duracion_ns) / 1_000_000_000.0
    except (TypeError, ValueError):
        return None
    if tokens <= 0 or segundos <= 0:
        return None
    return tokens / segundos


def _extract_speculative_stats(
    payload: dict | None,
) -> tuple[int | None, int | None, float | None, float | None]:
    """(draft_n, draft_n_accepted, predicted_per_second, prompt_per_second)
    straight from llama-server's own ``timings`` block — see RequestStat's
    draft_n/draft_n_accepted/predicted_tokens_per_second/
    prompt_tokens_per_second docstrings for why these are persisted.
    Only llama.cpp reports any of this; everything else returns all-None.
    """
    if not isinstance(payload, dict):
        return None, None, None, None
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        # Respuesta con forma de Ollama: no trae ``timings``, trae contadores y
        # duraciones en nanosegundos. Son los mismos numeros con otro nombre, y
        # sin esto /api/chat y /api/generate no guardaban ni una cifra de
        # rendimiento: 1.339 peticiones a ciegas en los 7 dias previos al
        # 2026-11-09, un 29% del trafico generativo.
        return (
            None,
            None,
            _ollama_tps(payload, "eval_count", "eval_duration"),
            _ollama_tps(payload, "prompt_eval_count", "prompt_eval_duration"),
        )

    def _int(key: str) -> int | None:
        value = timings.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _float(key: str) -> float | None:
        value = timings.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return (
        _int("draft_n"),
        _int("draft_n_accepted"),
        _float("predicted_per_second"),
        _float("prompt_per_second"),
    )


# Peticiones que solo hacen prefill y cuyo motor no devuelve ``timings``.
_PREFILL_ONLY_KINDS = {"embedding", "ollama_embedding"}


def _prefill_only_prompt_tps(
    request: Request, request_kind: str, input_tokens: int | None
) -> float | None:
    """tokens/s de prefill de un embedding: tokens de entrada entre lo que
    tardó el worker (``RequestRuntimeTelemetry.prefill_ms``, que anota el
    endpoint). No se usa ``duration_ms`` porque incluye la cola y la carga
    del modelo, y con un arranque en frío la cifra no significaría nada."""
    if request_kind not in _PREFILL_ONLY_KINDS or not input_tokens or input_tokens <= 0:
        return None
    telemetry = getattr(request.state, "runtime_telemetry", None)
    prefill_ms = getattr(telemetry, "prefill_ms", None)
    if not isinstance(prefill_ms, (int, float)) or prefill_ms <= 0:
        return None
    return round(input_tokens / (prefill_ms / 1000.0), 2)


def _extract_error_message(payload: dict | None, status_code: int) -> str | None:
    """Preserve the structured API error instead of reducing it to ``HTTP 5xx``."""
    if status_code < 400:
        return None
    if not isinstance(payload, dict):
        return f"HTTP {status_code}"

    error: object = payload.get("error")
    if error is None and isinstance(payload.get("detail"), dict):
        error = payload["detail"].get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        code = str(error.get("code") or "").strip()
        if message and code and code not in message:
            return f"{message} [{code}]"
        if message:
            return message
        if code:
            return code
    elif isinstance(error, str) and error.strip():
        return error.strip()

    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return f"HTTP {status_code}"


class StatsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that records usage statistics for /v1/* and /api/* requests.

    For non-streaming JSON responses, token counts are extracted from usage payloads.
    Streaming responses record request-level latency/errors but token counts may be
    unavailable depending on upstream chunk format.

    Records only. Admission (in-flight cap, maintenance, circuit breaker)
    is ``ocabra.core.admission.AdmissionMiddleware``, registered just outside
    this one (REF-08): it leaves the parsed body, the start time and the
    in-flight id in ``request.state``, and passes its rejections to
    :func:`record_admission_rejection`.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if not _should_track(path):
            return await call_next(request)

        admitted_at = request.scope.get("state", {}).get(admission.STARTED_STATE_KEY)
        if admitted_at is not None:
            start, started_at = admitted_at
        else:
            start, started_at = time.monotonic(), datetime.now(UTC)
        request.state.runtime_telemetry = RequestRuntimeTelemetry(started_monotonic=start)
        # call_next corre la app en un contexto copiado de este: ahi la ve
        # ``_do_ensure_loaded`` para anotar la espera de carga.
        telemetry_token = current_runtime_telemetry.set(request.state.runtime_telemetry)
        if admission.PAYLOAD_STATE_KEY in request.scope.get("state", {}):
            request_payload = request.scope["state"][admission.PAYLOAD_STATE_KEY]
        else:  # no admission layer in front (some tests): read it here
            request_payload = await _extract_request_payload(request)
        request_kind = _classify_request_kind(path)
        try:
            mm = request.app.state.model_manager
        except AttributeError:
            mm = None

        try:
            try:
                response = await call_next(request)
            finally:
                # Quien llame a la app despues en este mismo contexto (el
                # procesador de batches por ASGI) no debe heredarla.
                current_runtime_telemetry.reset(telemetry_token)
        except asyncio.CancelledError:
            # Admin or owner cancelled via /ocabra/stats/requests/{id}/cancel.
            # ``CancelledError`` is BaseException on 3.8+, so the generic
            # ``except Exception`` below never catches it. Recorded as a 499 —
            # the canonical "client closed request" code — so the UI shows the
            # cancelled turn. Admission releases the in-flight slot.
            model_id = _extract_model_id(request=request, body=request_payload)
            if model_id:
                spawn(
                    _record_stat(
                        request=request,
                        model_id=model_id,
                        started_at=started_at,
                        duration_ms=(time.monotonic() - start) * 1000,
                        error_message="cancelled",
                        status_code=499,
                        endpoint_path=path,
                        request_kind=request_kind,
                        input_tokens=None,
                        output_tokens=None,
                    )
                )
            raise
        except Exception as exc:
            # Generation timeout (504) and circuit breaker (503) get their
            # response from admission (``dispatch_error_response``), anything
            # else is a 500; recorded here with the same status.
            mapped = admission.dispatch_error_response(exc, path)
            model_id = _extract_model_id(request=request, body=request_payload)
            if model_id:
                spawn(
                    _record_stat(
                        request=request,
                        model_id=model_id,
                        started_at=started_at,
                        duration_ms=(time.monotonic() - start) * 1000,
                        error_message=str(exc),
                        status_code=mapped.status_code if mapped is not None else 500,
                        endpoint_path=path,
                        request_kind=request_kind,
                        input_tokens=None,
                        output_tokens=None,
                    )
                )
            raise
        inflight_request_id = getattr(request.state, "inflight_request_id", None)

        # Detect streaming by content-type (SSE or NDJSON) or by response
        # shape (any Starlette response that exposes ``body_iterator``). The
        # last check is critical: ``BaseHTTPMiddleware`` wraps every downstream
        # response in a private ``starlette.middleware.base._StreamingResponse``
        # so a FastAPI ``StreamingResponse`` is NOT recognised by isinstance
        # anymore. Without the body_iterator fallback, audio/*, video/* and
        # file downloads fell through to the non-streaming branch and were
        # fully buffered via ``response.body``, which turned the TTS endpoint
        # from true streaming into "wait, then dump everything at once"
        # (observed TTFB == total duration).
        content_type = response.headers.get("content-type", "")
        is_streaming = (
            "text/event-stream" in content_type
            or "x-ndjson" in content_type
            or content_type.startswith("audio/")
            or content_type.startswith("video/")
            or isinstance(response, StreamingResponse)
        )
        # NOTE: do NOT fall back to ``hasattr(response, "body_iterator")`` —
        # BaseHTTPMiddleware wraps every downstream response in a
        # ``_StreamingResponse`` that exposes ``body_iterator`` even for
        # application/json. Treating those as streaming leaves token counts
        # NULL because we never read the final usage block. The explicit
        # content-type checks above already cover genuine streams.

        if is_streaming:
            # Bloque 20 fix (2026-09-19): the streaming path used to store
            # the raw ``model_id`` from the body — i.e. for a router request
            # it recorded ``gemma4:26b`` (router name) instead of the target
            # the resolver actually served. Mirror the non-streaming path's
            # override so ``/stats/routing`` and ``request_stats`` reflect
            # reality for the 99% of traffic that uses stream=true.
            model_id = _extract_model_id(request=request, body=request_payload)
            fed_target, fed_router = _federated_router_attribution(request)
            resolved_model_id = getattr(request.state, "resolved_model_id", None) or fed_target
            if resolved_model_id:
                model_id = resolved_model_id
            via_router_profile_id = (
                getattr(request.state, "via_router_profile_id", None) or fed_router
            )
            status_code = response.status_code
            original_iterator = response.body_iterator

            async def tee_and_record():
                # Parsed as it passes, once per line (BUG-36): only the audio
                # of TTS/music is kept, for its length (``audio_seconds``).
                acc = _StreamAccumulator(content_type, keep_body=request_kind in _AUDIO_KINDS)
                error_msg: str | None = None
                stream_cancelled = False
                last_progress_at = 0.0
                first_token_reported = False
                try:
                    async for chunk in original_iterator:
                        if isinstance(chunk, str):
                            chunk = chunk.encode("utf-8")
                        acc.feed(chunk)
                        _mark_stream_first_token_from(request, acc)
                        if (
                            not first_token_reported
                            and mm
                            and inflight_request_id
                            and _stream_ttft_known(request)
                        ):
                            first_token_reported = True
                            mm.mark_request_first_token(inflight_request_id)
                        yield chunk
                        # Live progress for the model/GPU card "is this hung
                        # or still working" indicator (2026-10-29). The count
                        # is already running; the throttle (~2/s) only spares
                        # the model manager's lock.
                        if mm and inflight_request_id:
                            now_mono = time.monotonic()
                            if now_mono - last_progress_at >= 0.5:
                                last_progress_at = now_mono
                                mm.update_request_progress(
                                    inflight_request_id,
                                    output_tokens_estimate=acc.output_tokens(
                                        include_reasoning=True
                                    ),
                                )
                except asyncio.CancelledError:
                    # Once response headers have been sent, a downstream
                    # disconnect cancels this body iterator rather than the
                    # call_next() await above. CancelledError inherits from
                    # BaseException, so the generic handler does not see it.
                    # Preserve partial usage while marking the request as the
                    # canonical 499 instead of a misleading successful 200.
                    stream_cancelled = True
                    error_msg = "stream_cancelled"
                    raise
                except Exception as exc:
                    error_msg = str(exc)
                    raise
                finally:
                    # Cancelling an outer async generator does not guarantee
                    # that the iterator it was consuming is closed.  That is
                    # especially important here: the inner OpenAI/Ollama
                    # stream owns the fair-slot/KV lease.  If a client drops
                    # after response headers, Starlette cancels this tee; an
                    # unclosed inner iterator would keep that lease forever
                    # and make every later request look one slot busier.
                    #
                    # Close on every exit.  ``aclose`` is idempotent for async
                    # generators and also lets backend response iterators
                    # release their HTTP connection promptly on errors.
                    close_iterator = getattr(original_iterator, "aclose", None)
                    if close_iterator is not None:
                        try:
                            await close_iterator()
                        except Exception:  # noqa: BLE001 - cleanup must not hide the stream result
                            logger.exception("stream_iterator_close_failed", model_id=model_id)
                    duration_ms = (time.monotonic() - start) * 1000
                    if model_id:
                        acc.finish()
                        stream_error = acc.stream_error()
                        recorded_status = status_code
                        if stream_cancelled:
                            recorded_status = 499
                        elif error_msg:
                            recorded_status = 500
                        elif stream_error is not None:
                            error_msg, error_code = stream_error
                            recorded_status = _stream_error_status(error_code)
                        last_payload = acc.last_payload()
                        _update_engine_runtime_telemetry(request, last_payload)
                        in_tok, out_tok, cached_tok, reason_tok = _extract_usage_tokens_detailed(
                            last_payload, request_kind, request_payload
                        )
                        generation_ms = _extract_generation_ms(last_payload)
                        draft_n, draft_n_accepted, predicted_tps, prompt_tps = (
                            _extract_speculative_stats(last_payload)
                        )
                        # Streaming without a usage payload (e.g. no stream_options.
                        # include_usage): approximate from the streamed content and
                        # the request so the row isn't left as "—".
                        if request_kind in _TEXT_GEN_KINDS:
                            if out_tok is None:
                                out_tok = acc.output_tokens()
                            if in_tok is None:
                                in_tok = _count_input_tokens(request_payload, request_kind)
                        work_size_meta = _extract_work_size_meta(
                            request_kind, request_payload, last_payload
                        )
                        spawn(
                            _record_stat(
                                request=request,
                                model_id=model_id,
                                started_at=started_at,
                                duration_ms=duration_ms,
                                error_message=error_msg
                                or (f"HTTP {recorded_status}" if recorded_status >= 400 else None),
                                status_code=recorded_status,
                                endpoint_path=path,
                                request_kind=request_kind,
                                input_tokens=in_tok,
                                output_tokens=out_tok,
                                cached_tokens=cached_tok,
                                reasoning_tokens=reason_tok,
                                generation_ms=generation_ms,
                                draft_n=draft_n,
                                draft_n_accepted=draft_n_accepted,
                                predicted_tokens_per_second=predicted_tps,
                                prompt_tokens_per_second=prompt_tps,
                                work_size_meta=work_size_meta,
                                via_router_profile_id=via_router_profile_id,
                                output_audio=(
                                    acc.audio(content_type)
                                    if request_kind in _AUDIO_KINDS and recorded_status < 400
                                    else None
                                ),
                            )
                        )

            response.body_iterator = tee_and_record()
            return response

        # Non-streaming: buffer body, extract tokens.
        duration_ms = (time.monotonic() - start) * 1000
        response_payload, response = await _extract_response_payload_and_rebuild(response)
        error_message = _extract_error_message(response_payload, response.status_code)

        model_id = _extract_model_id(request=request, body=request_payload)
        # Bloque 20 — when a router redirected the request, use the target
        # profile that actually served the traffic as ``model_id``. That
        # way ``request_stats.model_id`` reflects reality and the
        # /stats/routing view breaks down redirects by real target.
        fed_target, fed_router = _federated_router_attribution(request)
        resolved_model_id = getattr(request.state, "resolved_model_id", None) or fed_target
        if resolved_model_id:
            model_id = resolved_model_id
        if model_id:
            in_tok, out_tok, cached_tok, reason_tok = _extract_usage_tokens_detailed(
                response_payload, request_kind=request_kind, request_payload=request_payload
            )
            generation_ms = _extract_generation_ms(response_payload)
            _update_engine_runtime_telemetry(request, response_payload)
            draft_n, draft_n_accepted, predicted_tps, prompt_tps = _extract_speculative_stats(
                response_payload
            )
            if prompt_tps is None:
                prompt_tps = _prefill_only_prompt_tps(request, request_kind, in_tok)
            # Transcription output: JSON formats are proxied above; for non-JSON
            # (srt/vtt/text) count words from the raw body. Empty/silence → 0
            # (not "—") so the row is consistent.
            if request_kind == "audio_transcription" and out_tok is None:
                if response_payload is None:
                    raw_body = getattr(response, "body", b"") or b""
                    out_tok = _approx_text_tokens(
                        _strip_subtitle_markup(raw_body.decode("utf-8", errors="ignore"))
                    )
                out_tok = out_tok or 0
            work_size_meta = _extract_work_size_meta(
                request_kind, request_payload, response_payload
            )
            # Transcripciones: si la respuesta no trae la duración del audio
            # (formato json o texto), la que el worker mandó en su cabecera.
            header_audio_s = getattr(request.state, "audio_seconds", None)
            if request_kind == "audio_transcription" and isinstance(header_audio_s, (int, float)):
                if not (isinstance(work_size_meta, dict) and work_size_meta.get("audio_seconds")):
                    work_size_meta = {
                        **(work_size_meta or {}),
                        "audio_seconds": float(header_audio_s),
                    }
            # Bloque 20 — router attribution. ``resolve_profile`` stashes
            # the router profile_id on ``request.state`` (BaseHTTPMiddleware
            # breaks contextvar propagation across ``call_next``, but
            # request.state is shared). NULL when the request didn't go
            # through a router — most direct calls.
            via_router = getattr(request.state, "via_router_profile_id", None) or fed_router
            spawn(
                _record_stat(
                    request=request,
                    model_id=model_id,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    error_message=error_message,
                    status_code=response.status_code,
                    endpoint_path=path,
                    request_kind=request_kind,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cached_tokens=cached_tok,
                    reasoning_tokens=reason_tok,
                    generation_ms=generation_ms,
                    draft_n=draft_n,
                    draft_n_accepted=draft_n_accepted,
                    predicted_tokens_per_second=predicted_tps,
                    prompt_tokens_per_second=prompt_tps,
                    work_size_meta=work_size_meta,
                    via_router_profile_id=via_router,
                    output_audio=(
                        (
                            getattr(response, "body", b"") or b"",
                            response.headers.get("content-type", ""),
                        )
                        if request_kind in _AUDIO_KINDS and response.status_code < 400
                        else None
                    ),
                )
            )

        return response


def _client_addr(request: Request) -> str | None:
    """Real client IP: X-Real-IP, then X-Forwarded-For (first hop), falling
    back to the direct peer.

    2026-10-31: whatever sits in front of Caddy for the public domain
    (reported against an opencode/ollama-python client showing up as a LAN
    IP in the dashboard) sends BOTH headers, but only X-Real-IP carries the
    genuine original client — X-Forwarded-For is set to that proxy's own
    address instead of being a real forwarding chain. X-Forwarded-For used
    to be checked first, so the dashboard showed the proxy's IP every time.
    X-Real-IP is a single trusted value by convention (not an
    attacker/intermediary-extensible chain like X-Forwarded-For), so
    preferring it first is also the safer default in general, not just a
    fix for this specific proxy's behavior.
    """
    xri = request.headers.get("x-real-ip")
    if xri:
        return xri.strip()[:64] or None
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()[:64] or None
    client = request.client
    return client.host if client else None


async def _extract_request_payload(request: Request) -> dict | None:
    """Read and cache JSON request body when present.

    Uploads and media (multipart, audio, video, image, octet-stream) are not
    read (BUG-37): they can never be a JSON object, and reading them pulled a
    whole audio file into memory a second time just for a parse that fails.
    Everything else is read as before; Starlette hands the endpoint the same
    cached bytes, so a JSON body is not held twice.
    """
    if admission.body_is_binary(request.headers.get("content-type", "")):
        return None
    try:
        body = await request.json()
        return body if isinstance(body, dict) else None
    except Exception:
        return None


def _extract_model_id(request: Request, body: dict | None) -> str | None:
    if body:
        model_id = body.get("model")
        if model_id:
            return str(model_id)

    model_id_from_state = getattr(request.state, "stats_model_id", None)
    if model_id_from_state:
        return str(model_id_from_state)

    return None


def _extract_last_payload_from_stream(body: bytes, content_type: str) -> dict | None:
    """Extract token-count fields from the final chunk of a streaming response.

    - SSE (text/event-stream): searches for the last `data: {...}` chunk that
      contains an OpenAI-style `usage` object.
    - NDJSON (application/x-ndjson): searches for the line with `"done": true`
      which carries Ollama's `prompt_eval_count` / `eval_count` fields.
    """
    if not body:
        return None
    text = body.decode("utf-8", errors="ignore")

    if "event-stream" in content_type:
        last_with_usage = None
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
                if isinstance(chunk, dict) and chunk.get("usage"):
                    last_with_usage = chunk
            except Exception:
                pass
        return last_with_usage

    if "ndjson" in content_type:
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if isinstance(chunk, dict) and chunk.get("done") is True:
                    return chunk
            except Exception:
                pass

    return None


def _extract_stream_error(body: bytes, content_type: str) -> tuple[str, str | None] | None:
    """Extract an error envelope emitted after streaming headers were sent."""
    if not body:
        return None
    text = body.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    if "event-stream" in content_type:
        lines = [line[6:].strip() for line in lines if line.startswith("data: ")]

    for line in reversed(lines):
        if not line or line == "[DONE]":
            continue
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or "error" not in payload:
            continue
        error = payload["error"]
        if isinstance(error, dict):
            return str(error.get("message") or "Streaming inference failed"), error.get("code")
        return str(error), None
    return None


def _stream_error_status(code: str | None) -> int:
    """Map an in-band streaming error to the status recorded in request stats."""
    return {
        "insufficient_vram": 409,
        "model_at_capacity": 429,
        "model_load_failed": 503,
        "model_load_timeout": 503,
        "generation_timeout": 504,
        "upstream_invalid_request": 400,
        "upstream_rate_limited": 429,
        "upstream_server_error": 502,
        "stream_error": 502,
    }.get(code, 500)


async def _extract_response_payload_and_rebuild(response: Response) -> tuple[dict | None, Response]:
    if isinstance(response, StreamingResponse):
        return None, response

    body = getattr(response, "body", b"") or b""
    if not body and getattr(response, "body_iterator", None) is not None:
        chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        body = b"".join(chunks)
        response = Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=response.background,
        )

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type or not body:
        return None, response

    try:
        payload = json.loads(body)
        if isinstance(payload, dict):
            return payload, response
    except Exception:
        pass
    return None, response


_AUDIO_KINDS = {"tts", "music"}


async def _audio_duration_s(
    body: bytes, content_type: str, total_size: int | None = None
) -> float | None:
    """Seconds of audio in a generated response body, or None.

    WAV from its header, raw PCM (24 kHz mono 16-bit, the OpenAI speech
    default) from its size, anything else (mp3, opus, aac, flac) with
    ffprobe — a TTS response is a few hundred KB, and this runs after the
    response was sent.

    ``total_size``: the streamed body was longer than the streaming tee
    keeps (``_StreamAccumulator._MAX_AUDIO_BODY``) and ``body`` is only its
    head. WAV and PCM still come out right from the header and the size;
    a compressed format cannot be measured from a head, so it gives None.
    """
    if not body:
        return None
    truncated = total_size is not None and total_size > len(body)
    full_size = total_size if truncated else len(body)
    ctype = (content_type or "").split(";")[0].strip().lower()
    if body[:4] == b"RIFF" and body[8:12] == b"WAVE":
        import struct

        try:
            byte_rate = struct.unpack("<I", body[28:32])[0]
            data_at = body.find(b"data", 12)
            if byte_rate > 0 and data_at > 0:
                declared = struct.unpack("<I", body[data_at + 4 : data_at + 8])[0]
                size = full_size - (data_at + 8)
                # A streamed WAV declares a placeholder size: trust the bytes.
                frames = size if declared in (0, 0xFFFFFFFF) or declared > size else declared
                return round(frames / byte_rate, 3)
        except struct.error:
            return None
    if ctype in ("audio/pcm", "audio/l16"):
        return round(full_size / (24000 * 2), 3)
    if not ctype.startswith("audio/") or truncated:
        return None
    import shutil

    if not shutil.which("ffprobe"):
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", "-i", "pipe:0",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        out, _ = await asyncio.wait_for(proc.communicate(body), timeout=5)
        value = float(out.decode().strip() or 0)
        return round(value, 3) if value > 0 else None
    except (TimeoutError, ValueError, OSError):
        return None


async def _record_stat(
    request: Request,
    model_id: str,
    started_at: datetime,
    duration_ms: float,
    error_message: str | None,
    status_code: int,
    endpoint_path: str,
    request_kind: str,
    input_tokens: int | None,
    output_tokens: int | None,
    remote_node_id: str | None = None,
    work_size_meta: dict | None = None,
    via_router_profile_id: str | None = None,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    generation_ms: int | None = None,
    draft_n: int | None = None,
    draft_n_accepted: int | None = None,
    predicted_tokens_per_second: float | None = None,
    prompt_tokens_per_second: float | None = None,
    output_audio: tuple[bytes, str] | tuple[bytes, str, int] | None = None,
) -> None:
    """Write a RequestStat row to the database and update Prometheus counters.

    ``output_audio``: the generated audio (body, content type) of a TTS or
    music request, whose length goes into ``work_size_meta.audio_seconds``.
    """
    try:
        if output_audio is not None:
            seconds = await _audio_duration_s(*output_audio)
            if seconds:
                work_size_meta = {**(work_size_meta or {}), "audio_seconds": seconds}
        gpu_index: int | None = None
        serving_gpus: list[int] = []
        backend_type: str | None = None
        energy_wh: float | None = None
        # ``model_id`` of the row is the logical model (``models.model_key``),
        # whatever name the client used — profile, alias, deployment id; that
        # name goes to ``requested_model``. Before migration 0034 the column
        # mixed public names, canonical ids and worker keys.
        requested_model = model_id
        stat_model_id = model_id
        try:
            mm = request.app.state.model_manager
            resolved_model_id = await _resolve_inflight_model_id(request, model_id)
            state = await mm.get_state(resolved_model_id or model_id)
            if state is None:
                # Fallback: resolve by backend_model_id alias (e.g. "mistral:7b" → "ollama/mistral:7b")
                states = await mm.list_states(include_derived=True)
                state = next(
                    (
                        s
                        for s in states
                        if s.model_id == resolved_model_id or s.backend_model_id == model_id
                    ),
                    None,
                )
            if state:
                backend_type = state.backend_type
                if getattr(state, "model_key", None):
                    stat_model_id = state.model_key
                # The worker that served it: a preset's worker
                # (``<model>::<hash>``) holds the GPUs, not the base model,
                # which is usually not loaded at all — 86 % of the chat rows
                # had no GPU and therefore no energy (2026-11-18).
                worker_key = getattr(request.state, "runtime_worker_key", None) or getattr(
                    request.state, "inflight_model_id", None
                )
                worker_state = await mm.get_state(worker_key) if worker_key else None
                if worker_state is not None and worker_state.current_gpu:
                    serving_gpus = list(worker_state.current_gpu)
                elif state.current_gpu:
                    serving_gpus = list(state.current_gpu)
                if serving_gpus:
                    gpu_index = serving_gpus[0]
                elif backend_type == "ollama":
                    # Ollama manages its own GPU assignment; use preferred_gpu or the
                    # system default so energy estimates are at least approximated.
                    gpu_index = (
                        state.preferred_gpu
                        if state.preferred_gpu is not None
                        else settings.default_gpu_index
                    )
        except Exception:
            pass

        # Estimate energy from the power draw of the GPUs that served it ×
        # its duration (every GPU of a split model, not only the first). It is
        # the whole card's draw: concurrent requests on it each count it.
        if gpu_index is not None:
            try:
                gm = request.app.state.gpu_manager
                watts = 0.0
                for index in serving_gpus or [gpu_index]:
                    gpu_state = gm._states.get(index)
                    if gpu_state and gpu_state.power_draw_w > 0:
                        watts += float(gpu_state.power_draw_w)
                if watts > 0:
                    energy_wh = watts * duration_ms / 1000.0 / 3600.0
            except Exception:
                pass

        from ocabra.api.metrics import record_request, record_tokens
        from ocabra.database import AsyncSessionLocal
        from ocabra.db.stats import RequestStat

        record_request(
            model_id=model_id,
            duration_s=max(duration_ms, 0.0) / 1000.0,
            status="error" if error_message else "ok",
        )
        record_tokens(
            model_id=model_id,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
        )

        # Extract user_id set by auth dependency (stored on request.state by get_current_user).
        auth_user = getattr(request.state, "auth_user", None)
        user_id = auth_user.user_id if auth_user and not auth_user.is_anonymous else None

        import uuid as _uuid

        parsed_user_id: _uuid.UUID | None = None
        if user_id:
            try:
                parsed_user_id = _uuid.UUID(str(user_id))
            except (ValueError, AttributeError):
                pass

        key_group_id = auth_user.key_group_id if auth_user else None
        parsed_group_id: _uuid.UUID | None = None
        if key_group_id:
            try:
                parsed_group_id = _uuid.UUID(str(key_group_id))
            except (ValueError, AttributeError):
                pass

        api_key_name = auth_user.api_key_name if auth_user else None

        client_addr = _client_addr(request)
        user_agent = request.headers.get("user-agent") or None
        try:
            from ocabra.api.openai._deps import is_batch_origin_request

            is_batch = is_batch_origin_request(request)
        except Exception:  # noqa: BLE001 — stats recording must never 500 a real request
            is_batch = False
        if user_agent and len(user_agent) > 512:
            user_agent = user_agent[:512]

        # Federation: check if the request was proxied to a remote node.
        if remote_node_id is None:
            remote_node_id = getattr(request.state, "federation_remote_node_id", None)
        remote_node_name, inbound_from, federation_route = _federation_fields(
            request, auth_user, remote_node_id
        )

        # Pull agent context from the executor's contextvars (only set during
        # an agent invocation).  This stamps the *root* request_stat row with
        # the agent_id so the stats endpoints can filter by agent.
        agent_id_ctx: _uuid.UUID | None = None
        try:
            from ocabra.agents.executor import current_agent_id

            agent_id_ctx = current_agent_id.get()
        except Exception:
            agent_id_ctx = None

        # Feed en vivo del dashboard: la petición terminada, unos segundos.
        # Nunca con IP ni user agent: lo ve cualquier usuario.
        try:
            from ocabra.stats import live_feed

            telemetry_now = getattr(request.state, "runtime_telemetry", None)
            live_feed.record_finished(
                {
                    "requestId": getattr(request.state, "inflight_request_id", None),
                    "model": requested_model,
                    "backendType": backend_type,
                    "gpuIndex": gpu_index,
                    "requestKind": request_kind,
                    "statusCode": status_code,
                    # Zero data: who asked and what went wrong (error text can
                    # quote the input) stay out of the feed too.
                    "error": None
                    if privacy.zero_data()
                    else (error_message or None) and str(error_message)[:300],
                    "username": None
                    if privacy.zero_data()
                    else (getattr(auth_user, "username", None) if auth_user else None),
                    "apiKeyName": None if privacy.zero_data() else api_key_name,
                    "durationMs": int(duration_ms),
                    "inputTokens": input_tokens,
                    "cachedTokens": cached_tokens,
                    "outputTokens": output_tokens,
                    "promptTokensPerSecond": prompt_tokens_per_second,
                    "predictedTokensPerSecond": predicted_tokens_per_second,
                    "ttftMs": getattr(telemetry_now, "ttft_ms", None),
                    "queueMs": getattr(telemetry_now, "queue_ms", None),
                    # Transcripciones: con durationMs da el factor de tiempo real.
                    # Federation: where it ran (None = here), who sent it
                    # to us, and why it went where it did.
                    "node": remote_node_name,
                    "inboundFrom": inbound_from,
                    "route": federation_route,
                    "audioSeconds": (
                        round(float(work_size_meta["audio_seconds"]), 1)
                        if isinstance(work_size_meta, dict)
                        and isinstance(work_size_meta.get("audio_seconds"), (int, float))
                        else None
                    ),
                }
            )
        except Exception:  # noqa: BLE001 — el feed nunca rompe el registro
            pass

        if privacy.zero_data():
            # Nothing about this request is stored (core/privacy.py): only
            # its metrics, in memory, for the last 10 (stats/recent_memory.py).
            try:
                import uuid as _uuid

                from ocabra.stats import recent_memory

                recent_memory.record(
                    {
                        "id": str(_uuid.uuid4()),
                        "modelId": stat_model_id,
                        "requestedModel": requested_model,
                        "backendType": backend_type,
                        "requestKind": request_kind,
                        "endpointPath": endpoint_path,
                        "statusCode": status_code,
                        "startedAt": started_at.isoformat() if started_at else None,
                        "durationMs": int(duration_ms),
                        "inputTokens": input_tokens,
                        "outputTokens": output_tokens,
                        "cachedTokens": cached_tokens,
                        "promptTokensPerSecond": prompt_tokens_per_second,
                        "predictedTokensPerSecond": predicted_tokens_per_second,
                        "node": remote_node_name,
                        "inboundFrom": inbound_from,
                        "route": federation_route,
                        "viaRouter": via_router_profile_id,
                        "runtimeTelemetry": (
                            telemetry.as_dict()
                            if isinstance(
                                (telemetry := getattr(request.state, "runtime_telemetry", None)),
                                RequestRuntimeTelemetry,
                            )
                            else None
                        ),
                        "inFlight": False,
                    }
                )
            except Exception:  # noqa: BLE001 — never break the request for this
                pass
            return

        async with AsyncSessionLocal() as session:
            stat = RequestStat(
                model_id=stat_model_id,
                requested_model=requested_model,
                backend_type=backend_type,
                request_kind=request_kind,
                endpoint_path=endpoint_path,
                status_code=status_code,
                gpu_index=gpu_index,
                started_at=started_at,
                duration_ms=int(duration_ms),
                generation_ms=generation_ms,
                draft_n=draft_n,
                draft_n_accepted=draft_n_accepted,
                predicted_tokens_per_second=predicted_tokens_per_second,
                prompt_tokens_per_second=prompt_tokens_per_second,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                energy_wh=energy_wh,
                error=error_message,
                user_id=parsed_user_id,
                group_id=parsed_group_id,
                api_key_name=api_key_name,
                client_addr=client_addr,
                user_agent=user_agent,
                remote_node_id=remote_node_id,
                remote_node_name=remote_node_name,
                federation_inbound_from=inbound_from,
                federation_route=federation_route,
                agent_id=agent_id_ctx,
                # Bloque 20 — Duration estimator + Router audit.
                work_size_meta=work_size_meta,
                runtime_telemetry=(
                    telemetry.as_dict()
                    if isinstance(
                        (telemetry := getattr(request.state, "runtime_telemetry", None)),
                        RequestRuntimeTelemetry,
                    )
                    else None
                ),
                via_router_profile_id=via_router_profile_id,
                runtime_preset=getattr(request.state, "runtime_preset_key", None),
                runtime_key=getattr(request.state, "runtime_key", None),
                effective_context=getattr(request.state, "runtime_effective_context", None),
                is_batch=is_batch,
            )
            session.add(stat)
            await session.commit()
    except Exception as exc:
        logger.warning("stats_write_failed", error=str(exc))


def record_admission_rejection(conn: Any, rejected: admission.AdmissionRejection) -> None:
    """Record a request admission turned away (503 maintenance, 429 cap).

    ``AdmissionMiddleware``'s ``on_reject`` (wired in ``ocabra.main``): the
    request never reaches this middleware, so the row is written from here,
    the same one it always was — keyed by the worker, no token counts.
    """
    spawn(
        _record_stat(
            request=conn,
            model_id=rejected.model_id,
            started_at=rejected.started_at,
            duration_ms=rejected.duration_ms,
            error_message=rejected.error_code,
            status_code=rejected.status_code,
            endpoint_path=rejected.path,
            request_kind=rejected.request_kind,
            input_tokens=None,
            output_tokens=None,
        )
    )


async def record_realtime_request(
    websocket: object,
    *,
    session_id: str,
    model_id: str,
    started_at: datetime,
    duration_ms: float,
    request_kind: str,
    status_code: int,
    error_message: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    work_size_meta: dict | None = None,
) -> None:
    """Persist one Realtime session/backend operation and emit its audit log.

    A Starlette WebSocket exposes the same app/state/headers/client attributes
    consumed by ``_record_stat``. Keeping this adapter here makes Realtime use
    the same attribution, energy and Prometheus paths as ordinary HTTP calls.
    """
    await _record_stat(
        request=websocket,  # type: ignore[arg-type]
        model_id=model_id,
        started_at=started_at,
        duration_ms=duration_ms,
        error_message=error_message,
        status_code=status_code,
        endpoint_path="/v1/realtime",
        request_kind=request_kind,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        work_size_meta=work_size_meta,
    )

    state = getattr(websocket, "state", None)
    auth_user = getattr(state, "auth_user", None)
    logger.info(
        "realtime_request_recorded",
        session_id=session_id,
        model_id=model_id,
        request_kind=request_kind,
        status_code=status_code,
        duration_ms=int(duration_ms),
        username=None if privacy.zero_data() else getattr(auth_user, "username", None),
        api_key_name=None if privacy.zero_data() else getattr(auth_user, "api_key_name", None),
        client_addr=None if privacy.zero_data() else _client_addr(websocket),  # type: ignore[arg-type]
        error=None if privacy.zero_data() else error_message,
    )


async def record_direct_request(
    request: Request,
    *,
    model_id: str,
    started_at: datetime,
    duration_ms: float,
    endpoint_path: str,
    request_kind: str,
    status_code: int,
    error_message: str | None = None,
    work_size_meta: dict | None = None,
) -> None:
    """Record one request from its endpoint, for paths the middleware skips.

    ``/ocabra/video/upscale`` is outside ``/v1`` and uploads whole video
    segments, which the middleware would read into memory again; it records
    itself instead (BUG-20).
    """
    await _record_stat(
        request=request,
        model_id=model_id,
        started_at=started_at,
        duration_ms=duration_ms,
        error_message=error_message,
        status_code=status_code,
        endpoint_path=endpoint_path,
        request_kind=request_kind,
        input_tokens=None,
        output_tokens=None,
        work_size_meta=work_size_meta,
    )


def _federated_router_attribution(request: Request) -> tuple[str | None, str | None]:
    """``(target, router)`` when a router request was proxied to a peer.

    The local path sets ``resolved_model_id``/``via_router_profile_id`` on
    ``request.state``; a request a peer served never reaches it, so the stats
    recorded the router's first *local* target (ornith for the pod's
    nemotron) and no router. ``resolve_federated`` notes both instead.
    """
    info = getattr(request.state, "federation_info", None) or {}
    if info.get("decision") != "remote":
        return None, None
    return info.get("router_target"), info.get("router")


def _federation_fields(
    request: Request, auth_user: Any, remote_node_id: str | None
) -> tuple[str | None, str | None, str | None]:
    """``(remote_node_name, inbound_from, route)`` for a finished request.

    Read from ``request.state.federation_info`` (filled by
    ``resolve_federated`` and the proxy helpers) and the federation manager:
    the serving peer's *name* (its id alone loses meaning once the peer is
    deleted), the peer that sent us the request, and the decision reason —
    ``fallback_local`` when a peer was chosen but failed and we served it.
    """
    info = getattr(request.state, "federation_info", None) or {}
    fm = getattr(getattr(request, "app", None), "state", None)
    fm = getattr(fm, "federation_manager", None)
    peers = []
    if fm is not None:
        try:
            peers = fm.get_all_peers()
        except Exception:  # noqa: BLE001 — stats must not fail on this
            peers = []

    name = None
    if remote_node_id:
        name = info.get("peer_name") or next(
            (p.name for p in peers if p.peer_id == str(remote_node_id)), None
        )

    inbound = None
    hop = info.get("inbound_hop")
    if hop:
        inbound = next(
            (p.name for p in peers if str(p.remote_node_id or "") == hop or p.peer_id == hop),
            None,
        )
    username = str(getattr(auth_user, "username", "") or "")
    if inbound is None and username.startswith("federation:"):
        inbound = username.split(":", 1)[1] or None
    if inbound is None and hop:
        inbound = hop[:256]

    route = None
    if info.get("fallback"):
        route = "fallback_local"
    elif info.get("reason") and (remote_node_id or info.get("reason") != "local_loaded"):
        route = str(info["reason"])[:64]
    return name, inbound, route


