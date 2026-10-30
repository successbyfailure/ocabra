"""
Stats middleware — records inference request metrics.

Tracks OpenAI-compatible `/v1/*` and Ollama-compatible inference routes under
`/api/*` so the stats page reflects real usage regardless of client protocol.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ocabra.config import settings
from ocabra.core.model_manager_helpers import compute_worker_key
from ocabra.core.runtime_coordinator import _text_size
from ocabra.core.worker_pool import InferenceTimeoutError, WorkerDegradedError


async def _resolve_inflight_auth_meta(request: Request) -> dict:
    """Best-effort auth lookup for the stats middleware.

    Runs BEFORE ``call_next``, so ``request.state.auth_user`` isn't populated
    yet (that happens inside endpoint handlers). We re-resolve the bearer
    token here solely so ``/stats/recent`` can show *who* is driving the
    currently in-flight request instead of a bare "anónimo" row. Failures
    are swallowed — this never gates the request.
    """
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        return {}
    token = header[7:].strip()
    if not token:
        return {}
    try:
        from ocabra.api._deps_auth import _resolve_api_key
        from ocabra.database import AsyncSessionLocal
    except Exception:  # noqa: BLE001
        return {}
    try:
        async with AsyncSessionLocal() as session:
            ctx = await _resolve_api_key(token, session)
    except Exception:  # noqa: BLE001
        return {}
    if ctx is None or ctx.is_anonymous:
        return {}
    return {
        "user_id": ctx.user_id,
        "username": ctx.username,
        "api_key_name": ctx.api_key_name,
    }

logger = structlog.get_logger(__name__)

_TRACKED_API_PATHS = {
    "/api/chat",
    "/api/generate",
    "/api/embeddings",
    "/api/embed",
}


def _classify_request_kind(path: str) -> str:
    mapping = {
        "/v1/chat/completions": "chat",
        "/v1/completions": "completion",
        "/v1/embeddings": "embedding",
        "/v1/images/generations": "image_generation",
        "/v1/images/edits": "image_generation",
        "/v1/audio/transcriptions": "audio_transcription",
        "/v1/audio/speech": "tts",
        "/v1/rerank": "rerank",
        "/api/chat": "ollama_chat",
        "/api/generate": "ollama_generate",
        "/api/embeddings": "ollama_embedding",
        "/api/embed": "ollama_embedding",
    }
    return mapping.get(path, "other")


def _approx_text_tokens(text: object) -> int | None:
    """Rough token estimate (~word count) for endpoints without a usage payload."""
    if isinstance(text, str) and text.strip():
        return len(text.split())
    return None


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
        text = request_payload.get("input") if request_payload else None
        if isinstance(text, str) and text:
            return {"text_chars": len(text)}
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
        return meta or None

    if request_kind == "audio_transcription":
        # Whisper surfaces ``duration`` (seconds of decoded audio) on the
        # verbose_json response. Faster-whisper and Nvidia backends do the
        # same. When absent, we leave it blank; a follow-up server-side
        # probe would be more effort than it's worth for now.
        if response_payload is None:
            return None
        for k in ("duration", "audio_seconds"):
            v = response_payload.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return {"audio_seconds": float(v)}
        return None

    # ACE-Step music generation lands under the openai /v1/audio/speech
    # path with a ``duration_seconds`` field; other TTS callers only care
    # about text_chars (above).
    if request_kind == "tts" and request_payload:
        dur = request_payload.get("duration_seconds") or request_payload.get("duration")
        if isinstance(dur, (int, float)) and dur > 0:
            return {"duration_seconds": float(dur)}

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
    elif request_kind in ("rerank", "ollama_embedding"):
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
_TEXT_GEN_KINDS = {"chat", "completion", "ollama_chat", "ollama_generate"}


def _count_input_tokens(request_payload: dict | None, request_kind: str) -> int | None:
    """Approx input tokens (~word count) from the request, for text-gen kinds whose
    streamed response lacked a usage payload.

    Includes ``tools`` (function-calling schemas), same as the context-preset
    estimator (``estimate_request_token_budget`` / ``_text_size`` in
    runtime_coordinator.py). 2026-10-29: without this, a tool-heavy request
    (e.g. Home Assistant's large function schemas) showed as "~128 tok
    entrada" live while the real prompt — confirmed via the preset
    selector's own ``required_tokens`` — was 18,240. That gap made a request
    legitimately busy on a huge prompt look suspiciously idle."""
    if not isinstance(request_payload, dict):
        return None
    if request_kind in ("chat", "ollama_chat"):
        total = 0
        for m in request_payload.get("messages") or []:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, str):
                total += len(content.split())
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total += len(part["text"].split())
        _tools_chars, tools_words = _text_size(request_payload.get("tools"))
        total += tools_words
        return total or None
    if request_kind in ("completion", "ollama_generate"):
        return _approx_text_tokens(request_payload.get("prompt"))
    return None


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
    return words if found else None


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
                reasoning_text = (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                )
            if reasoning_text:
                reason_tok = _approx_text_tokens(reasoning_text)

    # Whisper-style: {"text": "..."} — use word count as output proxy.
    if request_kind == "audio_transcription" and out_tok is None and isinstance(payload, dict):
        out_tok = _approx_text_tokens(payload.get("text"))

    in_tok, out_tok = _apply_token_kind_defaults(
        in_tok, out_tok, request_kind, request_payload
    )
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
        return None, None, None, None

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
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import asyncio

        path = request.url.path
        if not _should_track(path):
            return await call_next(request)

        start = time.monotonic()
        started_at = datetime.now(UTC)
        request_payload = await _extract_request_payload(request)
        request_kind = _classify_request_kind(path)

        # Track in-flight requests so the pressure eviction loop can avoid
        # evicting models that are currently serving a request.
        requested_model_id = _extract_model_id(request=request, body=request_payload)
        inflight_model_id = await _resolve_inflight_model_id(request, requested_model_id)
        inflight_request_id: str | None = None
        try:
            mm = request.app.state.model_manager
        except AttributeError:
            mm = None
        if mm and inflight_model_id and mm.is_in_maintenance(inflight_model_id):
            # Maintenance window: reject new requests fast and clear so
            # whoever is testing against this model doesn't have real
            # traffic contending for the same GPU/worker. Requests already
            # admitted before the window opened are untouched — this check
            # runs before try_begin_request ever marks one in-flight.
            asyncio.create_task(
                _record_stat(
                    request=request,
                    model_id=inflight_model_id,
                    started_at=started_at,
                    duration_ms=(time.monotonic() - start) * 1000,
                    error_message="model_maintenance",
                    status_code=503,
                    endpoint_path=path,
                    request_kind=request_kind,
                    input_tokens=None,
                    output_tokens=None,
                )
            )
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "30"},
                content={
                    "error": {
                        "message": (
                            f"Model '{inflight_model_id}' is under maintenance. "
                            "Please retry shortly."
                        ),
                        "type": "server_error",
                        "code": "model_maintenance",
                    }
                },
            )
        if mm and inflight_model_id:
            auth_meta = await _resolve_inflight_auth_meta(request)
            inflight_request_id = mm.try_begin_request(
                inflight_model_id,
                settings.max_inflight_per_model,
                user_id=auth_meta.get("user_id"),
                username=auth_meta.get("username"),
                api_key_name=auth_meta.get("api_key_name"),
                # ``requested_model_id`` is the public ``model`` field the
                # client sent — a profile id ("qwen3.8:27b") or router id
                # ("recomendado-ha-chatbots"). We keep it alongside the
                # resolved worker key so the UI can show the friendly name
                # instead of ``ollama/qwen3.8:27b::<hash>``.
                profile_id=requested_model_id,
                # Approximate, available immediately (exact prompt_tokens
                # isn't known until the backend responds) — same heuristic
                # used for the completed-request row further down.
                input_tokens_estimate=_count_input_tokens(request_payload, request_kind),
            )
            if inflight_request_id is None:
                # Admission control: the worker is already at capacity. Reject
                # rather than pile on so one client's burst can't starve others.
                effective_cap = mm.effective_inflight_cap(
                    inflight_model_id, settings.max_inflight_per_model
                )
                asyncio.create_task(
                    _record_stat(
                        request=request,
                        model_id=inflight_model_id,
                        started_at=started_at,
                        duration_ms=(time.monotonic() - start) * 1000,
                        error_message="model_at_capacity",
                        status_code=429,
                        endpoint_path=path,
                        request_kind=request_kind,
                        input_tokens=None,
                        output_tokens=None,
                    )
                )
                # A low-concurrency worker (e.g. a single-slot Ollama model)
                # drains its queue far slower than the flat 2s this used to
                # suggest unconditionally — tell low-slot callers to back off
                # longer so they don't hammer an already-saturated worker.
                retry_after_s = 2 if effective_cap > 4 else 10
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry_after_s)},
                    content={
                        "error": {
                            "message": (
                                f"Model '{inflight_model_id}' is at capacity "
                                f"({effective_cap} concurrent requests for this "
                                "worker). Please retry shortly."
                            ),
                            "type": "rate_limit_error",
                            "code": "model_at_capacity",
                        }
                    },
                )
            request.state.inflight_model_id = inflight_model_id
            request.state.inflight_request_id = inflight_request_id

        try:
            response = await call_next(request)
        except InferenceTimeoutError as exc:
            if mm and inflight_model_id:
                mm.end_request(
                    getattr(request.state, "inflight_model_id", inflight_model_id),
                    inflight_request_id,
                )
            model_id = _extract_model_id(request=request, body=request_payload)
            if model_id:
                asyncio.create_task(
                    _record_stat(
                        request=request,
                        model_id=model_id,
                        started_at=started_at,
                        duration_ms=(time.monotonic() - start) * 1000,
                        error_message=str(exc),
                        status_code=504,
                        endpoint_path=path,
                        request_kind=request_kind,
                        input_tokens=None,
                        output_tokens=None,
                    )
                )
            if path.startswith("/api/"):
                return JSONResponse(
                    status_code=504,
                    content={"error": str(exc)},
                )
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "server_error",
                        "code": "generation_timeout",
                    }
                },
            )
        except WorkerDegradedError as exc:
            # Circuit breaker tripped: this worker just failed several times
            # in a row in a way that means its process died (see
            # worker_pool._looks_like_dead_backend). Fail fast with a clear
            # 503 instead of letting the request sit behind a dead runner for
            # a full upstream timeout only to fail the same way anyway.
            if mm and inflight_model_id:
                mm.end_request(
                    getattr(request.state, "inflight_model_id", inflight_model_id),
                    inflight_request_id,
                )
            model_id = _extract_model_id(request=request, body=request_payload)
            if model_id:
                asyncio.create_task(
                    _record_stat(
                        request=request,
                        model_id=model_id,
                        started_at=started_at,
                        duration_ms=(time.monotonic() - start) * 1000,
                        error_message=str(exc),
                        status_code=503,
                        endpoint_path=path,
                        request_kind=request_kind,
                        input_tokens=None,
                        output_tokens=None,
                    )
                )
            if path.startswith("/api/"):
                return JSONResponse(
                    status_code=503,
                    headers={"Retry-After": str(settings.worker_circuit_breaker_cooldown_s)},
                    content={"error": str(exc)},
                )
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": str(settings.worker_circuit_breaker_cooldown_s)},
                content={
                    "error": {
                        "message": str(exc),
                        "type": "server_error",
                        "code": "worker_degraded",
                    }
                },
            )
        except asyncio.CancelledError:
            # Admin or owner cancelled via /ocabra/stats/requests/{id}/cancel.
            # ``CancelledError`` is BaseException on 3.8+, so the generic
            # ``except Exception`` below never catches it. Clean up the
            # in-flight slot (otherwise the counter leaks until the ActiveRequest
            # ages out) and record the stat as a 499 — the canonical "client
            # closed request" code — so the UI shows the cancelled turn.
            if mm and inflight_model_id:
                mm.end_request(
                    getattr(request.state, "inflight_model_id", inflight_model_id),
                    inflight_request_id,
                )
            model_id = _extract_model_id(request=request, body=request_payload)
            if model_id:
                asyncio.create_task(
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
            if mm and inflight_model_id:
                mm.end_request(
                    getattr(request.state, "inflight_model_id", inflight_model_id),
                    inflight_request_id,
                )
            model_id = _extract_model_id(request=request, body=request_payload)
            if model_id:
                asyncio.create_task(
                    _record_stat(
                        request=request,
                        model_id=model_id,
                        started_at=started_at,
                        duration_ms=(time.monotonic() - start) * 1000,
                        error_message=str(exc),
                        status_code=500,
                        endpoint_path=path,
                        request_kind=request_kind,
                        input_tokens=None,
                        output_tokens=None,
                    )
                )
            raise

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
            resolved_model_id = getattr(request.state, "resolved_model_id", None)
            if resolved_model_id:
                model_id = resolved_model_id
            via_router_profile_id = getattr(request.state, "via_router_profile_id", None)
            status_code = response.status_code
            original_iterator = response.body_iterator

            async def tee_and_record():
                chunks: list[bytes] = []
                error_msg: str | None = None
                last_progress_at = 0.0
                try:
                    async for chunk in original_iterator:
                        if isinstance(chunk, str):
                            chunk = chunk.encode("utf-8")
                        chunks.append(chunk)
                        yield chunk
                        # Live progress for the model/GPU card "is this hung
                        # or still working" indicator (2026-10-29). Throttled
                        # to ~2/s — re-parsing the whole accumulated body on
                        # every chunk would be O(n^2) over a long stream.
                        if mm and inflight_request_id:
                            now_mono = time.monotonic()
                            if now_mono - last_progress_at >= 0.5:
                                last_progress_at = now_mono
                                out_tok = _stream_output_tokens(
                                    b"".join(chunks), content_type, include_reasoning=True
                                )
                                mm.update_request_progress(
                                    inflight_request_id,
                                    output_tokens_estimate=out_tok,
                                )
                except Exception as exc:
                    error_msg = str(exc)
                    raise
                finally:
                    duration_ms = (time.monotonic() - start) * 1000
                    if mm and inflight_model_id:
                        mm.end_request(
                            getattr(
                                request.state, "inflight_model_id", inflight_model_id
                            ),
                            inflight_request_id,
                        )
                    if model_id:
                        all_body = b"".join(chunks)
                        stream_error = _extract_stream_error(all_body, content_type)
                        recorded_status = status_code
                        if error_msg:
                            recorded_status = 500
                        elif stream_error is not None:
                            error_msg, error_code = stream_error
                            recorded_status = _stream_error_status(error_code)
                        last_payload = _extract_last_payload_from_stream(all_body, content_type)
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
                                out_tok = _stream_output_tokens(all_body, content_type)
                            if in_tok is None:
                                in_tok = _count_input_tokens(request_payload, request_kind)
                        work_size_meta = _extract_work_size_meta(
                            request_kind, request_payload, last_payload
                        )
                        asyncio.create_task(
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
                            )
                        )

            response.body_iterator = tee_and_record()
            return response

        # Non-streaming: end in-flight immediately, buffer body, extract tokens.
        if mm and inflight_model_id:
            mm.end_request(
                getattr(request.state, "inflight_model_id", inflight_model_id),
                inflight_request_id,
            )

        duration_ms = (time.monotonic() - start) * 1000
        response_payload, response = await _extract_response_payload_and_rebuild(response)
        error_message = _extract_error_message(response_payload, response.status_code)

        model_id = _extract_model_id(request=request, body=request_payload)
        # Bloque 20 — when a router redirected the request, use the target
        # profile that actually served the traffic as ``model_id``. That
        # way ``request_stats.model_id`` reflects reality and the
        # /stats/routing view breaks down redirects by real target.
        resolved_model_id = getattr(request.state, "resolved_model_id", None)
        if resolved_model_id:
            model_id = resolved_model_id
        if model_id:
            in_tok, out_tok, cached_tok, reason_tok = _extract_usage_tokens_detailed(
                response_payload, request_kind=request_kind, request_payload=request_payload
            )
            generation_ms = _extract_generation_ms(response_payload)
            draft_n, draft_n_accepted, predicted_tps, prompt_tps = (
                _extract_speculative_stats(response_payload)
            )
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
            # Bloque 20 — router attribution. ``resolve_profile`` stashes
            # the router profile_id on ``request.state`` (BaseHTTPMiddleware
            # breaks contextvar propagation across ``call_next``, but
            # request.state is shared). NULL when the request didn't go
            # through a router — most direct calls.
            via_router = getattr(request.state, "via_router_profile_id", None)
            asyncio.create_task(
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
                )
            )

        return response


def _client_addr(request: Request) -> str | None:
    """Real client IP: X-Forwarded-For (first hop) / X-Real-IP behind Caddy,
    falling back to the direct peer."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()[:64] or None
    xri = request.headers.get("x-real-ip")
    if xri:
        return xri.strip()[:64] or None
    client = request.client
    return client.host if client else None


def _should_track(path: str) -> bool:
    if path.startswith("/v1/"):
        return True
    return path in _TRACKED_API_PATHS


async def _extract_request_payload(request: Request) -> dict | None:
    """Read and cache JSON request body when present."""
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


async def _resolve_inflight_model_id(request: Request, requested: str | None) -> str | None:
    """Resolve a public profile/model id to the worker key used by ModelManager.

    Statistics retain the public id from the request body. Only lifecycle and
    admission tracking use this resolved key.
    """
    if not requested:
        return None

    registry = getattr(request.app.state, "profile_registry", None)
    if registry is not None:
        try:
            profile = await registry.get(requested)
            if profile is None:
                # Hidden compatibility aliases (e.g. "gemma4:26b" -> the
                # renamed "gemma4:26b-llamacpp" profile after a backend
                # migration) aren't profile_ids themselves, so a plain
                # registry.get() misses them. Without this, stats blamed
                # the alias's OLD backend_model_id match in the fallback
                # below — attributing every request to the wrong backend
                # even though the real request correctly used the new one.
                # See resolve_profile() in api/openai/_deps.py for the
                # same alias-then-direct-lookup order this mirrors.
                alias_resolver = getattr(registry, "resolve_alias", None)
                alias_target = await alias_resolver(requested) if alias_resolver is not None else None
                profile = alias_target[0] if alias_target is not None else None
            if profile is not None and profile.enabled:
                if getattr(profile, "profile_kind", "execution") == "router":
                    for target_id in profile.routing_targets or []:
                        target = await registry.get(target_id)
                        if target is not None and target.enabled and target.base_model_id:
                            return compute_worker_key(
                                target.base_model_id,
                                target.load_overrides,
                            )
                    return None
                return compute_worker_key(profile.base_model_id, profile.load_overrides)

            # Legacy canonical model ids resolve through their default profile.
            if "/" in requested:
                profiles = await registry.list_by_model(requested)
                profile = next(
                    (item for item in profiles if item.enabled and item.is_default),
                    None,
                )
                if profile is None:
                    profile = next((item for item in profiles if item.enabled), None)
                if profile is not None:
                    return compute_worker_key(
                        profile.base_model_id,
                        profile.load_overrides,
                    )
        except Exception as exc:  # noqa: BLE001 - tracking must not block inference
            logger.warning(
                "inflight_model_resolution_failed",
                requested_model_id=requested,
                error=str(exc),
            )

    model_manager = getattr(request.app.state, "model_manager", None)
    if model_manager is not None:
        try:
            state = await model_manager.get_state(requested)
            if state is not None:
                return state.model_id
        except Exception:  # noqa: BLE001 - fall back to the public id
            pass

    return requested


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
) -> None:
    """Write a RequestStat row to the database and update Prometheus counters."""
    try:
        gpu_index: int | None = None
        backend_type: str | None = None
        energy_wh: float | None = None
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
                        if s.model_id == resolved_model_id
                        or s.backend_model_id == model_id
                    ),
                    None,
                )
            if state:
                backend_type = state.backend_type
                if state.current_gpu:
                    gpu_index = state.current_gpu[0]
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

        # Estimate energy from current GPU power draw × request duration.
        if gpu_index is not None:
            try:
                gm = request.app.state.gpu_manager
                gpu_state = gm._states.get(gpu_index)
                if gpu_state and gpu_state.power_draw_w > 0:
                    energy_wh = gpu_state.power_draw_w * duration_ms / 1000.0 / 3600.0
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

        # Pull agent context from the executor's contextvars (only set during
        # an agent invocation).  This stamps the *root* request_stat row with
        # the agent_id so the stats endpoints can filter by agent.
        agent_id_ctx: _uuid.UUID | None = None
        try:
            from ocabra.agents.executor import current_agent_id

            agent_id_ctx = current_agent_id.get()
        except Exception:
            agent_id_ctx = None

        async with AsyncSessionLocal() as session:
            stat = RequestStat(
                model_id=model_id,
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
                agent_id=agent_id_ctx,
                # Bloque 20 — Duration estimator + Router audit.
                work_size_meta=work_size_meta,
                via_router_profile_id=via_router_profile_id,
                runtime_preset=getattr(request.state, "runtime_preset_key", None),
                runtime_key=getattr(request.state, "runtime_key", None),
                effective_context=getattr(
                    request.state, "runtime_effective_context", None
                ),
                is_batch=is_batch,
            )
            session.add(stat)
            await session.commit()
    except Exception as exc:
        logger.warning("stats_write_failed", error=str(exc))


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
        username=getattr(auth_user, "username", None),
        api_key_name=getattr(auth_user, "api_key_name", None),
        client_addr=_client_addr(websocket),  # type: ignore[arg-type]
        error=error_message,
    )
