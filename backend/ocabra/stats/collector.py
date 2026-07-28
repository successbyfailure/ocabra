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
from ocabra.core.worker_pool import InferenceTimeoutError

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
    streamed response lacked a usage payload."""
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
        return total or None
    if request_kind in ("completion", "ollama_generate"):
        return _approx_text_tokens(request_payload.get("prompt"))
    return None


def _stream_output_tokens(body: bytes, content_type: str) -> int | None:
    """Approx output tokens (~word count) from the streamed content deltas, used when
    a streaming response carried no usage payload."""
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
    in_tok: int | None = None
    out_tok: int | None = None

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

    # Whisper-style: {"text": "..."} — use word count as output proxy.
    if request_kind == "audio_transcription" and out_tok is None and isinstance(payload, dict):
        out_tok = _approx_text_tokens(payload.get("text"))

    return _apply_token_kind_defaults(in_tok, out_tok, request_kind, request_payload)


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
        if mm and inflight_model_id:
            inflight_request_id = mm.try_begin_request(
                inflight_model_id, settings.max_inflight_per_model
            )
            if inflight_request_id is None:
                # Admission control: the worker is already at capacity. Reject
                # rather than pile on so one client's burst can't starve others.
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
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": "2"},
                    content={
                        "error": {
                            "message": (
                                f"Model '{inflight_model_id}' is at capacity "
                                f"({settings.max_inflight_per_model} concurrent "
                                "requests). Please retry shortly."
                            ),
                            "type": "rate_limit_error",
                            "code": "model_at_capacity",
                        }
                    },
                )

        try:
            response = await call_next(request)
        except InferenceTimeoutError as exc:
            if mm and inflight_model_id:
                mm.end_request(inflight_model_id, inflight_request_id)
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
        except Exception as exc:
            if mm and inflight_model_id:
                mm.end_request(inflight_model_id, inflight_request_id)
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

        # Detect streaming by content-type (SSE or NDJSON).
        # For streaming we tee the body_iterator: chunks go to the client in real-time
        # while we buffer them to extract token counts from the final chunk.
        # end_request is moved to the generator's finally block so it fires only
        # after the last byte is delivered (or the client disconnects).
        content_type = response.headers.get("content-type", "")
        is_streaming = (
            "text/event-stream" in content_type
            or "x-ndjson" in content_type
            or isinstance(response, StreamingResponse)
        )

        if is_streaming:
            model_id = _extract_model_id(request=request, body=request_payload)
            status_code = response.status_code
            original_iterator = response.body_iterator

            async def tee_and_record():
                chunks: list[bytes] = []
                error_msg: str | None = None
                try:
                    async for chunk in original_iterator:
                        if isinstance(chunk, str):
                            chunk = chunk.encode("utf-8")
                        chunks.append(chunk)
                        yield chunk
                except Exception as exc:
                    error_msg = str(exc)
                    raise
                finally:
                    duration_ms = (time.monotonic() - start) * 1000
                    if mm and inflight_model_id:
                        mm.end_request(inflight_model_id, inflight_request_id)
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
                        in_tok, out_tok = _extract_usage_tokens(
                            last_payload, request_kind, request_payload
                        )
                        # Streaming without a usage payload (e.g. no stream_options.
                        # include_usage): approximate from the streamed content and
                        # the request so the row isn't left as "—".
                        if request_kind in _TEXT_GEN_KINDS:
                            if out_tok is None:
                                out_tok = _stream_output_tokens(all_body, content_type)
                            if in_tok is None:
                                in_tok = _count_input_tokens(request_payload, request_kind)
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
                            )
                        )

            response.body_iterator = tee_and_record()
            return response

        # Non-streaming: end in-flight immediately, buffer body, extract tokens.
        if mm and inflight_model_id:
            mm.end_request(inflight_model_id, inflight_request_id)

        duration_ms = (time.monotonic() - start) * 1000
        error_message = f"HTTP {response.status_code}" if response.status_code >= 400 else None

        response_payload, response = await _extract_response_payload_and_rebuild(response)

        model_id = _extract_model_id(request=request, body=request_payload)
        if model_id:
            in_tok, out_tok = _extract_usage_tokens(
                response_payload, request_kind=request_kind, request_payload=request_payload
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
            if profile is not None and profile.enabled:
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
) -> None:
    """Write a RequestStat row to the database and update Prometheus counters."""
    try:
        gpu_index: int | None = None
        backend_type: str | None = None
        energy_wh: float | None = None
        try:
            mm = request.app.state.model_manager
            state = await mm.get_state(model_id)
            if state is None:
                # Fallback: resolve by backend_model_id alias (e.g. "mistral:7b" → "ollama/mistral:7b")
                states = await mm.list_states()
                state = next((s for s in states if s.backend_model_id == model_id), None)
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
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                energy_wh=energy_wh,
                error=error_message,
                user_id=parsed_user_id,
                group_id=parsed_group_id,
                api_key_name=api_key_name,
                client_addr=client_addr,
                user_agent=user_agent,
                remote_node_id=remote_node_id,
                agent_id=agent_id_ctx,
            )
            session.add(stat)
            await session.commit()
    except Exception as exc:
        logger.warning("stats_write_failed", error=str(exc))
