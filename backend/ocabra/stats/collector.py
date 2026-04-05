"""
Stats middleware — records inference request metrics.

Tracks OpenAI-compatible `/v1/*` and Ollama-compatible inference routes under
`/api/*` so the stats page reflects real usage regardless of client protocol.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone

import structlog
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ocabra.config import settings

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
        "/v1/audio/transcriptions": "audio_transcription",
        "/v1/audio/speech": "tts",
        "/api/chat": "ollama_chat",
        "/api/generate": "ollama_generate",
        "/api/embeddings": "ollama_embedding",
        "/api/embed": "ollama_embedding",
    }
    return mapping.get(path, "other")


def _extract_usage_tokens(
    payload: dict | None,
    request_kind: str = "",
) -> tuple[int | None, int | None]:
    if not payload:
        return None, None

    usage = payload.get("usage") if isinstance(payload, dict) else None
    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens")
        if input_tokens is None:
            input_tokens = usage.get("input_tokens")

        output_tokens = usage.get("completion_tokens")
        if output_tokens is None:
            output_tokens = usage.get("output_tokens")

        try:
            return (
                int(input_tokens) if input_tokens is not None else None,
                int(output_tokens) if output_tokens is not None else None,
            )
        except (TypeError, ValueError):
            pass

    # Ollama-style normalized responses.
    prompt_eval_count = payload.get("prompt_eval_count")
    eval_count = payload.get("eval_count")
    if prompt_eval_count is not None or eval_count is not None:
        try:
            return (
                int(prompt_eval_count) if prompt_eval_count is not None else None,
                int(eval_count) if eval_count is not None else None,
            )
        except (TypeError, ValueError):
            return None, None

    # Whisper-style: {"text": "..."}  — use word count as output proxy.
    if request_kind == "audio_transcription":
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return None, len(text.split())

    return None, None


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
        started_at = datetime.now(timezone.utc)
        request_payload = await _extract_request_payload(request)
        request_kind = _classify_request_kind(path)

        # Track in-flight requests so the pressure eviction loop can avoid
        # evicting models that are currently serving a request.
        inflight_model_id = _extract_model_id(request=request, body=request_payload)
        try:
            mm = request.app.state.model_manager
        except AttributeError:
            mm = None
        if mm and inflight_model_id:
            mm.begin_request(inflight_model_id)

        try:
            response = await call_next(request)
        except Exception as exc:
            if mm and inflight_model_id:
                mm.end_request(inflight_model_id)
            model_id = _extract_model_id(request=request, body=request_payload)
            if model_id:
                asyncio.create_task(_record_stat(
                    request=request, model_id=model_id, started_at=started_at,
                    duration_ms=(time.monotonic() - start) * 1000,
                    error_message=str(exc), status_code=500,
                    endpoint_path=path, request_kind=request_kind,
                    input_tokens=None, output_tokens=None,
                ))
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
                        mm.end_request(inflight_model_id)
                    if model_id:
                        all_body = b"".join(chunks)
                        last_payload = _extract_last_payload_from_stream(all_body, content_type)
                        in_tok, out_tok = _extract_usage_tokens(last_payload, request_kind)
                        asyncio.create_task(_record_stat(
                            request=request, model_id=model_id, started_at=started_at,
                            duration_ms=duration_ms,
                            error_message=error_msg or (f"HTTP {status_code}" if status_code >= 400 else None),
                            status_code=status_code,
                            endpoint_path=path, request_kind=request_kind,
                            input_tokens=in_tok, output_tokens=out_tok,
                        ))

            response.body_iterator = tee_and_record()
            return response

        # Non-streaming: end in-flight immediately, buffer body, extract tokens.
        if mm and inflight_model_id:
            mm.end_request(inflight_model_id)

        duration_ms = (time.monotonic() - start) * 1000
        error_message = f"HTTP {response.status_code}" if response.status_code >= 400 else None

        response_payload, response = await _extract_response_payload_and_rebuild(response)

        model_id = _extract_model_id(request=request, body=request_payload)
        if model_id:
            in_tok, out_tok = _extract_usage_tokens(response_payload, request_kind=request_kind)
            asyncio.create_task(_record_stat(
                request=request, model_id=model_id, started_at=started_at,
                duration_ms=duration_ms, error_message=error_message,
                status_code=response.status_code, endpoint_path=path,
                request_kind=request_kind, input_tokens=in_tok, output_tokens=out_tok,
            ))

        return response


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
                    gpu_index = state.preferred_gpu if state.preferred_gpu is not None else settings.default_gpu_index
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

        record_request(model_id=model_id, duration_s=max(duration_ms, 0.0) / 1000.0, status="error" if error_message else "ok")
        record_tokens(model_id=model_id, input_tokens=int(input_tokens or 0), output_tokens=int(output_tokens or 0))

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
            )
            session.add(stat)
            await session.commit()
    except Exception as exc:
        logger.warning("stats_write_failed", error=str(exc))
