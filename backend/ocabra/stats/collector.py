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

logger = structlog.get_logger(__name__)

_TRACKED_API_PATHS = {
    "/api/chat",
    "/api/generate",
    "/api/embeddings",
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
    }
    return mapping.get(path, "other")


def _extract_usage_tokens(payload: dict | None) -> tuple[int | None, int | None]:
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
    try:
        return (
            int(prompt_eval_count) if prompt_eval_count is not None else None,
            int(eval_count) if eval_count is not None else None,
        )
    except (TypeError, ValueError):
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
        path = request.url.path
        if not _should_track(path):
            return await call_next(request)

        start = time.monotonic()
        started_at = datetime.now(timezone.utc)
        request_payload = await _extract_request_payload(request)

        error_message: str | None = None
        response: Response | None = None
        try:
            response = await call_next(request)
        except Exception as exc:
            error_message = str(exc)
            model_id = _extract_model_id(request=request, body=request_payload)
            if model_id:
                import asyncio

                asyncio.create_task(
                    _record_stat(
                        request=request,
                        model_id=model_id,
                        started_at=started_at,
                        duration_ms=(time.monotonic() - start) * 1000,
                        error_message=error_message,
                        status_code=500,
                        endpoint_path=path,
                        request_kind=_classify_request_kind(path),
                        input_tokens=None,
                        output_tokens=None,
                    )
                )
            raise

        duration_ms = (time.monotonic() - start) * 1000
        if response.status_code >= 400:
            error_message = f"HTTP {response.status_code}"

        response_payload, response = await _extract_response_payload_and_rebuild(response)

        model_id = _extract_model_id(request=request, body=request_payload)
        if model_id:
            input_tokens, output_tokens = _extract_usage_tokens(response_payload)

            import asyncio

            asyncio.create_task(
                _record_stat(
                    request=request,
                    model_id=model_id,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    error_message=error_message,
                    status_code=response.status_code,
                    endpoint_path=path,
                    request_kind=_classify_request_kind(path),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )

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
        try:
            mm = request.app.state.model_manager
            state = await mm.get_state(model_id)
            if state:
                backend_type = state.backend_type
                if state.current_gpu:
                    gpu_index = state.current_gpu[0]
        except Exception:
            pass

        from ocabra.api.metrics import record_request, record_tokens
        from ocabra.database import AsyncSessionLocal
        from ocabra.db.stats import RequestStat

        record_request(model_id=model_id, duration_s=max(duration_ms, 0.0) / 1000.0, status="error" if error_message else "ok")
        record_tokens(model_id=model_id, input_tokens=int(input_tokens or 0), output_tokens=int(output_tokens or 0))

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
                error=error_message,
            )
            session.add(stat)
            await session.commit()
    except Exception as exc:
        logger.warning("stats_write_failed", error=str(exc))
