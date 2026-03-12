"""
Stats middleware — records inference request metrics.

Tracks OpenAI-compatible `/v1/*` and Ollama-compatible inference routes under
`/api/*` so the stats page reflects real usage regardless of client protocol.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)

_TRACKED_API_PATHS = {
    "/api/chat",
    "/api/generate",
    "/api/embeddings",
}


class StatsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that records usage statistics for /v1/* requests.

    For non-streaming responses, token counts are extracted from the JSON body.
    For streaming responses, the middleware records duration but cannot reliably
    extract token counts (the caller's last chunk would contain usage data).
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not _should_track(request.url.path):
            return await call_next(request)

        start = time.monotonic()
        started_at = datetime.now(timezone.utc)
        request_payload = await _extract_request_payload(request)

        error_message: str | None = None
        try:
            response = await call_next(request)
        except Exception as exc:
            error_message = str(exc)
            model_id = _extract_model_id_from_payload(request_payload)
            if model_id:
                import asyncio
                asyncio.create_task(
                    _record_stat(request, model_id, started_at, (time.monotonic() - start) * 1000, error_message)
                )
            raise
        duration_ms = (time.monotonic() - start) * 1000
        if response.status_code >= 400:
            error_message = f"HTTP {response.status_code}"

        model_id = _extract_model_id_from_payload(request_payload)
        if model_id:
            import asyncio
            asyncio.create_task(
                _record_stat(request, model_id, started_at, duration_ms, error_message)
            )

        return response


def _should_track(path: str) -> bool:
    if path.startswith("/v1/"):
        return True
    return path in _TRACKED_API_PATHS


async def _extract_request_payload(request: Request) -> dict | None:
    """Read and cache the request JSON body if present."""
    try:
        body = await request.json()
        return body if isinstance(body, dict) else None
    except Exception:
        return None


def _extract_model_id_from_payload(body: dict | None) -> str | None:
    if not body:
        return None
    model_id = body.get("model")
    return str(model_id) if model_id else None


async def _record_stat(
    request: Request,
    model_id: str,
    started_at: datetime,
    duration_ms: float,
    error_message: str | None = None,
) -> None:
    """Write a RequestStat row to the database."""
    try:
        gpu_index: int | None = None
        try:
            mm = request.app.state.model_manager
            state = await mm.get_state(model_id)
            if state and state.current_gpu:
                gpu_index = state.current_gpu[0]
        except Exception:
            pass

        from ocabra.database import AsyncSessionLocal
        from ocabra.db.stats import RequestStat

        async with AsyncSessionLocal() as session:
            stat = RequestStat(
                model_id=model_id,
                gpu_index=gpu_index,
                started_at=started_at,
                duration_ms=int(duration_ms),
                error=error_message,
            )
            session.add(stat)
            await session.commit()
    except Exception as exc:
        logger.warning("stats_write_failed", error=str(exc))
