"""
Stats middleware — records request metrics for all /v1/* endpoints.

Writes to the request_stats table: model_id, gpu_index, duration_ms,
prompt_tokens, completion_tokens, energy_wh.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)


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
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000

        model_id = await _extract_model_id(request)
        if model_id:
            # Fire-and-forget stats write
            import asyncio
            asyncio.create_task(
                _record_stat(request, model_id, duration_ms)
            )

        return response


async def _extract_model_id(request: Request) -> str | None:
    """Try to extract model ID from a cached request body."""
    try:
        body = await request.json()
        return body.get("model")
    except Exception:
        return None


async def _record_stat(request: Request, model_id: str, duration_ms: float) -> None:
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
                duration_ms=int(duration_ms),
            )
            session.add(stat)
            await session.commit()
    except Exception as exc:
        logger.warning("stats_write_failed", error=str(exc))
