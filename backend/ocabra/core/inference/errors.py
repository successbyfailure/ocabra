"""OpenAI-shaped error constructors for the inference request path.

These build :class:`fastapi.HTTPException` with the ``{"error": {...}}``
``detail`` shape OpenAI clients expect. They live in ``core`` (not ``api``)
because the resolution/loading helpers that raise them also live in ``core``
after REF-09; keeping the constructors here avoids the layer inversion where
``core`` imported from ``api``. The ``api`` layer re-exports them via
``ocabra.api.openai._deps`` so handlers are unaffected.

Note: distinct from :mod:`ocabra.api._errors` (REF-10), which *formats* an
already-raised ``detail`` into a family-specific response body. This module
*constructs* the exception.
"""

from __future__ import annotations

import json

import httpx
import structlog
from fastapi import HTTPException

logger = structlog.get_logger(__name__)


class EngineSuffixedModelName(HTTPException):
    """El cliente pidio un modelo nombrando el motor (``gemma4:26b-ollama``).

    Es un 404, pero **no** uno cualquiera: los handlers de Ollama capturan el
    404 de ``resolve_profile`` y reintentan por la via legacy, que acaba
    respondiendo un "not found" pelado y se come el mensaje que dice los dos
    nombres validos. Al ser una clase aparte, cada handler puede re-lanzarla
    antes de entrar en ese fallback.
    """


def engine_suffixed_error(
    requested: str, model_name: str, deployment_id: str
) -> EngineSuffixedModelName:
    """404 que nombra las dos salidas, para arreglarlo sin leer el changelog."""
    return EngineSuffixedModelName(
        status_code=404,
        detail={
            "error": {
                "message": (
                    f"The model '{requested}' does not exist. Use "
                    f"'{model_name}', or '{deployment_id}' if you need that "
                    "specific engine."
                ),
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


def _openai_error(
    message: str,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
    status_code: int = 400,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        headers=headers,
    )


def router_pick_error(exc: Exception) -> HTTPException:
    """HTTP answer for a router that found no target.

    No target with the capability the request needs → 400
    ``model_not_capable``, the same a plain model without it answers. Any
    other reason (all disabled, all missing) → 503 ``router_no_target``.
    """
    from ocabra.core.router_resolver import NoCapableTargetError

    if isinstance(exc, NoCapableTargetError):
        return _openai_error(
            str(exc),
            "invalid_request_error",
            param="model",
            code="model_not_capable",
        )
    return _openai_error(
        str(exc),
        "server_error",
        param="model",
        code="router_no_target",
        status_code=503,
    )


def raise_upstream_http_error(exc: httpx.HTTPStatusError) -> None:
    """
    Translate worker HTTP errors into OpenAI-compatible API errors.

    Preserves upstream status code and error payload when possible, and the
    headers a client acts on (``Retry-After``): they used to be dropped, so a
    backend's own 429/503 lost its retry hint (BUG-19).
    """
    status_code = exc.response.status_code
    body_text = exc.response.text
    headers = {
        name: exc.response.headers[name]
        for name in _UPSTREAM_HEADERS_KEPT
        if name in exc.response.headers
    } or None

    try:
        parsed = json.loads(body_text) if body_text else None
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        if "detail" in parsed:
            raise HTTPException(status_code=status_code, detail=parsed["detail"], headers=headers)
        if "error" in parsed:
            raise HTTPException(status_code=status_code, detail=parsed, headers=headers)

    message = body_text.strip() if body_text else str(exc)
    raise _openai_error(
        message,
        "invalid_request_error" if 400 <= status_code < 500 else "server_error",
        status_code=status_code,
        headers=headers,
    )


#: Upstream response headers worth passing on with an error.
_UPSTREAM_HEADERS_KEPT = ("retry-after",)


def raise_inference_dispatch_error(exc: Exception) -> None:
    """Translate ``InferenceTimeoutError``/``WorkerDegradedError`` from
    ``WorkerPool.forward_request`` into the same stable codes the streaming
    path already produces via ``stream_error_details`` — neither is an
    ``httpx.HTTPStatusError``, so a bare ``except httpx.HTTPStatusError``
    around the non-streaming ``forward_request`` call lets both escape
    uncaught into main.py's generic ``Exception`` handler: a 500 with
    ``code`` set to the raw Python exception class name, no ``Retry-After``,
    and no match to any documented status (confirmed 2026-10-31 — both are
    transient and should look like it).
    """
    from ocabra.core.fair_slot_gate import KVCapacityError
    from ocabra.core.worker_pool import (
        InferenceTimeoutError,
        WorkerDegradedError,
        WorkerGoneError,
    )

    if isinstance(exc, KVCapacityError):
        raise _openai_error(
            str(exc),
            "invalid_request_error",
            code="context_length_exceeded",
            status_code=400,
        ) from exc

    if isinstance(exc, InferenceTimeoutError):
        raise _openai_error(
            str(exc),
            "server_error",
            code="generation_timeout",
            status_code=504,
            headers={"Retry-After": "5"},
        ) from exc
    if isinstance(exc, WorkerGoneError):
        raise _openai_error(
            f"{exc}: it was unloaded while the request waited. Retry.",
            "server_error",
            code="worker_unloaded",
            status_code=503,
            headers={"Retry-After": "5"},
        ) from exc
    if isinstance(exc, WorkerDegradedError):
        from ocabra.config import settings

        raise _openai_error(
            str(exc),
            "server_error",
            code="worker_degraded",
            status_code=503,
            headers={"Retry-After": str(settings.worker_circuit_breaker_cooldown_s)},
        ) from exc
    raise exc


def stream_error_details(exc: Exception) -> tuple[str, str]:
    """Return a client-safe message and stable code for an in-band SSE error.

    Streaming response headers have already been sent when worker forwarding
    fails, so the upstream HTTP status cannot be returned directly. Preserve
    the worker's JSON error message in the SSE envelope and encode its status
    class in a stable oCabra code for request statistics.
    """
    from ocabra.core.fair_slot_gate import KVCapacityError
    from ocabra.core.worker_pool import (
        InferenceTimeoutError,
        WorkerDegradedError,
        WorkerGoneError,
    )

    if isinstance(exc, KVCapacityError):
        return str(exc), "context_length_exceeded"

    if isinstance(exc, InferenceTimeoutError):
        return str(exc), "generation_timeout"

    if isinstance(exc, WorkerDegradedError):
        return str(exc), "worker_degraded"

    if isinstance(exc, WorkerGoneError):
        return str(exc), "worker_unloaded"

    from ocabra.core.worker_pool import VisionNoRoomError

    if isinstance(exc, VisionNoRoomError):
        return str(exc), "vision_no_room"

    if not isinstance(exc, httpx.HTTPStatusError):
        return str(exc), "stream_error"

    status_code = exc.response.status_code
    body_text = exc.response.text
    message = ""
    try:
        parsed = json.loads(body_text) if body_text else None
    except (TypeError, ValueError):
        parsed = None

    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        error = parsed.get("error")
        if isinstance(detail, str):
            message = detail
        elif isinstance(detail, dict):
            message = str(detail.get("message") or "")
        if not message and isinstance(error, str):
            message = error
        elif not message and isinstance(error, dict):
            message = str(error.get("message") or "")

    if not message:
        message = body_text.strip() if body_text else str(exc)

    if status_code == 429:
        code = "upstream_rate_limited"
    elif 400 <= status_code < 500:
        code = "upstream_invalid_request"
    else:
        code = "upstream_server_error"

    logger.warning(
        "worker_stream_http_error",
        upstream_status=status_code,
        upstream_code=code,
        error_message=message[:1000],
        worker_url=str(exc.request.url),
    )
    return message, code
