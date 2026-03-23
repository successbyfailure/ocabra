"""
Shared dependencies for OpenAI API endpoints.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import structlog
from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from ocabra.core.model_manager import ModelManager, ModelState

logger = structlog.get_logger(__name__)

_ENSURE_LOAD_TIMEOUT_S = 180


def _openai_error(message: str, error_type: str, param: str | None = None, code: str | None = None, status_code: int = 400) -> HTTPException:
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
    )


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


async def resolve_model(model_manager: ModelManager, model_id: str) -> tuple[str, ModelState | None]:
    """
    Resolve a model id by canonical id or backend_model_id alias.

    Resolution order:
    1) Exact canonical match (model_id)
    2) First state whose backend_model_id equals requested value
    """
    requested = str(model_id or "").strip()
    if not requested:
        return "", None

    exact = await model_manager.get_state(requested)
    if exact is not None:
        return requested, exact

    states = await model_manager.list_states()
    for state in states:
        if state.backend_model_id == requested:
            return state.model_id, state

    return requested, None


async def ensure_loaded(model_manager: ModelManager, model_id: str) -> ModelState:
    """
    Ensure a model is LOADED before forwarding a request.
    Triggers on-demand loading if CONFIGURED or UNLOADED.
    Waits up to 180s if already LOADING.
    """
    from ocabra.core.model_manager import ModelStatus

    resolved_model_id, state = await resolve_model(model_manager, model_id)
    if state is None:
        raise _openai_error(
            f"The model '{model_id}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    if state.status == ModelStatus.LOADED:
        state.last_request_at = datetime.now(UTC)
        return state

    if state.status in (ModelStatus.CONFIGURED, ModelStatus.UNLOADED):
        try:
            await model_manager.load(resolved_model_id)
        except Exception as exc:
            # Surface resource/scheduler failures as clear client-visible errors.
            from ocabra.core.scheduler import InsufficientVRAMError

            if isinstance(exc, InsufficientVRAMError):
                raise _openai_error(
                    str(exc),
                    "invalid_request_error",
                    code="insufficient_vram",
                    status_code=409,
                ) from exc
            raise _openai_error(
                f"Failed to load model '{resolved_model_id}': {exc}",
                "server_error",
                code="model_load_failed",
                status_code=503,
            ) from exc
        # Verify it loaded
        state = await model_manager.get_state(resolved_model_id)
        if state and state.status == ModelStatus.LOADED:
            state.last_request_at = datetime.now(UTC)
            return state

    if state and state.status == ModelStatus.LOADING:
        for _ in range(_ENSURE_LOAD_TIMEOUT_S):
            await asyncio.sleep(1)
            state = await model_manager.get_state(resolved_model_id)
            if state and state.status == ModelStatus.LOADED:
                state.last_request_at = datetime.now(UTC)
                return state
        raise _openai_error(
            f"Model '{resolved_model_id}' did not finish loading in time.",
            "server_error",
            code="model_load_timeout",
            status_code=503,
        )

    raise _openai_error(
        f"Model '{resolved_model_id}' is not available (status: {state.status.value if state else 'unknown'}).",
        "server_error",
        code="model_unavailable",
        status_code=503,
    )


def check_capability(state: ModelState, capability: str, endpoint: str) -> None:
    """Raise 400 if the model lacks the required capability."""
    if not getattr(state.capabilities, capability, False):
        raise _openai_error(
            f"The model '{state.model_id}' does not support {endpoint}.",
            "invalid_request_error",
            param="model",
            code="model_not_capable",
        )


def to_backend_body(state: ModelState, body: dict) -> dict:
    """Copy request payload and normalize model field to backend-native model id."""
    payload = dict(body)
    payload["model"] = state.backend_model_id
    return payload


def raise_upstream_http_error(exc: httpx.HTTPStatusError) -> None:
    """
    Translate worker HTTP errors into OpenAI-compatible API errors.

    Preserves upstream status code and error payload when possible.
    """
    status_code = exc.response.status_code
    body_text = exc.response.text

    try:
        parsed = json.loads(body_text) if body_text else None
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        if "detail" in parsed:
            raise HTTPException(status_code=status_code, detail=parsed["detail"])
        if "error" in parsed:
            raise HTTPException(status_code=status_code, detail=parsed)

    message = body_text.strip() if body_text else str(exc)
    raise _openai_error(
        message,
        "invalid_request_error" if 400 <= status_code < 500 else "server_error",
        status_code=status_code,
    )
