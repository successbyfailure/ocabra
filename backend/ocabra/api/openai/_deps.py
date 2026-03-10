"""
Shared dependencies for OpenAI API endpoints.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from ocabra.core.model_manager import ModelManager, ModelState, ModelStatus

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


def get_model_manager(request: Request) -> "ModelManager":
    return request.app.state.model_manager


async def ensure_loaded(model_manager: "ModelManager", model_id: str) -> "ModelState":
    """
    Ensure a model is LOADED before forwarding a request.
    Triggers on-demand loading if CONFIGURED or UNLOADED.
    Waits up to 180s if already LOADING.
    """
    from ocabra.core.model_manager import ModelStatus

    state = await model_manager.get_state(model_id)
    if state is None:
        raise _openai_error(
            f"The model '{model_id}' does not exist.",
            "invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    if state.status == ModelStatus.LOADED:
        return state

    if state.status in (ModelStatus.CONFIGURED, ModelStatus.UNLOADED):
        await model_manager.load(model_id)
        # Verify it loaded
        state = await model_manager.get_state(model_id)
        if state and state.status == ModelStatus.LOADED:
            return state

    if state and state.status == ModelStatus.LOADING:
        for _ in range(_ENSURE_LOAD_TIMEOUT_S):
            await asyncio.sleep(1)
            state = await model_manager.get_state(model_id)
            if state and state.status == ModelStatus.LOADED:
                return state
        raise _openai_error(
            f"Model '{model_id}' did not finish loading in time.",
            "server_error",
            code="model_load_timeout",
            status_code=503,
        )

    raise _openai_error(
        f"Model '{model_id}' is not available (status: {state.status.value if state else 'unknown'}).",
        "server_error",
        code="model_unavailable",
        status_code=503,
    )


def check_capability(state: "ModelState", capability: str, endpoint: str) -> None:
    """Raise 400 if the model lacks the required capability."""
    if not getattr(state.capabilities, capability, False):
        raise _openai_error(
            f"The model '{state.model_id}' does not support {endpoint}.",
            "invalid_request_error",
            param="model",
            code="model_not_capable",
        )
