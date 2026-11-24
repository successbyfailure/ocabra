"""FastAPI ``Depends`` providers for the OpenAI-compatible endpoints.

Only the dependency-injection providers live here (app-state accessors and
auth). The request-path logic they used to sit next to was split out by REF-06
into :mod:`ocabra.core.inference` (resolution/loading/admission/reasoning/body)
and :mod:`ocabra.api.sse`; :mod:`ocabra.api.openai._deps` re-exports everything
so existing handler imports keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Request

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.core.inference.errors import _openai_error

if TYPE_CHECKING:
    from ocabra.core.model_manager import ModelManager
    from ocabra.core.profile_registry import ProfileRegistry


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


async def get_openai_user(
    user: UserContext = Depends(get_current_user),
) -> UserContext:
    """Resolve auth for OpenAI-compatible endpoints.

    Delegates to ``get_current_user`` which already handles:
    - Bearer API key resolution.
    - Cookie JWT resolution.
    - Anonymous access when ``require_api_key_openai=False``.
    - HTTP 401 when ``require_api_key_openai=True`` and no credentials provided.

    Returns:
        Resolved :class:`UserContext` for the caller.

    Raises:
        HTTPException 401: When authentication is required but missing or invalid.
    """
    return user


def get_profile_registry(request: Request) -> ProfileRegistry:
    """Return the :class:`ProfileRegistry` stored on ``app.state``."""
    registry = getattr(request.app.state, "profile_registry", None)
    if registry is None:
        raise _openai_error(
            "Profile registry not available.",
            "server_error",
            code="service_unavailable",
            status_code=503,
            headers={"Retry-After": "5"},
        )
    return registry


def get_federation_manager(request: Request):
    """Return the FederationManager from app state, or None if disabled."""
    return getattr(request.app.state, "federation_manager", None)
