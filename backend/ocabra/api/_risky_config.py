"""Model options that run arbitrary code are for ``system_admin`` only (SEG-04).

``trust_remote_code`` runs the Python a Hugging Face repo ships; ``extra_args``
goes straight onto vLLM's command line; a tool-parser plugin is a Python file
vLLM imports; TensorRT-LLM's ``python_bin``/``serve_bin``/``serve_module`` pick
what gets executed. Any of them lets whoever sets it run code in the api
container. They used to be open to ``model_manager``, like every other option.

The check is a middleware over every ``POST``/``PUT``/``PATCH`` under
``/ocabra/`` with a JSON body, so it covers all the ways a configuration gets
in: models, logical models, registry and downloads, profile load overrides,
federation registrations. Only *setting* one of them to an active value needs
the role; turning it off does not. The model editor sends a JSON merge patch
(only what changed), so editing other fields of a model that already has one of
them is not affected.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)

#: Option names (snake and camel case, as the backends accept both).
RISKY_KEYS = frozenset(
    {
        "trust_remote_code",
        "trustRemoteCode",
        "extra_args",
        "extraArgs",
        "tool_parser_plugin",
        "toolParserPlugin",
        "python_bin",
        "pythonBin",
        "serve_bin",
        "serveBin",
        "serve_module",
        "serveModule",
    }
)
_METHODS = frozenset({"POST", "PUT", "PATCH"})


def _active(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | dict):
        return bool(value)
    return value is not None


def risky_settings(payload: Any, path: str = "") -> list[str]:
    """Dotted paths of every risky option ``payload`` sets to an active value."""
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            where = f"{path}.{key}" if path else str(key)
            if key in RISKY_KEYS and _active(value):
                found.append(where)
            found.extend(risky_settings(value, where))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            found.extend(risky_settings(item, f"{path}[{index}]"))
    return found


class RiskyConfigMiddleware(BaseHTTPMiddleware):
    """403 for a non-admin request that sets a code-running option."""

    async def dispatch(self, request: Request, call_next):
        if (
            request.method not in _METHODS
            or not request.url.path.startswith("/ocabra/")
            or "json" not in request.headers.get("content-type", "")
        ):
            return await call_next(request)
        body = await request.body()
        if not body:
            return await call_next(request)
        try:
            found = risky_settings(json.loads(body))
        except ValueError:
            return await call_next(request)  # the endpoint answers malformed JSON
        if not found:
            return await call_next(request)

        from fastapi import HTTPException

        from ocabra.api._deps_auth import get_current_user

        try:
            user = await get_current_user(request)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        if user.role != "system_admin":
            logger.warning(
                "risky_config_refused", user=user.username, path=request.url.path, options=found
            )
            return JSONResponse(
                {
                    "detail": (
                        "These options run arbitrary code and need the system_admin role: "
                        + ", ".join(found)
                    )
                },
                status_code=403,
            )
        return await call_next(request)
