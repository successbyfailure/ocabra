"""Muestreo del fabricante en las ejecuciones de benchmark que lo piden.

Los runners de cada app mandan su propio muestreo (IFEval temperatura 0, BFCL
0.001…) y no se pueden tocar uno a uno. Cuando una ejecución pide
``sampling: "recommended"``, el gestor de benchmarks activa un override para el
perfil que está midiendo; este middleware lo engancha a las peticiones que
traen la clave de servicio de los benchmarks, y ``merge_profile_defaults`` lo
aplica por encima del cuerpo del cliente. Las peticiones de usuarios nunca lo
ven: la clave de servicio solo la tiene el runner.
"""

from __future__ import annotations

import hmac
from contextvars import ContextVar
from typing import Any

current_sampling_override: ContextVar[dict[str, Any] | None] = ContextVar(
    "benchmark_sampling_override", default=None
)

# Solo muestreo: el tope de salida y lo demás sigue siendo el del benchmark.
OVERRIDE_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "dry_multiplier",
    }
)


def apply_override(profile_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Pisa el muestreo del cuerpo si hay override activo para este perfil."""
    override = current_sampling_override.get()
    if not override or override.get("profile") != profile_id:
        return body
    params = {
        key: value for key, value in (override.get("params") or {}).items() if key in OVERRIDE_KEYS
    }
    return {**body, **params}


def _bearer(scope: dict[str, Any]) -> str | None:
    for name, value in scope.get("headers") or []:
        if name == b"authorization":
            text = value.decode("latin-1")
            if text.lower().startswith("bearer "):
                return text[7:].strip()
    return None


class BenchmarkSamplingMiddleware:
    """Marca las peticiones del runner de benchmarks con el override activo."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        manager = getattr(getattr(scope.get("app"), "state", None), "benchmark_manager", None)
        override = getattr(manager, "sampling_override", None) if manager else None
        if not override:
            return await self.app(scope, receive, send)
        token = _bearer(scope)
        key = override.get("api_key") or ""
        if not token or not key or not hmac.compare_digest(token.encode(), key.encode()):
            return await self.app(scope, receive, send)
        marker = current_sampling_override.set(override)
        try:
            return await self.app(scope, receive, send)
        finally:
            current_sampling_override.reset(marker)
