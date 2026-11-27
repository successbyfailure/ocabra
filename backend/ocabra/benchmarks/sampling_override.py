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
import json
from contextvars import ContextVar
from typing import Any

# Los overrides activos de las ejecuciones en curso (puede haber varias a la
# vez, una por máquina); cada uno lleva el perfil al que se aplica.
current_sampling_override: ContextVar[list[dict[str, Any]] | dict[str, Any] | None] = ContextVar(
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
    active = current_sampling_override.get()
    candidates = [active] if isinstance(active, dict) else list(active or [])
    override = next((item for item in candidates if item.get("profile") == profile_id), None)
    if override is None:
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


def _active_overrides(manager: Any) -> list[dict[str, Any]]:
    if manager is None:
        return []
    listed = getattr(manager, "sampling_overrides", None)
    if listed is not None:
        return [item for item in listed if item]
    single = getattr(manager, "sampling_override", None)
    return [single] if single else []


# Lo que se anota de cada petición del runner (nunca el contenido): los
# parámetros de muestreo y de salida, para saber con qué se midió de verdad.
OBSERVED_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "max_tokens",
    "max_completion_tokens",
    "num_predict",
    "num_ctx",
    "reasoning_effort",
)
_OBSERVED_PATHS = (
    "/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/api/chat",
    "/api/generate",
)


def observed_params(body: dict[str, Any]) -> dict[str, Any]:
    """Los parámetros de muestreo de una petición (OpenAI u Ollama ``options``)."""
    out: dict[str, Any] = {}
    sources = [body]
    if isinstance(body.get("options"), dict):
        sources.append(body["options"])
    for source in sources:
        for key in OBSERVED_KEYS:
            value = source.get(key)
            numeric = isinstance(value, (int, float, bool))
            if numeric or (key == "reasoning_effort" and isinstance(value, str)):
                out[key] = value
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and isinstance(kwargs.get("enable_thinking"), bool):
        out["enable_thinking"] = kwargs["enable_thinking"]
    if isinstance(body.get("tools"), list):
        out["tools"] = True
    return out


def _node_of(path: str) -> str:
    if path.startswith("/_node/"):
        from urllib.parse import unquote

        return unquote(path[len("/_node/") :].split("/", 1)[0])
    return ""


class BenchmarkSamplingMiddleware:
    """Marca las peticiones del runner de benchmarks con el override activo."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        manager = getattr(getattr(scope.get("app"), "state", None), "benchmark_manager", None)
        receive = await self._observe(scope, receive, manager)
        overrides = _active_overrides(manager)
        if not overrides:
            return await self.app(scope, receive, send)
        token = _bearer(scope)
        mine = [
            item
            for item in overrides
            if token
            and item.get("api_key")
            and hmac.compare_digest(token.encode(), str(item["api_key"]).encode())
        ]
        if not mine:
            return await self.app(scope, receive, send)
        marker = current_sampling_override.set(mine)
        try:
            return await self.app(scope, receive, send)
        finally:
            current_sampling_override.reset(marker)

    async def _observe(self, scope, receive, manager):
        """Anota el muestreo de las peticiones del runner y devuelve el ``receive`` a usar."""
        observe = getattr(manager, "observe_sampling", None)
        key = getattr(manager, "service_api_key", None)
        path = str(scope.get("path") or "")
        if (
            not callable(observe)
            or not key
            or scope.get("method") != "POST"
            or not path.endswith(_OBSERVED_PATHS)
        ):
            return receive
        token = _bearer(scope)
        if not token or not hmac.compare_digest(token.encode(), str(key).encode()):
            return receive
        chunks: list[bytes] = []
        early: dict | None = None
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                early = message  # desconexión antes del cuerpo: que la vea la app
                break
            chunks.append(message.get("body") or b"")
            if not message.get("more_body"):
                break
        body = b"".join(chunks)
        if early is not None:
            queued = [early]

            async def replay_disconnect():
                return queued.pop(0) if queued else await receive()

            return replay_disconnect
        try:
            data = json.loads(body) if body else None
            if isinstance(data, dict) and isinstance(data.get("model"), str):
                observe(_node_of(path), data["model"], observed_params(data))
        except Exception:  # noqa: BLE001 — anotar nunca rompe la petición
            pass
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return replay
