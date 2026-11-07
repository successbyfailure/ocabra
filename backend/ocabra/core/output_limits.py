"""Shared resolution of profile output-token limits."""

from __future__ import annotations

from typing import Any


def resolve_max_output_tokens(
    profile: Any,
    *,
    context_length: int = 0,
    backend_limit: int = 0,
) -> int | None:
    """Return the explicit enforceable output cap, bounded by context.

    A total context window is not an output limit. Profile policy wins over a
    backend-declared cap; zero or missing values mean that no independent cap
    is known and callers must only enforce prompt + output <= context.
    """

    policy = getattr(profile, "runtime_policy", None)
    policy = policy if isinstance(policy, dict) else {}
    raw_limit = policy.get("max_output_tokens")
    if raw_limit is None:
        raw_limit = backend_limit
    try:
        limit = int(raw_limit or 0)
    except (TypeError, ValueError):
        return None
    if limit <= 0:
        return None
    if context_length > 0:
        return min(limit, context_length)
    return limit


#: Campos con los que un cliente OpenAI fija su propio tope de salida.
_OPENAI_OUTPUT_FIELDS = ("max_completion_tokens", "max_tokens")


def _client_set_a_limit(payload: dict, *, ollama_native: bool) -> bool:
    """Whether the caller already chose its own output limit."""
    if ollama_native:
        options = payload.get("options")
        if isinstance(options, dict) and options.get("num_predict") is not None:
            return True
        return payload.get("num_predict") is not None
    return any(payload.get(field) is not None for field in _OPENAI_OUTPUT_FIELDS)


def apply_declared_output_limit(
    profile: Any,
    payload: dict,
    *,
    ollama_native: bool = False,
    context_length: int = 0,
    fallback_limit: int = 0,
) -> int | None:
    """Inyecta el tope de salida declarado del perfil cuando el cliente no fija uno.

    ``max_output_tokens`` se usaba solo para dos cosas: anunciarlo en
    ``/v1/models`` y rechazar peticiones que pidiesen mas. Nunca se enviaba al
    backend, asi que una peticion sin ``max_tokens`` generaba sin limite alguno
    y el tope declarado era decorativo.

    Incidente del 2026-11-07: gemma4:26b (``max_output_tokens`` 65536) recibio
    una peticion sin ``max_tokens``, entro en una generacion desbocada y
    produjo **132.263 tokens** durante 26 minutos hasta que el timeout de
    inferencia de oCabra la corto con un 504 y el cliente se quedo sin
    respuesta. Con el tope inyectado, el mismo caso termina acotado y con
    ``finish_reason: "length"``, que es un resultado que el cliente puede usar.

    Devuelve el limite aplicado, o ``None`` si no habia que tocar nada.
    """
    if _client_set_a_limit(payload, ollama_native=ollama_native):
        return None
    # Acotado por el contexto efectivo del preset elegido: varios perfiles
    # declaran 65536 de salida y tienen presets de 32k o 40960, asi que el tope
    # declarado supera la ventana entera. Mientras no se inyectaba era inocuo;
    # inyectandolo sin acotar se pediria mas salida que contexto disponible.
    ctx = max(0, int(context_length or 0))
    limit = resolve_max_output_tokens(profile, context_length=ctx)
    if limit is None:
        # El perfil no declara tope. Se usa el global para que ningun modelo
        # generativo quede sin limite: nueve perfiles no lo declaran, entre
        # ellos qwen3:32b y qwen3:8b. Ojo, esto NO se usa para rechazar
        # peticiones que pidan mas; solo para rellenar cuando el cliente calla.
        try:
            respaldo = int(fallback_limit or 0)
        except (TypeError, ValueError):
            respaldo = 0
        if respaldo <= 0:
            return None
        limit = min(respaldo, ctx) if ctx > 0 else respaldo
    if ollama_native:
        options = payload.get("options")
        payload["options"] = {**options, "num_predict": limit} if isinstance(options, dict) else {
            "num_predict": limit
        }
    else:
        payload["max_tokens"] = limit
    return limit
