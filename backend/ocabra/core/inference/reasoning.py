"""``reasoning_effort``/``think`` handling and backend-body normalization.

Translates the client's OpenAI-style reasoning/thinking controls into the
per-backend knobs, normalizes ``tool_calls``/``content`` shapes, and assembles
the payload forwarded to a strict upstream (:func:`to_backend_body`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ocabra.core.model_ref import is_llama_cpp_family

if TYPE_CHECKING:
    from ocabra.core.model_manager import ModelState


def _thinking_disable_intent(body: dict) -> bool | None:
    """Infer whether the client wants reasoning/thinking disabled.

    Returns True (disable), False (force enable) or None (unspecified) by
    inspecting the various spellings clients use: Ollama's native ``think``,
    vLLM's ``chat_template_kwargs.enable_thinking`` and the OpenAI-style
    ``reasoning`` / ``reasoning_effort`` fields.
    """
    if isinstance(body.get("think"), bool):
        return not body["think"]

    cct = body.get("chat_template_kwargs")
    if isinstance(cct, dict) and isinstance(cct.get("enable_thinking"), bool):
        return not cct["enable_thinking"]

    reasoning = body.get("reasoning")
    if isinstance(reasoning, bool):
        return not reasoning
    if isinstance(reasoning, str) and reasoning.strip().lower() in {"none", "off", "false"}:
        return True

    effort = body.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip().lower() in {"none", "off", "false"}:
        return True

    return None


def _apply_thinking_controls(state: ModelState, payload: dict) -> None:
    """Map the client's thinking-disable intent onto backend-specific knobs.

    Backends disagree on how (and whether) reasoning can be turned off:

    - **vLLM / sglang** honour it through the chat template, so we set
      ``chat_template_kwargs.enable_thinking = false``. This genuinely
      suppresses the ``<think>`` block for models that support the flag
      (e.g. Qwen3) and is silently ignored by templates that don't.
    - **Ollama** cannot disable reasoning through its OpenAI-compatible
      endpoint for current reasoning models: a string ``reasoning`` 400s
      (it expects an ``openai.Reasoning`` object) and ``reasoning_effort``
      only relocates the chain-of-thought inline. So we inject nothing and
      just drop the vLLM-only ``chat_template_kwargs`` to avoid confusing the
      upstream. Disabling reasoning there needs a prompt-level switch (e.g.
      Qwen's ``/no_think``) or an upstream Ollama fix (ollama/ollama#15288).
    """
    backend_type = getattr(state, "backend_type", "")
    if backend_type == "ollama":
        payload.pop("chat_template_kwargs", None)
        return
    if backend_type in {"vllm", "sglang"} and _thinking_disable_intent(payload) is True:
        cct = payload.get("chat_template_kwargs")
        cct = dict(cct) if isinstance(cct, dict) else {}
        cct.setdefault("enable_thinking", False)
        payload["chat_template_kwargs"] = cct


_REASONING_EFFORT_LEVELS = {"low", "medium", "high"}
_REASONING_EFFORT_OFF = {"off", "none", "disabled", "no", "false"}
_REASONING_EFFORT_DEFAULT = {"", "default"}
# 2026-11-05: a caller that never sets reasoning_effort/reasoning_budget/
# max_tokens can leave a reasoning model thinking indefinitely — confirmed
# live against gemma4:26b, a request with no cap at all generated 14k+
# tokens non-stop until oCabra's 1500s dispatch timeout killed it, three
# retries in a row. "medium" gives the model's own chat template a bounded
# hint instead of running with no guidance whatsoever. Hardcoded last-resort
# fallback if settings/per-model config ever holds something invalid.
DEFAULT_REASONING_EFFORT = "medium"


def _resolve_default_reasoning_effort(state: ModelState) -> str:
    """Pick the effort level to use when a caller doesn't set one.

    Precedence: the model's own ``extra_config.default_reasoning_effort``
    (set via the model edit UI) over the global ``GET/PATCH /ocabra/config``
    ``defaultReasoningEffort`` setting. Falls back to
    :data:`DEFAULT_REASONING_EFFORT` if neither resolves to a valid level.
    """
    from ocabra.config import settings

    extra_config = getattr(state, "extra_config", None)
    per_model = None
    if isinstance(extra_config, dict):
        per_model = extra_config.get("default_reasoning_effort") or extra_config.get(
            "defaultReasoningEffort"
        )
    candidate = str(per_model if per_model else settings.default_reasoning_effort).strip().lower()
    if candidate in _REASONING_EFFORT_LEVELS or candidate in _REASONING_EFFORT_OFF:
        return candidate
    return DEFAULT_REASONING_EFFORT


def _reasoning_budget_for_level(level: str, *, implicit: bool = False) -> int | None:
    """Tope duro de tokens de pensamiento para un nivel de ``reasoning_effort``.

    ``None`` = ese nivel no lleva tope (``high`` por defecto: quien pide
    esfuerzo alto quiere justamente que piense sin prisa). ``implicit`` es
    una peticion que no mando nivel: lleva ``reasoning_budget_default_tokens``
    en vez del tope del nivel por defecto.
    """
    from ocabra.config import settings

    if implicit:
        try:
            budget = int(settings.reasoning_budget_default_tokens or 0)
        except (TypeError, ValueError):
            return None
        return budget if budget > 0 else None

    por_nivel = {
        "low": settings.reasoning_budget_low_tokens,
        "medium": settings.reasoning_budget_medium_tokens,
        "high": settings.reasoning_budget_high_tokens,
    }
    try:
        budget = int(por_nivel.get(level) or 0)
    except (TypeError, ValueError):
        return None
    return budget if budget > 0 else None


def _output_allowance(payload: dict) -> int | None:
    """Tokens de salida que tiene esta peticion, si se sabe.

    ``apply_declared_output_limit`` ya ha corrido para cuando se llama a
    :func:`_apply_reasoning_effort`, asi que el tope esta en el payload: el
    del cliente si lo mando, y si no el declarado por el modelo.
    """
    opciones = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    for valor in (
        payload.get("max_completion_tokens"),
        payload.get("max_tokens"),
        opciones.get("num_predict"),
    ):
        if valor is None:
            continue
        try:
            entero = int(valor)
        except (TypeError, ValueError):
            continue
        if entero > 0:
            return entero
    return None


def _reasoning_budget_capped(level: str, payload: dict, *, implicit: bool = False) -> int | None:
    """Tope de pensamiento del nivel, recortado para que quepa la respuesta.

    El presupuesto del nivel no puede pasarse del total de salida: si se
    permite pensar mas de lo que cabe, el pensamiento se come la respuesta
    entera y el cliente recibe ``content`` vacio con
    ``finish_reason: "length"``. Detectado en el benchmark del 2026-11-08
    (``ravenx`` con ``medium`` daba "JSON inutilizable"): el nivel concedia
    16.384 tokens de pensamiento sobre una salida de 4.096, asi que el
    truncado estaba garantizado. Pasaba igual con ``low`` (2.048) por debajo
    de 2.048 de salida, y ``high`` no llevaba tope ninguno.

    Se reserva para la respuesta un cuarto de la salida (minimo 256 tokens).
    Con salida holgada el nivel manda sin cambios: ``medium`` sobre 65.536
    sigue siendo 16.384.
    """
    derivado = _reasoning_budget_for_level(level, implicit=implicit)
    salida = _output_allowance(payload)
    if salida is None:
        # Sin tope conocido no hay nada de lo que proteger a la respuesta.
        return derivado
    reserva = max(256, salida // 4)
    tope = max(1, salida - reserva)
    if derivado is None:
        # ``high`` no declara presupuesto a proposito (quien pide esfuerzo
        # alto quiere que piense sin prisa), pero sin tope el pensamiento
        # puede agotar la salida igualmente. Con un tope conocido, se acota.
        return tope
    return min(derivado, tope)


def _apply_reasoning_effort(state: ModelState, payload: dict) -> None:
    """Translate the OpenAI-style ``reasoning_effort`` field into whatever the
    target backend understands. The field is already in the ChatCompletionRequest
    schema; here we fan it out:

    - **vLLM**: forwarded as ``extra_body.chat_template_kwargs.reasoning_effort``
      (vLLM 0.9+ wires that through to the chat template; harmless on older
      versions because unknown kwargs are ignored).
    - **Ollama ≥0.5**: translated to the native ``think`` field
      (``True`` for low/medium/high — Ollama has a single on/off switch —
      and ``False`` for off/none/disabled).
    - **llama.cpp (native)**: the requested level is forwarded as
      ``reasoning_effort`` and paired with the per-request Jinja override
      ``chat_template_kwargs.enable_thinking``. The boolean is required when
      the worker was launched with ``--reasoning off``: a level alone does not
      override that launch default.
    - **Everything else**: silently dropped — no-op for backends with no
      native reasoning control.

    When the caller omits the field (or sends ``default``/``""``) and the
    model advertises ``capabilities.reasoning``, the effective default from
    :func:`_resolve_default_reasoning_effort` is substituted instead of
    leaving reasoning completely unbounded. A non-reasoning model (or an
    explicit ``off``/``low``/``high``) is untouched.

    The input ``reasoning_effort`` key is always popped from the forwarded
    payload so unknown backends don't 400 on it.
    """
    from ocabra.config import settings

    raw = payload.pop("reasoning_effort", None)
    value = "" if raw is None else str(raw).strip().lower()
    implicit = False
    if value in _REASONING_EFFORT_DEFAULT:
        if not getattr(state.capabilities, "reasoning", False):
            return
        value = _resolve_default_reasoning_effort(state)
        # Only the global default is "the client asked for nothing": a level
        # an admin set for this model is a deliberate choice and gets that
        # level's budget.
        extra = state.extra_config if isinstance(state.extra_config, dict) else {}
        implicit = not (
            extra.get("default_reasoning_effort") or extra.get("defaultReasoningEffort")
        )
    if value in _REASONING_EFFORT_OFF:
        enabled = False
    elif value in _REASONING_EFFORT_LEVELS:
        enabled = True
    else:
        # Unknown value — leave the payload alone and let the backend 400.
        payload["reasoning_effort"] = raw
        return

    backend = getattr(state, "backend_type", "") or ""
    if backend == "vllm":
        extra = payload.get("extra_body")
        if not isinstance(extra, dict):
            extra = {}
        template_kwargs = extra.get("chat_template_kwargs")
        if not isinstance(template_kwargs, dict):
            template_kwargs = {}
        # Mismo tratamiento que la familia llama.cpp: vLLM pasa
        # ``chat_template_kwargs`` tal cual a la plantilla, asi que valen las
        # mismas reglas.
        #
        # ``enable_thinking`` es obligatorio para apagar el pensamiento: la
        # plantilla de Qwen 3.5 valida el nivel dentro de
        # ``{% if enable_thinking is undefined or enable_thinking is true %}``,
        # asi que sin esa clave entra a validar igual y un "off" explota.
        # Comprobado renderizando la plantilla oficial de Qwen3.8-27B: tanto
        # "off" como "high" dan "Unexpected reasoning effort"; con
        # ``enable_thinking=False`` y con "xhigh" pasan. Antes se enviaba
        # ``reasoning_effort="off"`` a pelo, que es justo el caso que rompe.
        template_kwargs.setdefault("enable_thinking", enabled)
        if enabled:
            template_kwargs.setdefault(
                "reasoning_effort",
                settings.reasoning_effort_template_aliases.get(value, value),
            )
        extra["chat_template_kwargs"] = template_kwargs
        payload["extra_body"] = extra
    elif backend == "ollama":
        # Ollama 0.5+ only has a boolean switch; carry it through. When the
        # payload goes to ``/api/chat`` the ``think`` key sits at the top
        # level; when it goes to the OpenAI-compat ``/v1/chat/completions``
        # Ollama passes unknown top-level keys to the chat template so the
        # same key still works.
        payload.setdefault("think", enabled)
    elif is_llama_cpp_family(backend):
        # El vocabulario de la plantilla no es el de OpenAI: Qwen 3.5 valida
        # (xhigh, medium, low) y lanza excepcion con cualquier otra cosa, asi
        # que "high" hay que traducirlo antes de mandarlo a cualquiera de los
        # dos sitios donde llama-server lo lee.
        nivel_plantilla = settings.reasoning_effort_template_aliases.get(value, value)
        template_kwargs = payload.get("chat_template_kwargs")
        template_kwargs = dict(template_kwargs) if isinstance(template_kwargs, dict) else {}
        template_kwargs["enable_thinking"] = enabled
        if enabled:
            # El nivel tiene que ir DENTRO de chat_template_kwargs: llama-server
            # no reenvia a la plantilla los campos de primer nivel (comprobado
            # en vivo — ``enable_thinking`` de primer nivel se ignora y dentro
            # de chat_template_kwargs funciona). Enviandolo solo arriba, las
            # plantillas que si leen el nivel caian a su propio default, que en
            # Qwen 3.5 es ``xhigh``: esos modelos recibian siempre el esfuerzo
            # maximo pidieras lo que pidieras. El alias traduce el vocabulario
            # de OpenAI al de la plantilla (high -> xhigh).
            template_kwargs["reasoning_effort"] = nivel_plantilla
        payload["chat_template_kwargs"] = template_kwargs
        # Arriba va TAMBIEN el nivel traducido, no el crudo. ``reasoning_effort``
        # de primer nivel es un parametro nativo de llama-server y **gana** al
        # de ``chat_template_kwargs``: comprobado en vivo contra el worker de
        # qwen3.8:27b, ``{top: high, kwargs: xhigh}`` da 500 ("Unexpected
        # reasoning effort high") igual que ``high`` a secas, y solo pasa
        # cuando arriba va ``xhigh``. Enviar aqui el crudo rompia
        # ``reasoning_effort="high"`` en los tres modelos Qwen 3.5 del parque,
        # cuyo vocabulario es (xhigh, medium, low).
        payload["reasoning_effort"] = nivel_plantilla if enabled else "none"
        if enabled:
            # Chaos starts with reasoning_format=none so ordinary requests do
            # not expose empty think tags. An explicit reasoning request wants
            # the standard split field consumed by OpenAI-compatible clients.
            payload.setdefault("reasoning_format", "deepseek")
            # El nivel, por si solo, no limita nada: ni la plantilla de gemma4
            # menciona ``reasoning_effort`` (verificado leyendo la plantilla que
            # publica el propio llama-server en /props), asi que "low",
            # "medium" y "high" daban resultados identicos. Se traduce a
            # ``reasoning_budget_tokens``, que es el unico tope duro: al
            # agotarse, llama.cpp corta el pensamiento y fuerza la respuesta.
            # Sin esto, una peticion sin max_tokens podia pensar hasta agotar
            # la salida (incidente 2026-11-07: 132.263 tokens en 26 minutos).
            # Un presupuesto explicito del cliente manda sobre el derivado.
            if "reasoning_budget_tokens" not in payload and "reasoning_budget" not in payload:
                derivado = _reasoning_budget_capped(value, payload, implicit=implicit)
                if derivado is not None:
                    payload["reasoning_budget_tokens"] = derivado


def _apply_reasoning_budget(state: ModelState, payload: dict) -> None:
    """Translate oCabra's ``reasoning_budget`` extension into llama-server's
    native per-request field — a hard token cap on thinking — when
    exhausted, llama.cpp cuts reasoning short and forces the final answer,
    unlike ``reasoning_effort`` which is only a soft hint the chat template
    may ignore).

    2026-10-29: added after a production incident where gemma4 (llama_cpp,
    no budget cap at all — ``--reasoning-budget`` defaults to -1,
    unrestricted) reasoned for 28k+ tokens on an uncapped request and burned
    through the entire context preset before writing an answer, leaving
    ``content`` empty. No native equivalent on other backends — silently
    dropped there, same pattern as ``reasoning_effort``.

    The wire key is ``reasoning_budget_tokens``, NOT ``reasoning_budget`` —
    confirmed by reading the strings in ``libllama-server-impl.so`` (the
    CLI flag ``--reasoning-budget`` doesn't map 1:1 to its own JSON key; the
    binary groups it with ``thinking_budget_tokens``,
    ``reasoning_budget_start_tag`` etc., all suffixed ``_tokens``/``_tag``).
    An initial implementation forwarded the CLI-flag-shaped name
    (``reasoning_budget``) directly, which the server silently ignored — a
    request with budget=200 still reasoned for thousands of tokens. Fixed
    and verified live the same day: budget=200 against qwen3.8:27b produced
    674 chars of reasoning and a clean ``finish_reason: "stop"`` (vs. ~16.6k
    chars / several minutes unconstrained).
    """
    raw = payload.pop("reasoning_budget", None)
    if raw is None:
        return
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        return
    if is_llama_cpp_family(getattr(state, "backend_type", "")):
        payload.setdefault("reasoning_budget_tokens", budget)


def _normalize_tool_call_entry(tool_call: object) -> object:
    """Backfill the OpenAI-required ``type``/stringified-``arguments`` shape
    on a ``tool_calls`` entry before forwarding it to a strict backend.

    Mirrors ``ocabra.api.ollama.chat._normalize_tool_call_for_backend`` —
    same fix, same root cause (llama-server's OpenAI-compatible endpoint
    rejects outright with "Missing tool call type" if a historical
    ``tool_calls`` entry omits ``type`` or has ``arguments`` as a parsed
    object instead of a JSON string), applied here defensively for
    OpenAI-compatible callers too: found live 2026-10-30 via an external
    agent hitting /v1/chat/completions directly against
    ``llama_cpp/gemma4-26b`` with exactly this shape.
    """
    if not isinstance(tool_call, dict):
        return tool_call
    out = dict(tool_call)
    out.setdefault("type", "function")
    function = out.get("function")
    if isinstance(function, dict):
        function = dict(function)
        if isinstance(function.get("arguments"), dict):
            function["arguments"] = json.dumps(function["arguments"])
        out["function"] = function
    return out


def _flatten_text_only_content(messages: object) -> object:
    """Colapsa las ``content parts`` de solo texto en la cadena llana que
    esperan las plantillas de chat.

    Varias plantillas asumen que ``message.content`` es una cadena y llaman a
    ``.startswith()`` sobre ella. La que Ollama empaqueta para qwen3 es una
    revision anterior a la oficial de Qwen, que añadio guardas
    ``message.content is string`` justamente por esto: comprobado renderizando
    las dos, con ``content`` como lista mas ``tools`` la de Ollama muere con
    "'list object' has no attribute 'startswith'" y la oficial funciona. Afecta
    a qwen3:32b y qwen3:8b, que comparten plantilla.

    Aplanar aqui lo arregla para cualquier plantilla con esa debilidad, sin
    tener que mantener copias de plantillas ajenas que se quedarian viejas en
    silencio. El prompt resultante es identico: se comprobo contra el
    ``/apply-template`` del propio llama-server que une las partes de texto
    **sin separador**, y que una sola parte equivale byte a byte a la cadena.

    Solo se aplana cuando TODAS las partes son de texto. Una lista con
    imagenes, audio o video se deja intacta: esas rutas necesitan la estructura.
    """
    if not isinstance(messages, list):
        return messages
    out = []
    for raw in messages:
        if not isinstance(raw, dict) or not isinstance(raw.get("content"), list):
            out.append(raw)
            continue
        partes = raw["content"]
        if not partes or not all(
            isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
            for p in partes
        ):
            out.append(raw)
            continue
        msg = dict(raw)
        msg["content"] = "".join(p["text"] for p in partes)
        out.append(msg)
    return out


def _normalize_messages_for_backend(messages: object) -> object:
    if not isinstance(messages, list):
        return messages
    out = []
    for raw in messages:
        if not isinstance(raw, dict) or not isinstance(raw.get("tool_calls"), list):
            out.append(raw)
            continue
        msg = dict(raw)
        msg["tool_calls"] = [_normalize_tool_call_entry(tc) for tc in msg["tool_calls"]]
        out.append(msg)
    return out


def to_backend_body(state: ModelState, body: dict) -> dict:
    """Copy request payload and normalize model field to backend-native model id."""
    payload = dict(body)
    # oCabra request extensions are consumed by the coordinator and must not
    # leak into strict OpenAI-compatible upstreams.
    payload.pop("ocabra", None)
    payload["model"] = state.backend_model_id
    if "messages" in payload:
        payload["messages"] = _normalize_messages_for_backend(payload["messages"])
        if is_llama_cpp_family(getattr(state, "backend_type", "")):
            # Solo la familia llama.cpp: es donde se ha comprobado que una
            # plantilla desfasada revienta con content como lista. vLLM y
            # Ollama parsean las content parts ellos mismos.
            payload["messages"] = _flatten_text_only_content(payload["messages"])
    if payload.get("stream") is True:
        stream_options = payload.get("stream_options")
        if isinstance(stream_options, dict):
            payload["stream_options"] = {**stream_options, "include_usage": True}
        else:
            payload["stream_options"] = {"include_usage": True}
    _apply_reasoning_effort(state, payload)
    _apply_reasoning_budget(state, payload)
    _apply_thinking_controls(state, payload)
    return payload
