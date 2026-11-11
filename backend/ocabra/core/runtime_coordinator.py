"""Request-time coordination for profile runtime presets."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any

import structlog

from ocabra.config import settings
from ocabra.core.model_manager_helpers import compute_worker_key, resolve_llama_cpp_option
from ocabra.core.model_ref import is_llama_cpp_family
from ocabra.core.output_limits import resolve_max_output_tokens
from ocabra.core.runtime_preset_capacity import runtime_capacity_cache
from ocabra.core.runtime_presets import (
    ContextCapacityError,
    ResolvedRuntimePreset,
    select_runtime_preset,
)

logger = structlog.get_logger(__name__)

_IMAGE_TOKEN_RESERVATION = 128


@dataclass(frozen=True)
class RuntimePlan:
    worker_key: str
    load_overrides: dict[str, Any] | None
    preferred_gpu: int | None
    load_policy: str | None
    preset_key: str | None
    runtime_key: str | None
    effective_context: int | None
    required_tokens: int | None
    selection_reason: str | None
    # Lo que se reserva del pool KV, que NO es lo mismo que el presupuesto con
    # el que se elige ventana: cuando el tamaño real no se ha podido medir, el
    # presupuesto se deja en la capacidad entera para quedarse con la ventana
    # mayor y no rechazar, pero reservar el pool entero deja los demas slots
    # fuera hasta que esa peticion termine. Ver ``refine_required_tokens``.
    kv_reservation_tokens: int | None = None
    # Per-request context the chosen worker really has, when it was changed in
    # place (ModelState.live_context) and is smaller than the preset's. The
    # caller grows it back before serving a request that needs more.
    live_effective_context: int | None = None


class OutputTokenLimitError(ContextCapacityError):
    """An explicit output budget exceeds the model's enforceable hard cap."""

    def __init__(self, *, parameter: str, requested: int, maximum: int) -> None:
        self.parameter = parameter
        self.requested = requested
        self.maximum = maximum
        super().__init__(
            f"{parameter}={requested} exceeds the model maximum output of {maximum} tokens"
        )


def _explicit_output_budget(body: dict[str, Any]) -> tuple[str, int] | None:
    """Return the OpenAI output-limit field that controls this request."""

    for parameter in ("max_completion_tokens", "max_tokens"):
        raw = body.get(parameter)
        if raw is None:
            continue
        try:
            return parameter, int(raw)
        except (TypeError, ValueError):
            return None
    return None


def _text_size(value: Any) -> tuple[int, int]:
    """Return (characters, words) from the textual portions of an OpenAI body."""

    if isinstance(value, str):
        return len(value), len(value.split())
    if isinstance(value, list):
        parts = [_text_size(item) for item in value]
        return sum(item[0] for item in parts), sum(item[1] for item in parts)
    if not isinstance(value, dict):
        return (0, 0)
    chars = words = 0
    for key, item in value.items():
        if key in {"type", "image_url", "input_audio", "video_url"}:
            continue
        item_chars, item_words = _text_size(item)
        chars += item_chars
        words += item_words
    return chars, words


def _image_count(value: Any) -> int:
    """Count image inputs without decoding or fetching their payloads."""
    if isinstance(value, list):
        return sum(_image_count(item) for item in value)
    if not isinstance(value, dict):
        return 0
    if value.get("type") in {"image_url", "input_image"}:
        return 1
    count = 0
    images = value.get("images")
    if isinstance(images, list):
        count += len(images)
    for key, item in value.items():
        if key not in {"image_url", "images"}:
            count += _image_count(item)
    return count


def estimate_prompt_tokens(body: dict[str, Any]) -> int:
    """Conservatively estimate prompt tokens without the output reservation.

    Exact tokenization is model-specific and may require loading tokenizer
    code. The estimator intentionally rounds upward and includes structural
    overhead, tool schemas and image placeholders.
    """
    if isinstance(body.get("messages"), list):
        source: Any = body["messages"]
        structural_overhead = 16 * len(body["messages"])
    else:
        source = body.get("prompt", body.get("input", ""))
        structural_overhead = 8
    chars, words = _text_size(source)
    tools_chars, tools_words = _text_size(body.get("tools", []))
    chars += tools_chars
    words += tools_words
    prompt_estimate = max(math.ceil(chars / 3), math.ceil(words * 1.5))
    prompt_estimate = math.ceil(prompt_estimate * 1.15) + structural_overhead
    prompt_estimate += _image_count(source) * _IMAGE_TOKEN_RESERVATION
    return max(1, prompt_estimate)


def output_reservation(body: dict[str, Any], default_reservation: int) -> int:
    """Tokens de salida que hay que reservar para esta peticion.

    Factorizado aparte porque el planificador necesita recombinarlo con un
    conteo de prompt *exacto* cuando lo consigue del tokenizador del worker.
    """
    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    output_value = body.get(
        "max_completion_tokens",
        body.get(
            "max_tokens",
            options.get("num_predict", default_reservation),
        ),
    )
    try:
        return max(1, int(output_value))
    except (TypeError, ValueError):
        return default_reservation


def _cuerpo_para_plantilla(body: dict[str, Any]) -> dict[str, Any]:
    """Lo minimo que ``/apply-template`` necesita para renderizar el prompt."""

    salida: dict[str, Any] = {}
    if isinstance(body.get("messages"), list):
        salida["messages"] = body["messages"]
    if body.get("tools"):
        salida["tools"] = body["tools"]
    if body.get("tool_choice") is not None:
        salida["tool_choice"] = body["tool_choice"]
    return salida


def _prompt_text(body: dict[str, Any]) -> str:
    """Texto plano del prompt, para pedir su conteo exacto al tokenizador."""
    piezas: list[str] = []
    if isinstance(body.get("messages"), list):
        for mensaje in body["messages"]:
            if not isinstance(mensaje, dict):
                continue
            contenido = mensaje.get("content")
            if isinstance(contenido, str):
                piezas.append(contenido)
            elif isinstance(contenido, list):
                for parte in contenido:
                    if isinstance(parte, dict) and isinstance(parte.get("text"), str):
                        piezas.append(parte["text"])
            # Las llamadas a herramientas viajan fuera de ``content``, y en un
            # cliente de codigo son la mitad del prompt: sus argumentos llevan
            # ficheros enteros. Sin esto el conteo "exacto" salia a la mitad
            # (medido 102.200 contra 196.549 reales en flint:40b el
            # 2026-11-09), y la reserva de KV se quedaba corta justo con las
            # peticiones mas grandes.
            for llamada in mensaje.get("tool_calls") or []:
                if not isinstance(llamada, dict):
                    continue
                funcion = llamada.get("function")
                if isinstance(funcion, dict):
                    for clave in ("name", "arguments"):
                        valor = funcion.get(clave)
                        if isinstance(valor, str):
                            piezas.append(valor)
                        elif valor is not None:
                            piezas.append(json.dumps(valor, ensure_ascii=False))
    else:
        origen = body.get("prompt", body.get("input", ""))
        if isinstance(origen, str):
            piezas.append(origen)
        elif isinstance(origen, list):
            piezas.extend(x for x in origen if isinstance(x, str))
    if body.get("tools"):
        piezas.append(json.dumps(body["tools"], ensure_ascii=False))
    return "\n".join(piezas)


async def exact_prompt_tokens(model_manager: Any, profile: Any, body: dict[str, Any]) -> int | None:
    """Conteo exacto de tokens pedido al worker del propio modelo.

    Los heuristicos de caracteres y palabras no sirven para decidir un rechazo:
    medidos contra el tokenizador real, ``chars/3`` va de 0,94x a 2,42x del
    valor verdadero segun el contenido (subestima en prosa densa, sobreestima
    2,4x con palabras largas). Cuando el worker ya esta cargado, el que cuenta
    bien esta a un POST de distancia.

    Devuelve ``None`` si no hay ningun runtime de este modelo cargado, si el
    backend no es de la familia llama.cpp o si la llamada falla: en esos casos
    el que llama NO debe rechazar por estimacion.
    """
    base = getattr(profile, "base_model_id", None)
    if not base:
        return None
    try:
        estados = await model_manager.list_states(include_derived=True)
    except Exception:  # noqa: BLE001 — es una optimizacion, nunca un fallo
        return None
    for estado in estados:
        model_id = getattr(estado, "model_id", "")
        # Cualquier runtime del mismo modelo base comparte tokenizador, asi que
        # vale tambien un derivado (``<base>::<hash>``) de otro preset.
        if model_id != base and not model_id.startswith(f"{base}::"):
            continue
        if getattr(getattr(estado, "status", None), "value", "") != "loaded":
            continue
        if not is_llama_cpp_family(getattr(estado, "backend_type", "") or ""):
            continue
        worker = getattr(estado, "worker_info", None)
        puerto = getattr(worker, "port", None)
        if not puerto:
            continue
        try:
            import httpx

            async with httpx.AsyncClient(timeout=settings.tokenize_timeout_s) as cliente:
                # Primero se le pide al propio worker que aplique su plantilla
                # de chat. Lo que devuelve es EXACTAMENTE el prompt que vera el
                # modelo —marcas de rol, formato de las llamadas a herramientas,
                # esquemas de ``tools`` renderizados—, asi que contarlo da el
                # numero de verdad y no una aproximacion. Medido el 2026-11-09
                # contra flint:40b con un turno de agente de codigo: la
                # plantilla aplicada da 7.833 tokens y el motor cuenta 7.833;
                # concatenar el texto a mano daba 8.490, un 8,4% de mas.
                renderizado = None
                try:
                    plantilla = await cliente.post(
                        f"http://127.0.0.1:{int(puerto)}/apply-template",
                        json=_cuerpo_para_plantilla(body),
                    )
                    if plantilla.status_code == 200:
                        valor = plantilla.json().get("prompt")
                        if isinstance(valor, str) and valor:
                            renderizado = valor
                except Exception:  # noqa: BLE001 — build sin /apply-template
                    renderizado = None

                respuesta = await cliente.post(
                    f"http://127.0.0.1:{int(puerto)}/tokenize",
                    json={
                        "content": renderizado if renderizado is not None else _prompt_text(body)
                    },
                )
            respuesta.raise_for_status()
            tokens = respuesta.json().get("tokens")
            if not isinstance(tokens, list):
                return None
        except Exception as exc:  # noqa: BLE001
            logger.info("exact_tokenize_failed", model_id=model_id, error=str(exc))
            return None
        if renderizado is not None:
            return len(tokens)
        # Sin plantilla aplicada, el prompt va pelado: se suma el mismo margen
        # estructural que usa el heuristico para las marcas de rol.
        estructural = 16 * len(body["messages"]) if isinstance(body.get("messages"), list) else 8
        return len(tokens) + estructural
    return None


def estimate_request_token_budget(
    body: dict[str, Any], *, default_output_reservation: int | None = None
) -> int:
    """Conservatively estimate prompt plus reserved output tokens.

    Exact tokenization is model-specific and may require loading tokenizer code.
    The estimator uses both character and word bounds with headroom, then adds
    the client's complete output reservation. It intentionally rounds upward.

    ``default_output_reservation`` overrides the global
    ``settings.runtime_preset_default_output_reservation_tokens`` fallback.
    """
    default_reservation = (
        default_output_reservation
        if default_output_reservation is not None
        else settings.runtime_preset_default_output_reservation_tokens
    )
    prompt_estimate = estimate_prompt_tokens(body)
    reserved_output = output_reservation(body, default_reservation)
    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    estimated = max(1, prompt_estimate + reserved_output)
    try:
        requested_context = max(0, int(options.get("num_ctx") or 0))
    except (TypeError, ValueError):
        requested_context = 0
    return max(estimated, requested_context)


def _reserva_kv(
    required_tokens: int, medido: bool, capacity: int, *, parallelism: int | None
) -> int:
    """Lo que esta peticion reserva del pool KV compartido.

    Con el tamaño medido se reserva eso y ya esta. Sin medir, ``required_tokens``
    viene inflado a proposito —la capacidad entera— para quedarse con la ventana
    mayor y no rechazar por una estimacion; reservarlo tal cual deja el pool a
    cero y los otros slots fuera hasta que termine. Visto el 2026-11-09 en
    ``flint:40b``: la tarjeta marcaba 256k/256k de KV ocupado con una sola
    peticion dentro, y sus cuatro slots quedaban servidos de uno en uno.

    Sin medida se reserva la parte que le tocaria a un slot, que es justo para
    lo que el preset se dimensiono. Si de verdad necesita mas, el pool unificado
    se lo da: para eso es unificado.
    """

    if medido:
        return max(0, int(required_tokens))
    if capacity <= 0:
        return max(0, int(required_tokens))
    return max(1, int(capacity) // max(1, int(parallelism or 1)))


async def refine_required_tokens(
    model_manager: Any,
    profile: Any,
    body: dict[str, Any],
    *,
    required_tokens: int,
    capacity: int,
    default_output_reservation: int | None,
) -> tuple[int, bool]:
    """Afina el presupuesto de tokens antes de que pueda provocar un rechazo.

    Un rechazo por tamaño no puede decidirse con el heuristico: medido contra
    el tokenizador real va de 0,94x a 2,42x del valor verdadero segun el
    contenido. Asi que el heuristico sirve para ELEGIR ventana —equivocarse
    ahi solo carga una mas grande de lo necesario— pero antes de rechazar:

    1. Si hay un runtime de este modelo cargado, se le pide el conteo exacto a
       su tokenizador y se decide con ese numero.
    2. Si no hay ninguno, no se rechaza: se devuelve justo la capacidad, se
       intenta la peticion y, si de verdad no cabe, lo dice el motor, que es
       el unico que cuenta bien.

    ``capacity`` es con lo que se va a comparar aguas abajo: la ventana del
    preset mayor cuando el perfil tiene presets, o el pool KV del propio
    worker cuando no los tiene. Con ``capacity <= 0`` no hay nada que afinar.

    Devuelve ``(presupuesto, medido)``. El segundo valor importa porque el
    presupuesto sirve para dos cosas distintas y solo una admite "no lo se":
    elegir ventana (ahi conviene pasarse) y reservar KV (ahi pasarse deja al
    resto de slots sin sitio).
    """
    if capacity <= 0 or required_tokens <= capacity:
        return required_tokens, True
    exacto = await exact_prompt_tokens(model_manager, profile, body)
    if exacto is not None:
        # Misma resolucion que estimate_request_token_budget: el valor del
        # perfil si lo declara, el global si no. Un ``or 0`` aqui dejaria sin
        # reserva de salida a quien no manda max_tokens.
        reserva_por_defecto = (
            default_output_reservation
            if default_output_reservation is not None
            else settings.runtime_preset_default_output_reservation_tokens
        )
        medido = exacto + output_reservation(body, reserva_por_defecto)
        logger.info(
            "token_budget_measured",
            profile_id=getattr(profile, "profile_id", None),
            estimated=required_tokens,
            measured=medido,
            largest_context=capacity,
        )
        return medido, True
    logger.info(
        "token_budget_unverified_attempt",
        profile_id=getattr(profile, "profile_id", None),
        estimated=required_tokens,
        largest_context=capacity,
    )
    return capacity, False


def _requested_preset_key(body: dict[str, Any]) -> str | None:
    extension = body.get("ocabra")
    if not isinstance(extension, dict):
        return None
    value = extension.get("context_preset")
    return str(value).strip() if value else None


def _apply_llama_cpp_preset_placement(
    overrides: dict[str, Any],
    state: Any,
    preset: ResolvedRuntimePreset,
    gpu_states: list[Any],
) -> None:
    """Translate a resolved preset placement into concrete llama.cpp flags.

    ``_runtime.gpu_indices`` drives admission, but llama.cpp also needs its
    CUDA visibility and split flags changed on the derived worker. Without
    this translation a dual preset cloned ``disabled_gpus=[0]`` from a normal
    3090-only base profile and silently remained single-GPU.
    """
    if not is_llama_cpp_family(getattr(state, "backend_type", "")):
        return
    selected = list(dict.fromkeys(int(index) for index in preset.gpu_indices))
    if not selected:
        return

    extra = state.extra_config if isinstance(state.extra_config, dict) else {}
    section = copy.deepcopy(extra.get("llama_cpp") or {})
    profile_section = overrides.get("llama_cpp")
    if isinstance(profile_section, dict):
        section.update(copy.deepcopy(profile_section))

    known_indices = sorted(int(gpu.index) for gpu in gpu_states)
    raw_disabled = section.get("disabled_gpus", section.get("disabledGpus", []))
    try:
        prior_disabled = {int(index) for index in (raw_disabled or [])}
    except (TypeError, ValueError):
        prior_disabled = set()
    prior_visible = [index for index in known_indices if index not in prior_disabled]
    section["disabled_gpus"] = [index for index in known_indices if index not in selected]
    # A measured override belongs to one concrete placement. Keep it only
    # when the preset uses that exact GPU set; otherwise use the planner.
    if prior_visible != selected:
        section["vram_estimate_mb"] = None

    if len(selected) == 1:
        section.update(
            {
                "split_mode": "none",
                "split_strategy": None,
                "tensor_split": None,
                "main_gpu": None,
            }
        )
    else:
        totals = {int(gpu.index): int(gpu.total_vram_mb) for gpu in gpu_states}
        smallest = min((totals.get(index, 1) for index in selected), default=1)
        raw_split = section.get("tensor_split", section.get("tensorSplit"))
        calibrated_split = (
            prior_visible == selected
            and isinstance(raw_split, list)
            and len(raw_split) == len(selected)
        )
        ratios = (
            [float(value) for value in raw_split]
            if calibrated_split
            else [totals.get(index, smallest) / max(1, smallest) for index in selected]
        )
        raw_main_gpu = section.get("main_gpu", section.get("mainGpu"))
        try:
            configured_main_gpu = int(raw_main_gpu)
        except (TypeError, ValueError):
            configured_main_gpu = -1
        main_gpu = (
            configured_main_gpu
            if calibrated_split and 0 <= configured_main_gpu < len(selected)
            else max(
                range(len(selected)),
                key=lambda pos: totals.get(selected[pos], 0),
            )
        )
        # Multi-GPU presets used to force "layer". A profile that asks for
        # "row" or "tensor" keeps it; anything else still becomes "layer".
        configured_split_mode = (
            str(section.get("split_mode", section.get("splitMode")) or "").strip().lower()
        )
        split_mode = (
            configured_split_mode if configured_split_mode in {"row", "tensor"} else "layer"
        )
        section.update(
            {
                "split_mode": split_mode,
                "split_strategy": "evenly",
                "tensor_split": ratios,
                # llama.cpp interprets --main-gpu relative to CUDA_VISIBLE_DEVICES.
                "main_gpu": main_gpu,
            }
        )

    overrides["llama_cpp"] = section


def _probe_vision_mb(policy: dict, preset_key: str) -> dict[str, int] | None:
    probes = policy.get("context_probe") if isinstance(policy, dict) else None
    probe = probes.get(preset_key) if isinstance(probes, dict) else None
    vision = probe.get("vision") if isinstance(probe, dict) else None
    cost = vision.get("cost_mb") if isinstance(vision, dict) else None
    if not isinstance(cost, dict) or not cost:
        return None
    try:
        return {str(gpu): int(mb) for gpu, mb in cost.items()}
    except (TypeError, ValueError):
        return None


async def plan_profile_runtime(
    profile: Any,
    model_manager: Any,
    request_body: dict[str, Any] | None,
) -> RuntimePlan:
    """Resolve and select the runtime variant for one profile request."""

    body = request_body or {}
    raw_policy = getattr(profile, "runtime_policy", None)
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    raw_reservation = policy.get("default_output_reservation_tokens")
    try:
        default_output_reservation = int(raw_reservation) if raw_reservation is not None else None
    except (TypeError, ValueError):
        default_output_reservation = None
    required_tokens = estimate_request_token_budget(
        body, default_output_reservation=default_output_reservation
    )
    explicit_output = _explicit_output_budget(body)
    declared_max_output = resolve_max_output_tokens(profile)
    if explicit_output is not None and declared_max_output is not None:
        parameter, requested_output = explicit_output
        if requested_output > declared_max_output:
            raise OutputTokenLimitError(
                parameter=parameter,
                requested=requested_output,
                maximum=declared_max_output,
            )

    enabled_presets = [
        preset for preset in getattr(profile, "runtime_presets", []) if preset.enabled
    ]
    if not enabled_presets:
        worker_key = compute_worker_key(profile.base_model_id, profile.load_overrides)
        # Sin presets no hay seleccion de ventana, pero `required_tokens` SI
        # viaja hasta la puerta de admision (`fairness_kv_budget`), que lo
        # compara con el pool KV del worker y rechaza con 400. Afinarlo aqui
        # tambien: sin esto, los cuatro modelos sin presets —gemma4:26b,
        # gemma4:12b, mistral:7b y ravenx— seguian rechazando por estimacion.
        # Reproducido 2026-11-07 contra gemma4:26b: un prompt de 230.011
        # tokens reales se estimaba en 705.372 y la puerta lo rechazaba con
        # "request needs 705372 KV tokens but the pool holds 262144", 3,07x
        # el tamaño verdadero.
        # ``get_state`` con getattr: afinar es una mejora, no un requisito, y
        # por esta via pasan ModelManager mas ligeros que no lo implementan.
        # Sin estado no se puede saber el tamaño del pool, asi que se deja el
        # presupuesto como estaba.
        buscar_estado = getattr(model_manager, "get_state", None)
        pool_kv = 0
        if callable(buscar_estado):
            try:
                sin_presets = await buscar_estado(profile.base_model_id)
                if sin_presets is not None:
                    pool_kv = int(resolve_llama_cpp_option(sin_presets, "ctx_size", 0) or 0)
            except (TypeError, ValueError, AttributeError):
                pool_kv = 0
        required_tokens, medido = await refine_required_tokens(
            model_manager,
            profile,
            body,
            required_tokens=required_tokens,
            capacity=pool_kv,
            default_output_reservation=default_output_reservation,
        )
        reserva_kv = _reserva_kv(required_tokens, medido, pool_kv, parallelism=None)
        return RuntimePlan(
            worker_key=worker_key,
            load_overrides=profile.load_overrides,
            preferred_gpu=None,
            load_policy=None,
            preset_key=None,
            runtime_key=None,
            effective_context=None,
            required_tokens=required_tokens,
            selection_reason=None,
            kv_reservation_tokens=reserva_kv,
        )

    state = await model_manager.get_state(profile.base_model_id)
    if state is None:
        raise ContextCapacityError(f"base model '{profile.base_model_id}' is not configured")
    gpu_manager = model_manager.gpu_manager
    gpu_states = await gpu_manager.get_all_states() if gpu_manager is not None else []
    rows = await runtime_capacity_cache.resolve(profile, state, gpu_states)
    rows_by_key = {row["preset_key"]: row for row in rows}

    candidates: list[ResolvedRuntimePreset] = []
    preset_by_key = {preset.preset_key: preset for preset in enabled_presets}
    for preset in enabled_presets:
        row = rows_by_key.get(preset.preset_key)
        if not row or not row.get("resolved") or not row.get("fits"):
            continue
        candidates.append(
            ResolvedRuntimePreset(
                preset_key=preset.preset_key,
                effective_context=int(row["effective_context"]),
                parallelism=int(row.get("effective_parallelism") or preset.parallelism),
                kv_cache_dtype=preset.kv_cache_dtype,
                placement_policy=preset.placement_policy,
                gpu_indices=tuple(row.get("target_gpus") or ()),
                execution_policy=preset.execution_policy,
                enabled=preset.enabled,
                is_default=preset.is_default,
                priority=preset.priority,
                runtime_key=row.get("runtime_key"),
            )
        )

    if not candidates:
        logger.warning(
            "runtime_plan_no_candidates",
            profile_id=profile.profile_id,
            base_model_id=profile.base_model_id,
            enabled_preset_keys=[preset.preset_key for preset in enabled_presets],
            fits_by_key={row["preset_key"]: row.get("fits") for row in rows},
        )
        raise ContextCapacityError("profile has no enabled runtime preset that fits")

    # Un rechazo por tamano no puede decidirse con el heuristico: medido
    # contra el tokenizador real va de 0,94x a 2,42x del valor verdadero. Asi
    # que el heuristico sirve para ELEGIR ventana (equivocarse ahi solo carga
    # una ventana mas grande de lo necesario) pero antes de RECHAZAR se busca
    # un numero de verdad:
    #
    #   1. Si hay un runtime de este modelo cargado, se le pide el conteo
    #      exacto a su tokenizador y se decide con eso.
    #   2. Si no hay ninguno, no se rechaza: se intenta con la ventana mas
    #      grande disponible y, si de verdad no cabe, lo dice el motor, que es
    #      el unico que cuenta bien.
    #
    # Reportado 2026-11-07: un prompt de ~32k tokens reales se estimaba en
    # ~55k y qwen3:32b (ventana de 40.960) lo rechazaba con 400 aunque cabia.
    explicit_key = _requested_preset_key(body)

    # La capacidad contra la que se afina tiene que ser la MISMA contra la que
    # se va a decidir el rechazo. Si el cliente pide un preset concreto, el
    # rechazo se compara con ESE preset, no con el mayor disponible: afinar
    # contra el mayor dejaba pasar sin medir un presupuesto que luego no cabia
    # en el preset pedido, y peor, el recorte "no rechaces por estimacion"
    # podia FABRICAR el fallo —recortaba a la ventana mayor (40.960) y el
    # preset pedido (30.720) no la cubria—.
    #
    # Reproducido 2026-11-08 contra qwen3:32b con el preset `normal` (30.720):
    #   prompt de ~19.000 tokens reales, estimado en 34.311
    #   34.311 < 40.960 (mayor)  -> no se afinaba
    #   34.311 > 30.720 (pedido) -> 400, con el worker cargado y sitio de sobra
    explicit_candidate = (
        next((item for item in candidates if item.preset_key == explicit_key), None)
        if explicit_key
        else None
    )
    capacidad_decisiva = (
        explicit_candidate.effective_context
        if explicit_candidate is not None
        else max(item.effective_context for item in candidates)
    )
    required_tokens, medido = await refine_required_tokens(
        model_manager,
        profile,
        body,
        required_tokens=required_tokens,
        capacity=capacidad_decisiva,
        default_output_reservation=default_output_reservation,
    )

    loaded_keys: set[str] = set()
    live_context_by_key: dict[str, int] = {}
    for runtime_state in await model_manager.list_states(include_derived=True):
        if runtime_state.status.value != "loaded":
            continue
        extra = runtime_state.extra_config if isinstance(runtime_state.extra_config, dict) else {}
        metadata = extra.get("_runtime")
        if isinstance(metadata, dict) and metadata.get("runtime_key"):
            loaded_keys.add(str(metadata["runtime_key"]))
            live = getattr(runtime_state, "live_context", None)
            if isinstance(live, dict) and live.get("n_ctx_slot"):
                live_context_by_key[str(metadata["runtime_key"])] = int(live["n_ctx_slot"])

    if explicit_key:
        explicit = explicit_candidate
        if explicit is None:
            raise ContextCapacityError(
                f"runtime preset '{explicit_key}' is disabled, unavailable, or does not fit"
            )
        if explicit.effective_context < required_tokens:
            raise ContextCapacityError(
                f"runtime preset '{explicit_key}' cannot cover {required_tokens} tokens"
            )
        chosen = explicit
        reason = "explicit"
    else:
        selected = select_runtime_preset(
            candidates,
            required_tokens=required_tokens,
            loaded_runtime_keys=loaded_keys,
            selection=str(policy.get("selection", "prefer_loaded")),
        )
        chosen = selected.preset
        reason = selected.reason

    logger.info(
        "runtime_plan_selected",
        profile_id=profile.profile_id,
        base_model_id=profile.base_model_id,
        preset_key=chosen.preset_key,
        selection_reason=reason,
        required_tokens=required_tokens,
        effective_context=chosen.effective_context,
        parallelism=chosen.parallelism,
        gpu_indices=list(chosen.gpu_indices),
        candidate_presets=[item.preset_key for item in candidates],
        fits_by_preset={row["preset_key"]: bool(row.get("fits")) for row in rows},
        loaded_runtime_keys=sorted(loaded_keys),
        explicit=bool(explicit_key),
    )

    default_candidate = next((item for item in candidates if item.is_default), chosen)

    def _overrides_for(candidate: ResolvedRuntimePreset) -> dict[str, Any]:
        candidate_preset = preset_by_key[candidate.preset_key]
        measured_contexts = policy.get("validated_context_tokens")
        measured_context = (
            measured_contexts.get(candidate.preset_key)
            if isinstance(measured_contexts, dict)
            else None
        )
        try:
            capacity_validated = int(measured_context) == candidate.effective_context
        except (TypeError, ValueError):
            capacity_validated = False
        measured_vram_by_preset = policy.get("validated_vram_mb")
        measured_vram = (
            measured_vram_by_preset.get(candidate.preset_key)
            if capacity_validated and isinstance(measured_vram_by_preset, dict)
            else None
        )
        try:
            capacity_vram_mb = max(0, int(measured_vram or 0))
        except (TypeError, ValueError):
            capacity_vram_mb = 0
        candidate_overrides = dict(profile.load_overrides or {})
        _apply_llama_cpp_preset_placement(candidate_overrides, state, candidate, gpu_states)
        unified_pool = False
        if is_llama_cpp_family(getattr(state, "backend_type", "")):
            section = candidate_overrides.get("llama_cpp")
            raw_unified = (
                section.get("kv_unified", section.get("kvUnified"))
                if isinstance(section, dict)
                else None
            )
            if raw_unified is None:
                raw_unified = resolve_llama_cpp_option(state, "kv_unified", False)
            unified_pool = (
                raw_unified.strip().lower() in {"1", "true", "yes", "on"}
                if isinstance(raw_unified, str)
                else bool(raw_unified)
            )
            if isinstance(section, dict):
                # Admission estimates run before resolve_use_case(). Materialize
                # the exact physical pool here so the scheduler never sizes a
                # normal one-GPU runtime using the base model's dual-GPU pool.
                section["ctx_size"] = (
                    candidate.effective_context
                    if unified_pool
                    else candidate.effective_context * candidate.parallelism
                )
                section["parallel_slots"] = candidate.parallelism
        candidate_overrides["use_case"] = {
            "context": candidate.effective_context,
            "slots": candidate.parallelism,
            "kv_dtype": candidate_preset.kv_cache_dtype,
        }
        if unified_pool:
            candidate_overrides["use_case"]["context_is_pool"] = True
        candidate_overrides["_runtime"] = {
            "profile_id": profile.profile_id,
            "base_model_id": profile.base_model_id,
            "preset_key": candidate_preset.preset_key,
            "runtime_key": candidate.runtime_key,
            "effective_context": candidate.effective_context,
            "context_is_pool": unified_pool,
            "capacity_validated": capacity_validated,
            "capacity_vram_mb": capacity_vram_mb or None,
            "requested_parallelism": candidate_preset.parallelism,
            "parallelism": candidate.parallelism,
            "kv_cache_dtype": candidate_preset.kv_cache_dtype,
            "placement_policy": candidate_preset.placement_policy,
            "execution_policy": candidate_preset.execution_policy,
            # Multi-GPU hint para los backends con pool shared (Ollama). El
            # coordinator no sabe pools, pero sí sabe que este preset pide >1 GPU.
            # ``_select_pool`` del backend usa esta lista para elegir un pool que
            # exponga todas estas GPUs juntas (``multi_gpu=true`` + ``gpus`` ⊇).
            "gpu_indices": list(candidate.gpu_indices),
            # What the vision projector took on each GPU when this preset was
            # probed (runtime_policy.context_probe): lets the worker make just
            # enough room before the first image instead of guessing.
            "context_probe_vision_mb": _probe_vision_mb(policy, candidate_preset.preset_key),
            "tracking_worker_key": compute_worker_key(
                profile.base_model_id,
                profile.load_overrides,
            ),
            "default_context": default_candidate.effective_context,
            "restore_default_when_idle": bool(policy.get("restore_default_when_idle", True)),
            "downgrade_grace_seconds": int(policy.get("downgrade_grace_seconds", 600)),
            "minimum_residency_seconds": int(policy.get("minimum_residency_seconds", 120)),
            "small_requests_extend_large_ttl": bool(
                policy.get("small_requests_extend_large_ttl", False)
            ),
        }
        return candidate_overrides

    preset = preset_by_key[chosen.preset_key]
    overrides = _overrides_for(chosen)
    if not chosen.is_default:
        default_overrides = _overrides_for(default_candidate)
        overrides["_runtime"]["default_worker_key"] = compute_worker_key(
            profile.base_model_id, default_overrides
        )
        overrides["_runtime"]["default_load_overrides"] = default_overrides
    worker_key = compute_worker_key(profile.base_model_id, overrides)
    preferred_gpu = chosen.gpu_indices[0] if len(chosen.gpu_indices) == 1 else None
    # load_policy lives on the base model (ModelConfig), not per-preset
    # (2026-10-30 — presets used to carry their own load_policy, which
    # unconditionally overrode the base model's: setting a model to "pin"
    # had no real effect once a request resolved through a profile, since
    # every preset defaulted to "warm"/"on_demand" and that value always
    # won). Only the DEFAULT preset for a profile inherits the base model's
    # residency policy; any other preset (a request that needed a bigger
    # context than usual) is always on_demand — a one-off excursion outside
    # the steady-state configuration shouldn't become pinned/warm residency.
    worker_load_policy = state.load_policy.value if chosen.is_default else "on_demand"
    return RuntimePlan(
        worker_key=worker_key,
        load_overrides=overrides,
        preferred_gpu=preferred_gpu,
        load_policy=worker_load_policy,
        preset_key=preset.preset_key,
        runtime_key=chosen.runtime_key,
        effective_context=chosen.effective_context,
        required_tokens=required_tokens,
        selection_reason=reason,
        kv_reservation_tokens=_reserva_kv(
            required_tokens,
            medido,
            chosen.effective_context,
            parallelism=chosen.parallelism,
        ),
        live_effective_context=(
            live_context_by_key.get(chosen.runtime_key)
            if chosen.runtime_key is not None
            and live_context_by_key.get(chosen.runtime_key, chosen.effective_context)
            < chosen.effective_context
            else None
        ),
    )
