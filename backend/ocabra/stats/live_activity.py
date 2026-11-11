"""Estado en vivo de las peticiones para el dashboard (2026-11-09).

``/ocabra/models/activity`` ya contaba peticiones en vuelo por modelo; aqui se
calcula lo que el dashboard nuevo necesita encima de eso, sin tocar la
admision:

* **Fase real de cada peticion**: en cola (con su puesto en el turno real del
  ``FairSlotGate``), prefill o generacion. El prefill trae progreso de verdad
  en llama.cpp (``return_progress``); en el resto de backends se infiere del
  primer token.
* **t/s de prefill y de generacion por peticion**, medidos sobre su propia
  fase. El ``tokensPerSecond`` historico divide la salida entre TODO el tiempo
  transcurrido (cola + prefill incluidos) y por eso infravalora.
* **Estimaciones**: duracion esperada de cada peticion (regresion del
  ``DurationEstimator``) y espera de la cola simulando la liberacion de slots.
* **Contexto por modelo**: niveles de contexto por colocacion (1 o 2 GPU), el
  activo, y lo reservado/ocupado por cada peticion. Ver la seccion "Barra de
  contexto" de ``docs/tasks/dashboard-redesign-plan.md``.

Todo es de solo lectura y tolerante a fallos: un dato que no se puede calcular
sale como ``None``, nunca rompe el endpoint.
"""

from __future__ import annotations

import asyncio
import heapq
import re
import time
from typing import Any

import httpx
import structlog

from ocabra.core.model_ref import is_llama_cpp_family

logger = structlog.get_logger(__name__)

# Backends con KV paginada compartida (sin parcelas por peticion).
_PAGED_BACKENDS = frozenset({"vllm", "sglang", "tensorrt_llm"})
# Backends con contexto reservado por slot al cargar.
_SLOT_BACKENDS = frozenset({"ollama", "bitnet"})

# Estimaciones por peticion: no cambian mientras vive, y el estimador es
# async. Se podan las que ya no estan en vuelo.
_estimate_cache: dict[str, int | None] = {}


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return int(value)
    return None


def _rate(tokens: int | None, seconds: float | None) -> float | None:
    if not tokens or not seconds or seconds <= 0:
        return None
    return round(tokens / seconds, 1)


def gate_requests_by_id(worker_pool: Any, keys: tuple[str, ...]) -> tuple[dict, dict, int | None]:
    """Leases y cola del gate por ``request_id`` para los worker keys dados."""
    leases: dict[str, dict] = {}
    queued: dict[str, dict] = {}
    peek = getattr(worker_pool, "peek_gate_requests", None)
    if not callable(peek):
        return leases, queued, None
    total_queued: int | None = None
    for key in keys:
        try:
            snap = peek(key)
        except Exception:  # noqa: BLE001 — diagnostico, nunca rompe la card
            snap = None
        if not snap:
            continue
        for lease in snap.get("leases", ()):
            if lease.get("requestId"):
                leases[lease["requestId"]] = lease
        for waiter in snap.get("queued", ()):
            if waiter.get("requestId"):
                queued[waiter["requestId"]] = waiter
        total_queued = (total_queued or 0) + len(snap.get("queued", ()))
    return leases, queued, total_queued


def request_live_detail(
    req: Any,
    *,
    now: float,
    lease: dict | None,
    queued: dict | None,
    load_phase: str | None,
    has_gate: bool,
    prefill_rate: tuple[float, str] | None = None,
) -> dict[str, Any]:
    """Campos nuevos de una peticion en vuelo (se suman a los de siempre).

    ``prefill_rate``: (tokens/s, fuente) de prefill de ese modelo segun su
    historial, para estimar el avance cuando el backend no lo informa.
    """
    elapsed_s = max(0.0, now - req.started_at)
    first_token_at = getattr(req, "first_token_at", None)
    streaming = getattr(req, "streaming", None)
    total = getattr(req, "prompt_tokens_total", None)
    cached = getattr(req, "prompt_tokens_cached", None)
    processed = getattr(req, "prompt_tokens_processed", None)
    progress_ms = getattr(req, "prompt_progress_ms", None)
    output = req.output_tokens_estimate

    admitted_at: float | None = None
    if lease is not None:
        admitted_at = now - (lease.get("heldMs") or 0) / 1000
    elif not has_gate:
        # Sin gate (backend sin admision por slots): entra directa.
        admitted_at = req.started_at

    if queued is not None:
        stage = "queued"
    elif load_phase in ("loading", "waiting_for_gpu"):
        stage = load_phase
    elif first_token_at is not None:
        stage = "generating"
    elif streaming is False:
        # Sin streaming no hay ningun evento hasta el final.
        stage = "processing"
    elif getattr(req, "prompt_progress_tracked", False) and total is None:
        # Admitida por oCabra pero el backend aun no ha empezado con ella.
        stage = "backend_wait"
    else:
        stage = "prefill"

    # Prefill: real si llama.cpp lo informa; si no, desde la admision hasta
    # el primer token sobre el tamaño estimado del prompt.
    prefill_tps: float | None = None
    prefill_source: str | None = None
    if total is not None and processed is not None and progress_ms:
        computed = max(0, processed - (cached or 0))
        prefill_tps = _rate(computed, progress_ms / 1000)
        prefill_source = "live"
    elif first_token_at is not None and admitted_at is not None:
        prompt = req.input_tokens_estimate
        prefill_tps = _rate(prompt, first_token_at - admitted_at)
        prefill_source = "estimate"

    generation_tps: float | None = None
    if first_token_at is not None and output:
        last = getattr(req, "last_activity_at", None) or now
        generation_tps = _rate(output, max(0.0, last - first_token_at))

    prefill_pct: float | None = None
    if total and processed is not None:
        prefill_pct = round(min(100.0, 100.0 * processed / total), 1)
    elif first_token_at is not None:
        prefill_pct = 100.0
    elif (
        stage == "prefill"
        and prefill_rate is not None
        and admitted_at is not None
        and req.input_tokens_estimate
    ):
        # Sin progreso del backend (Ollama, vLLM...): avance esperado al ritmo
        # de prefill medido para este modelo. Nunca llega al 100 % por si
        # solo: lo cierra el primer token real.
        rate, source = prefill_rate
        expected = rate * max(0.0, now - admitted_at)
        prefill_pct = round(min(99.0, 100.0 * expected / req.input_tokens_estimate), 1)
        prefill_tps = rate
        prefill_source = source

    return {
        "stage": stage,
        "streaming": streaming,
        "requestKind": getattr(req, "request_kind", None),
        "profileId": getattr(req, "profile_id", None),
        "apiKeyName": getattr(req, "api_key_name", None),
        "source": getattr(req, "source", None),
        "startedAt": req.started_at,
        "elapsedMs": int(elapsed_s * 1000),
        "admittedAfterMs": (
            int(max(0.0, admitted_at - req.started_at) * 1000) if admitted_at is not None else None
        ),
        "firstTokenAfterMs": (
            int(max(0.0, first_token_at - req.started_at) * 1000)
            if first_token_at is not None
            else None
        ),
        "kvReservedTokens": _int_or_none(lease.get("kvTokens"))
        if lease
        else (_int_or_none(queued.get("kvTokens")) if queued else None),
        "queuePosition": queued.get("position") if queued else None,
        # Solo para calcular; se retiran antes de responder (``strip_internal``):
        # la clave de usuario del gate (FIFO por usuario en la simulación), la
        # clave de los ritmos medidos y si el backend informa del progreso.
        "userKey": queued.get("userKey") if queued else None,
        "rateKey": getattr(req, "profile_id", None) or getattr(req, "model_id", None),
        "progressTracked": bool(getattr(req, "prompt_progress_tracked", False)),
        "queuedMs": queued.get("waitedMs") if queued else None,
        "prompt": {
            "totalTokens": total if total is not None else req.input_tokens_estimate,
            "cachedTokens": cached,
            "processedTokens": processed,
            "source": "live" if total is not None else "estimate",
        },
        "prefillPct": prefill_pct,
        "prefillTokensPerSecond": prefill_tps,
        "prefillSource": prefill_source,
        "outputTokens": output,
        "maxOutputTokens": getattr(req, "max_output_tokens", None),
        "generationTokensPerSecond": generation_tps,
    }


async def attach_estimates(
    details: list[dict[str, Any]],
    *,
    estimator: Any,
    model_id: str,
    backend_type: str | None,
    slots: int | None,
    kv_capacity: int | None = None,
    kv_reserved: int | None = None,
    rates: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Duracion esperada por peticion y cuando entra cada una de la cola.

    La cola se simula por eventos con las mismas dos condiciones que mira el
    gate: un slot libre y KV suficiente para la reserva de la peticion. En
    cada instante entran, en orden de turno, las que caben; una cabeza que no
    cabe bloquea las siguientes de SU usuario (FIFO por usuario) pero deja
    pasar las de otros, como ``FairSlotGate._dispatch_locked``. Asi una
    peticion pequeña puede adelantar a una grande que espera KV, y la UI lo
    muestra (``predictedStartOrder``) en vez de prometer el orden de turno.
    """
    estimate = getattr(estimator, "estimate", None)
    for detail in details:
        # Cuando llama.cpp da el total real del prompt se re-estima con el: el
        # recuento aproximado puede quedarse en una quinta parte (18k frente a
        # 70k tokens reales en produccion) y dejar el "restante" en 0 con el
        # prefill al 26 %.
        prompt = detail.get("prompt") or {}
        rid = (
            f"{detail['requestId']}:{prompt.get('totalTokens')}"
            if prompt.get("source") == "live" and prompt.get("totalKnown", True)
            else detail["requestId"]
        )
        if rid not in _estimate_cache:
            expected_ms: int | None = None
            if callable(estimate):
                try:
                    est = await estimate(
                        model_id,
                        backend_type=backend_type,
                        input_tokens=detail["prompt"]["totalTokens"],
                        max_tokens=detail.get("maxOutputTokens"),
                    )
                    if est is not None and est.expected_s and est.expected_s > 0:
                        expected_ms = int(est.expected_s * 1000)
                except Exception:  # noqa: BLE001 — sin estimacion, no sin card
                    expected_ms = None
            _estimate_cache[rid] = expected_ms
        expected_ms = _estimate_cache[rid]
        # Con ritmos medidos del modelo manda la duración física: el
        # estimador queda solo de respaldo.
        physical = physical_duration_ms(detail, (rates or {}).get(detail.get("rateKey") or ""))
        if physical:
            expected_ms = physical
        detail["expectedDurationMs"] = expected_ms
        if detail["stage"] == "queued":
            detail["estimatedRemainingMs"] = None
            continue
        if expected_ms is None:
            # Sin estimación general puede haber ritmos en vivo de la propia
            # petición: con ellos sí hay restante.
            running_ms = detail["elapsedMs"] - (detail.get("admittedAfterMs") or 0)
            live_left = _live_remaining_ms(detail, (rates or {}).get(detail.get("rateKey") or ""))
            if live_left is not None and live_left != _UNKNOWN:
                detail["estimatedRemainingMs"] = live_left
                detail["expectedDurationMs"] = running_ms + live_left
            else:
                detail["estimatedRemainingMs"] = None
            continue
        # Lo esperado cuenta desde la admision, no desde la llegada.
        running_ms = detail["elapsedMs"] - (detail.get("admittedAfterMs") or 0)
        remaining = max(0, expected_ms - running_ms)
        if detail["stage"] in ("backend_wait", "loading", "waiting_for_gpu"):
            # Aun no ha empezado (espera al backend, a la carga del modelo o a
            # que se libere la GPU): le queda, como poco, toda su duracion. Antes
            # el restante contaba hacia atras durante una carga de un minuto y
            # llegaba a 0 sin haber procesado nada.
            remaining = expected_ms
        live_left = _live_remaining_ms(detail, (rates or {}).get(detail.get("rateKey") or ""))
        if live_left == _UNKNOWN:
            # El prompt es más largo de lo estimado y el backend no da su
            # total: mejor sin restante que uno inventado.
            detail["expectedDurationMs"] = None
            remaining = None
        elif live_left is not None:
            # Con ritmos medidos EN ESTA petición mandan ellos: el histórico
            # del modelo puede ser 3 veces más lento (gemma4: 726 t/s de
            # mediana frente a 2.300 t/s en vivo) y el restante salía inflado.
            # La duración esperada cuadra con él: transcurrido + restante.
            remaining = live_left
            detail["expectedDurationMs"] = running_ms + live_left
        detail["estimatedRemainingMs"] = remaining

    queued = sorted(
        (d for d in details if d["stage"] == "queued"),
        key=lambda d: d.get("queuePosition") or 0,
    )
    if not queued:
        return
    for detail in queued:
        detail["queueWaitEstimateMs"] = None
        detail["predictedStartOrder"] = None
    running = [d for d in details if d["stage"] != "queued"]
    # Sin estimacion de lo que esta en marcha no hay base para la espera:
    # mejor ``None`` que un "0 s" falso.
    if any(d.get("estimatedRemainingMs") is None for d in running):
        return
    fallback = next((d["expectedDurationMs"] for d in details if d.get("expectedDurationMs")), None)

    n_slots = max(1, int(slots or 1))
    use_kv = bool(kv_capacity)
    # (fin, reserva KV) de lo que ocupa slot. Si hay mas en marcha que slots
    # (llama.cpp encola por dentro) cuentan como ocupando hasta que acaban.
    busy: list[tuple[float, int]] = [
        (float(d["estimatedRemainingMs"]), int(d.get("kvReservedTokens") or 0)) for d in running
    ]
    heapq.heapify(busy)
    if use_kv:
        reserved = kv_reserved if kv_reserved is not None else sum(kv for _, kv in busy)
        free_kv = max(0, int(kv_capacity) - int(reserved))
    else:
        free_kv = 0
    pending = list(queued)
    now_ms = 0.0
    order = 0
    while pending:
        blocked_users: set[str] = set()
        still: list[dict[str, Any]] = []
        for item in pending:
            user = str(item.get("userKey") or item.get("username") or item["requestId"])
            kv = int(item.get("kvReservedTokens") or 0)
            fits_slot = len(busy) < n_slots
            fits_kv = (not use_kv) or kv <= free_kv
            duration = item.get("expectedDurationMs") or fallback
            if user in blocked_users or not (fits_slot and fits_kv) or duration is None:
                blocked_users.add(user)
                still.append(item)
                continue
            item["queueWaitEstimateMs"] = int(now_ms)
            item["predictedStartOrder"] = order
            order += 1
            heapq.heappush(busy, (now_ms + float(duration), kv))
            if use_kv:
                free_kv -= kv
        pending = still
        if not pending or not busy:
            # Lo que queda no cabe ni con todo libre (o le falta duracion):
            # no hay estimacion honesta para ello.
            break
        finish, kv = heapq.heappop(busy)
        now_ms = max(now_ms, finish)
        if use_kv:
            free_kv += kv
        # Libera tambien lo que acabe en el mismo instante.
        while busy and busy[0][0] <= now_ms:
            _, kv = heapq.heappop(busy)
            if use_kv:
                free_kv += kv


_UNKNOWN = -1


def _live_remaining_ms(detail: dict[str, Any], rates: dict[str, Any] | None) -> int | None:
    """Restante con los ritmos medidos en esta petición, si los hay.

    Devuelve ms, ``None`` si no hay ritmos en vivo (se usa la estimación
    general) o ``_UNKNOWN`` si se sabe que no se puede estimar: sin streaming,
    cuando lo procesado ya supera la estimación del prompt, no hay forma de
    saber cuánto falta (verificado en producción: el restante se quedaba
    clavado en 1,4 s durante 15 s de prefill).
    """
    prompt = detail.get("prompt") or {}
    typical = (rates or {}).get("output")
    asked = detail.get("maxOutputTokens")
    expected_out = min(asked, typical) if asked and typical else (asked or typical)
    out_so_far = detail.get("outputTokens") or 0
    detail["remainingIsUpperBound"] = False
    if expected_out and out_so_far >= expected_out:
        # Ya va por encima de la salida típica del modelo: el único límite
        # conocido es su tope. Restante "como mucho" hasta él; sin tope, no
        # se sabe. Antes daba 0 con la petición a medias (9k de 16,4k tokens).
        if asked and asked > out_so_far:
            expected_out = asked
            detail["remainingIsUpperBound"] = True
        else:
            expected_out = None
    hist_gen = (rates or {}).get("generation")
    stage = detail.get("stage")
    if (
        stage == "prefill"
        and prompt.get("source") == "live"
        and detail.get("prefillSource") == "live"
    ):
        tps = detail.get("prefillTokensPerSecond")
        if not tps:
            return None
        processed = prompt.get("processedTokens") or 0
        if prompt.get("totalKnown") is False:
            estimated = detail.get("estimatedPromptTokens") or 0
            if processed >= estimated:
                return _UNKNOWN
            left = estimated - processed
        else:
            left = max(0, (prompt.get("totalTokens") or 0) - processed)
        gen_ms = expected_out / hist_gen * 1000 if expected_out and hist_gen else 0
        return int(left / tps * 1000 + gen_ms)
    if stage == "generating":
        # Ritmo en vivo si ya lo hay; si no (primer instante), el histórico de
        # GENERACIÓN del modelo. Antes caía a la duración física completa y el
        # restante saltaba de 1 s a 43 s al empezar a generar.
        tps = detail.get("generationTokensPerSecond") or hist_gen
        out = detail.get("outputTokens") or 0
        if tps and expected_out:
            return int(max(0, expected_out - out) / tps * 1000)
    return None


def prune_estimates(live_ids: set[str]) -> None:
    for key in [k for k in _estimate_cache if k.split(":", 1)[0] not in live_ids]:
        _estimate_cache.pop(key, None)


# ── Contexto ────────────────────────────────────────────────────────────────


def context_mode(state: Any) -> str | None:
    """Como reserva contexto el backend del modelo (``None`` = no aplica)."""
    caps = getattr(state, "capabilities", None)
    if caps is not None and not (
        getattr(caps, "chat", False) or getattr(caps, "completion", False)
    ):
        return None
    backend = str(getattr(state, "backend_type", "") or "")
    if is_llama_cpp_family(backend):
        from ocabra.core.runtime_preset_capacity import _uses_unified_kv_pool

        try:
            return "pool" if _uses_unified_kv_pool(state) else "slots"
        except Exception:  # noqa: BLE001
            return "slots"
    if backend in _SLOT_BACKENDS:
        return "slots"
    if backend in _PAGED_BACKENDS:
        return "paged"
    return None


def _pick_profile(profiles: list[Any], model_id: str, preferred_profile_id: str | None) -> Any:
    candidates = [
        p
        for p in profiles
        if getattr(p, "base_model_id", None) == model_id
        and getattr(p, "profile_kind", "execution") == "execution"
        and getattr(p, "enabled", True)
    ]
    if not candidates:
        return None
    for p in candidates:
        if preferred_profile_id and p.profile_id == preferred_profile_id:
            return p
    with_presets = [p for p in candidates if any(r.enabled for r in p.runtime_presets or ())]
    for pool in (with_presets, candidates):
        for p in pool:
            if getattr(p, "is_default", False):
                return p
        if pool:
            return pool[0]
    return None


async def model_context(
    state: Any,
    *,
    runtime_meta: dict | None,
    gate: dict | None,
    request_details: list[dict[str, Any]],
    profiles: list[Any],
    gpu_states: list[Any],
    worker_port: int | None = None,
    live_context: dict | None = None,
) -> dict[str, Any] | None:
    """Objeto ``context`` normalizado de un modelo cargado.

    ``live_context``: ``ModelState.live_context`` del worker si su contexto se
    cambió en vivo (v0.65.0, sin recargar pesos). Manda sobre el tamaño del
    preset, que sigue siendo el de carga.
    """
    mode = context_mode(state)
    if mode is None:
        return None
    paged_usage_pct = await vllm_kv_usage_pct(worker_port) if mode == "paged" else None

    from ocabra.core.model_manager_helpers import resolve_llama_cpp_option
    from ocabra.core.runtime_preset_capacity import runtime_capacity_cache

    meta = runtime_meta or {}
    tiers: list[dict[str, Any]] = []
    profile = _pick_profile(profiles, state.model_id, meta.get("profile_id"))
    if profile is not None and any(r.enabled for r in profile.runtime_presets or ()):
        try:
            rows = await runtime_capacity_cache.resolve(profile, state, gpu_states)
        except Exception:  # noqa: BLE001 — sin niveles, la barra aun sirve
            logger.debug("live_context_resolve_failed", model_id=state.model_id)
            rows = []
        for row in rows:
            if not row.get("resolved") or not row.get("fits") or not row.get("effective_context"):
                continue
            tiers.append(
                {
                    "presetKey": row.get("preset_key"),
                    "gpus": list(row.get("target_gpus") or []),
                    "maxTokens": int(row["effective_context"]),
                    "slots": row.get("effective_parallelism"),
                }
            )
    tiers.sort(key=lambda t: (t["maxTokens"], len(t["gpus"])))

    active_tokens: int | None = None
    if meta.get("effective_context"):
        active_tokens = int(meta["effective_context"])
    elif gate and gate.get("kv_capacity_tokens") and mode == "pool":
        active_tokens = int(gate["kv_capacity_tokens"])
    else:
        try:
            ctx = (
                resolve_llama_cpp_option(state, "ctx_size", 0)
                if is_llama_cpp_family(state.backend_type)
                else 0
            )
            active_tokens = int(ctx or 0) or None
        except Exception:  # noqa: BLE001
            active_tokens = None
    nominal_tokens = active_tokens
    live = live_context if isinstance(live_context, dict) else None
    if live:
        try:
            live_pool = int(live.get("n_ctx") or 0)
            live_slot = int(live.get("n_ctx_slot") or 0)
            live_parallel = int(live.get("n_parallel") or 0)
        except (TypeError, ValueError):
            live_pool = live_slot = live_parallel = 0
        # Pool unificado: una petición puede usar todo el pool. Por slots: su
        # parcela.
        if mode == "pool" and live_pool:
            active_tokens = live_pool
        elif live_slot or (live_pool and live_parallel):
            active_tokens = live_slot or live_pool // live_parallel
    active_gpus = list(meta.get("gpu_indices") or getattr(state, "current_gpu", []) or [])
    if not tiers and (nominal_tokens or active_tokens):
        # Sin presets, el único nivel es el tamaño con el que se cargó.
        tiers = [
            {
                "presetKey": meta.get("preset_key"),
                "gpus": active_gpus,
                "maxTokens": nominal_tokens or active_tokens,
                "slots": meta.get("parallelism"),
            }
        ]

    single = [t["maxTokens"] for t in tiers if len(t["gpus"]) == 1]
    scale = max([t["maxTokens"] for t in tiers] + [active_tokens or 0]) or None

    segments: list[dict[str, Any]] = []
    for detail in request_details:
        if detail.get("stage") == "queued":
            continue
        prompt = detail.get("prompt") or {}
        used = prompt.get("processedTokens") if prompt.get("source") == "live" else None
        if used is None and detail.get("stage") == "generating":
            used = prompt.get("totalTokens")
        if used is not None:
            used += detail.get("outputTokens") or 0
        segments.append(
            {
                "requestId": detail.get("requestId"),
                "reservedTokens": detail.get("kvReservedTokens"),
                "usedTokens": used,
            }
        )

    return {
        "mode": mode,
        "scaleMaxTokens": scale,
        "activeMaxTokens": active_tokens,
        # Cambiado en vivo: el tamaño con el que se cargó, para que la UI lo diga.
        "liveResized": bool(live) and nominal_tokens != active_tokens,
        "nominalMaxTokens": nominal_tokens if live else None,
        "activePresetKey": meta.get("preset_key"),
        "activeGpus": active_gpus,
        "singleGpuMaxTokens": max(single) if single else None,
        "tiers": tiers,
        "reservedTokens": _int_or_none(gate.get("kv_reserved_tokens")) if gate else None,
        "segments": segments,
        "usagePct": paged_usage_pct if mode == "paged" else None,
        "source": "live" if any(s["usedTokens"] is not None for s in segments) else "estimate",
    }


# ── vLLM: ocupacion del pool KV ─────────────────────────────────────────────

# vLLM >= 0.10 (motor v1) publica ``vllm:kv_cache_usage_perc``; el motor v0
# lo llamaba ``vllm:gpu_cache_usage_perc``. Ambos van de 0 a 1.
_VLLM_KV_METRIC = re.compile(
    r"^vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$",
    re.MULTILINE,
)
_METRICS_TTL_S = 2.0
_metrics_cache: dict[int, tuple[float, float | None]] = {}


def parse_vllm_kv_usage(text: str) -> float | None:
    """Porcentaje (0-100) de KV ocupada segun el texto Prometheus de vLLM.

    Con varias series (una por motor en data-parallel) se toma la mayor: es
    la que antes se queda sin sitio.
    """
    values = []
    for match in _VLLM_KV_METRIC.finditer(text or ""):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    if not values:
        return None
    return round(max(0.0, min(1.0, max(values))) * 100, 1)


async def vllm_kv_usage_pct(port: int | None) -> float | None:
    """Lee ``/metrics`` del worker vLLM (cache de 2 s, timeout corto)."""
    if not port:
        return None
    now = time.monotonic()
    cached = _metrics_cache.get(port)
    if cached and now - cached[0] < _METRICS_TTL_S:
        return cached[1]
    value: float | None = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(0.8)) as client:
            resp = await client.get(f"http://127.0.0.1:{port}/metrics")
            if resp.status_code == 200:
                value = parse_vllm_kv_usage(resp.text)
    except (httpx.HTTPError, OSError):
        value = None
    _metrics_cache[port] = (now, value)
    return value


# ── VRAM real por GPU de cada modelo ────────────────────────────────────────

_PROC_TREE_TTL_S = 5.0
_proc_tree_cache: dict[int, tuple[float, set[int]]] = {}


def _process_tree(pid: int) -> set[int]:
    """PID del worker y todos sus descendientes (cache de 5 s).

    Algunos backends lanzan un wrapper de Python que a su vez arranca el
    proceso que reserva la VRAM; NVML atribuye la memoria al hijo.
    """
    now = time.monotonic()
    cached = _proc_tree_cache.get(pid)
    if cached and now - cached[0] < _PROC_TREE_TTL_S:
        return cached[1]
    pids = {pid}
    try:
        import psutil

        pids |= {child.pid for child in psutil.Process(pid).children(recursive=True)}
    except Exception:  # noqa: BLE001 — proceso muerto o sin permiso
        pass
    _proc_tree_cache[pid] = (now, pids)
    return pids


async def vram_by_gpu(
    worker_pids: dict[str, list[int]], gpu_states: list[Any]
) -> dict[str, dict[str, int]]:
    """MB que cada modelo ocupa en cada GPU, medidos por NVML.

    ``worker_pids``: modelo -> PIDs de sus workers. Se cruzan con los procesos
    que NVML ve en cada tarjeta. Sustituye al reparto "a partes iguales" de un
    modelo con tensor-split, que podia equivocarse por gigas (nemotron: 9,6 GB
    en la 3060 y 21,4 en la 3090, no 15,4 en cada una). Los modelos que
    comparten proceso (Ollama) no se atribuyen: NVML no distingue cual es.
    """
    if not worker_pids or not gpu_states:
        return {}
    trees = await asyncio.to_thread(
        lambda: {
            model: set().union(*(_process_tree(p) for p in pids))
            for model, pids in worker_pids.items()
        }
    )
    owners: dict[int, list[str]] = {}
    for model, pids in trees.items():
        for pid in pids:
            owners.setdefault(pid, []).append(model)
    result: dict[str, dict[str, int]] = {}
    for gpu in gpu_states:
        for proc in getattr(gpu, "processes", []) or []:
            models = owners.get(int(getattr(proc, "pid", 0) or 0), [])
            if len(models) != 1:
                continue
            per_gpu = result.setdefault(models[0], {})
            key = str(gpu.index)
            per_gpu[key] = per_gpu.get(key, 0) + int(getattr(proc, "used_vram_mb", 0) or 0)
    return result


# ── Ritmo de prefill por modelo (para backends sin progreso) ───────────────

_PREFILL_RATE_TTL_S = 60.0


async def _measured_model_rates(keys: list[str]) -> dict[str, dict[str, float]]:
    """Ritmos medidos por modelo en las últimas 6 h (medianas) y salida típica.

    ``prefill``: tokens/s de prefill · ``generation``: tokens/s de generación ·
    ``output``: percentil 75 de tokens de salida. Solo cuentan peticiones que
    los informan (llama.cpp y Ollama al terminar).
    """
    import sqlalchemy as sa

    from ocabra.database import AsyncSessionLocal
    from ocabra.db.stats import RequestStat

    since = sa.func.now() - sa.text("interval '6 hours'")

    def median(col):
        return sa.func.percentile_cont(0.5).within_group(col)

    stmt = (
        sa.select(
            RequestStat.model_id,
            median(
                sa.case(
                    (RequestStat.prompt_tokens_per_second > 0, RequestStat.prompt_tokens_per_second)
                )
            ),
            median(
                sa.case(
                    (
                        RequestStat.predicted_tokens_per_second > 0,
                        RequestStat.predicted_tokens_per_second,
                    )
                )
            ),
            sa.func.percentile_cont(0.75).within_group(
                sa.case((RequestStat.output_tokens > 0, RequestStat.output_tokens))
            ),
        )
        .where(RequestStat.model_id.in_(keys), RequestStat.started_at > since)
        .group_by(RequestStat.model_id)
    )
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(stmt)).all()
    out: dict[str, dict[str, float]] = {}
    for model, prefill, generation, output in rows:
        values = {
            name: float(value)
            for name, value in (
                ("prefill", prefill),
                ("generation", generation),
                ("output", output),
            )
            if value
        }
        if values:
            out[model] = values
    return out


_model_rates_cache: dict[str, tuple[float, dict[str, float]]] = {}


async def model_rates(keys: set[str], estimator: Any) -> dict[str, dict[str, Any]]:
    """Ritmos por modelo con su fuente, cacheados 60 s (también los vacíos).

    Devuelve ``{modelo: {"prefill": (t/s, fuente), "generation": t/s,
    "output": tokens}}``. El prefill cae a la regresión del estimador si no
    hay medidas; generación y salida solo existen medidas.
    """
    now = time.monotonic()
    missing = [
        k
        for k in keys
        if k
        and (k not in _model_rates_cache or now - _model_rates_cache[k][0] > _PREFILL_RATE_TTL_S)
    ]
    if missing:
        try:
            measured = await asyncio.wait_for(_measured_model_rates(missing), timeout=2.0)
        except Exception:  # noqa: BLE001 — sin BD: solo la regresión
            measured = {}
        for key in missing:
            _model_rates_cache[key] = (now, measured.get(key, {}))
    calibrations = getattr(estimator, "_calibrations", {}) or {}
    result: dict[str, dict[str, Any]] = {}
    for key in keys:
        if key not in _model_rates_cache:
            continue
        values = _model_rates_cache[key][1]
        entry: dict[str, Any] = {}
        if values.get("prefill"):
            entry["prefill"] = (round(values["prefill"], 1), "historical")
        else:
            regression = getattr(calibrations.get(key), "chat_regression", None)
            a_in = getattr(regression, "a_in", None)
            if a_in and a_in > 0:
                entry["prefill"] = (round(1000.0 / a_in, 1), "regression")
        if values.get("generation"):
            entry["generation"] = round(values["generation"], 1)
        if values.get("output"):
            entry["output"] = int(values["output"])
        if entry:
            result[key] = entry
    return result


async def prefill_rates(keys: set[str], estimator: Any) -> dict[str, tuple[float, str]]:
    """Solo el ritmo de prefill de ``model_rates`` (compatibilidad)."""
    rates = await model_rates(keys, estimator)
    return {k: v["prefill"] for k, v in rates.items() if v.get("prefill")}


def physical_duration_ms(detail: dict[str, Any], rates: dict[str, Any] | None) -> int | None:
    """Duración esperada con los ritmos medidos del propio modelo.

    prompt ÷ ritmo de prefill + salida esperada ÷ ritmo de generación. La
    salida esperada es el ``max_tokens`` pedido acotado por el p75 de salida
    del modelo (o el p75 si no hay tope). El estimador de duración ajusta una
    recta global y con prompts largos se quedaba muy corto: 15 s para un
    prompt de 57k tokens que solo de prefill tardaba más de un minuto.
    """
    if not rates:
        return None
    prefill = rates.get("prefill")
    prompt_tokens = (detail.get("prompt") or {}).get("totalTokens")
    if not prefill or not prompt_tokens:
        return None
    ms = prompt_tokens / prefill[0] * 1000
    generation = rates.get("generation")
    typical = rates.get("output")
    asked = detail.get("maxOutputTokens")
    expected_out = min(asked, typical) if asked and typical else (asked or typical)
    if generation and expected_out:
        ms += expected_out / generation * 1000
    return int(ms)


def strip_internal(details: list[dict[str, Any]]) -> None:
    """Quita lo que solo hacia falta para calcular (no sale del servidor)."""
    for detail in details:
        for key in ("userKey", "rateKey", "progressTracked", "estimatedPromptTokens"):
            detail.pop(key, None)


# ── Peticiones sin streaming en llama.cpp: progreso desde /slots ────────────

STALL_SECONDS = 15.0
_SLOTS_TTL_S = 1.0
_slots_cache: dict[int, tuple[float, list[dict[str, Any]] | None]] = {}
# request_id -> seguimiento entre consultas: primera muestra, último cambio y
# primer token visto, para sacar ritmos y detectar parones.
_slot_track: dict[str, dict[str, float]] = {}


async def llama_slots(port: int | None) -> list[dict[str, Any]] | None:
    """``GET /slots`` del llama-server (caché de 1 s, timeout corto)."""
    if not port:
        return None
    now = time.monotonic()
    cached = _slots_cache.get(port)
    if cached and now - cached[0] < _SLOTS_TTL_S:
        return cached[1]
    value: list[dict[str, Any]] | None = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(0.8)) as client:
            resp = await client.get(f"http://127.0.0.1:{port}/slots")
            if resp.status_code == 200 and isinstance(resp.json(), list):
                value = resp.json()
    except (httpx.HTTPError, OSError, ValueError):
        value = None
    _slots_cache[port] = (now, value)
    return value


def _slot_counts(slot: dict[str, Any]) -> tuple[int, int, int, int]:
    total = int(slot.get("n_prompt_tokens") or 0)
    processed = int(slot.get("n_prompt_tokens_processed") or 0)
    cached = int(slot.get("n_prompt_tokens_cache") or 0)
    next_token = slot.get("next_token")
    if isinstance(next_token, list):
        next_token = next_token[0] if next_token else {}
    decoded = int((next_token or {}).get("n_decoded") or 0)
    return total, processed, cached, decoded


def apply_slot_progress(
    details: list[dict[str, Any]], slots: list[dict[str, Any]] | None, *, now: float
) -> None:
    """Progreso real de las peticiones a llama.cpp que no lo traen en el stream.

    Las peticiones sin streaming (la mayoría del tráfico de benchmarks y de
    muchos clientes) no reciben ``prompt_progress``, pero ``/slots`` las ve. Se
    emparejan así: los slots cuyo total de prompt coincide con el de una
    petición que sí trae progreso se descartan; los demás, por orden de tarea
    (``id_task``), se asignan a las peticiones sin progreso por orden de
    admisión. Una admitida sin slot procesando espera al backend.
    """
    untracked = [
        d
        for d in details
        if d["stage"] in ("prefill", "processing", "backend_wait")
        and not d.get("progressTracked")
        and (d.get("prompt") or {}).get("source") != "live"
        and d.get("admittedAfterMs") is not None
    ]
    if not untracked or not slots:
        return
    tracked_totals: dict[int, int] = {}
    for d in details:
        prompt = d.get("prompt") or {}
        if prompt.get("source") == "live" and prompt.get("totalTokens"):
            tracked_totals[prompt["totalTokens"]] = tracked_totals.get(prompt["totalTokens"], 0) + 1
    free: list[dict[str, Any]] = []
    for slot in sorted(
        (s for s in slots if s.get("is_processing")), key=lambda s: s.get("id_task") or 0
    ):
        total = _slot_counts(slot)[0]
        if tracked_totals.get(total):
            tracked_totals[total] -= 1
            continue
        free.append(slot)
    untracked.sort(key=lambda d: (d.get("startedAt") or 0) + (d.get("admittedAfterMs") or 0) / 1000)
    for detail, slot in zip(untracked, free, strict=False):
        total, processed, cached, decoded = _slot_counts(slot)
        rid = detail["requestId"]
        track = _slot_track.setdefault(
            rid, {"t0": now, "p0": processed, "progress": -1.0, "changed": now}
        )
        progress = float(processed + decoded)
        if progress != track["progress"]:
            track["progress"] = progress
            track["changed"] = now
        # OJO: en /slots, ``n_prompt_tokens`` es lo cargado en el slot hasta
        # ahora (va 512 por delante de lo procesado), NO el total del prompt:
        # visto en producción con un prompt de 43.912 tokens que /slots
        # mostraba como 5.128, 11.272, ... El total real solo lo da
        # ``prompt_progress`` en streaming. Aquí lo procesado es real y el
        # total, la estimación (nunca menos de lo ya procesado); sin total
        # fiable no se da porcentaje.
        done_prompt = cached + processed
        estimated_total = (detail.get("prompt") or {}).get("totalTokens") or 0
        detail["prompt"] = {
            "totalTokens": max(estimated_total, done_prompt) or None,
            "cachedTokens": cached,
            "processedTokens": done_prompt,
            "source": "live",
            "totalKnown": False,
        }
        # Interno: la estimación original, para saber cuándo se ha superado.
        detail["estimatedPromptTokens"] = estimated_total
        detail["prefillPct"] = None
        detail["prefillSource"] = "live"
        detail["lastActivitySecondsAgo"] = round(now - track["changed"], 1)
        if decoded > 0:
            detail["stage"] = "generating"
            detail["prefillPct"] = 100.0
            if "t_first" not in track:
                track["t_first"] = now
                track["d0"] = float(decoded)
            detail["outputTokens"] = decoded
            elapsed = now - track["t_first"]
            rate = round((decoded - track["d0"]) / elapsed, 1) if elapsed >= 1 else None
            detail["generationTokensPerSecond"] = rate or detail.get("generationTokensPerSecond")
            # Instante del primer token: /slots no lo da. Si la primera vez que
            # la vemos ya llevaba tokens, se retrocede al ritmo medido (sin
            # ritmo aún, el momento en que se vio). Sin esto la barra de la
            # petición salía vacía: no sabía dónde empezaba la generación.
            first_token = track["t_first"]
            if rate and track["d0"] > 0:
                first_token = track["t_first"] - track["d0"] / rate
            started = detail.get("startedAt") or now
            admitted_ms = detail.get("admittedAfterMs") or 0
            detail["firstTokenAfterMs"] = max(admitted_ms, int((first_token - started) * 1000))
        elif processed > 0 or cached > 0:
            detail["stage"] = "prefill"
            elapsed = now - track["t0"]
            if elapsed >= 1 and processed > track["p0"]:
                detail["prefillTokensPerSecond"] = round((processed - track["p0"]) / elapsed, 1)
        else:
            detail["stage"] = "backend_wait"
        detail["progressTracked"] = True
    # Admitidas sin slot que les corresponda: esperan dentro del backend.
    for detail in untracked[len(free) :]:
        if detail["stage"] in ("prefill", "processing"):
            detail["stage"] = "backend_wait"
            detail["prefillPct"] = None
            detail["prefillTokensPerSecond"] = None
            detail["prefillSource"] = None


def mark_stalls(details: list[dict[str, Any]]) -> None:
    """``stalled``: hay una señal de vida que vigilar y lleva 15 s parada.

    Solo cuenta lo que emite eventos: la generación en streaming y el prefill
    con progreso real (``prompt_progress`` o ``/slots``). Una petición sin
    streaming o un prefill de Ollama no emiten nada hasta el final, y marcarlas
    como paradas era el falso "sin actividad del worker" que salía siempre.
    """
    for detail in details:
        stage = detail.get("stage")
        live_prompt = (detail.get("prompt") or {}).get("source") == "live"
        watched = (
            stage == "generating" and (detail.get("streaming") or detail.get("progressTracked"))
        ) or (stage == "prefill" and live_prompt)
        detail["stalled"] = bool(
            watched and (detail.get("lastActivitySecondsAgo") or 0) >= STALL_SECONDS
        )


def prune_slot_track(live_ids: set[str]) -> None:
    for rid in [rid for rid in _slot_track if rid not in live_ids]:
        _slot_track.pop(rid, None)


# ── Estado de carga del modelo ──────────────────────────────────────────────

_LOAD_TIME_TTL_S = 300.0
_load_time_cache: tuple[float, dict[str, int]] | None = None


async def typical_load_ms() -> dict[str, int]:
    """Mediana de duración de carga por modelo canónico (últimos 30 días).

    ``model_load_stats`` guarda el worker (``modelo::hash`` si es un preset);
    se agrupa por el modelo canónico. Caché de 5 min.
    """
    global _load_time_cache
    now = time.monotonic()
    if _load_time_cache and now - _load_time_cache[0] < _LOAD_TIME_TTL_S:
        return _load_time_cache[1]
    result: dict[str, int] = {}
    try:
        import sqlalchemy as sa

        from ocabra.database import AsyncSessionLocal
        from ocabra.db.stats import ModelLoadStat

        canonical = sa.func.split_part(ModelLoadStat.model_id, "::", 1)
        stmt = (
            sa.select(
                canonical, sa.func.percentile_cont(0.5).within_group(ModelLoadStat.duration_ms)
            )
            .where(
                ModelLoadStat.started_at > sa.func.now() - sa.text("interval '30 days'"),
                ModelLoadStat.duration_ms > 0,
            )
            .group_by(canonical)
        )

        async def run() -> list:
            async with AsyncSessionLocal() as session:
                return (await session.execute(stmt)).all()

        rows = await asyncio.wait_for(run(), timeout=2.0)
        result = {model: int(ms) for model, ms in rows if ms}
    except Exception:  # noqa: BLE001 — sin BD: sin estimación de carga
        result = {}
    _load_time_cache = (now, result)
    return result


def model_phase(
    mm: Any,
    state: Any,
    worker_keys: list[str],
    *,
    now: float,
    typical_ms: dict[str, int],
) -> dict[str, Any]:
    """Estado del modelo para la píldora de la card.

    ``phase``: ``waiting_for_gpu`` (desalojando a ``waitingOn`` para hacer
    sitio), ``waiting_for_service_gpu`` (GPU retenida por el servicio
    ``waitingOn``), ``loading`` o ``ready``. Con carga en curso, cuánto lleva y
    cuánto suele tardar.
    """
    phase, waiting_on = "ready", None
    keys = [state.model_id, *worker_keys]
    for key in keys:
        try:
            blocker = mm.service_gpu_blocker(key)
        except Exception:  # noqa: BLE001
            blocker = None
        if blocker:
            phase, waiting_on = "waiting_for_service_gpu", blocker
            break
        try:
            key_phase, detail = mm.get_load_phase(key)
        except Exception:  # noqa: BLE001
            continue
        if key_phase == "waiting_for_gpu":
            phase, waiting_on = key_phase, detail
            break
        if key_phase == "loading":
            phase = "loading"
    if (
        phase == "ready"
        and getattr(getattr(state, "status", None), "value", getattr(state, "status", None))
        == "loading"
    ):
        phase = "loading"
    started = [t for t in (mm.load_started_at(k) for k in keys) if t]
    return {
        "phase": phase,
        "waitingOn": waiting_on,
        "loadingForMs": int((now - min(started)) * 1000) if started else None,
        "typicalLoadMs": typical_ms.get(state.model_id),
    }
