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
        # Solo para la simulacion de cola (FIFO por usuario); se retira antes
        # de responder, ver ``strip_internal``.
        "userKey": queued.get("userKey") if queued else None,
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
            if prompt.get("source") == "live"
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
        detail["expectedDurationMs"] = expected_ms
        if expected_ms is None or detail["stage"] == "queued":
            detail["estimatedRemainingMs"] = None
            continue
        # Lo esperado cuenta desde la admision, no desde la llegada.
        running_ms = detail["elapsedMs"] - (detail.get("admittedAfterMs") or 0)
        remaining = max(0, expected_ms - running_ms)
        if detail["stage"] == "backend_wait":
            # Aun no ha empezado: le queda, como poco, toda su duracion.
            remaining = expected_ms
        # Con progreso real, nunca menos que el prefill que falta a su ritmo.
        tps = detail.get("prefillTokensPerSecond")
        if (
            detail["stage"] == "prefill"
            and prompt.get("source") == "live"
            and tps
            and prompt.get("totalTokens")
        ):
            left = max(0, prompt["totalTokens"] - (prompt.get("processedTokens") or 0))
            remaining = max(remaining, int(left / tps * 1000))
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
) -> dict[str, Any] | None:
    """Objeto ``context`` normalizado de un modelo cargado."""
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
    active_gpus = list(meta.get("gpu_indices") or getattr(state, "current_gpu", []) or [])
    if not tiers and active_tokens:
        tiers = [
            {
                "presetKey": meta.get("preset_key"),
                "gpus": active_gpus,
                "maxTokens": active_tokens,
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
_prefill_rate_cache: dict[str, tuple[float, tuple[float, str] | None]] = {}


async def _measured_prefill_rates(keys: list[str]) -> dict[str, float]:
    """Mediana de ``prompt_tokens_per_second`` de las ultimas 6 h por modelo."""
    import sqlalchemy as sa

    from ocabra.database import AsyncSessionLocal
    from ocabra.db.stats import RequestStat

    since = sa.func.now() - sa.text("interval '6 hours'")
    stmt = (
        sa.select(
            RequestStat.model_id,
            sa.func.percentile_cont(0.5).within_group(RequestStat.prompt_tokens_per_second),
        )
        .where(
            RequestStat.model_id.in_(keys),
            RequestStat.started_at > since,
            RequestStat.prompt_tokens_per_second > 0,
        )
        .group_by(RequestStat.model_id)
    )
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(stmt)).all()
    return {model: float(rate) for model, rate in rows if rate}


async def prefill_rates(keys: set[str], estimator: Any) -> dict[str, tuple[float, str]]:
    """(tokens/s, fuente) de prefill para cada modelo pedido.

    Fuente ``historical``: mediana medida en sus peticiones recientes (llama.cpp
    y Ollama la informan al terminar). Si no hay, ``regression``: la pendiente
    por token de entrada que ajusta el ``DurationEstimator``. Cache de 60 s,
    tambien para los que no tienen dato, para no consultar la BD cada segundo.
    """
    now = time.monotonic()
    missing = [
        k
        for k in keys
        if k
        and (k not in _prefill_rate_cache or now - _prefill_rate_cache[k][0] > _PREFILL_RATE_TTL_S)
    ]
    if missing:
        try:
            measured = await asyncio.wait_for(_measured_prefill_rates(missing), timeout=2.0)
        except Exception:  # noqa: BLE001 — sin BD: se intenta con la regresion
            measured = {}
        calibrations = getattr(estimator, "_calibrations", {}) or {}
        for key in missing:
            value: tuple[float, str] | None = None
            if measured.get(key):
                value = (round(measured[key], 1), "historical")
            else:
                cal = calibrations.get(key)
                regression = getattr(cal, "chat_regression", None)
                a_in = getattr(regression, "a_in", None)
                if a_in and a_in > 0:
                    value = (round(1000.0 / a_in, 1), "regression")
            _prefill_rate_cache[key] = (now, value)
    return {
        k: _prefill_rate_cache[k][1]
        for k in keys
        if k in _prefill_rate_cache and _prefill_rate_cache[k][1]
    }


def strip_internal(details: list[dict[str, Any]]) -> None:
    """Quita lo que solo hacia falta para calcular (no sale del servidor)."""
    for detail in details:
        detail.pop("userKey", None)
