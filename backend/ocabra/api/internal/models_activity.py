"""Live per-model activity payload for the dashboard (REF-07 split).

``build_models_activity`` is the body of ``GET /ocabra/models/activity`` and is
also pushed every second over ``/ocabra/ws`` to clients subscribed to
``models_activity`` (see ``api/internal/ws.py``). Extracted verbatim from
``api/internal/models.py``; behaviour is unchanged.
"""

import time
from typing import Any

from ocabra.config import settings
from ocabra.core.model_manager_helpers import resolve_worker_slots as _resolve_worker_slots
from ocabra.core.model_ref import is_llama_cpp_family
from ocabra.stats import live_activity, live_feed


def _canonical_runtime_id(model_id: str, canonical_ids: set[str]) -> str | None:
    """Match a runtime worker to a catalog id, preserving GGUF ``repo::file``."""

    matches = [
        candidate
        for candidate in canonical_ids
        if model_id == candidate or model_id.startswith(f"{candidate}::")
    ]
    return max(matches, key=len) if matches else None


def _live_totals(details: list[dict]) -> dict:
    """Totales del dashboard nuevo: por fase y t/s medidos en su propia fase."""
    stages: dict[str, int] = {}
    prefill = 0.0
    generation = 0.0
    for detail in details:
        stage = detail.get("stage") or "unknown"
        stages[stage] = stages.get(stage, 0) + 1
        if stage == "prefill" and detail.get("prefillTokensPerSecond"):
            prefill += float(detail["prefillTokensPerSecond"])
        if stage == "generating" and detail.get("generationTokensPerSecond"):
            generation += float(detail["generationTokensPerSecond"])
    return {
        "stages": stages,
        "queuedCount": stages.get("queued", 0),
        "prefillTokensPerSecondTotal": round(prefill, 1) if prefill else None,
        "generationTokensPerSecondTotal": round(generation, 1) if generation else None,
    }


async def build_models_activity(request: Any) -> dict:
    """Body of ``get_models_activity``; ``request`` only needs ``.app.state``
    (the WebSocket passes itself)."""
    mm = request.app.state.model_manager
    # Read-only: the fair-share gate is the only place that knows the real KV
    # pool occupancy and the queue depth behind a busy worker. Both were
    # already computed for the per-model status endpoint and nothing surfaced
    # them on the dashboard, where "N en curso" could not distinguish a busy
    # worker from one that is about to queue or reject.
    worker_pool = getattr(request.app.state, "worker_pool", None)
    estimator = getattr(request.app.state, "duration_estimator", None)
    live_ids: set[str] = set()
    snapshot = mm.inflight_snapshot()
    ages = mm.active_request_ages()
    sources = mm.activity_sources()
    now_ts = time.time()
    requests_by_key: dict[str, list] = {}
    for req in mm.active_requests_snapshot():
        requests_by_key.setdefault(req.model_id, []).append(req)
    # Ritmo de prefill para las peticiones cuyo backend no informa del avance
    # (todo lo que no es llama.cpp): una consulta cacheada por modelo.
    rate_keys = {
        req.profile_id or req.model_id
        for reqs in requests_by_key.values()
        for req in reqs
        if getattr(req, "prompt_tokens_total", None) is None
        and getattr(req, "first_token_at", None) is None
    }
    # Ritmos medidos por modelo: prefill (para estimar el avance donde el
    # backend no lo informa) y prefill + generación + salida típica (para
    # estimar duraciones). Una consulta cacheada por modelo.
    all_rate_keys = {
        req.profile_id or req.model_id for reqs in requests_by_key.values() for req in reqs
    }
    measured_rates = (
        await live_activity.model_rates(all_rate_keys, estimator) if all_rate_keys else {}
    )
    prefill_rates = {
        key: value["prefill"]
        for key, value in measured_rates.items()
        if key in rate_keys and value.get("prefill")
    }
    # ``list_states()`` (sin include_derived) devuelve solo los canónicos,
    # pero ``begin_request`` registra con el worker-key derivado del
    # preset activo (``ollama/qwen3.8:27b::<hash>``). Para que el badge
    # del card del modelo canónico cuente las peticiones de sus
    # derivados, suma todos los inflights cuyo ``model_id`` empieza por
    # ``canonical.model_id + "::"``.
    states = await mm.list_states()
    # Derivados cargados (``<canonical>::<hash>``). Nos interesan por dos
    # motivos: (1) agregarles las peticiones en vuelo al canonical, y
    # (2) leer el ``_runtime.parallelism`` del preset activo, que vive en
    # el derivado y no en el canonical state.
    all_states = await mm.list_states(include_derived=True)
    canonical_ids = {state.model_id for state in states}
    runtime_by_canonical: dict[str, int] = {}
    preset_by_canonical: dict[str, dict] = {}
    runtime_meta_by_canonical: dict[str, dict] = {}
    # Workers derivados cargados por modelo, tengan o no peticiones: hacen
    # falta para leer su puerto (/metrics de vLLM) y sus PID (VRAM por GPU).
    loaded_worker_keys_by_canonical: dict[str, list[str]] = {}
    loading_worker_keys_by_canonical: dict[str, list[str]] = {}
    # Contexto cambiado en vivo (v0.65.0) del worker cargado: manda sobre el
    # tamaño del preset en la barra de contexto.
    live_context_by_canonical: dict[str, dict] = {}
    for s in all_states:
        base = _canonical_runtime_id(s.model_id, canonical_ids)
        if base is None or base == s.model_id:
            continue
        meta = s.extra_config.get("_runtime") if isinstance(s.extra_config, dict) else None
        if not isinstance(meta, dict):
            continue
        try:
            p = int(meta.get("parallelism") or 0)
        except (TypeError, ValueError):
            p = 0
        if p > 0:
            # Guardar el mayor por si hay varios derivados activos (distintos
            # presets) — es la capacidad real de slots concurrentes.
            runtime_by_canonical[base] = max(runtime_by_canonical.get(base, 0), p)
        # El preset activo y su contexto efectivo tambien viven solo en el
        # derivado. Sin ellos la barra de KV es un numero sin referencia: no
        # se sabe si 240k de pool vienen del preset de 32k o del de 128k.
        derived_status = getattr(s.status, "value", s.status)
        if derived_status == "loaded":
            runtime_meta_by_canonical[base] = meta
            loaded_worker_keys_by_canonical.setdefault(base, []).append(s.model_id)
            if isinstance(getattr(s, "live_context", None), dict):
                live_context_by_canonical[base] = s.live_context
        elif derived_status == "loading":
            # El preset que se está cargando: la barra de contexto y el
            # estado de la card ya lo muestran durante la carga.
            runtime_meta_by_canonical.setdefault(base, meta)
            loading_worker_keys_by_canonical.setdefault(base, []).append(s.model_id)
        preset_key = meta.get("preset_key")
        effective_context = meta.get("effective_context")
        if preset_key or effective_context:
            preset_by_canonical[base] = {
                "presetKey": str(preset_key) if preset_key else None,
                "effectiveContext": (
                    int(effective_context)
                    if isinstance(effective_context, (int, float)) and effective_context
                    else None
                ),
            }
    derived_keys_by_canonical: dict[str, list[str]] = {}
    for key in snapshot:
        base = _canonical_runtime_id(key, canonical_ids)
        if base is None or base == key:
            continue
        derived_keys_by_canonical.setdefault(base, []).append(key)
    activity: dict[str, dict] = {}
    for state in states:
        # begin_request may key by the canonical id, the raw backend model
        # name, or a derived worker-key (``<canonical>::<hash>``) when the
        # active preset creates a dedicated runner.
        keys = (
            state.model_id,
            state.backend_model_id,
            *derived_keys_by_canonical.get(state.model_id, ()),
        )
        count = sum(snapshot.get(k, 0) for k in keys)
        if count > 0:
            oldest = max((ages.get(k, 0.0) for k in keys), default=0.0)
            source_counts: dict[str, int] = {}
            for key in keys:
                for source, source_count in sources.get(key, {}).items():
                    source_counts[source] = max(source_counts.get(source, 0), source_count)
            # Concurrency the backend will actually serve in parallel. Only
            # ``_runtime.parallelism`` (set by the coordinator, reflects the
            # active preset) is a real measure of that — it's set for
            # backends with genuine concurrent-slot serving (llama.cpp,
            # vLLM). Backends with no such concept (e.g. whisper, which just
            # thread-offloads via asyncio.to_thread with no managed slot
            # pool) leave this ``None``.
            #
            # ``settings.max_inflight_per_model`` used to be the fallback
            # here, but that's an admission-control ceiling ("reject with
            # 429 past this many queued"), not a measure of how many run
            # concurrently — conflating the two showed e.g. whisper models
            # as "1/32 procesando" (32 = the default cap), implying 32-way
            # parallelism that doesn't exist. Leaving ``slots=None`` lets
            # clients render an honest "N en curso" instead of a fabricated
            # ratio (found live 2026-10-30 chasing exactly that "32").
            # Derived-state parallelism first (the active preset, most
            # accurate), then the shared resolver — which also covers the
            # configured llama.cpp ``parallel_slots`` and piper's declared
            # ceiling, and is the same function admission and the fair-share
            # gate use, so the card can no longer show a slot count that
            # disagrees with the one requests are admitted against.
            slots: int | None = runtime_by_canonical.get(state.model_id) or None
            if slots is None:
                slots = _resolve_worker_slots(state)
            # Per-request progress for the "is this hung or still working"
            # indicator on the model/GPU card (2026-10-29). ``tokensPerSecond``
            # is None until at least one output token has streamed — division
            # by elapsed time before that is meaningless, not just zero.
            requests_detail: list[dict] = []
            gate_leases, gate_queued, _ = live_activity.gate_requests_by_id(worker_pool, keys)
            has_gate = bool(gate_leases or gate_queued)
            for key in keys:
                if not requests_by_key.get(key):
                    continue
                # Same phase for every request piled up behind the same
                # worker_key — computed once per key, not per request.
                load_phase, waiting_on = mm.get_load_phase(key)
                for req in requests_by_key.get(key, ()):
                    elapsed_s = max(0.0, now_ts - req.started_at)
                    tokens_per_second = None
                    if req.output_tokens_estimate and elapsed_s > 0:
                        tokens_per_second = round(req.output_tokens_estimate / elapsed_s, 1)
                    if load_phase == "loading":
                        phase = "loading"
                    elif load_phase == "waiting_for_gpu":
                        phase = "waiting_for_gpu"
                    elif req.output_tokens_estimate is None:
                        phase = "prompt_eval"
                    else:
                        phase = "generating"
                    requests_detail.append(
                        {
                            "requestId": req.request_id,
                            "username": req.username,
                            "elapsedSeconds": round(elapsed_s, 1),
                            "inputTokensEstimate": req.input_tokens_estimate,
                            "outputTokensEstimate": req.output_tokens_estimate,
                            "tokensPerSecond": tokens_per_second,
                            "lastActivitySecondsAgo": round(
                                max(0.0, now_ts - (req.last_activity_at or req.started_at)), 1
                            ),
                            "phase": phase,
                            "waitingOnModelId": waiting_on,
                            **live_activity.request_live_detail(
                                req,
                                now=now_ts,
                                lease=gate_leases.get(req.request_id),
                                queued=gate_queued.get(req.request_id),
                                load_phase=load_phase,
                                has_gate=has_gate,
                                prefill_rate=prefill_rates.get(req.profile_id or req.model_id),
                            ),
                        }
                    )
            # Fair-share gate snapshot. Keyed by whichever id actually runs
            # the worker, so try the same candidates as the in-flight count.
            admission = None
            if worker_pool is not None:
                for key in keys:
                    try:
                        gate = worker_pool.peek_queue_state(key)
                    except Exception:  # noqa: BLE001 -- read-only, never break the card
                        gate = None
                    if gate is not None:
                        admission = {
                            "slots": gate["slots"],
                            "queuedRequests": gate["queued"],
                            "slotSaturated": gate["slot_saturated"],
                            "computeSaturated": gate.get("compute_saturated", False),
                            "admissionLimit": gate.get("admission_limit", gate["slots"]),
                            "kvCapacityTokens": gate["kv_capacity_tokens"],
                            "kvReservedTokens": gate["kv_reserved_tokens"],
                            "kvFreeTokens": gate["kv_free_tokens"],
                        }
                        break
            # The gate's slot count is authoritative for backends that really
            # serve concurrent slots, so it is a legitimate fallback when
            # ``_runtime.parallelism`` is missing. It is NOT a replacement for
            # the deliberate ``None`` above: backends with no slot pool never
            # create a gate, so they still report None and clients keep
            # rendering an honest "N en curso" instead of a made-up ratio.
            if slots is None and admission is not None:
                gate_slots = admission.get("slots")
                if isinstance(gate_slots, int) and gate_slots > 0:
                    slots = gate_slots

            # Peticiones a llama.cpp sin progreso en el stream (sin streaming):
            # su avance real sale de /slots del llama-server.
            if is_llama_cpp_family(state.backend_type) and any(
                not d.get("progressTracked") and d["stage"] != "queued" for d in requests_detail
            ):
                port = None
                get_worker = getattr(worker_pool, "get_worker", None)
                for key in keys:
                    info = get_worker(key) if callable(get_worker) else None
                    if info is not None and getattr(info, "port", None):
                        port = info.port
                        break
                try:
                    slots_now = await live_activity.llama_slots(port)
                    live_activity.apply_slot_progress(requests_detail, slots_now, now=time.time())
                except Exception:  # noqa: BLE001 — sin slots, se queda con lo estimado
                    pass
            # whisper: avance real de cada transcripción desde /progress de
            # sus workers (también el gemelo -diarized, que va por otro
            # worker key y otro puerto).
            if state.backend_type == "whisper" and requests_detail:
                get_worker = getattr(worker_pool, "get_worker", None)
                audio_jobs: dict = {}
                for key in keys:
                    info = get_worker(key) if callable(get_worker) else None
                    if info is not None and getattr(info, "port", None):
                        try:
                            audio_jobs.update(await live_activity.whisper_progress(info.port))
                        except Exception:  # noqa: BLE001 — sin avance, se queda estimado
                            pass
                live_activity.apply_audio_progress(requests_detail, audio_jobs)
            requests_detail.sort(
                key=lambda d: (
                    d["stage"] == "queued",
                    d.get("queuePosition") or 0,
                    -d["elapsedMs"],
                )
            )
            await live_activity.attach_estimates(
                requests_detail,
                estimator=estimator,
                model_id=state.model_id,
                backend_type=state.backend_type,
                slots=slots,
                kv_capacity=(admission or {}).get("kvCapacityTokens"),
                kv_reserved=(admission or {}).get("kvReservedTokens"),
                rates=measured_rates,
            )
            live_activity.mark_stalls(requests_detail)
            live_activity.strip_internal(requests_detail)
            live_ids.update(d["requestId"] for d in requests_detail)

            # Aggregate phase mix. Per-request phases were already exposed but
            # only readable one line at a time; the mix is what tells you at a
            # glance whether a model is generating or stuck in prompt eval.
            phases: dict[str, int] = {}
            tps_total = 0.0
            generating_count = 0
            for detail in requests_detail:
                phase_name = str(detail.get("phase") or "unknown")
                phases[phase_name] = phases.get(phase_name, 0) + 1
                tps = detail.get("tokensPerSecond")
                if isinstance(tps, (int, float)) and tps > 0:
                    tps_total += float(tps)
                    generating_count += 1

            entry = {
                "inFlight": count,
                "oldestSeconds": round(oldest, 1),
                "sources": source_counts,
                "slots": slots,
                "requests": requests_detail,
                "phases": phases,
                # Sum across requests that are actually emitting tokens, plus
                # how many those are. The count matters: tokensPerSecond is
                # null outside the ``generating`` phase, so a model saturated
                # with prompt-eval work would otherwise read as 0 t/s while
                # being at full load.
                "tokensPerSecondTotal": round(tps_total, 1) if generating_count else None,
                "generatingCount": generating_count,
                **_live_totals(requests_detail),
            }
            if admission is not None:
                entry["admission"] = admission
            preset_info = preset_by_canonical.get(state.model_id)
            if preset_info:
                entry["runtimePreset"] = preset_info["presetKey"]
                entry["effectiveContext"] = preset_info["effectiveContext"]
            activity[state.model_id] = entry
    live_activity.prune_estimates(live_ids)
    live_activity.prune_slot_track(live_ids)
    contexts: dict[str, dict] = {}
    registry = getattr(request.app.state, "profile_registry", None)
    profiles: list = []
    if registry is not None:
        try:
            profiles = await registry.list_all()
        except Exception:  # noqa: BLE001
            live_activity.logger.debug("models_activity_profiles_failed", exc_info=True)
            profiles = []
    gpu_manager = getattr(mm, "gpu_manager", None)
    gpu_states: list = []
    if gpu_manager is not None:
        try:
            gpu_states = await gpu_manager.get_all_states()
        except Exception:  # noqa: BLE001
            live_activity.logger.debug("models_activity_gpu_states_failed", exc_info=True)
            gpu_states = []
    worker_pids: dict[str, list[int]] = {}
    model_states: dict[str, dict] = {}
    typical_load = await live_activity.typical_load_ms()
    for state in states:
        derived_meta = runtime_meta_by_canonical.get(state.model_id)
        status_value = getattr(state.status, "value", state.status)
        worker_keys = [
            *loaded_worker_keys_by_canonical.get(state.model_id, ()),
            *loading_worker_keys_by_canonical.get(state.model_id, ()),
            *derived_keys_by_canonical.get(state.model_id, ()),
        ]
        try:
            phase_info = live_activity.model_phase(
                mm, state, worker_keys, now=now_ts, typical_ms=typical_load
            )
        except Exception:  # noqa: BLE001
            live_activity.logger.debug(
                "models_activity_model_phase_failed", model_id=state.model_id, exc_info=True
            )
            phase_info = None
        if phase_info and (status_value in ("loaded", "loading") or phase_info["phase"] != "ready"):
            model_states[state.model_id] = phase_info
        loaded = status_value in ("loaded", "loading")
        if not loaded and derived_meta is None:
            continue
        worker_port: int | None = None
        get_worker = getattr(worker_pool, "get_worker", None)
        if callable(get_worker):
            for key in (state.model_id, *loaded_worker_keys_by_canonical.get(state.model_id, ())):
                try:
                    info = get_worker(key)
                except Exception:  # noqa: BLE001
                    info = None
                if info is None:
                    continue
                worker_port = worker_port or getattr(info, "port", None)
                pid = getattr(info, "pid", 0)
                # Ollama comparte un proceso entre modelos: NVML no sabe cual.
                if pid and state.backend_type != "ollama":
                    worker_pids.setdefault(state.model_id, []).append(int(pid))
        gate_raw = None
        if worker_pool is not None and hasattr(worker_pool, "peek_queue_state"):
            for key in (state.model_id, *derived_keys_by_canonical.get(state.model_id, ())):
                try:
                    gate_raw = worker_pool.peek_queue_state(key)
                except Exception:  # noqa: BLE001
                    gate_raw = None
                if gate_raw is not None:
                    break
        try:
            ctx = await live_activity.model_context(
                state,
                runtime_meta=derived_meta,
                gate=gate_raw,
                request_details=(activity.get(state.model_id) or {}).get("requests", []),
                profiles=profiles,
                gpu_states=gpu_states,
                worker_port=worker_port,
                live_context=live_context_by_canonical.get(state.model_id)
                or (
                    state.live_context
                    if isinstance(getattr(state, "live_context", None), dict)
                    else None
                ),
            )
        except Exception:  # noqa: BLE001 — una barra menos, no un 500
            live_activity.logger.debug(
                "live_context_failed", model_id=state.model_id, exc_info=True
            )
            ctx = None
        if ctx is not None:
            contexts[state.model_id] = ctx
    try:
        vram_by_gpu = await live_activity.vram_by_gpu(worker_pids, gpu_states)
    except Exception:  # noqa: BLE001 — sin desglose, la UI reparte a partes iguales
        live_activity.logger.debug("models_activity_vram_by_gpu_failed", exc_info=True)
        vram_by_gpu = {}
    backend_by_model = {state.model_id: state.backend_type for state in states}
    try:
        measured_ram = await live_activity.ram_by_model(
            {
                model: pids
                for model, pids in worker_pids.items()
                if live_activity.wants_host_ram(backend_by_model.get(model))
            }
        )
        ram_by_model = {
            model: mb
            for model, mb in measured_ram.items()
            if live_activity.keep_host_ram(backend_by_model.get(model), mb)
        }
    except Exception:  # noqa: BLE001
        live_activity.logger.debug("models_activity_ram_by_model_failed", exc_info=True)
        ram_by_model = {}
    return {
        "activity": activity,
        "contexts": contexts,
        # MB medidos por NVML de cada modelo en cada GPU ({modelo: {gpu: MB}}).
        "vramByGpu": vram_by_gpu,
        # MB de RAM del host (RSS en vivo) de los modelos que la usan a
        # propósito aunque corran en GPU, como Strata.
        "ramByModel": ram_by_model,
        # Terminadas en los últimos 20 s (incluidos los 429): las muy cortas
        # no llegarían a verse entre dos refrescos.
        "finished": live_feed.recent_finished(),
        # Estado de cada modelo activo o esperando: cargando (desde cuándo y
        # cuánto suele tardar), esperando espacio (a quién desaloja) o
        # esperando una GPU retenida por un servicio.
        "modelStates": model_states,
        "stuckThresholdSeconds": int(getattr(settings, "busy_timeout_seconds", 300)),
        # Same delivery as stuckThresholdSeconds: the card needs it to show the
        # idle-unload countdown, and shipping it here avoids threading server
        # config through GpuCard/Dashboard just for one number.
        "idleTimeoutSeconds": int(getattr(settings, "idle_timeout_seconds", 300)),
    }
