"""Calculos del dashboard en vivo (``ocabra.stats.live_activity``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocabra.backends.base import BackendCapabilities
from ocabra.stats import live_activity as la

NOW = 1_000_000.0


def _req(**overrides):
    base = dict(
        request_id="r",
        started_at=NOW - 10,
        input_tokens_estimate=2000,
        output_tokens_estimate=None,
        last_activity_at=NOW,
        first_token_at=None,
        streaming=True,
        prompt_tokens_total=None,
        prompt_tokens_cached=None,
        prompt_tokens_processed=None,
        prompt_progress_ms=None,
        prompt_progress_tracked=False,
        request_kind="chat",
        profile_id="p",
        api_key_name=None,
        source="api",
        max_output_tokens=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _detail(req, **kw):
    params = dict(now=NOW, lease=None, queued=None, load_phase="ready", has_gate=True)
    params.update(kw)
    return {"requestId": req.request_id, **la.request_live_detail(req, **params)}


def test_queued_request_carries_turn_position_and_reservation():
    d = _detail(_req(), queued={"position": 2, "waitedMs": 4000, "kvTokens": 8000})
    assert d["stage"] == "queued"
    assert (d["queuePosition"], d["queuedMs"], d["kvReservedTokens"]) == (2, 4000, 8000)
    assert d["admittedAfterMs"] is None


def test_llama_cpp_prefill_uses_real_progress():
    req = _req(
        prompt_tokens_total=10_000,
        prompt_tokens_cached=2_000,
        prompt_tokens_processed=6_000,
        prompt_progress_ms=2_000,
    )
    d = _detail(req, lease={"heldMs": 3000, "kvTokens": 12_000})
    assert d["stage"] == "prefill"
    assert d["prefillPct"] == 60.0
    # (6000 procesados - 2000 de cache) / 2 s
    assert d["prefillTokensPerSecond"] == 2000.0
    assert d["prefillSource"] == "live"
    assert d["prompt"] == {
        "totalTokens": 10_000,
        "cachedTokens": 2_000,
        "processedTokens": 6_000,
        "source": "live",
    }
    assert d["admittedAfterMs"] == 7000


def test_generation_rate_only_counts_generation_time():
    """10 s en total, pero el primer token llego a los 8 s: 100 tokens en los
    2 s de generacion son 50 t/s, no los 10 t/s del calculo antiguo."""
    req = _req(first_token_at=NOW - 2, output_tokens_estimate=100)
    d = _detail(req, lease={"heldMs": 9000})
    assert d["stage"] == "generating"
    assert d["generationTokensPerSecond"] == 50.0
    # Prefill inferido: 2000 tokens estimados entre admision (-9 s) y primer token (-2 s)
    assert d["prefillTokensPerSecond"] == pytest.approx(285.7, abs=0.1)
    assert d["prefillSource"] == "estimate"
    assert d["prefillPct"] == 100.0


def test_non_streaming_request_is_processing_not_eternal_prefill():
    d = _detail(_req(streaming=False), has_gate=False)
    assert d["stage"] == "processing"
    assert d["admittedAfterMs"] == 0


def test_loading_wins_over_prefill():
    assert _detail(_req(), load_phase="loading")["stage"] == "loading"


class _Estimator:
    def __init__(self, seconds_by_input):
        self.seconds_by_input = seconds_by_input

    async def estimate(self, model_id, *, backend_type, input_tokens, max_tokens):
        return SimpleNamespace(expected_s=self.seconds_by_input[input_tokens])


@pytest.mark.asyncio
async def test_queue_wait_simulates_slot_release_in_turn_order():
    la._estimate_cache.clear()
    # 2 slots: uno acaba en 2 s, el otro en 6 s. Cola: q0 (dura 5 s), q1 (dura 1 s).
    details = [
        {
            "requestId": "p1",
            "stage": "generating",
            "elapsedMs": 8000,
            "admittedAfterMs": 0,
            "prompt": {"totalTokens": 1},
            "maxOutputTokens": None,
        },
        {
            "requestId": "p2",
            "stage": "prefill",
            "elapsedMs": 4000,
            "admittedAfterMs": 0,
            "prompt": {"totalTokens": 2},
            "maxOutputTokens": None,
        },
        {
            "requestId": "q1",
            "stage": "queued",
            "queuePosition": 1,
            "elapsedMs": 1000,
            "prompt": {"totalTokens": 4},
            "maxOutputTokens": None,
        },
        {
            "requestId": "q0",
            "stage": "queued",
            "queuePosition": 0,
            "elapsedMs": 3000,
            "prompt": {"totalTokens": 3},
            "maxOutputTokens": None,
        },
    ]
    est = _Estimator({1: 10, 2: 10, 3: 5, 4: 1})
    await la.attach_estimates(details, estimator=est, model_id="m", backend_type="x", slots=2)
    by_id = {d["requestId"]: d for d in details}
    assert by_id["p1"]["estimatedRemainingMs"] == 2000
    assert by_id["p2"]["estimatedRemainingMs"] == 6000
    # q0 toma el slot que se libera a los 2 s; q1 el de los 6 s (q0 acaba a los 7).
    assert by_id["q0"]["queueWaitEstimateMs"] == 2000
    assert by_id["q1"]["queueWaitEstimateMs"] == 6000
    assert by_id["q0"]["estimatedRemainingMs"] is None


@pytest.mark.asyncio
async def test_queue_wait_unknown_when_running_work_has_no_estimate():
    la._estimate_cache.clear()
    details = [
        {
            "requestId": "p",
            "stage": "generating",
            "elapsedMs": 1,
            "admittedAfterMs": 0,
            "prompt": {"totalTokens": 1},
            "maxOutputTokens": None,
        },
        {
            "requestId": "q",
            "stage": "queued",
            "queuePosition": 0,
            "elapsedMs": 1,
            "prompt": {"totalTokens": 1},
            "maxOutputTokens": None,
        },
    ]
    await la.attach_estimates(details, estimator=None, model_id="m", backend_type="x", slots=1)
    assert details[1]["queueWaitEstimateMs"] is None


def _state(backend="llamacpp_mks", *, chat=True, extra=None, gpus=(1,)):
    return SimpleNamespace(
        model_id=f"{backend}/m",
        backend_type=backend,
        capabilities=BackendCapabilities(chat=chat),
        extra_config=extra
        if extra is not None
        else {"llama_cpp": {"kv_unified": True, "ctx_size": 262144}},
        current_gpu=list(gpus),
    )


def test_context_mode_by_backend():
    assert la.context_mode(_state()) == "pool"
    assert la.context_mode(_state(extra={"llama_cpp": {"kv_unified": False}})) == "slots"
    assert la.context_mode(_state("ollama", extra={})) == "slots"
    assert la.context_mode(_state("vllm", extra={})) == "paged"
    assert la.context_mode(_state("whisper", extra={})) is None
    assert la.context_mode(_state(chat=False)) is None


@pytest.mark.asyncio
async def test_context_marks_single_gpu_ceiling_on_two_gpu_scale(monkeypatch):
    """El caso del usuario: 96k en la 3090, 256k en las dos."""
    rows = [
        {
            "preset_key": "normal",
            "resolved": True,
            "fits": True,
            "effective_context": 98304,
            "effective_parallelism": 4,
            "target_gpus": [1],
        },
        {
            "preset_key": "extended",
            "resolved": True,
            "fits": True,
            "effective_context": 262144,
            "effective_parallelism": 4,
            "target_gpus": [0, 1],
        },
        {
            "preset_key": "huge",
            "resolved": True,
            "fits": False,
            "effective_context": 524288,
            "effective_parallelism": 4,
            "target_gpus": [0, 1],
        },
    ]

    async def fake_resolve(profile, state, gpu_states, force=False):
        return rows

    from ocabra.core import runtime_preset_capacity

    monkeypatch.setattr(runtime_preset_capacity.runtime_capacity_cache, "resolve", fake_resolve)
    state = _state()
    profile = SimpleNamespace(
        profile_id="qwen",
        base_model_id=state.model_id,
        profile_kind="execution",
        enabled=True,
        is_default=True,
        runtime_presets=[SimpleNamespace(enabled=True)],
    )
    details = [
        {
            "requestId": "a",
            "stage": "generating",
            "kvReservedTokens": 40_000,
            "outputTokens": 500,
            "prompt": {"totalTokens": 30_000, "processedTokens": 30_000, "source": "live"},
        },
        {
            "requestId": "b",
            "stage": "prefill",
            "kvReservedTokens": 20_000,
            "outputTokens": None,
            "prompt": {"totalTokens": 15_000, "processedTokens": None, "source": "estimate"},
        },
        {
            "requestId": "c",
            "stage": "queued",
            "kvReservedTokens": 9_000,
            "prompt": {"totalTokens": 1},
        },
    ]
    ctx = await la.model_context(
        state,
        runtime_meta={
            "profile_id": "qwen",
            "preset_key": "normal",
            "effective_context": 98304,
            "gpu_indices": [1],
        },
        gate={"kv_capacity_tokens": 98304, "kv_reserved_tokens": 60_000},
        request_details=details,
        profiles=[profile],
        gpu_states=[],
    )
    assert ctx["mode"] == "pool"
    assert ctx["scaleMaxTokens"] == 262144
    assert ctx["singleGpuMaxTokens"] == 98304
    assert ctx["activeMaxTokens"] == 98304
    assert ctx["activePresetKey"] == "normal"
    assert [t["presetKey"] for t in ctx["tiers"]] == ["normal", "extended"]  # "huge" no cabe
    assert ctx["reservedTokens"] == 60_000
    assert ctx["segments"] == [
        {"requestId": "a", "reservedTokens": 40_000, "usedTokens": 30_500},
        {"requestId": "b", "reservedTokens": 20_000, "usedTokens": None},
    ]
    assert ctx["source"] == "live"


@pytest.mark.asyncio
async def test_context_without_presets_is_single_tier_from_ctx_size():
    state = _state(gpus=(1,))
    ctx = await la.model_context(
        state, runtime_meta=None, gate=None, request_details=[], profiles=[], gpu_states=[]
    )
    assert ctx["tiers"] == [{"presetKey": None, "gpus": [1], "maxTokens": 262144, "slots": None}]
    assert ctx["scaleMaxTokens"] == ctx["singleGpuMaxTokens"] == 262144


# ── Deudas cerradas el 2026-11-09 ──────────────────────────────────────────


def test_parse_vllm_kv_usage_supports_both_metric_names_and_takes_the_fullest_engine():
    v1 = (
        "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.\n"
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25\n'
        'vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.6\n'
        'vllm:num_requests_running{engine="0"} 3.0\n'
    )
    assert la.parse_vllm_kv_usage(v1) == 60.0
    assert la.parse_vllm_kv_usage("vllm:gpu_cache_usage_perc 0.125\n") == 12.5
    assert la.parse_vllm_kv_usage("vllm:num_requests_running 1\n") is None
    assert la.parse_vllm_kv_usage("") is None


@pytest.mark.asyncio
async def test_paged_context_reads_vllm_metrics(monkeypatch):
    async def fake_usage(port):
        assert port == 18123
        return 42.0

    monkeypatch.setattr(la, "vllm_kv_usage_pct", fake_usage)
    state = _state("vllm", extra={})
    ctx = await la.model_context(
        state,
        runtime_meta={"effective_context": 32768, "gpu_indices": [1]},
        gate=None,
        request_details=[],
        profiles=[],
        gpu_states=[],
        worker_port=18123,
    )
    assert ctx["mode"] == "paged"
    assert ctx["usagePct"] == 42.0


@pytest.mark.asyncio
async def test_vram_by_gpu_uses_measured_per_process_memory(monkeypatch):
    """El reparto real de nemotron visto en producción: 9,6 GB en la 3060 y
    21,4 en la 3090, no la mitad en cada una. El hijo del worker cuenta; un
    proceso compartido por dos modelos (Ollama) no se atribuye a ninguno."""
    trees = {100: {100, 101}, 200: {200}, 300: {300}}
    monkeypatch.setattr(la, "_process_tree", lambda pid: trees[pid])
    proc = lambda pid, mb: SimpleNamespace(pid=pid, used_vram_mb=mb)  # noqa: E731
    gpus = [
        SimpleNamespace(index=0, processes=[proc(101, 9566), proc(200, 2184), proc(300, 500)]),
        SimpleNamespace(index=1, processes=[proc(101, 21406), proc(999, 700)]),
    ]
    result = await la.vram_by_gpu(
        {"nemotron": [100], "whisper": [200], "a": [300], "b": [300]},
        gpus,
    )
    assert result == {
        "nemotron": {"0": 9566, "1": 21406},
        "whisper": {"0": 2184},
    }


def _queued(rid, pos, kv, ms, user):
    return {
        "requestId": rid,
        "stage": "queued",
        "queuePosition": pos,
        "elapsedMs": 1,
        "kvReservedTokens": kv,
        "userKey": user,
        "prompt": {"totalTokens": 1},
        "maxOutputTokens": None,
        "_ms": ms,
    }


@pytest.mark.asyncio
async def test_queue_simulation_lets_a_small_request_pass_one_that_does_not_fit_kv():
    """Pool de 100k con 70k reservados y 4 slots: la cabeza (50k) no cabe y
    espera a que acabe la que está en marcha; la siguiente (10k, otro
    usuario) cabe y entra ya. La segunda de "big" respeta el FIFO de su
    usuario aunque quepa."""
    la._estimate_cache.clear()
    running = {
        "requestId": "r",
        "stage": "generating",
        "elapsedMs": 2000,
        "admittedAfterMs": 0,
        "kvReservedTokens": 70_000,
        "prompt": {"totalTokens": 1},
        "maxOutputTokens": None,
    }
    details = [
        running,
        _queued("big1", 0, 50_000, 5000, "big"),
        _queued("small", 1, 10_000, 3000, "small"),
        _queued("big2", 2, 5_000, 1000, "big"),
    ]
    durations = {"r": 12_000, "big1": 5000, "small": 3000, "big2": 1000}

    class Est:
        async def estimate(self, model_id, *, backend_type, input_tokens, max_tokens):
            return None

    for d in details:
        la._estimate_cache[d["requestId"]] = durations[d["requestId"]]
    await la.attach_estimates(
        details,
        estimator=Est(),
        model_id="m",
        backend_type="x",
        slots=4,
        kv_capacity=100_000,
        kv_reserved=70_000,
    )
    by_id = {d["requestId"]: d for d in details}
    assert by_id["small"]["queueWaitEstimateMs"] == 0
    assert by_id["small"]["predictedStartOrder"] == 0
    # "r" acaba a los 10 s (12 - 2 transcurridos) y libera 70k.
    assert by_id["big1"]["queueWaitEstimateMs"] == 10_000
    assert by_id["big2"]["queueWaitEstimateMs"] == 10_000
    assert [by_id[k]["predictedStartOrder"] for k in ("big1", "big2")] == [1, 2]
    la.strip_internal(details)
    assert all("userKey" not in d for d in details)


@pytest.mark.asyncio
async def test_queue_request_bigger_than_the_whole_pool_gets_no_estimate():
    la._estimate_cache.clear()
    la._estimate_cache.update({"r": 4000, "huge": 1000})
    details = [
        {
            "requestId": "r",
            "stage": "generating",
            "elapsedMs": 0,
            "admittedAfterMs": 0,
            "kvReservedTokens": 10,
            "prompt": {"totalTokens": 1},
            "maxOutputTokens": None,
        },
        _queued("huge", 0, 200_000, 1000, "u"),
    ]
    await la.attach_estimates(
        details,
        estimator=None,
        model_id="m",
        backend_type="x",
        slots=2,
        kv_capacity=100_000,
        kv_reserved=10,
    )
    assert details[1]["queueWaitEstimateMs"] is None
    assert details[1]["predictedStartOrder"] is None


def test_prefill_without_backend_progress_uses_measured_model_rate():
    """Ollama/vLLM no informan del prefill: avanza al ritmo medido del modelo
    desde la admisión y no pasa del 99 % hasta que llega el primer token."""
    req = _req(input_tokens_estimate=10_000)
    d = _detail(req, lease={"heldMs": 4000}, prefill_rate=(1000.0, "historical"))
    assert d["stage"] == "prefill"
    assert d["prefillPct"] == 40.0
    assert d["prefillTokensPerSecond"] == 1000.0
    assert d["prefillSource"] == "historical"
    assert d["prompt"]["processedTokens"] is None  # no se inventa un dato real
    late = _detail(req, lease={"heldMs": 60_000}, prefill_rate=(1000.0, "historical"))
    assert late["prefillPct"] == 99.0


@pytest.mark.asyncio
async def test_prefill_rates_fall_back_to_estimator_regression(monkeypatch):
    la._model_rates_cache.clear()

    async def no_db(keys):
        raise RuntimeError("sin base de datos")

    monkeypatch.setattr(la, "_measured_model_rates", no_db)
    estimator = SimpleNamespace(
        _calibrations={"m": SimpleNamespace(chat_regression=SimpleNamespace(a_in=0.5))}
    )
    rates = await la.prefill_rates({"m", "sin-datos"}, estimator)
    assert rates == {"m": (2000.0, "regression")}


@pytest.mark.asyncio
async def test_model_rates_prefer_measured_values_and_cache_them(monkeypatch):
    la._model_rates_cache.clear()
    calls = []

    async def measured(keys):
        calls.append(sorted(keys))
        return {"m": {"prefill": 1234.56, "generation": 41.26, "output": 380.0}}

    monkeypatch.setattr(la, "_measured_model_rates", measured)
    estimator = SimpleNamespace(_calibrations={})
    expected = {"m": {"prefill": (1234.6, "historical"), "generation": 41.3, "output": 380}}
    assert await la.model_rates({"m"}, estimator) == expected
    assert await la.model_rates({"m"}, estimator) == expected
    assert calls == [["m"]]


def test_physical_duration_uses_model_rates_and_caps_output_by_typical():
    """57k tokens a 1000 t/s de prefill = 57 s; 128 de salida pedidos (por
    debajo del p75 de 380) a 40 t/s = 3,2 s. El estimador decía 15 s."""
    detail = {"prompt": {"totalTokens": 57_000}, "maxOutputTokens": 128}
    rates = {"prefill": (1000.0, "historical"), "generation": 40.0, "output": 380}
    assert la.physical_duration_ms(detail, rates) == 60_200
    # Sin tope: la salida típica del modelo.
    assert la.physical_duration_ms({"prompt": {"totalTokens": 1000}}, rates) == 10_500
    assert la.physical_duration_ms(detail, None) is None
    assert la.physical_duration_ms(detail, {"generation": 40.0}) is None


@pytest.mark.asyncio
async def test_estimate_prefers_physical_duration(monkeypatch):
    la._estimate_cache.clear()

    class Est:
        async def estimate(self, model_id, *, backend_type, input_tokens, max_tokens):
            return SimpleNamespace(expected_s=15.0)

    d = {
        "requestId": "r",
        "rateKey": "ravenx",
        "stage": "processing",
        "elapsedMs": 30_000,
        "admittedAfterMs": 0,
        "maxOutputTokens": 128,
        "prompt": {"totalTokens": 57_000, "source": "estimate"},
    }
    rates = {"ravenx": {"prefill": (1000.0, "historical"), "generation": 40.0, "output": 380}}
    await la.attach_estimates(
        [d], estimator=Est(), model_id="m", backend_type="x", slots=4, rates=rates
    )
    assert d["expectedDurationMs"] == 60_200
    assert d["estimatedRemainingMs"] == 30_200


# ── Hallazgos de la verificación en producción (2026-11-09) ─────────────────


def test_tracked_request_without_progress_waits_in_backend_not_prefill():
    """llama.cpp procesa los prompts de uno en uno: una petición admitida a la
    que se pidió progreso y aún no tiene ningún evento está esperando dentro
    del backend. Antes salía como "prefill 99 %" estimado durante 200 s."""
    req = _req(prompt_progress_tracked=True)
    d = _detail(req, lease={"heldMs": 5000, "kvTokens": 29416}, prefill_rate=(721.3, "historical"))
    assert d["stage"] == "backend_wait"
    assert d["prefillPct"] is None
    started = _req(
        prompt_progress_tracked=True,
        prompt_tokens_total=49558,
        prompt_tokens_cached=0,
        prompt_tokens_processed=0,
        prompt_progress_ms=0,
    )
    assert _detail(started, lease={"heldMs": 1000})["stage"] == "prefill"


@pytest.mark.asyncio
async def test_remaining_reestimated_with_real_prompt_and_never_below_prefill_left():
    """El recuento aproximado decía 18k tokens y eran 70k: con el total real
    se re-estima, y el restante no baja nunca de lo que falta de prefill."""
    la._estimate_cache.clear()
    asked = []

    class Est:
        async def estimate(self, model_id, *, backend_type, input_tokens, max_tokens):
            asked.append(input_tokens)
            return SimpleNamespace(expected_s=15.0)

    live = {
        "requestId": "r",
        "stage": "prefill",
        "elapsedMs": 20_000,
        "admittedAfterMs": 0,
        "prefillTokensPerSecond": 1000.0,
        "maxOutputTokens": 64,
        "prompt": {"totalTokens": 70_000, "processedTokens": 20_000, "source": "live"},
    }
    await la.attach_estimates([live], estimator=Est(), model_id="m", backend_type="x", slots=4)
    assert asked == [70_000]
    # 15 s estimados ya pasados, pero quedan 50k tokens a 1000 t/s.
    assert live["estimatedRemainingMs"] == 50_000

    waiting = {
        "requestId": "w",
        "stage": "backend_wait",
        "elapsedMs": 90_000,
        "admittedAfterMs": 0,
        "maxOutputTokens": None,
        "prompt": {"totalTokens": 9000, "source": "estimate"},
    }
    await la.attach_estimates([waiting], estimator=Est(), model_id="m", backend_type="x", slots=4)
    assert waiting["estimatedRemainingMs"] == 15_000


@pytest.mark.asyncio
async def test_remaining_does_not_count_down_while_the_model_loads():
    la._estimate_cache.clear()

    class Est:
        async def estimate(self, model_id, *, backend_type, input_tokens, max_tokens):
            return SimpleNamespace(expected_s=15.0)

    for stage in ("loading", "waiting_for_gpu"):
        d = {
            "requestId": stage,
            "stage": stage,
            "elapsedMs": 60_000,
            "admittedAfterMs": 0,
            "maxOutputTokens": None,
            "prompt": {"totalTokens": 9000, "source": "estimate"},
        }
        await la.attach_estimates([d], estimator=Est(), model_id="m", backend_type="x", slots=4)
        assert d["estimatedRemainingMs"] == 15_000


# ── 2026-11-10: peticiones sin streaming, parones y estado de carga ─────────


def _slot(id_task, total, processed, cached=0, decoded=0, processing=True):
    return {
        "id_task": id_task,
        "is_processing": processing,
        "n_prompt_tokens": total,
        "n_prompt_tokens_processed": processed,
        "n_prompt_tokens_cache": cached,
        "next_token": [{"n_decoded": decoded}],
    }


def _untracked(rid, started, stage="processing", streaming=False):
    return {
        "requestId": rid,
        "stage": stage,
        "streaming": streaming,
        "startedAt": started,
        "admittedAfterMs": 0,
        "progressTracked": False,
        "prompt": {
            "totalTokens": 50_000,
            "cachedTokens": None,
            "processedTokens": None,
            "source": "estimate",
        },
        "lastActivitySecondsAgo": 60.0,
    }


def test_slots_give_real_progress_to_non_streaming_requests():
    """Dos sin streaming y una en streaming con progreso propio. La de
    streaming se reconoce por su total y se descarta; las otras se asignan por
    orden de llegada. La tercera admitida sin slot espera al backend."""
    la._slot_track.clear()
    tracked = {
        "requestId": "s",
        "stage": "prefill",
        "progressTracked": True,
        "admittedAfterMs": 0,
        "prompt": {"totalTokens": 70_000, "processedTokens": 1000, "source": "live"},
    }
    first = _untracked("a", 100.0)
    second = _untracked("b", 101.0)
    third = _untracked("c", 102.0)
    slots = [
        _slot(7, 70_000, 1000),  # la de streaming
        _slot(3, 49_558, 20_000, cached=4096),
        _slot(5, 30_000, 30_000, decoded=12),
        _slot(9, 0, 0, processing=False),
    ]
    details = [tracked, first, second, third]
    la.apply_slot_progress(details, slots, now=1000.0)
    assert first["stage"] == "prefill"
    # /slots no da el total del prompt (n_prompt_tokens es lo cargado hasta
    # ahora): procesado real, total = la estimación (50k) y sin porcentaje.
    assert first["prompt"] == {
        "totalTokens": 50_000,
        "cachedTokens": 4096,
        "processedTokens": 24_096,
        "source": "live",
        "totalKnown": False,
    }
    assert first["prefillPct"] is None
    assert second["stage"] == "generating" and second["outputTokens"] == 12
    assert third["stage"] == "backend_wait"
    # Segunda consulta (el endpoint reconstruye los detalles cada vez): ritmo
    # medido entre las dos muestras del mismo slot.
    slots[1] = _slot(3, 49_558, 30_000, cached=4096)
    again = [dict(tracked), _untracked("a", 100.0), _untracked("b", 101.0), _untracked("c", 102.0)]
    la.apply_slot_progress(again, slots, now=1005.0)
    assert again[1]["prefillTokensPerSecond"] == 2000.0
    assert again[1]["lastActivitySecondsAgo"] == 0.0
    # Sin cambios en otra consulta: la actividad envejece.
    third_poll = [
        dict(tracked),
        _untracked("a", 100.0),
        _untracked("b", 101.0),
        _untracked("c", 102.0),
    ]
    la.apply_slot_progress(third_poll, slots, now=1020.0)
    assert third_poll[1]["lastActivitySecondsAgo"] == 15.0


def test_stall_flag_only_when_there_is_a_signal_to_watch():
    base = {"lastActivitySecondsAgo": 40.0}
    no_stream = {
        **base,
        "stage": "processing",
        "streaming": False,
        "prompt": {"source": "estimate"},
    }
    ollama_prefill = {
        **base,
        "stage": "prefill",
        "streaming": True,
        "prompt": {"source": "estimate"},
    }
    live_prefill = {**base, "stage": "prefill", "streaming": True, "prompt": {"source": "live"}}
    gen_stream = {**base, "stage": "generating", "streaming": True, "prompt": {"source": "live"}}
    fresh = {"lastActivitySecondsAgo": 2.0, "stage": "generating", "streaming": True, "prompt": {}}
    la.mark_stalls([no_stream, ollama_prefill, live_prefill, gen_stream, fresh])
    assert [d["stalled"] for d in (no_stream, ollama_prefill, live_prefill, gen_stream, fresh)] == [
        False,
        False,
        True,
        True,
        False,
    ]


def test_model_phase_reports_eviction_service_wait_and_load_timing():
    class MM:
        def __init__(self, phases, blockers=None, started=None):
            self.phases, self.blockers, self.started = phases, blockers or {}, started or {}

        def get_load_phase(self, key):
            return self.phases.get(key, ("ready", None))

        def service_gpu_blocker(self, key):
            return self.blockers.get(key)

        def load_started_at(self, key):
            return self.started.get(key)

    state = SimpleNamespace(model_id="m", status=SimpleNamespace(value="loading"))
    evicting = la.model_phase(
        MM({"m::h": ("waiting_for_gpu", "gemma4:26b")}),
        state,
        ["m::h"],
        now=100.0,
        typical_ms={"m": 40_000},
    )
    assert evicting["phase"] == "waiting_for_gpu" and evicting["waitingOn"] == "gemma4:26b"
    service = la.model_phase(MM({}, blockers={"m": "comfyui"}), state, [], now=100.0, typical_ms={})
    assert service == {
        "phase": "waiting_for_service_gpu",
        "waitingOn": "comfyui",
        "loadingForMs": None,
        "typicalLoadMs": None,
    }
    loading = la.model_phase(
        MM({"m::h": ("loading", None)}, started={"m::h": 70.0}),
        state,
        ["m::h"],
        now=100.0,
        typical_ms={"m": 40_000},
    )
    assert loading == {
        "phase": "loading",
        "waitingOn": None,
        "loadingForMs": 30_000,
        "typicalLoadMs": 40_000,
    }


def test_slot_total_never_below_processed():
    """Visto en producción: estimación de 10k para un prompt real de 43,9k."""
    la._slot_track.clear()
    d = _untracked("x", 1.0)
    d["prompt"]["totalTokens"] = 10_004
    la.apply_slot_progress([d], [_slot(1, 43_528, 43_016)], now=10.0)
    assert d["prompt"]["processedTokens"] == 43_016
    assert d["prompt"]["totalTokens"] == 43_016
    assert d["prefillPct"] is None
