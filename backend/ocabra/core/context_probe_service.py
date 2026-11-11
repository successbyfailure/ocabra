"""Run a context probe against a real preset (see ``context_probe.py``).

One probe at a time, as a background job the API polls:

1. Resolve the preset's load (its placement and llama.cpp options) and
   replace only its context with a small starting one, so the probe worker
   gets its own key and never collides with the preset's real worker.
2. Put the profile's model in maintenance (its requests get a clean 503) and
   unload whatever else sits on the preset's GPUs: the probe measures the
   cards empty. Pinned models that had to go are loaded back at the end.
3. Load the probe worker and grow its context in place to the limit.
4. Optionally fill the pool with one long prompt, to see the peak in use.
5. Optionally load the vision projector to measure what it costs; it is not
   part of the budget, and is loaded last because it cannot be unloaded.
6. Unload the probe worker, lift maintenance, reload what was evicted.
"""

from __future__ import annotations

import asyncio
import copy
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
import structlog

from ocabra.backends.llama_cpp_backend import LiveContextFailed
from ocabra.config import settings
from ocabra.core.context_probe import GRANULARITY, ProbeError, ProbeStep, run_probe
from ocabra.core.model_manager_helpers import compute_worker_key

logger = structlog.get_logger(__name__)

#: Context the probe worker starts with (never above the preset's own).
START_CONTEXT = 32768
#: Other processes may hold this much on a probed GPU without spoiling it.
_FOREIGN_TOLERANCE_MB = 256
_DRAIN_TIMEOUT_S = 60.0
_RELOAD_ATTEMPTS = 4
_RELOAD_BACKOFF_S = 5.0


@dataclass
class ProbeJob:
    job_id: str
    profile_id: str
    preset_key: str
    full_validation: bool
    measure_vision: bool
    status: str = (
        "pending"  # preparing, loading, probing, validating, vision, restoring, done, error
    )
    gpu_indices: list[int] = field(default_factory=list)
    context_is_pool: bool = True
    parallelism: int = 1
    steps: list[dict] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ContextProbeService:
    def __init__(self, model_manager, profile_registry, worker_pool) -> None:
        self._mm = model_manager
        self._registry = profile_registry
        self._pool = worker_pool
        self._lock = asyncio.Lock()
        self._jobs: dict[str, ProbeJob] = {}
        self._latest: dict[tuple[str, str], str] = {}

    # -- public API --------------------------------------------------------

    def busy(self) -> bool:
        return self._lock.locked()

    def get(self, job_id: str) -> ProbeJob | None:
        return self._jobs.get(job_id)

    def latest(self, profile_id: str, preset_key: str) -> ProbeJob | None:
        job_id = self._latest.get((profile_id, preset_key))
        return self._jobs.get(job_id) if job_id else None

    def start(
        self, profile_id: str, preset_key: str, *, full_validation: bool, measure_vision: bool
    ) -> ProbeJob:
        if self.busy():
            raise RuntimeError("another context probe is running")
        job = ProbeJob(
            job_id=uuid.uuid4().hex[:12],
            profile_id=profile_id,
            preset_key=preset_key,
            full_validation=full_validation,
            measure_vision=measure_vision,
        )
        self._jobs[job.job_id] = job
        self._latest[(profile_id, preset_key)] = job.job_id
        asyncio.get_running_loop().create_task(self._run(job), name=f"context-probe:{job.job_id}")
        return job

    # -- the job -----------------------------------------------------------

    async def _run(self, job: ProbeJob) -> None:
        async with self._lock:
            base_id: str | None = None
            probe_key: str | None = None
            pinned_evicted: list[str] = []
            try:
                job.status = "preparing"
                profile, overrides, gpus, start = await self._prepare_load(job)
                base_id = profile.base_model_id
                probe_key = compute_worker_key(base_id, overrides)
                job.gpu_indices = gpus

                self._mm.set_maintenance(base_id, True)
                pinned_evicted = await self._clear_gpus(job, gpus, keep=probe_key)

                job.status = "loading"
                from ocabra.api.openai._deps import ensure_worker_loaded

                state = await ensure_worker_loaded(
                    self._mm,
                    base_id,
                    probe_key,
                    overrides,
                    preferred_gpu=gpus[0] if len(gpus) == 1 else None,
                    load_policy="on_demand",
                )
                limits = self._limits(gpus, state)
                cap = await self._trained_context(probe_key) or 1048576

                job.status = "probing"
                result = await run_probe(
                    start_context=start,
                    cap=cap,
                    limit_mb=limits,
                    resize=lambda ctx: self._resize(probe_key, ctx),
                    measure=lambda: self._measure(probe_key, gpus),
                    on_step=lambda step: job.steps.append(_step_dict(step)),
                )
                report: dict[str, Any] = {
                    "max_context": result.max_context,
                    "used_mb": result.used_mb,
                    "limit_mb": result.limit_mb,
                    "kv_mb_per_1k": result.kv_mb_per_1k,
                    "capped_by": result.capped_by,
                    "trained_context": cap,
                    "margin_mb": int(settings.vram_buffer_mb),
                    "full_validated": False,
                }

                if job.full_validation:
                    job.status = "validating"
                    report.update(await self._validate_full(job, probe_key, gpus, result))

                if job.measure_vision:
                    job.status = "vision"
                    report.update(await self._measure_vision(job, probe_key, gpus, report))

                report["measured_at"] = time.time()
                job.result = report
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                logger.warning("context_probe_failed", job_id=job.job_id, error=str(exc))
                job.error = str(exc)
                job.status = "error"
            finally:
                restoring_from = job.status
                job.status = "restoring" if restoring_from == "done" else job.status
                await self._restore(base_id, probe_key, pinned_evicted)
                if restoring_from == "done":
                    job.status = "done"
                job.finished_at = time.time()
                logger.info(
                    "context_probe_finished",
                    job_id=job.job_id,
                    status=job.status,
                    max_context=(job.result or {}).get("max_context"),
                )

    async def _prepare_load(self, job: ProbeJob):
        from ocabra.core.runtime_coordinator import plan_profile_runtime

        profile = await self._registry.get(job.profile_id)
        if profile is None:
            raise ProbeError(f"profile '{job.profile_id}' not found")
        if not any(p.preset_key == job.preset_key for p in profile.runtime_presets or []):
            raise ProbeError(f"preset '{job.preset_key}' not found in '{job.profile_id}'")
        base_state = await self._mm.get_state(profile.base_model_id)
        backend = self._pool._backends.get(getattr(base_state, "backend_type", ""))
        if not getattr(backend, "SUPPORTS_LIVE_CONTEXT", False):
            raise ProbeError("only llama.cpp fork models can change their context live")
        if not settings.llama_cpp_live_context_enabled:
            raise ProbeError("live context is disabled (LLAMA_CPP_LIVE_CONTEXT_ENABLED)")

        plan = await plan_profile_runtime(
            profile,
            self._mm,
            {
                "messages": [{"role": "user", "content": "probe"}],
                "max_tokens": 16,
                "ocabra": {"context_preset": job.preset_key},
            },
        )
        overrides = copy.deepcopy(plan.load_overrides or {})
        runtime = overrides.setdefault("_runtime", {})
        gpus = [int(g) for g in runtime.get("gpu_indices") or []]
        if not gpus:
            raise ProbeError("the preset does not declare its GPUs")
        pool = bool(runtime.get("context_is_pool", True))
        parallel = max(1, int(runtime.get("parallelism") or 1))
        start = min(START_CONTEXT, int(plan.effective_context or START_CONTEXT))
        start = max(GRANULARITY, start // GRANULARITY * GRANULARITY)

        section = overrides.setdefault("llama_cpp", {})
        section["ctx_size"] = start
        use_case = overrides.setdefault("use_case", {})
        use_case["context"] = start if pool else max(1, start // parallel)
        runtime["effective_context"] = start
        runtime["capacity_validated"] = False
        runtime["capacity_vram_mb"] = None
        runtime["context_probe"] = True
        runtime.pop("default_worker_key", None)
        runtime.pop("default_load_overrides", None)
        job.context_is_pool = pool
        job.parallelism = parallel
        return profile, overrides, gpus, start

    async def _clear_gpus(self, job: ProbeJob, gpus: list[int], *, keep: str) -> list[str]:
        """Unload everything on the probed GPUs; returns the pinned ones."""
        from ocabra.core.model_manager import LoadPolicy, ModelStatus

        pinned: list[str] = []
        for state in await self._mm.list_states(include_derived=True):
            if state.model_id == keep or state.status != ModelStatus.LOADED:
                continue
            if not set(state.current_gpu or []) & set(gpus):
                continue
            deadline = time.monotonic() + _DRAIN_TIMEOUT_S
            while self._mm.is_busy(state.model_id):
                if time.monotonic() > deadline:
                    raise ProbeError(f"'{state.model_id}' kept serving requests on GPU {gpus}")
                await asyncio.sleep(1.0)
            await self._mm.unload(state.model_id, reason="context_probe")
            job.evicted.append(state.model_id)
            if state.load_policy == LoadPolicy.PIN:
                pinned.append(state.model_id)
        return pinned

    def _limits(self, gpus: list[int], state) -> dict[int, int]:
        """What the worker may hold on each GPU: the admission ceiling (which
        already keeps ``vram_buffer_mb`` aside) minus other processes."""
        gm = self._mm._gpu_manager
        pid = state.worker_info.pid if state.worker_info else None
        limits: dict[int, int] = {}
        for gpu in gpus:
            ceiling = gm.admission_ceiling_mb(gpu, physical=False)
            if ceiling is None:
                raise ProbeError(f"cannot read GPU {gpu}'s reserved memory")
            foreign = sum(mb for p, mb in gm.get_process_vram_mb(gpu).items() if p != pid)
            if foreign > _FOREIGN_TOLERANCE_MB:
                logger.info("context_probe_foreign_vram", gpu=gpu, foreign_mb=foreign)
            limits[gpu] = int(ceiling) - int(foreign)
        return limits

    async def _resize(self, probe_key: str, ctx: int) -> bool:
        try:
            await self._mm.resize_context(probe_key, context=ctx)
        except LiveContextFailed:
            return False
        return True

    async def _measure(self, probe_key: str, gpus: list[int]) -> dict[int, int]:
        state = self._mm._states[probe_key]
        usage = self._mm._worker_vram_by_gpu(state)
        return {gpu: int(usage.get(gpu, 0)) for gpu in gpus}

    async def _trained_context(self, probe_key: str) -> int | None:
        worker = self._pool.get_worker(probe_key)
        if worker is None:
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                data = (await client.get(f"http://127.0.0.1:{worker.port}/v1/models")).json()
            return int(data["data"][0]["meta"]["n_ctx_train"])
        except Exception:  # noqa: BLE001
            return None

    async def _validate_full(self, job, probe_key, gpus, result) -> dict:
        """Fill the pool with one long prompt and keep the peak in use."""
        worker = self._pool.get_worker(probe_key)
        target_tokens = int(result.max_context * 0.9)
        prompt = await self._prompt_of(worker.port, target_tokens)
        peak = dict(result.used_mb)
        done = asyncio.Event()

        async def sample() -> None:
            while not done.is_set():
                for gpu, mb in (await self._measure(probe_key, gpus)).items():
                    peak[gpu] = max(peak.get(gpu, 0), mb)
                await asyncio.sleep(0.25)

        sampler = asyncio.create_task(sample())
        try:
            async with httpx.AsyncClient(timeout=3600) as client:
                response = await client.post(
                    f"http://127.0.0.1:{worker.port}/completion",
                    json={"prompt": prompt, "n_predict": 8, "cache_prompt": False},
                )
                response.raise_for_status()
                prompt_n = (response.json().get("timings") or {}).get("prompt_n")
        finally:
            done.set()
            await sampler
        over = {g: peak[g] - result.limit_mb[g] for g in gpus if peak[g] > result.limit_mb[g]}
        report: dict[str, Any] = {
            "full_validated": True,
            "validation_prompt_tokens": prompt_n,
            "peak_mb": peak,
        }
        if over:
            # The peak in use went past the margin: back off by what it took.
            cut = max(over[g] / max(result.kv_mb_per_1k.get(g, 0) or 1, 0.01) * 1024 for g in over)
            reduced = max(GRANULARITY, int(result.max_context - cut) // GRANULARITY * GRANULARITY)
            reduced -= GRANULARITY
            if not await self._resize(probe_key, reduced):
                raise ProbeError("could not shrink after the full-pool test went over the margin")
            report["max_context"] = reduced
            report["used_mb"] = await self._measure(probe_key, gpus)
            report["reduced_by_validation"] = result.max_context - reduced
        return report

    async def _prompt_of(self, port: int, tokens: int) -> str:
        """Text of about ``tokens`` tokens, calibrated with the worker's tokenizer."""
        unit = (
            "En la reunión del lunes se revisaron los presupuestos del taller, el "
            "calendario de cursos y el estado de las impresoras {n}. "
        )
        chunk = "".join(unit.format(n=i) for i in range(200))
        async with httpx.AsyncClient(timeout=60) as client:
            counted = (
                await client.post(f"http://127.0.0.1:{port}/tokenize", json={"content": chunk})
            ).json().get("tokens") or []
        per_chunk = max(1, len(counted))
        repeats = max(1, tokens // per_chunk)
        return chunk * repeats

    async def _measure_vision(self, job, probe_key, gpus, report) -> dict:
        """Load the projector and record what it took on each GPU."""
        state = self._mm._states[probe_key]
        backend = self._pool._backends.get(state.backend_type)
        path = backend.vision_projector_path(state.backend_model_id) if backend else None
        if not path:
            return {"vision": None}
        size = await asyncio.to_thread(os.path.getsize, path)
        estimate = int(size / 1024 / 1024 * 1.1) + 256
        max_ctx = int(report["max_context"])
        used = await self._measure(probe_key, gpus)
        headroom = min(report["limit_mb"][g] - used[g] for g in gpus)
        ctx = max_ctx
        if headroom < estimate:
            worst = max(report["kv_mb_per_1k"].get(g, 0) for g in gpus) or 1.0
            ctx = max(
                GRANULARITY,
                int(max_ctx - (estimate - headroom) / worst * 1024) // GRANULARITY * GRANULARITY,
            )
            if not await self._resize(probe_key, ctx):
                raise ProbeError("could not shrink the context to make room for the projector")
        before = await self._measure(probe_key, gpus)
        await backend.activate_vision(state.backend_model_id)
        after = await self._measure(probe_key, gpus)
        cost = {g: max(0, after[g] - before[g]) for g in gpus}
        # Context that still fits with the projector loaded.
        kv = report["kv_mb_per_1k"]
        excess = [
            (report["used_mb"][g] + cost[g] - report["limit_mb"][g])
            / max(kv.get(g, 0), 0.01)
            * 1024
            for g in gpus
            if kv.get(g)
        ]
        with_vision = max_ctx - max([0.0, *excess])
        with_vision = max(GRANULARITY, int(with_vision) // GRANULARITY * GRANULARITY)
        job.steps.append(_step_dict(ProbeStep(ctx, after, True, f"vision loaded: {cost} MB")))
        return {
            "vision": {
                "projector": os.path.basename(path),
                "cost_mb": cost,
                "max_context_with_vision": with_vision,
            }
        }

    async def _restore(self, base_id, probe_key, pinned_evicted) -> None:
        if probe_key and probe_key in self._mm._states:
            state = self._mm._states[probe_key]
            gpus = list(state.current_gpu or [])
            held = sum(self._mm._worker_vram_by_gpu(state).values())
            try:
                snapshot = await self._mm._get_vram_free_snapshot(gpus) if gpus else None
                await self._mm.unload(probe_key, reason="context_probe")
                # The driver frees a 20 GB worker over a few hw-monitor cycles:
                # reloading the evicted pins before that gets a 409 (seen live
                # 2026-11-11: gemma4:26b was left in error on an empty 3090).
                if gpus and held:
                    await self._mm._wait_for_vram_released(
                        held,
                        baseline_free_mb=snapshot[0] if snapshot else None,
                        gpu_indices=gpus,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("context_probe_unload_failed", error=str(exc))
        if base_id:
            self._mm.set_maintenance(base_id, False)
        for model_id in pinned_evicted:
            for attempt in range(_RELOAD_ATTEMPTS):
                try:
                    await self._mm.load(model_id)
                    break
                except Exception as exc:  # noqa: BLE001 - its PIN watcher retries too
                    logger.warning(
                        "context_probe_reload_failed",
                        model_id=model_id,
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                    await asyncio.sleep(_RELOAD_BACKOFF_S)


def _step_dict(step: ProbeStep) -> dict:
    return {
        "context": step.context,
        "used_mb": dict(step.used_mb),
        "fits": step.fits,
        "note": step.note,
    }
