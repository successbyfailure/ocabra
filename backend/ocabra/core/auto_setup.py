"""Automatic runtime presets for a llama.cpp model on the devices the user picks.

Chosen when the model is downloaded (or later from the editor): which GPUs
it may use and whether the CPU may hold part of it. From that, and from
llama.cpp's own memory projection (``llama-fit-params``, see ``llama_fit``),
this module plans the presets ("rungs") and writes them:

- It fits on one GPU: ``normal`` on the biggest selected GPU with the largest
  context that fits; with two or more GPUs selected, also ``extended`` over
  all of them with the largest context up to the model's native one.
- It only fits across GPUs: ``normal`` over all selected GPUs.
- It does not fit in the selected GPUs: with the CPU allowed, the fewest MoE
  experts (``--n-cpu-moe``) or, for dense models, the fewest layers move to
  RAM; without the CPU, the plan says so and nothing is written.

A multi-GPU rung also gets a projection-derived context probe in
``runtime_policy.context_probe`` (same shape as a real probe), so it starts
leaving whisper's reserve free and grows in place when a request needs it
(``runtime_coordinator._apply_multi_gpu_reserve``) without a probe run first.

Budgets per GPU subtract what no oCabra model can have: memory owned by no
process, processes that are not oCabra workers (a desktop on the card, as on
Mapache), the scheduler buffer and the VRAM watchdog's margin. oCabra models
currently loaded are not subtracted: they can be evicted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import structlog

from ocabra.config import settings

logger = structlog.get_logger(__name__)

GRANULARITY = 4096
#: Smallest context worth a rung.
MIN_CONTEXT = 8192
#: Context a rung with weights in RAM aims for before trading speed for more.
TARGET_CONTEXT = 32768
DEFAULT_PARALLEL = 4


@dataclass
class GpuBudget:
    index: int
    name: str
    total_mb: int
    budget_mb: int


@dataclass
class Rung:
    key: str
    label: str
    gpus: list[int]
    context: int
    placement: dict[str, Any] = field(default_factory=dict)
    vram_by_gpu: dict[int, int] = field(default_factory=dict)
    host_mb: int = 0
    probe: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "gpus": self.gpus,
            "context": self.context,
            "placement": self.placement,
            "vram_by_gpu": self.vram_by_gpu,
            "host_mb": self.host_mb,
        }


@dataclass
class SetupPlan:
    rungs: list[Rung] = field(default_factory=list)
    needs_cpu: bool = False
    warnings: list[str] = field(default_factory=list)
    budgets: list[GpuBudget] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rungs": [r.as_dict() for r in self.rungs],
            "needs_cpu": self.needs_cpu,
            "warnings": self.warnings,
            "budgets": [b.__dict__ for b in self.budgets],
        }


def _gpu_value(gpu: Any, name: str, default: Any = 0) -> Any:
    return getattr(gpu, name, None) if not isinstance(gpu, dict) else gpu.get(name, default)


async def gpu_budgets(model_manager: Any, indices: list[int]) -> list[GpuBudget]:
    """What each selected GPU can give one model, if every oCabra model left.

    Counted by the scheduler's own rule (``get_free_vram_if_released`` with
    every oCabra model on that GPU), so the buffer, service reservations and
    processes oCabra does not own (a desktop) stay out. Matching workers by
    the PIDs of the polled process list failed on Mapache, where those are the
    host's PIDs and ``WorkerInfo.pid`` the container's: two loaded encoders
    counted as foreign memory, the budget fell to 147 MB and the editor and
    the planner found no configuration at all. The eviction threshold also
    caps the budget when it is stricter than the buffer (large cards).
    """
    gm = getattr(model_manager, "_gpu_manager", None)
    if gm is None:
        return []
    states = list(getattr(model_manager, "_states", {}).values())
    threshold = float(getattr(settings, "vram_eviction_threshold", 0.96) or 0.96)
    buffer = int(settings.vram_buffer_mb or 0)
    out: list[GpuBudget] = []
    for gpu in await gm.get_all_states():
        index = int(_gpu_value(gpu, "index"))
        if index not in indices:
            continue
        total = int(_gpu_value(gpu, "total_vram_mb") or 0)
        released = {
            str(getattr(s, "model_id", id(s))): getattr(
                getattr(s, "worker_info", None), "pid", None
            )
            for s in states
            if index in (getattr(s, "current_gpu", None) or [])
        }
        room = int(await gm.get_free_vram_if_released(index, released))
        # The buffer is already out of ``room``; the threshold only bites when
        # it reserves more than the buffer.
        room -= max(0, (total - buffer) - int(total * threshold))
        out.append(
            GpuBudget(
                index=index,
                name=str(_gpu_value(gpu, "name", "") or f"GPU {index}"),
                total_mb=total,
                budget_mb=max(0, room),
            )
        )
    out.sort(key=lambda b: b.budget_mb, reverse=True)
    return out


def materialized_extra(state: Any) -> dict[str, Any]:
    """``state.extra_config`` with the weight paths resolved, as at load time.

    A model registered from an Ollama tag stores no ``model_path``: oCabra
    resolves it from the artifact right before each load, so planning without
    this finds no weights (2026-11-16, gemma4:26b).
    """
    extra = copy.deepcopy(state.extra_config or {})
    artifact = getattr(state, "artifact", None)
    if not artifact:
        return extra
    try:
        from ocabra.core import artifacts as model_artifacts

        return model_artifacts.materialize(
            state.backend_type, artifact, extra, model_artifacts.ollama_root()
        )
    except Exception as exc:  # noqa: BLE001 - plan with what is there
        logger.info("auto_setup_materialize_failed", error=str(exc))
        return extra


class _Projector:
    """Projections for one model, with the base config plus overrides."""

    def __init__(self, backend: Any, state: Any) -> None:
        self._backend = backend
        self._state = state
        self._extra = materialized_extra(state)

    async def __call__(self, gpus: list[int], overrides: dict[str, Any]) -> dict | None:
        extra = copy.deepcopy(self._extra)
        section = dict(extra.get("llama_cpp") or {})
        section.pop("launch_ctx_size", None)
        # The devices are the user's choice here: the base config's own GPU
        # exclusions would hide them (gemma4:26b keeps GPU 0 disabled, and its
        # plan on GPU 0 projected everything into RAM).
        section.pop("disabled_gpus", None)
        section.pop("main_gpu", None)
        section.update({k: v for k, v in overrides.items() if v is not None})
        for key in ("n_cpu_moe", "tensor_split", "gpu_layers", "split_mode"):
            if key in overrides and overrides[key] is None:
                section.pop(key, None)
        extra["llama_cpp"] = section
        extra.pop("_runtime", None)
        return await self._backend.project_memory(self._state.backend_model_id, gpus, extra)


def _fits(projection: dict | None, budgets: dict[int, int]) -> bool:
    if not projection:
        return False
    per_gpu = projection.get("vram_by_gpu") or {}
    if not per_gpu and budgets:
        return False
    return all(int(mb) <= budgets.get(int(g), 0) for g, mb in per_gpu.items())


async def _max_context(
    project: _Projector,
    gpus: list[int],
    overrides: dict[str, Any],
    budgets: dict[int, int],
    native: int,
) -> tuple[int, dict | None]:
    """Largest context (multiple of GRANULARITY, ≤ native) that fits, or 0."""
    lo, hi = MIN_CONTEXT, max(MIN_CONTEXT, native)
    best, best_proj = 0, None
    proj = await project(gpus, {**overrides, "ctx_size": lo})
    if not _fits(proj, budgets):
        return 0, None
    best, best_proj = lo, proj
    proj = await project(gpus, {**overrides, "ctx_size": hi})
    if _fits(proj, budgets):
        return hi, proj
    lo_steps, hi_steps = lo // GRANULARITY, hi // GRANULARITY
    while hi_steps - lo_steps > 1:
        mid = (lo_steps + hi_steps) // 2
        proj = await project(gpus, {**overrides, "ctx_size": mid * GRANULARITY})
        if _fits(proj, budgets):
            lo_steps, best, best_proj = mid, mid * GRANULARITY, proj
        else:
            hi_steps = mid
    return best, best_proj


def _split(gpus: list[int], budgets: dict[int, int], mode: str = "layer") -> dict[str, Any]:
    """Multi-GPU split: ``mode`` comes from the model's base config (qwen3.8
    keeps ``tensor``, +9 % prefill with P2P/NCCL), else ``layer``."""
    if len(gpus) < 2:
        return {"split_mode": None, "tensor_split": None}
    return {
        "split_mode": mode,
        "tensor_split": [float(max(1, budgets.get(g, 1))) for g in gpus],
    }


async def _fit_on(
    project: _Projector,
    gpus: list[int],
    budgets: dict[int, int],
    native: int,
    *,
    layers: int,
    moe: bool,
    allow_cpu: bool,
    mode: str = "layer",
) -> tuple[int, dict[str, Any], dict | None]:
    """``(context, placement, projection)`` for ``gpus``; context 0 = no fit."""
    base = {"parallel_slots": DEFAULT_PARALLEL, **_split(gpus, budgets, mode)}
    full = {**base, "gpu_layers": 999, "n_cpu_moe": None}
    ctx, proj = await _max_context(project, gpus, full, budgets, native)
    if ctx or not allow_cpu or layers <= 0:
        return ctx, ({"gpu_layers": None, "n_cpu_moe": None} if ctx else {}), proj
    target = min(native, TARGET_CONTEXT)

    async def _fits_with(n: int) -> bool:
        trial = (
            {**base, "gpu_layers": 999, "n_cpu_moe": n}
            if moe
            else {**base, "gpu_layers": layers - n, "n_cpu_moe": None}
        )
        return _fits(await project(gpus, {**trial, "ctx_size": target}), budgets)

    # Fewest experts (MoE) or layers (dense) in RAM that reach the target.
    lo, hi = 0, layers
    if not await _fits_with(hi):
        n_off = layers
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if await _fits_with(mid):
                hi = mid
            else:
                lo = mid
        n_off = hi
    over = (
        {**base, "gpu_layers": 999, "n_cpu_moe": n_off}
        if moe
        else {**base, "gpu_layers": max(0, layers - n_off), "n_cpu_moe": None}
    )
    ctx, proj = await _max_context(project, gpus, over, budgets, native)
    placement = (
        {"gpu_layers": None, "n_cpu_moe": n_off}
        if moe
        else {"gpu_layers": max(0, layers - n_off), "n_cpu_moe": None}
    )
    return ctx, placement, proj


async def _probe_from_projection(
    project: _Projector,
    gpus: list[int],
    overrides: dict[str, Any],
    context: int,
    projection: dict,
    budgets: dict[int, int],
) -> dict[str, Any] | None:
    """A context probe built from two projections instead of a real run."""
    step = min(context // 2, 32768)
    if step < GRANULARITY:
        return None
    lower = await project(gpus, {**overrides, "ctx_size": context - step})
    if not lower:
        return None
    used = {str(g): int(mb) for g, mb in (projection.get("vram_by_gpu") or {}).items()}
    low = {str(g): int(mb) for g, mb in (lower.get("vram_by_gpu") or {}).items()}
    kv = {g: round(max(0.0, (used[g] - low.get(g, used[g])) / (step / 1024)), 2) for g in used}
    return {
        "pool_tokens": context,
        "context_tokens": context,
        "gpu_indices": gpus,
        "used_mb": used,
        "limit_mb": {str(g): int(budgets.get(g, 0)) for g in gpus},
        "kv_mb_per_1k": kv,
        "capped_by": "projection",
        "full_validated": False,
        "source": "llama_fit",
    }


async def plan_setup(
    model_manager: Any,
    backend: Any,
    state: Any,
    *,
    gpus: list[int],
    allow_cpu: bool,
) -> SetupPlan:
    """Plan the rungs for ``state`` on the selected devices (nothing written)."""
    plan = await _plan_setup(model_manager, backend, state, gpus=gpus, allow_cpu=allow_cpu)
    warn_ram(plan)
    return plan


async def _plan_setup(
    model_manager: Any,
    backend: Any,
    state: Any,
    *,
    gpus: list[int],
    allow_cpu: bool,
) -> SetupPlan:
    from ocabra.core import vram_planner as vp

    plan = SetupPlan()
    budgets_list = await gpu_budgets(model_manager, list(gpus))
    plan.budgets = budgets_list
    budgets = {b.index: b.budget_mb for b in budgets_list}
    project = _Projector(backend, state)

    section = materialized_extra(state).get("llama_cpp") or {}
    base_mode = str(section.get("split_mode") or "")
    mode = base_mode if base_mode in {"layer", "row", "tensor"} else "layer"
    model_file = section.get("model_path")
    arch = vp.arch_from_gguf(model_file) if model_file else None
    layers = int(getattr(arch, "layers", 0) or 0)
    moe = bool(getattr(arch, "expert_mb_by_layer", ()))
    native = int(getattr(arch, "context_length", 0) or section.get("ctx_size") or 32768)
    native = max(MIN_CONTEXT, native // GRANULARITY * GRANULARITY)

    if not budgets_list:
        if not allow_cpu:
            plan.warnings.append("No hay ninguna GPU seleccionada ni CPU.")
            return plan
        overrides = {"parallel_slots": DEFAULT_PARALLEL, "gpu_layers": 0}
        ctx = min(native, TARGET_CONTEXT)
        proj = await project([], {**overrides, "ctx_size": ctx})
        plan.rungs.append(
            Rung(
                key="normal",
                label=f"CPU · {ctx // 1024}k",
                gpus=[],
                context=ctx,
                placement={"gpu_layers": 0},
                host_mb=int((proj or {}).get("host_mb") or 0),
            )
        )
        return plan

    big = budgets_list[0].index
    all_gpus = [b.index for b in budgets_list]

    single_ctx, single_place, single_proj = await _fit_on(
        project, [big], budgets, native, layers=layers, moe=moe, allow_cpu=False, mode=mode
    )
    if single_ctx:
        plan.rungs.append(
            Rung(
                key="normal",
                label=f"{budgets_list[0].name.split()[-1]} · {single_ctx // 1024}k",
                gpus=[big],
                context=single_ctx,
                placement=single_place,
                vram_by_gpu=dict((single_proj or {}).get("vram_by_gpu") or {}),
                host_mb=int((single_proj or {}).get("host_mb") or 0),
            )
        )
        if len(all_gpus) > 1 and single_ctx < native:
            ctx, place, proj = await _fit_on(
                project,
                all_gpus,
                budgets,
                native,
                layers=layers,
                moe=moe,
                allow_cpu=False,
                mode=mode,
            )
            if ctx > single_ctx:
                overrides = {"parallel_slots": DEFAULT_PARALLEL, **_split(all_gpus, budgets, mode)}
                plan.rungs.append(
                    Rung(
                        key="extended",
                        label=f"Dual · {ctx // 1024}k",
                        gpus=all_gpus,
                        context=ctx,
                        placement={**place, **_split(all_gpus, budgets, mode)},
                        vram_by_gpu=dict((proj or {}).get("vram_by_gpu") or {}),
                        host_mb=int((proj or {}).get("host_mb") or 0),
                        probe=await _probe_from_projection(
                            project, all_gpus, overrides, ctx, proj or {}, budgets
                        ),
                    )
                )
        return plan

    # It does not fit on one GPU by itself: all selected GPUs, then the CPU.
    ctx, place, proj = await _fit_on(
        project, all_gpus, budgets, native, layers=layers, moe=moe, allow_cpu=allow_cpu, mode=mode
    )
    if not ctx:
        plan.needs_cpu = not allow_cpu
        speculative = section.get("speculative") if isinstance(section, dict) else None
        if allow_cpu and isinstance(speculative, dict) and speculative.get("spec_type"):
            plan.warnings.append(
                "No cabe con la decodificación especulativa: su borrador ocupa GPU. "
                "Quítala en Aceleración y vuelve a calcular."
            )
        else:
            plan.warnings.append(
                "No cabe en las GPU elegidas: marca también la CPU para dejar parte en RAM."
                if not allow_cpu
                else "No cabe ni con la CPU: el modelo es demasiado grande para estos dispositivos."
            )
        return plan
    multi = len(all_gpus) > 1
    placement = {**place, **(_split(all_gpus, budgets, mode) if multi else {})}
    in_ram = place.get("n_cpu_moe") or (place.get("gpu_layers") is not None)
    label = ("Dual" if multi else budgets_list[0].name.split()[-1]) + f" · {ctx // 1024}k"
    if in_ram:
        label += " · RAM"
    overrides = {"parallel_slots": DEFAULT_PARALLEL, **_split(all_gpus, budgets, mode)}
    plan.rungs.append(
        Rung(
            key="normal",
            label=label,
            gpus=all_gpus,
            context=ctx,
            placement=placement,
            vram_by_gpu=dict((proj or {}).get("vram_by_gpu") or {}),
            host_mb=int((proj or {}).get("host_mb") or 0),
            probe=(
                await _probe_from_projection(project, all_gpus, overrides, ctx, proj or {}, budgets)
                if multi and proj
                else None
            ),
        )
    )
    return plan


def _host_ram_mb() -> int:
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return 0


def warn_ram(plan: SetupPlan) -> None:
    """Flag rungs whose RAM share would not fit in this machine's memory."""
    total = _host_ram_mb()
    if not total:
        return
    for rung in plan.rungs:
        if rung.host_mb > total * 0.85:
            plan.warnings.append(
                f"«{rung.label}» deja {rung.host_mb // 1024} GB en RAM y la máquina "
                f"tiene {total // 1024} GB: puede no caber junto al resto."
            )


async def apply_setup(registry: Any, profile: Any, plan: SetupPlan) -> list[str]:
    """Replace ``profile``'s presets with the plan's rungs. Returns the keys."""
    from ocabra.database import AsyncSessionLocal

    if not plan.rungs:
        return []
    async with AsyncSessionLocal() as session:
        for preset in list(getattr(profile, "runtime_presets", None) or []):
            await registry.delete_runtime_preset(session, profile.profile_id, preset.preset_key)
        for priority, rung in enumerate(plan.rungs, start=1):
            placement = {
                k: v
                for k, v in rung.placement.items()
                if v is not None and k in {"split_mode", "tensor_split", "gpu_layers", "n_cpu_moe"}
            }
            await registry.create_runtime_preset(
                session,
                profile.profile_id,
                {
                    "preset_key": rung.key,
                    "label": rung.label,
                    "context_mode": "fixed",
                    "context_tokens": rung.context,
                    "parallelism": DEFAULT_PARALLEL,
                    "kv_cache_dtype": "q8",
                    "placement_policy": "forced" if rung.gpus else "auto",
                    "preferred_gpus": rung.gpus or None,
                    "enabled": True,
                    "is_default": priority == 1,
                    "priority": priority * 10,
                    "placement": placement or None,
                },
            )
        await session.commit()
    probes = {r.key: r.probe for r in plan.rungs if r.probe}
    if probes:
        fresh = await registry.get(profile.profile_id)
        policy = dict(getattr(fresh, "runtime_policy", None) or {})
        existing = dict(policy.get("context_probe") or {})
        existing.update(probes)
        policy["context_probe"] = existing
        async with AsyncSessionLocal() as session:
            await registry.update(session, profile.profile_id, {"runtime_policy": policy})
    logger.info(
        "auto_setup_applied",
        profile_id=profile.profile_id,
        rungs=[r.as_dict() for r in plan.rungs],
    )
    return [r.key for r in plan.rungs]
