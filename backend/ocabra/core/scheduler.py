from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
import structlog

from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.model_config import EvictionSchedule

logger = structlog.get_logger(__name__)


class InsufficientVRAMError(Exception):
    pass


class GPUScheduler:
    def __init__(self, gpu_manager, model_manager=None) -> None:
        self._gpu_manager = gpu_manager
        self._model_manager = model_manager
        self._last_schedule_run: dict[str, datetime] = {}

    def set_model_manager(self, model_manager) -> None:
        self._model_manager = model_manager

    async def find_gpu_for_model(
        self,
        vram_needed_mb: int,
        preferred_gpu: int | None = None,
        enforce_vllm_headroom: bool = False,
    ) -> list[int]:
        preferred = preferred_gpu if preferred_gpu is not None else settings.default_gpu_index
        pressure_threshold_pct = max(0.0, min(100.0, float(settings.vram_pressure_threshold_pct)))
        states = await self._gpu_manager.get_all_states()
        gpu_indices = list(range(len(states)))
        free_per_gpu = {i: await self._gpu_manager.get_free_vram(i) for i in gpu_indices}
        state_by_gpu = {state.index: state for state in states}

        def _has_vllm_headroom(gpu_idx: int) -> bool:
            if not enforce_vllm_headroom:
                return True
            state = state_by_gpu[gpu_idx]
            required_free_mb = int(state.total_vram_mb * settings.vllm_gpu_memory_utilization)
            return state.free_vram_mb >= required_free_mb

        def _is_under_pressure(gpu_idx: int) -> bool:
            state = state_by_gpu[gpu_idx]
            if state.total_vram_mb <= 0:
                return False
            free_pct = free_per_gpu[gpu_idx] * 100.0 / state.total_vram_mb
            return free_pct < pressure_threshold_pct

        healthy_gpu_indices = [i for i in gpu_indices if not _is_under_pressure(i)]
        total_free = sum(free_per_gpu.values())
        candidate_groups = [healthy_gpu_indices] if healthy_gpu_indices else []
        if not candidate_groups or healthy_gpu_indices != gpu_indices:
            candidate_groups.append(gpu_indices)

        for candidates in candidate_groups:
            ordered = [preferred] + [i for i in candidates if i != preferred]
            for gpu_idx in ordered:
                free = free_per_gpu[gpu_idx]
                if free >= vram_needed_mb and _has_vllm_headroom(gpu_idx):
                    logger.info(
                        "gpu_assigned",
                        gpu=gpu_idx,
                        vram_needed_mb=vram_needed_mb,
                        free_mb=free,
                    )
                    return [gpu_idx]

            tp_candidates = [i for i in candidates if _has_vllm_headroom(i)]
            total_free = sum(free_per_gpu[i] for i in tp_candidates)
            if total_free >= vram_needed_mb:
                logger.info(
                    "tensor_parallel_assigned",
                    gpus=tp_candidates,
                    vram_needed_mb=vram_needed_mb,
                    total_free_mb=total_free,
                )
                return tp_candidates

        raise InsufficientVRAMError(
            f"Need {vram_needed_mb} MB VRAM, only {total_free} MB available across all GPUs"
        )

    async def get_eviction_candidates(
        self,
        vram_needed_mb: int,
        target_gpu: int,
    ) -> list[str]:
        del vram_needed_mb, target_gpu
        import sqlalchemy as sa

        from ocabra.db.model_config import ModelConfig

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                sa.select(ModelConfig).where(
                    ModelConfig.load_policy.in_(["on_demand", "warm", "pin"])
                )
            )
            configs = result.scalars().all()

        policy_order = {"on_demand": 0, "warm": 1, "pin": 2}
        sorted_configs = sorted(
            configs,
            key=lambda c: policy_order.get(c.load_policy, 99),
        )
        return [c.model_id for c in sorted_configs]

    async def check_schedule_evictions(self) -> None:
        if self._model_manager is None:
            logger.debug("schedule_eviction_check_skipped", reason="missing_model_manager")
            return

        schedules = await self._load_enabled_schedules(actions={"evict_warm", "evict_all"})
        now = datetime.now(timezone.utc)
        scheduled_models: set[str] = set()

        for schedule in schedules:
            if not self._is_schedule_due(schedule, now):
                continue
            states = await self._resolve_schedule_states(schedule)
            for state in states:
                if state is None or state.model_id in scheduled_models:
                    continue
                if not self._should_evict_state(state, schedule.action):
                    continue
                scheduled_models.add(state.model_id)
                await self._model_manager.unload(
                    state.model_id,
                    reason=f"schedule:{schedule.action}",
                )

        logger.debug(
            "schedule_eviction_check",
            due_schedules=len(schedules),
            evicted_models=len(scheduled_models),
        )

    async def check_schedule_reloads(self) -> None:
        if self._model_manager is None:
            logger.debug("schedule_reload_check_skipped", reason="missing_model_manager")
            return

        schedules = await self._load_enabled_schedules(actions={"reload"})
        now = datetime.now(timezone.utc)
        reloaded_models: set[str] = set()

        for schedule in schedules:
            if not self._is_schedule_due(schedule, now):
                continue
            states = await self._resolve_schedule_states(schedule)
            for state in states:
                if state is None or state.model_id in reloaded_models:
                    continue
                if not self._should_reload_state(state):
                    continue
                reloaded_models.add(state.model_id)
                await self._model_manager.load(
                    state.model_id,
                    force_gpu=state.preferred_gpu,
                )

        logger.debug(
            "schedule_reload_check",
            due_schedules=len(schedules),
            reloaded_models=len(reloaded_models),
        )

    async def _load_enabled_schedules(self, actions: set[str] | None = None) -> list[EvictionSchedule]:
        query = sa.select(EvictionSchedule).where(EvictionSchedule.enabled.is_(True))
        if actions:
            query = query.where(EvictionSchedule.action.in_(sorted(actions)))

        async with AsyncSessionLocal() as session:
            result = await session.execute(query)
            return list(result.scalars().all())

    async def _resolve_schedule_states(self, schedule: EvictionSchedule) -> list:
        if self._model_manager is None:
            return []
        if schedule.model_id:
            state = await self._model_manager.get_state(schedule.model_id)
            return [state] if state is not None else []
        return await self._model_manager.list_states()

    def _should_evict_state(self, state, action: str) -> bool:
        from ocabra.core.model_manager import LoadPolicy, ModelStatus

        if state.status != ModelStatus.LOADED:
            return False
        if action == "evict_warm":
            return state.load_policy == LoadPolicy.WARM
        if action == "evict_all":
            return state.load_policy in {LoadPolicy.WARM, LoadPolicy.PIN}
        return False

    def _should_reload_state(self, state) -> bool:
        from ocabra.core.model_manager import LoadPolicy, ModelStatus

        if not state.auto_reload:
            return False
        if state.load_policy not in {LoadPolicy.WARM, LoadPolicy.PIN}:
            return False
        return state.status in {ModelStatus.CONFIGURED, ModelStatus.UNLOADED}

    def _is_schedule_due(self, schedule: EvictionSchedule, now: datetime) -> bool:
        current_minute = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
        schedule_id = str(schedule.id)
        if self._last_schedule_run.get(schedule_id) == current_minute:
            return False
        if not self._cron_matches(schedule.cron_expr, current_minute):
            return False
        self._last_schedule_run[schedule_id] = current_minute
        return True

    def _cron_matches(self, cron_expr: str, moment: datetime) -> bool:
        parts = str(cron_expr or "").split()
        if len(parts) != 5:
            logger.warning("invalid_schedule_cron", cron_expr=cron_expr)
            return False

        minute, hour, day, month, weekday = parts
        cron_weekday = (moment.weekday() + 1) % 7
        return all(
            (
                self._field_matches(minute, moment.minute, 0, 59),
                self._field_matches(hour, moment.hour, 0, 23),
                self._field_matches(day, moment.day, 1, 31),
                self._field_matches(month, moment.month, 1, 12),
                self._field_matches(weekday, cron_weekday, 0, 7, sunday_alias=True),
            )
        )

    def _field_matches(
        self,
        expr: str,
        value: int,
        minimum: int,
        maximum: int,
        sunday_alias: bool = False,
    ) -> bool:
        return any(
            self._part_matches(part.strip(), value, minimum, maximum, sunday_alias)
            for part in str(expr).split(",")
            if part.strip()
        )

    def _part_matches(
        self,
        expr: str,
        value: int,
        minimum: int,
        maximum: int,
        sunday_alias: bool,
    ) -> bool:
        if expr == "*":
            return True

        step = 1
        base = expr
        if "/" in expr:
            base, step_expr = expr.split("/", 1)
            try:
                step = int(step_expr)
            except ValueError:
                return False
            if step <= 0:
                return False

        if base == "*":
            return (value - minimum) % step == 0

        start, end = self._parse_range(base, minimum, maximum, sunday_alias)
        if start is None or end is None or not start <= value <= end:
            return False
        return (value - start) % step == 0

    def _parse_range(
        self,
        expr: str,
        minimum: int,
        maximum: int,
        sunday_alias: bool,
    ) -> tuple[int | None, int | None]:
        if "-" in expr:
            start_expr, end_expr = expr.split("-", 1)
            start = self._parse_int(start_expr, minimum, maximum, sunday_alias)
            end = self._parse_int(end_expr, minimum, maximum, sunday_alias)
            if start is None or end is None:
                return None, None
            return start, end

        parsed = self._parse_int(expr, minimum, maximum, sunday_alias)
        if parsed is None:
            return None, None
        return parsed, parsed

    def _parse_int(
        self,
        expr: str,
        minimum: int,
        maximum: int,
        sunday_alias: bool,
    ) -> int | None:
        try:
            parsed = int(expr)
        except ValueError:
            return None
        if sunday_alias and parsed == 7:
            parsed = 0
        if minimum <= parsed <= maximum:
            return parsed
        return None
