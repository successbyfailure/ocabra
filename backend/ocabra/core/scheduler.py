import structlog

from ocabra.config import settings

logger = structlog.get_logger(__name__)


class InsufficientVRAMError(Exception):
    pass


class GPUScheduler:
    def __init__(self, gpu_manager) -> None:
        self._gpu_manager = gpu_manager

    async def find_gpu_for_model(
        self,
        vram_needed_mb: int,
        preferred_gpu: int | None = None,
    ) -> list[int]:
        preferred = preferred_gpu if preferred_gpu is not None else settings.default_gpu_index
        states = await self._gpu_manager.get_all_states()
        gpu_indices = list(range(len(states)))

        # Try preferred first, then others
        ordered = [preferred] + [i for i in gpu_indices if i != preferred]
        for gpu_idx in ordered:
            free = await self._gpu_manager.get_free_vram(gpu_idx)
            if free >= vram_needed_mb:
                logger.info(
                    "gpu_assigned",
                    gpu=gpu_idx,
                    vram_needed_mb=vram_needed_mb,
                    free_mb=free,
                )
                return [gpu_idx]

        # Try tensor parallelism across all GPUs
        total_free = sum(
            await self._gpu_manager.get_free_vram(i) for i in gpu_indices
        )
        if total_free >= vram_needed_mb:
            logger.info(
                "tensor_parallel_assigned",
                gpus=gpu_indices,
                vram_needed_mb=vram_needed_mb,
                total_free_mb=total_free,
            )
            return gpu_indices

        raise InsufficientVRAMError(
            f"Need {vram_needed_mb} MB VRAM, only {total_free} MB available across all GPUs"
        )

    async def get_eviction_candidates(
        self,
        vram_needed_mb: int,
        target_gpu: int,
    ) -> list[str]:
        """
        Returns model_ids sorted by eviction priority (first = evict first).
        on_demand idle > on_demand recent > warm > pin
        """
        import sqlalchemy as sa

        from ocabra.database import AsyncSessionLocal
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
        """Called by APScheduler. Unloads warm/pin models during active eviction windows."""
        import sqlalchemy as sa
        from datetime import datetime

        from ocabra.database import AsyncSessionLocal
        from ocabra.db.model_config import EvictionSchedule

        now = datetime.now()
        # Simple implementation: check if current time falls in any enabled schedule
        # Full cron matching would use croniter library — placeholder for now
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                sa.select(EvictionSchedule).where(EvictionSchedule.enabled.is_(True))
            )
            schedules = result.scalars().all()

        logger.debug("schedule_eviction_check", active_schedules=len(schedules))

    async def check_schedule_reloads(self) -> None:
        """Called by APScheduler. Reloads pin/warm models after eviction window ends."""
        logger.debug("schedule_reload_check")
