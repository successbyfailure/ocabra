"""Persisted GPU power-cap + persistence-mode settings.

NVML clears the power limit and persistence mode on driver reload / host
reboot. To make the user's choices durable we record them in this table
keyed by NVML UUID (stable across reboots, hardware reorder, etc.) and
re-apply on API startup via :func:`reapply_persisted_settings`.

Channels: power limits go through the existing ``gpu:set_power_limit``
Redis pubsub so the privileged ``hw-monitor`` sidecar performs the NVML
call. Persistence mode is set inline via pynvml when possible (mostly a
no-op inside the unprivileged api container, but harmless).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import sqlalchemy as sa
import structlog
from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from ocabra.database import Base

logger = structlog.get_logger(__name__)


class GpuPowerSetting(Base):
    __tablename__ = "gpu_power_settings"

    gpu_uuid: Mapped[str] = mapped_column(String(80), primary_key=True)
    power_limit_w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    persistence_mode: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    last_known_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_known_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=sa.func.now(),
    )


def read_gpu_uuid(index: int) -> str | None:
    """Return the NVML UUID for *index*, or ``None`` if unavailable."""
    try:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode()
        return uuid
    except Exception:  # noqa: BLE001
        return None


def read_gpu_name(index: int) -> str | None:
    """Return the NVML product name for *index*, or ``None`` if unavailable."""
    try:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        return name
    except Exception:  # noqa: BLE001
        return None


async def upsert_setting(
    session: AsyncSession,
    *,
    gpu_uuid: str,
    power_limit_w: int | None = None,
    persistence_mode: bool | None = None,
    last_known_index: int | None = None,
    last_known_name: str | None = None,
) -> None:
    """Upsert a power setting row.

    ``None`` values for ``power_limit_w`` / ``persistence_mode`` mean *don't
    touch this column*. This way callers can update only the field they're
    changing without clobbering the other.
    """
    update_payload: dict[str, object] = {"updated_at": datetime.now(UTC)}
    if power_limit_w is not None:
        update_payload["power_limit_w"] = power_limit_w
    if persistence_mode is not None:
        update_payload["persistence_mode"] = persistence_mode
    if last_known_index is not None:
        update_payload["last_known_index"] = last_known_index
    if last_known_name is not None:
        update_payload["last_known_name"] = last_known_name

    insert_payload = {
        "gpu_uuid": gpu_uuid,
        "power_limit_w": power_limit_w,
        "persistence_mode": persistence_mode,
        "last_known_index": last_known_index,
        "last_known_name": last_known_name,
    }
    stmt = pg_insert(GpuPowerSetting).values(**insert_payload)
    stmt = stmt.on_conflict_do_update(
        index_elements=[GpuPowerSetting.gpu_uuid],
        set_=update_payload,
    )
    await session.execute(stmt)


async def load_all(session: AsyncSession) -> list[GpuPowerSetting]:
    result = await session.execute(sa.select(GpuPowerSetting))
    return list(result.scalars().all())


async def _wait_for_hw_monitor(timeout_s: float = 30.0, interval_s: float = 1.0) -> bool:
    """Block until ``hw-monitor:heartbeat`` is set in Redis, or *timeout_s*."""
    from ocabra.redis_client import get_key

    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        try:
            heartbeat = await get_key("hw-monitor:heartbeat")
        except Exception:  # noqa: BLE001
            heartbeat = None
        if heartbeat:
            return True
        if asyncio.get_event_loop().time() >= deadline:
            return False
        await asyncio.sleep(interval_s)


async def reapply_persisted_settings(
    session_factory,
    *,
    wait_for_hw_monitor: bool = True,
) -> dict[str, int]:
    """Read the saved settings and re-apply them to the live GPUs.

    Returns a counters dict ``{"applied", "skipped_no_match", "errors"}`` so
    the caller can log a single summary line.

    The function never raises — it logs warnings for unrecoverable cases
    (no NVML, no hw-monitor heartbeat, unknown UUID) and continues. Use
    ``session_factory`` rather than a session because we want to open the
    session inside our own scope (we may be called from a background task
    spawned by lifespan).
    """
    summary = {"applied": 0, "skipped_no_match": 0, "errors": 0}

    try:
        import pynvml
    except ImportError:
        logger.warning("gpu_power_reapply_skipped_no_pynvml")
        return summary

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        logger.warning("gpu_power_reapply_skipped_no_nvml", error=str(exc))
        return summary

    try:
        count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError as exc:
        logger.warning("gpu_power_reapply_skipped_count_failed", error=str(exc))
        return summary

    # Build the uuid → index lookup.
    uuid_to_index: dict[str, int] = {}
    for i in range(count):
        uuid = read_gpu_uuid(i)
        if uuid:
            uuid_to_index[uuid] = i

    async with session_factory() as session:
        rows = await load_all(session)

    if not rows:
        return summary

    # Power limits go through hw-monitor (privileged). Wait for its
    # heartbeat so the publish isn't dropped before any subscriber is ready.
    if wait_for_hw_monitor and any(r.power_limit_w for r in rows):
        ready = await _wait_for_hw_monitor()
        if not ready:
            logger.warning(
                "gpu_power_reapply_no_hw_monitor_heartbeat",
                detail="hw-monitor did not appear within timeout — caps NOT reapplied",
            )
            summary["errors"] += sum(1 for r in rows if r.power_limit_w)
            return summary

    from ocabra.redis_client import publish

    for row in rows:
        idx = uuid_to_index.get(row.gpu_uuid)
        if idx is None:
            logger.warning(
                "gpu_power_reapply_skipped_unknown_uuid",
                gpu_uuid=row.gpu_uuid,
                last_known_index=row.last_known_index,
                last_known_name=row.last_known_name,
                detail="GPU not found in current hardware inventory",
            )
            summary["skipped_no_match"] += 1
            continue

        try:
            if row.power_limit_w is not None:
                await publish(
                    "gpu:set_power_limit",
                    {"gpu_index": idx, "limit_w": int(row.power_limit_w)},
                )

            if row.persistence_mode is not None:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    mode = (
                        pynvml.NVML_FEATURE_ENABLED
                        if row.persistence_mode
                        else pynvml.NVML_FEATURE_DISABLED
                    )
                    pynvml.nvmlDeviceSetPersistenceMode(handle, mode)
                except pynvml.NVMLError as exc:
                    # Often "Insufficient Permissions" inside the api
                    # container — log but don't count as fatal since the
                    # power limit (the important one) goes through hw-monitor.
                    logger.warning(
                        "gpu_power_reapply_persistence_mode_failed",
                        gpu_uuid=row.gpu_uuid,
                        gpu_index=idx,
                        error=str(exc),
                    )

            logger.info(
                "gpu_power_setting_reapplied",
                gpu_uuid=row.gpu_uuid,
                gpu_index=idx,
                power_limit_w=row.power_limit_w,
                persistence_mode=row.persistence_mode,
            )
            summary["applied"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "gpu_power_reapply_failed",
                gpu_uuid=row.gpu_uuid,
                gpu_index=idx,
                error=str(exc),
            )
            summary["errors"] += 1

    # Update last_known_index / last_known_name now that we resolved them.
    async with session_factory() as session:
        for row in rows:
            idx = uuid_to_index.get(row.gpu_uuid)
            if idx is None:
                continue
            await upsert_setting(
                session,
                gpu_uuid=row.gpu_uuid,
                last_known_index=idx,
                last_known_name=read_gpu_name(idx),
            )
        await session.commit()

    return summary
