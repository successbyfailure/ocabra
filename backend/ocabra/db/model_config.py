import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ocabra.database import Base


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[str] = mapped_column(String(512), unique=True, nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(512))
    backend_type: Mapped[str] = mapped_column(String(64), nullable=False)
    load_policy: Mapped[str] = mapped_column(String(32), nullable=False, default="on_demand")
    auto_reload: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    preferred_gpu: Mapped[int | None] = mapped_column(Integer)
    extra_config: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class EvictionSchedule(Base):
    __tablename__ = "eviction_schedules"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # NULL = schedule global (aplica a todos los modelos)
    model_id: Mapped[str | None] = mapped_column(String(512), index=True)
    cron_expr: Mapped[str] = mapped_column(String(128), nullable=False)
    # evict_warm | evict_all | reload
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


GLOBAL_SCHEDULE_START_ACTION = "evict_all"
GLOBAL_SCHEDULE_END_ACTION = "reload"


def _normalize_days(days: Sequence[int]) -> list[int]:
    normalized = sorted({int(day) for day in days if 0 <= int(day) <= 6})
    if not normalized:
        raise ValueError("globalSchedules.days must contain at least one day between 0 and 6")
    return normalized


def _parse_time(value: str) -> tuple[int, int]:
    parts = str(value or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time value: {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"Invalid time value: {value!r}")
    return hour, minute


def _format_time(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def _days_to_cron_field(days: Sequence[int]) -> str:
    return ",".join(str(day) for day in _normalize_days(days))


def _shift_days(days: Sequence[int], offset: int) -> list[int]:
    return sorted({(int(day) + offset) % 7 for day in _normalize_days(days)})


def global_schedule_payload_to_rows(
    schedule_id: str,
    days: Sequence[int],
    start: str,
    end: str,
    enabled: bool,
) -> list[EvictionSchedule]:
    start_hour, start_minute = _parse_time(start)
    end_hour, end_minute = _parse_time(end)
    normalized_days = _normalize_days(days)
    start_total_minutes = start_hour * 60 + start_minute
    end_total_minutes = end_hour * 60 + end_minute
    end_days = normalized_days if end_total_minutes > start_total_minutes else _shift_days(normalized_days, 1)

    label = str(schedule_id).strip()
    if not label:
        raise ValueError("globalSchedules.id must be a non-empty string")

    return [
        EvictionSchedule(
            model_id=None,
            cron_expr=f"{start_minute} {start_hour} * * {_days_to_cron_field(normalized_days)}",
            action=GLOBAL_SCHEDULE_START_ACTION,
            label=label,
            enabled=bool(enabled),
        ),
        EvictionSchedule(
            model_id=None,
            cron_expr=f"{end_minute} {end_hour} * * {_days_to_cron_field(end_days)}",
            action=GLOBAL_SCHEDULE_END_ACTION,
            label=label,
            enabled=bool(enabled),
        ),
    ]


def _parse_day_field(field: str) -> list[int]:
    field = str(field or "").strip()
    if not field:
        raise ValueError("Empty cron weekday field")
    if field == "*":
        return list(range(7))

    days: set[int] = set()
    for part in field.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_expr, end_expr = token.split("-", 1)
            start = int(start_expr)
            end = int(end_expr)
            if start > end:
                raise ValueError(f"Unsupported wrapped cron weekday range: {field!r}")
            days.update(range(start, end + 1))
            continue
        days.add(int(token))
    return sorted(day % 7 for day in days)


def _cron_to_time_and_days(cron_expr: str) -> tuple[list[int], str]:
    minute, hour, _day, _month, weekday = str(cron_expr or "").split()
    return _parse_day_field(weekday), _format_time(int(hour), int(minute))


def _group_global_schedule_rows(rows: Sequence[EvictionSchedule]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        label = str(row.label or row.id)
        bucket = grouped.setdefault(label, {"enabled": True})
        bucket["enabled"] = bool(bucket["enabled"]) and bool(row.enabled)
        bucket.setdefault("label", label)
        if row.action == GLOBAL_SCHEDULE_START_ACTION:
            days, start = _cron_to_time_and_days(row.cron_expr)
            bucket["days"] = days
            bucket["start"] = start
        elif row.action == GLOBAL_SCHEDULE_END_ACTION:
            _days, end = _cron_to_time_and_days(row.cron_expr)
            bucket["end"] = end

    payloads: list[dict[str, object]] = []
    for label, bucket in grouped.items():
        days = bucket.get("days")
        start = bucket.get("start")
        end = bucket.get("end")
        if not isinstance(days, list) or not isinstance(start, str) or not isinstance(end, str):
            continue
        payloads.append(
            {
                "id": label,
                "days": days,
                "start": start,
                "end": end,
                "enabled": bool(bucket.get("enabled", True)),
            }
        )

    return sorted(payloads, key=lambda item: (str(item["start"]), str(item["id"])))


async def get_global_schedule_rows(session) -> list[EvictionSchedule]:
    import sqlalchemy as sa

    result = await session.execute(
        sa.select(EvictionSchedule).where(EvictionSchedule.model_id.is_(None))
    )
    return list(result.scalars().all())


async def replace_global_schedules(session, payloads: Sequence[dict[str, object]]) -> None:
    existing_rows = await get_global_schedule_rows(session)
    for row in existing_rows:
        await session.delete(row)

    new_rows: list[EvictionSchedule] = []
    for payload in payloads:
        new_rows.extend(
            global_schedule_payload_to_rows(
                schedule_id=str(payload["id"]),
                days=payload["days"],  # type: ignore[arg-type]
                start=str(payload["start"]),
                end=str(payload["end"]),
                enabled=bool(payload.get("enabled", True)),
            )
        )

    session.add_all(new_rows)


def global_schedule_rows_to_payload(rows: Sequence[EvictionSchedule]) -> list[dict[str, object]]:
    return _group_global_schedule_rows(rows)
